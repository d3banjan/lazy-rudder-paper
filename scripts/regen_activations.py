#!/usr/bin/env python3
"""regen_activations.py — regenerate the three activation-tensor archives.

Two analysis scripts (angular_fourier_delta_prime.py, two_point_correlator_delta.py)
consume pre-recorded residual-stream activation tensors:

    RESULTS_DIR/_orbit/410m_base_activations.pt
    RESULTS_DIR/_orbit/410m_sft_activations.pt
    RESULTS_DIR/_orbit/410m_dpo_activations.pt

These are NOT committed to the repo (they are ~100–500 MB each).  This script
regenerates them end-to-end from publicly available HuggingFace checkpoints:

    EleutherAI/pythia-410m              (base)
    lomahony/pythia-410m-helpful-sft    (SFT)
    lomahony/pythia-410m-helpful-dpo    (DPO)

Usage (from the papers/lazy-rudder directory, or repo root):
    uv run scripts/regen_activations.py

Environment / path override:
    LAZY_RUDDER_MODELS_DIR   — where to cache downloaded base models
    LAZY_RUDDER_RESULTS_DIR  — where to write the _orbit/ output tree
    (fallback: see scripts/_paths.py resolution order)

Hardware requirements:
    GPU with >=12 GB VRAM (RTX 3060 12GB is sufficient for Pythia-410M fp16)
    ~3–4 GB disk space for model weights + activation archives

Expected runtime (RTX 3060):
    ~5 min per model variant (base / sft / dpo) = ~15–20 min total
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the resolution logic used by analysis scripts
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from _paths import MODELS_DIR, RESULTS_DIR  # noqa: E402

ORBIT_DIR = RESULTS_DIR / "_orbit"

# ---------------------------------------------------------------------------
# Model specs: (local_subdir, hf_repo_id, output_pt_name)
# ---------------------------------------------------------------------------
MODEL_SPECS = [
    (
        "pythia-410m",
        "EleutherAI/pythia-410m",
        "410m_base_activations.pt",
    ),
    (
        "pythia-410m-sft",
        "lomahony/pythia-410m-helpful-sft",
        "410m_sft_activations.pt",
    ),
    (
        "pythia-410m-dpo",
        "lomahony/pythia-410m-helpful-dpo",
        "410m_dpo_activations.pt",
    ),
]

# Prompt data lives in cross-check/trained-model-battery/data/ in the
# original dev layout; resolve relative to this repo's root.
_PAPER_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT  = _PAPER_DIR.parent   # lean-mining/
_BATTERY_DATA = _REPO_ROOT / "cross-check" / "trained-model-battery" / "data"

PROMPT_SETS = {
    "entity_property":  _BATTERY_DATA / "entity_property_large.jsonl",
    "category_behavior": _BATTERY_DATA / "category_behavior_large.jsonl",
    "repeated_token":   _BATTERY_DATA / "repeated_token_large.jsonl",
}

N_LAYERS    = 24     # Pythia-410M
UNIT_TYPES  = ["residual", "ffn_post", "block_output"]
MAX_TOKENS  = 128
BATCH_SIZE  = 8
SEED        = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_model(hf_repo: str, local_dir: Path) -> None:
    """Download model weights from HuggingFace (idempotent via snapshot_download)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required.  Install via: uv add huggingface_hub"
        ) from exc

    print(f"  Fetching {hf_repo} -> {local_dir} (idempotent) ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_repo, local_dir=str(local_dir))
    print(f"  Done: {local_dir}")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_activations(model_dir: Path) -> dict:
    """Run forward passes and collect residual / ffn_post / block_output per layer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required.  Install via: uv sync"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: no GPU found; forward passes will be very slow on CPU.")

    print(f"  Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = model.to(device)

    # ---- hook registration ----
    captured: dict[str, list] = {u: [[] for _ in range(N_LAYERS)] for u in UNIT_TYPES}

    hooks = []
    for idx, block in enumerate(model.gpt_neox.layers):
        def _make_residual_hook(li: int):
            def _hook(_, inputs, __):
                h = inputs[0].detach().float().cpu()  # (B, T, d)
                captured["residual"][li].append(h.mean(dim=1))  # mean-pool over tokens -> (B, d)
            return _hook

        def _make_ffn_post_hook(li: int):
            def _hook(_, __, output):
                h = output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu()
                captured["ffn_post"][li].append(h.mean(dim=1))
            return _hook

        def _make_block_output_hook(li: int):
            def _hook(_, __, output):
                h = output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu()
                captured["block_output"][li].append(h.mean(dim=1))
            return _hook

        hooks.append(block.input_layernorm.register_forward_hook(_make_residual_hook(idx)))
        hooks.append(block.mlp.register_forward_hook(_make_ffn_post_hook(idx)))
        hooks.append(block.register_forward_hook(_make_block_output_hook(idx)))

    # ---- forward passes per prompt set ----
    prompt_sets_out: dict = {}
    for set_name, jsonl_path in PROMPT_SETS.items():
        if not jsonl_path.exists():
            print(f"  WARNING: prompt set {set_name} not found at {jsonl_path} — skipping.")
            continue
        rows = _read_jsonl(jsonl_path)
        print(f"  Set '{set_name}': {len(rows)} examples", flush=True)

        # reset captured buffers
        for u in UNIT_TYPES:
            for li in range(N_LAYERS):
                captured[u][li].clear()

        prompts = [r.get("prompt", r.get("text", "")) for r in rows]
        for start in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(device)
            with torch.no_grad():
                model(**enc)

        # collate into tensors
        import torch as _torch
        units_out: dict[str, list] = {}
        for u in UNIT_TYPES:
            layer_tensors = []
            for li in range(N_LAYERS):
                layer_tensors.append(_torch.cat(captured[u][li], dim=0))  # (N, d)
            units_out[u] = layer_tensors

        prompt_sets_out[set_name] = {"rows": rows, "units": units_out}

    for h in hooks:
        h.remove()
    del model
    if device == "cuda":
        import torch as _torch
        _torch.cuda.empty_cache()

    return prompt_sets_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import torch
    print("\n=== regen_activations.py ===")
    print(f"MODELS_DIR : {MODELS_DIR}")
    print(f"ORBIT_DIR  : {ORBIT_DIR}\n")

    ORBIT_DIR.mkdir(parents=True, exist_ok=True)

    for local_subdir, hf_repo, out_name in MODEL_SPECS:
        out_path = ORBIT_DIR / out_name
        print(f"\n--- {out_name} ---")

        if out_path.exists():
            print(f"  Already exists, skipping: {out_path}")
            continue

        model_dir = MODELS_DIR / local_subdir

        # 1. Download weights if missing
        if not model_dir.exists() or not any(model_dir.iterdir()):
            print(f"  Model not found locally; downloading from HF ...")
            _download_model(hf_repo, model_dir)
        else:
            print(f"  Using cached model weights at {model_dir}")

        # 2. Collect activations
        print(f"  Collecting activations (GPU forward passes) ...")
        prompt_sets = _collect_activations(model_dir)

        # 3. Save archive
        torch.save(prompt_sets, str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    print("\nDone. Re-run make analysis to compute delta / delta-prime results.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
