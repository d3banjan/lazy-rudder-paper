#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.40",
#   "peft>=0.10",
#   "datasets>=2.18",
#   "safetensors>=0.4",
# ]
# ///
"""cross_probe_score.py — T2.1 extension: teacher-forced reward-margin scorer.

Scores (prompt, chosen, rejected) pairs from a held-out probe dataset against
a given model, computing DPO reward margins (β=0.1).

Model specs:
  pythia_lora:<seed>:<run>   — loads Pythia base + LoRA adapter from snapshot
  qwen_fullweight            — loads lewtun/qwen2-1.5B-ultrafeedback-online-dpo
                               as a full-weight model (vs Qwen2-1.5B base)

Probe datasets:
  hh    — Anthropic/hh-rlhf test split
  uf    — HuggingFaceH4/ultrafeedback_binarized test_prefs split

Output:
  <out>/<model_spec>__<probe>.jsonl   — per-example records
  <out>/state_<model_spec>__<probe>.json  — resume state

Usage:
  uv run python scripts/cross_probe_score.py \\
      --model-spec pythia_lora:42:v2 --probe hh --out results/cross_probe --n 500
  uv run python scripts/cross_probe_score.py \\
      --model-spec qwen_fullweight --probe uf --out results/cross_probe --n 500
  uv run python scripts/cross_probe_score.py \\
      --model-spec qwen_fullweight --probe hh --out results/cross_probe --n 500
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Patch transformers torch.load safety check (workaround inherited from T1.2) ──
try:
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

log = logging.getLogger(__name__)

# ── Global constants ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR  = SCRIPT_DIR.parent

DPO_BETA         = 0.1
N_DEFAULT        = 500
CHECKPOINT_EVERY = 50
MAX_LENGTH       = 512

# ── Snapshot / cache paths (match behavior_geometry_link.py conventions) ──────

SNAP = Path(
    "/home/debanjan/.cache/huggingface/hub"
    "/models--d3banjan--lazy-rudder-checkpoints"
    "/snapshots/8ac4c033fbdd21eff7cc38bfbe6acbfcf53cfeec"
)

MODELS_BASE = Path(
    "/home/debanjan/Code/Research/lean-mining"
    "/cross-check/trained-model-battery/models"
)

HF_CACHE = Path("/home/debanjan/.cache/huggingface/hub")

# Qwen model HF cache paths (confirmed in PROVENANCE.md)
QWEN_BASE_SNAPSHOT = (
    HF_CACHE / "models--Qwen--Qwen2-1.5B"
    / "snapshots" / "8a16abf2848eda07cc5253dec660bf1ce007ad7a"
)
QWEN_DPO_SNAPSHOT = (
    HF_CACHE / "models--lewtun--qwen2-1.5B-ultrafeedback-online-dpo"
    / "snapshots" / "6750e4e383493166cdd8f47a6ebb7a3b79c0a7c6"
)

# Pythia LoRA adapter specs — keyed by (seed, run_tag) matching T1.2 MODEL_SPECS
PYTHIA_LORA_SPECS: dict[tuple[int, str], dict] = {
    (42, "v2"): [
        {
            "model_size": "70m",
            "base_dir":   MODELS_BASE / "pythia-70m",
            "adapter_dir": SNAP / "_leak_70m" / "v2" / "checkpoints" / "checkpoint-800",
        },
        {
            "model_size": "160m",
            "base_dir":   MODELS_BASE / "pythia-160m",
            "adapter_dir": SNAP / "_leak_160m" / "v2" / "checkpoints" / "checkpoint-800",
        },
        {
            "model_size": "410m",
            "base_dir":   MODELS_BASE / "pythia-410m",
            "adapter_dir": SNAP / "_leak" / "v2" / "checkpoints" / "checkpoint-800",
        },
        {
            "model_size": "1b",
            "base_dir":   MODELS_BASE / "pythia-1b",
            "adapter_dir": SNAP / "_leak_1b" / "v2" / "checkpoints" / "checkpoint-800",
        },
        {
            "model_size": "1b_s117",
            "base_dir":   MODELS_BASE / "pythia-1b",
            "adapter_dir": SNAP / "_leak_1b_seed117" / "v2" / "checkpoints" / "checkpoint-800",
        },
    ],
}

BATCH_SIZES = {"70m": 16, "160m": 16, "410m": 8, "1b": 4, "1b_s117": 4, "qwen": 4}


# ── Dataset normalization ──────────────────────────────────────────────────────

def normalize_pair(row: dict, dataset_name: str) -> tuple[str, str, str] | None:
    """Extract (prompt, chosen_response, rejected_response) from a dataset row.

    Handles two formats:
    - hh-rlhf: chosen/rejected are full-conversation strings with '\\n\\nAssistant:' separator
    - ultrafeedback_binarized: chosen/rejected are list-of-turn-dicts
    """
    if dataset_name == "hh":
        chosen_text   = row.get("chosen", "")
        rejected_text = row.get("rejected", "")
        sep = "\n\nAssistant:"

        if sep in chosen_text:
            parts    = chosen_text.rsplit(sep, 1)
            prompt   = parts[0] + sep
            chosen_r = parts[1].strip()
        else:
            prompt   = chosen_text[:256]
            chosen_r = chosen_text[256:]

        if sep in rejected_text:
            rejected_r = rejected_text.rsplit(sep, 1)[1].strip()
        else:
            rejected_r = rejected_text[256:]

        if len(chosen_r) < 10 or len(rejected_r) < 10:
            return None
        return prompt, chosen_r, rejected_r

    elif dataset_name == "uf":
        # ultrafeedback_binarized: chosen and rejected are lists of {"role": ..., "content": ...}
        chosen_turns   = row.get("chosen", [])
        rejected_turns = row.get("rejected", [])

        if not chosen_turns or not rejected_turns:
            return None

        # Extract prompt: all turns up to (not including) last assistant turn in chosen
        # The format is alternating user/assistant; last turn is the assistant response.
        def turns_to_text(turns: list) -> str:
            parts = []
            for t in turns:
                role    = t.get("role", "user")
                content = t.get("content", "")
                parts.append(f"{role.capitalize()}: {content}")
            return "\n\n".join(parts)

        # Prompt = everything except the last assistant turn
        chosen_prompt_turns = chosen_turns[:-1]
        chosen_r_turn       = chosen_turns[-1]
        rejected_r_turn     = rejected_turns[-1]

        if chosen_r_turn.get("role") != "assistant":
            # Malformed; skip
            return None
        if rejected_r_turn.get("role") != "assistant":
            return None

        prompt     = turns_to_text(chosen_prompt_turns) + "\n\nAssistant:"
        chosen_r   = chosen_r_turn.get("content", "").strip()
        rejected_r = rejected_r_turn.get("content", "").strip()

        if len(chosen_r) < 10 or len(rejected_r) < 10:
            return None
        return prompt, chosen_r, rejected_r

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name!r}. Must be 'hh' or 'uf'.")


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_probe_examples(probe: str, n: int) -> list[tuple[int, str, str, str, str]]:
    """Load probe dataset, return list of (idx, prompt_hash, prompt, chosen, rejected)."""
    from datasets import load_dataset  # noqa: PLC0415

    if probe == "hh":
        log.info(f"Loading Anthropic/hh-rlhf test split (target={n})...")
        ds = load_dataset("Anthropic/hh-rlhf", split="test")
    elif probe == "uf":
        log.info(f"Loading HuggingFaceH4/ultrafeedback_binarized test_prefs split (target={n})...")
        ds = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized",
            split="test_prefs",
            trust_remote_code=False,
        )
    else:
        raise ValueError(f"Unknown probe: {probe!r}")

    examples = []
    for i, row in enumerate(ds):
        if len(examples) >= n:
            break
        parsed = normalize_pair(row, probe)
        if parsed is None:
            continue
        prompt, chosen_r, rejected_r = parsed
        prompt_hash = hashlib.sha1(prompt.encode()).hexdigest()[:10]
        examples.append((i, prompt_hash, prompt, chosen_r, rejected_r))

    log.info(f"  loaded {len(examples)} usable examples")
    return examples


# ── Log-probability computation (verbatim from behavior_geometry_link.py) ─────

def compute_logp(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = MAX_LENGTH,
    device: str = "cuda",
) -> tuple[float, int]:
    """Compute log P(response | prompt). Returns (logp, n_tokens)."""
    import torch  # noqa: PLC0415

    full_text = prompt + response
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)

    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_len = prompt_enc["input_ids"].shape[1]
    n_tokens   = input_ids.shape[1] - prompt_len

    if n_tokens <= 0:
        return float("nan"), 0

    with torch.inference_mode():
        out    = model(input_ids=input_ids)
        logits = out.logits  # [1, seq_len, vocab]

    shift_logits = logits[0, prompt_len - 1 : input_ids.shape[1] - 1, :]
    shift_labels = input_ids[0, prompt_len:]

    log_probs  = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs[torch.arange(n_tokens), shift_labels]
    return token_logps.sum().item(), n_tokens


# ── State management ──────────────────────────────────────────────────────────

def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"version": 1, "last_example_idx": -1, "completed": False}


def save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)
    try:
        with open(state_path) as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass


# ── Single-model scorer ───────────────────────────────────────────────────────

def score_model_pair(
    model_label: str,
    base_model,
    dpo_model,
    tokenizer,
    examples: list[tuple[int, str, str, str, str]],
    resume_from: int,
    jsonl_path: Path,
    state_path: Path,
    device: str,
) -> None:
    """Score all examples with pre-loaded model pair, append to jsonl_path."""
    import torch  # noqa: PLC0415

    state = load_state(state_path)
    fh    = open(jsonl_path, "a")

    try:
        for i, (ex_idx, prompt_hash, prompt, chosen_r, rejected_r) in enumerate(examples):
            if i <= resume_from:
                continue

            try:
                base_chosen_lp,  base_chosen_n  = compute_logp(base_model, tokenizer, prompt, chosen_r,   device=device)
                base_rejected_lp, base_rejected_n = compute_logp(base_model, tokenizer, prompt, rejected_r, device=device)
                dpo_chosen_lp,   dpo_chosen_n   = compute_logp(dpo_model,  tokenizer, prompt, chosen_r,   device=device)
                dpo_rejected_lp, dpo_rejected_n  = compute_logp(dpo_model,  tokenizer, prompt, rejected_r, device=device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning(f"[{model_label}] OOM at example {i}; skipping.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

            record = {
                "prompt_id":          ex_idx,
                "prompt_text_hash":   prompt_hash,
                "logp_chosen_dpo":    dpo_chosen_lp,
                "logp_rejected_dpo":  dpo_rejected_lp,
                "logp_chosen_base":   base_chosen_lp,
                "logp_rejected_base": base_rejected_lp,
                "n_tokens_chosen":    dpo_chosen_n,
                "n_tokens_rejected":  dpo_rejected_n,
            }
            fh.write(json.dumps(record) + "\n")

            if (i + 1) % CHECKPOINT_EVERY == 0:
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
                state["last_example_idx"] = i
                save_state(state_path, state)
                log.info(f"[{model_label}] checkpoint at {i+1}/{len(examples)}")

        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
        state["last_example_idx"] = len(examples) - 1
        state["completed"]        = True
        save_state(state_path, state)

    except KeyboardInterrupt:
        fh.flush()
        state["last_example_idx"] = i if "i" in dir() else resume_from
        save_state(state_path, state)
        fh.close()
        raise
    finally:
        fh.close()


# ── LoRA chain runner ─────────────────────────────────────────────────────────

def run_pythia_lora_chain(
    seed: int,
    run_tag: str,
    probe: str,
    out_dir: Path,
    examples: list,
) -> None:
    """Score all checkpoints in the Pythia LoRA chain against the given probe."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415

    spec_list = PYTHIA_LORA_SPECS.get((seed, run_tag))
    if spec_list is None:
        raise ValueError(f"No Pythia LoRA spec for seed={seed}, run_tag={run_tag}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for spec in spec_list:
        model_size  = spec["model_size"]
        base_dir    = spec["base_dir"]
        adapter_dir = spec["adapter_dir"]
        model_label = f"pythia_{model_size}_s{seed}_{run_tag}"

        job_key    = f"pythia_lora_{seed}_{run_tag}_{model_size}__{probe}"
        jsonl_path = out_dir / f"{job_key}.jsonl"
        state_path = out_dir / f"state_{job_key}.json"

        state = load_state(state_path)
        if state.get("completed"):
            log.info(f"[{model_label}/{probe}] Already completed — skipping.")
            continue

        resume_from = state.get("last_example_idx", -1)
        log.info(f"[{model_label}/{probe}] Resuming from example {resume_from + 1}.")

        log.info(f"[{model_label}] Loading tokenizer from {base_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info(f"[{model_label}] Loading base model (fp16)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_dir), torch_dtype=torch.float16,
            low_cpu_mem_usage=True, local_files_only=True,
        ).to(device)
        base_model.eval()

        log.info(f"[{model_label}] Loading DPO adapter...")
        dpo_model = AutoModelForCausalLM.from_pretrained(
            str(base_dir), torch_dtype=torch.float16,
            low_cpu_mem_usage=True, local_files_only=True,
        )
        dpo_model = PeftModel.from_pretrained(
            dpo_model, str(adapter_dir), local_files_only=True,
        ).to(device)
        dpo_model.eval()

        score_model_pair(
            model_label, base_model, dpo_model, tokenizer,
            examples, resume_from, jsonl_path, state_path, device,
        )

        log.info(f"[{model_label}/{probe}] Done. Freeing GPU memory.")
        del base_model, dpo_model
        torch.cuda.empty_cache()
        gc.collect()


# ── Full-weight Qwen runner ───────────────────────────────────────────────────

def run_qwen_fullweight(
    probe: str,
    out_dir: Path,
    examples: list,
) -> None:
    """Score Qwen2-1.5B DPO (full-weight) against the given probe."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_label = f"qwen_fullweight__{probe}"
    job_key     = model_label
    jsonl_path  = out_dir / f"{job_key}.jsonl"
    state_path  = out_dir / f"state_{job_key}.json"

    state = load_state(state_path)
    if state.get("completed"):
        log.info(f"[{model_label}] Already completed — skipping.")
        return

    resume_from = state.get("last_example_idx", -1)
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[{model_label}] device={device}, resume_from={resume_from + 1}")

    # Qwen cached snapshots are weights-only (no tokenizer.json locally).
    # Load tokenizer by HF repo ID (will use cache if available, else download ~1MB).
    log.info("[qwen] Loading tokenizer from HF Hub (Qwen/Qwen2-1.5B)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("[qwen] Loading Qwen2-1.5B base (fp16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_BASE_SNAPSHOT),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    base_model.eval()

    log.info("[qwen] Loading lewtun DPO model (fp16)...")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_DPO_SNAPSHOT),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    dpo_model.eval()

    score_model_pair(
        model_label, base_model, dpo_model, tokenizer,
        examples, resume_from, jsonl_path, state_path, device,
    )

    log.info("[qwen] Done. Freeing GPU memory.")
    del base_model, dpo_model
    torch.cuda.empty_cache()
    gc.collect()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_model_spec(spec_str: str) -> dict:
    """Parse --model-spec string.
    Forms:
      pythia_lora:<seed>:<run_tag>   e.g. pythia_lora:42:v2
      qwen_fullweight
    """
    if spec_str == "qwen_fullweight":
        return {"kind": "qwen_fullweight"}
    if spec_str.startswith("pythia_lora:"):
        parts = spec_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected pythia_lora:<seed>:<run_tag>, got: {spec_str!r}")
        _, seed_str, run_tag = parts
        return {"kind": "pythia_lora", "seed": int(seed_str), "run_tag": run_tag}
    raise ValueError(f"Unknown model spec: {spec_str!r}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log.info(f"=== cross_probe_score.py started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-spec", required=True,
        help="Model spec string: 'pythia_lora:<seed>:<run_tag>' or 'qwen_fullweight'",
    )
    parser.add_argument(
        "--probe", required=True, choices=["hh", "uf"],
        help="Probe dataset: 'hh' (Anthropic/hh-rlhf) or 'uf' (ultrafeedback_binarized)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for JSONL + state files",
    )
    parser.add_argument(
        "--n", type=int, default=N_DEFAULT,
        help=f"Number of held-out test pairs (default: {N_DEFAULT})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"model-spec={args.model_spec!r}  probe={args.probe!r}  n={args.n}  out={out_dir}")

    spec = parse_model_spec(args.model_spec)

    log.info(f"Loading probe dataset '{args.probe}'...")
    examples = load_probe_examples(args.probe, args.n)

    if spec["kind"] == "pythia_lora":
        run_pythia_lora_chain(
            seed=spec["seed"],
            run_tag=spec["run_tag"],
            probe=args.probe,
            out_dir=out_dir,
            examples=examples,
        )
    elif spec["kind"] == "qwen_fullweight":
        run_qwen_fullweight(
            probe=args.probe,
            out_dir=out_dir,
            examples=examples,
        )
    else:
        raise ValueError(f"Unknown kind: {spec['kind']}")

    log.info("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
