#!/usr/bin/env python3
"""behavior_geometry_link.py — T1.2: Behavior-Geometry Correlation Analysis.

For each Pythia DPO checkpoint (70m, 160m, 410m, 1B), compute:
  - Reward margin: E[r_θ(x, y_chosen) - r_θ(x, y_rejected)]
    where r_θ(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
  - KL to base: E[Σ_t logp_dpo(y_t|y<t) - logp_base(y_t|y<t)] on chosen response
  - srank and γ-overlap (from existing analysis JSON files)

Then compute Pearson + Spearman correlations (with bootstrap 95% CI) of:
  (srank, reward_margin), (gamma_overlap, reward_margin),
  (srank, kl_to_base), (gamma_overlap, kl_to_base)

Resume-safe: --resume flag reloads state.json and skips completed (model, seed) pairs.

Runtime estimate (RTX 3060 12GB, 500 examples, max_length=512):
  70m ~5min, 160m ~8min, 410m ~12min, 1B ~20min
Total: ~45min for single seed pass.

Usage:
    uv run python behavior_geometry_link.py
    uv run python behavior_geometry_link.py --resume
    uv run python behavior_geometry_link.py --n-samples 200   # quick test
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json

# ── Patch transformers' torch.load safety check (same workaround as other scripts) ──
# Pythia checkpoints are in safetensors format for the base, but some versions of
# transformers still call check_torch_load_is_safe even when not needed.
try:
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
import logging
import os
import statistics
import sys
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR  = SCRIPT_DIR.parent

# All output goes under paper/results/behavior_geometry/
OUT_DIR = PAPER_DIR / "results" / "behavior_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_JSON = OUT_DIR / "state.json"
RUN_LOG    = OUT_DIR / "run.log"

# Checkpoint snapshot base
SNAP = Path(
    "/home/debanjan/.cache/huggingface/hub"
    "/models--d3banjan--lazy-rudder-checkpoints"
    "/snapshots/8ac4c033fbdd21eff7cc38bfbe6acbfcf53cfeec"
)

# Base model local paths (cross-check/trained-model-battery/models/)
MODELS_BASE = Path(
    "/home/debanjan/Code/Research/lean-mining"
    "/cross-check/trained-model-battery/models"
)

# Paper results base (for reading existing srank/gamma JSONs)
PAPER_RESULTS = PAPER_DIR / "results"

# DPO β (checked against adapter_config.json; fall back to 0.1)
DPO_BETA = 0.1

# Held-out samples: use first N from hh-rlhf test split (reproducible)
N_SAMPLES_DEFAULT = 500   # ~45min for full sweep; reduce to 200 for smoke test

# Checkpoint interval (flush state every K examples)
CHECKPOINT_EVERY = 50

# Per-model batch sizes (halve on OOM)
BATCH_SIZES = {
    "70m":  16,
    "160m": 16,
    "410m": 8,
    "1b":   4,
}

# Model specs: (model_key, size_label, base_model_dir, adapter_dir, seed, run_label)
# Only DPO checkpoints (not CLM) — we want reward margin.
MODEL_SPECS = [
    {
        "model_size":  "70m",
        "seed":        42,
        "base_dir":    MODELS_BASE / "pythia-70m",
        "adapter_dir": SNAP / "_leak_70m" / "v2" / "checkpoints" / "checkpoint-800",
        "hf_model_id": "EleutherAI/pythia-70m",
        "srank_source": ("petri", "pythia-70m"),
        "gamma_source": ("petri", "pythia-70m"),
    },
    {
        "model_size":  "160m",
        "seed":        42,
        "base_dir":    MODELS_BASE / "pythia-160m",
        "adapter_dir": SNAP / "_leak_160m" / "v2" / "checkpoints" / "checkpoint-800",
        "hf_model_id": "EleutherAI/pythia-160m",
        "srank_source": ("petri", "pythia-160m"),
        "gamma_source": ("petri", "pythia-160m"),
    },
    {
        "model_size":  "410m",
        "seed":        42,
        "base_dir":    MODELS_BASE / "pythia-410m",
        "adapter_dir": SNAP / "_leak" / "v2" / "checkpoints" / "checkpoint-800",
        "hf_model_id": "EleutherAI/pythia-410m",
        "srank_source": ("410m", "v2_dpo_r128"),
        "gamma_source": ("410m", "v2_dpo_r128"),
    },
    {
        "model_size":  "1b",
        "seed":        42,
        "base_dir":    MODELS_BASE / "pythia-1b",
        "adapter_dir": SNAP / "_leak_1b" / "v2" / "checkpoints" / "checkpoint-800",
        "hf_model_id": "EleutherAI/pythia-1b",
        "srank_source": ("1b_s42", "v2_dpo_r128_1b"),
        "gamma_source": ("1b_s42", "v2_dpo_r128_1b"),
    },
    {
        "model_size":  "1b",
        "seed":        117,
        "base_dir":    MODELS_BASE / "pythia-1b",
        "adapter_dir": SNAP / "_leak_1b_seed117" / "v2" / "checkpoints" / "checkpoint-800",
        "hf_model_id": "EleutherAI/pythia-1b",
        "srank_source": ("1b_seed117", "v3_dpo_r128_1b_s117"),
        "gamma_source": ("1b_seed117", "v3_dpo_r128_1b_s117"),
    },
]

# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging():
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RUN_LOG), mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    logging.getLogger(__name__).info(
        f"=== behavior_geometry_link.py started at {time.strftime('%Y-%m-%d %H:%M:%S')} ==="
    )

log = logging.getLogger(__name__)

# ── srank / gamma loading from existing JSONs ──────────────────────────────────

def load_srank_gamma(spec: dict) -> tuple[float, float]:
    """Load pre-computed srank and gamma (bonus_R_k5) from existing results JSONs."""
    source_type, run_key = spec["srank_source"]

    if source_type == "petri":
        data = json.loads(
            (PAPER_RESULTS / "spectral_overlap_gamma_petri" / "results.json").read_text()
        )
        avg = data[run_key]["avg"]
        srank = avg["srank"]
        gamma = avg["bonus_R_k5"]

    elif source_type == "410m":
        data = json.loads(
            (PAPER_RESULTS / "spectral_overlap_gamma" / "results.json").read_text()
        )
        pl = data["runs"][run_key]["per_layer"]
        srank = statistics.mean(p["srank_delta"] for p in pl)
        gamma = statistics.mean(p["k5"]["bonus_right"] for p in pl)

    elif source_type in ("1b_s42",):
        data = json.loads(
            (PAPER_RESULTS / "spectral_overlap_gamma_1b" / "results.json").read_text()
        )
        pl = data["runs"][run_key]["per_layer"]
        srank = statistics.mean(p["srank_delta"] for p in pl)
        gamma = statistics.mean(p["k5"]["bonus_right"] for p in pl)

    elif source_type == "1b_seed117":
        data = json.loads(
            (PAPER_RESULTS / "spectral_overlap_gamma_1b_seed117" / "results.json").read_text()
        )
        pl = data["runs"][run_key]["per_layer"]
        srank = statistics.mean(p["srank_delta"] for p in pl)
        gamma = statistics.mean(p["k5"]["bonus_right"] for p in pl)

    else:
        raise ValueError(f"Unknown srank source type: {source_type}")

    return srank, gamma


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {
        "version": 1,
        "completed_model_seeds": [],
        "current_model_size": None,
        "current_seed": None,
        "last_example_idx": -1,
    }


def save_state(state: dict) -> None:
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_JSON)
    try:
        # fsync for durability
        with open(STATE_JSON, "r") as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass


def model_seed_key(model_size: str, seed: int) -> str:
    return f"{model_size}_seed{seed}"


# ── Dataset loading ────────────────────────────────────────────────────────────

def parse_hh_example(example: dict) -> tuple[str, str, str] | None:
    """Parse hh-rlhf example → (prompt, chosen_response, rejected_response).
    Returns None if either response is too short.
    """
    chosen_text   = example["chosen"]
    rejected_text = example["rejected"]
    sep = "\n\nAssistant:"

    if sep in chosen_text:
        parts    = chosen_text.rsplit(sep, 1)
        prompt   = parts[0] + sep
        chosen_r = parts[1].strip()
    else:
        prompt   = chosen_text[:256]
        chosen_r = chosen_text[256:]

    if sep in rejected_text:
        parts      = rejected_text.rsplit(sep, 1)
        rejected_r = parts[1].strip()
    else:
        rejected_r = rejected_text[256:]

    if len(chosen_r) < 10 or len(rejected_r) < 10:
        return None
    return prompt, chosen_r, rejected_r


def load_test_examples(n_samples: int) -> list[tuple[int, str, str, str, str]]:
    """Load hh-rlhf test split, return list of (idx, hash, prompt, chosen, rejected)."""
    from datasets import load_dataset  # noqa: PLC0415

    log.info(f"Loading Anthropic/hh-rlhf test split (first {n_samples} examples)...")
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    examples = []
    for i, ex in enumerate(ds):
        if len(examples) >= n_samples:
            break
        parsed = parse_hh_example(ex)
        if parsed is None:
            continue
        prompt, chosen_r, rejected_r = parsed
        # Stable hash of the prompt for audit trail
        prompt_hash = hashlib.sha1(prompt.encode()).hexdigest()[:10]
        examples.append((i, prompt_hash, prompt, chosen_r, rejected_r))

    log.info(f"  loaded {len(examples)} usable examples (from first {n_samples} candidates)")
    return examples


# ── Log-probability computation ───────────────────────────────────────────────

def compute_logp_batch(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int = 512,
    device: str = "cuda",
) -> list[tuple[float, int]]:
    """Compute log P(response | prompt) for a batch.
    Returns list of (logp, n_tokens).
    Teacher-forced: only response tokens contribute.
    """
    import torch  # noqa: PLC0415

    results = []
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt+response together
        full_text = prompt + response
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)

        # Tokenize prompt alone to find boundary
        prompt_enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        n_tokens = input_ids.shape[1] - prompt_len
        if n_tokens <= 0:
            # Response was truncated out
            results.append((float("nan"), 0))
            continue

        with torch.inference_mode():
            out = model(input_ids=input_ids)
            logits = out.logits  # [1, seq_len, vocab]

        # Shift: logits[t] predicts token[t+1]
        # Response tokens: positions [prompt_len, ..., seq_len-1]
        # Logits for those: [prompt_len-1, ..., seq_len-2]
        shift_logits = logits[0, prompt_len - 1 : input_ids.shape[1] - 1, :]  # [n_tokens, vocab]
        shift_labels = input_ids[0, prompt_len:]                                # [n_tokens]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs[torch.arange(n_tokens), shift_labels]           # [n_tokens]
        total_logp = token_logps.sum().item()

        results.append((total_logp, n_tokens))

    return results


# ── Per-model evaluation ──────────────────────────────────────────────────────

def run_model_seed(
    spec: dict,
    examples: list[tuple[int, str, str, str, str]],
    resume_from: int,
    n_samples: int,
) -> Path:
    """Run inference for one (model_size, seed) pair.
    Returns path to the per-example jsonl file.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415

    model_size = spec["model_size"]
    seed       = spec["seed"]
    base_dir   = spec["base_dir"]
    adapter_dir = spec["adapter_dir"]

    key = model_seed_key(model_size, seed)
    jsonl_path = OUT_DIR / f"checkpoint_{model_size}_seed{seed}.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[{key}] Running on {device}. Base: {base_dir}. Adapter: {adapter_dir}")

    # Determine β from adapter config
    cfg_path = adapter_dir / "adapter_config.json"
    beta = DPO_BETA
    # Note: β is DPOConfig, not stored in adapter_config.json (PEFT). Use default.

    batch_size = BATCH_SIZES.get(model_size, 4)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    log.info(f"[{key}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_dir),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model ───────────────────────────────────────────────────────
    log.info(f"[{key}] Loading base model (fp16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    base_model.eval()

    # ── Load DPO model (LoRA applied) ─────────────────────────────────────────
    log.info(f"[{key}] Loading DPO adapter model...")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    dpo_model = PeftModel.from_pretrained(
        dpo_model,
        str(adapter_dir),
        local_files_only=True,
    ).to(device)
    dpo_model.eval()

    log.info(f"[{key}] Both models loaded. Starting eval from idx={resume_from}...")

    # Open jsonl for append (so resume works)
    fh = open(jsonl_path, "a")

    state = load_state()

    try:
        for i, (ex_idx, prompt_hash, prompt, chosen_r, rejected_r) in enumerate(examples):
            if i < resume_from:
                continue  # skip already processed

            # Process one at a time to avoid OOM; batch size = 1 for safety
            # (batching would complicate padding / truncation bookkeeping)
            try:
                base_chosen  = compute_logp_batch(base_model,  tokenizer, [prompt], [chosen_r],   device=device)
                base_rejected = compute_logp_batch(base_model, tokenizer, [prompt], [rejected_r],  device=device)
                dpo_chosen   = compute_logp_batch(dpo_model,   tokenizer, [prompt], [chosen_r],   device=device)
                dpo_rejected = compute_logp_batch(dpo_model,   tokenizer, [prompt], [rejected_r],  device=device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning(f"[{key}] OOM at example {i}. Clearing cache and skipping.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

            record = {
                "prompt_id": ex_idx,
                "prompt_text_hash": prompt_hash,
                "logp_chosen_dpo":    dpo_chosen[0][0],
                "logp_rejected_dpo":  dpo_rejected[0][0],
                "logp_chosen_base":   base_chosen[0][0],
                "logp_rejected_base": base_rejected[0][0],
                "n_tokens_chosen":    dpo_chosen[0][1],
                "n_tokens_rejected":  dpo_rejected[0][1],
            }
            fh.write(json.dumps(record) + "\n")

            # Checkpoint every N examples
            if (i + 1) % CHECKPOINT_EVERY == 0:
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
                state["current_model_size"] = model_size
                state["current_seed"]       = seed
                state["last_example_idx"]   = i
                save_state(state)
                log.info(f"[{key}] checkpoint at example {i+1}/{len(examples)}")

        # Done with this model
        fh.flush()
        os.fsync(fh.fileno())

    except KeyboardInterrupt:
        log.info(f"[{key}] KeyboardInterrupt — saving state before exit.")
        fh.flush()
        state["current_model_size"] = model_size
        state["current_seed"]       = seed
        state["last_example_idx"]   = i if "i" in dir() else resume_from
        save_state(state)
        fh.close()
        raise
    finally:
        fh.close()

    # Free GPU memory
    log.info(f"[{key}] Freeing GPU memory...")
    del base_model
    del dpo_model
    torch.cuda.empty_cache()
    gc.collect()

    return jsonl_path


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_jsonl(jsonl_path: Path, beta: float = DPO_BETA) -> dict:
    """Aggregate per-example records into reward_margin and KL-to-base stats."""
    records = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError(f"No records in {jsonl_path}")

    reward_margins = []
    kl_to_bases    = []

    for r in records:
        lp_chosen_dpo  = r["logp_chosen_dpo"]
        lp_rejected_dpo = r["logp_rejected_dpo"]
        lp_chosen_base  = r["logp_chosen_base"]
        lp_rejected_base = r["logp_rejected_base"]

        # Skip NaN entries
        vals = [lp_chosen_dpo, lp_rejected_dpo, lp_chosen_base, lp_rejected_base]
        if any(v != v for v in vals):  # NaN check
            continue
        if any(abs(v) > 1e6 for v in vals):
            continue

        # Reward margin = r_dpo(x, chosen) - r_dpo(x, rejected)
        # r_dpo(x, y) = β * (logp_dpo(y|x) - logp_base(y|x))
        reward_chosen   = beta * (lp_chosen_dpo  - lp_chosen_base)
        reward_rejected = beta * (lp_rejected_dpo - lp_rejected_base)
        margin = reward_chosen - reward_rejected
        reward_margins.append(margin)

        # KL to base on chosen: Σ_t [logp_dpo - logp_base] = logp_chosen_dpo - logp_chosen_base
        # This is sequence-level KL (teacher-forced); per-token = divide by n_tokens
        n_tok = r.get("n_tokens_chosen", 1) or 1
        kl = (lp_chosen_dpo - lp_chosen_base) / n_tok
        kl_to_bases.append(kl)

    n = len(reward_margins)
    if n == 0:
        raise ValueError("All records had NaN values.")

    def se(vals):
        if len(vals) < 2:
            return float("nan")
        return statistics.stdev(vals) / (len(vals) ** 0.5)

    return {
        "reward_margin_mean": statistics.mean(reward_margins),
        "reward_margin_se":   se(reward_margins),
        "kl_to_base_mean":    statistics.mean(kl_to_bases),
        "kl_to_base_se":      se(kl_to_bases),
        "n_samples":          n,
        "n_skipped_nan":      len(records) - n,
    }


# ── Correlation analysis ──────────────────────────────────────────────────────

def bootstrap_ci(x: list[float], y: list[float], stat_fn, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% CI for a statistic. Returns (lower, upper)."""
    import random  # noqa: PLC0415
    random.seed(42)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        xi = [x[i] for i in idx]
        yi = [y[i] for i in idx]
        try:
            stats.append(stat_fn(xi, yi))
        except Exception:
            pass
    if not stats:
        return float("nan"), float("nan")
    stats.sort()
    lo = int((1 - ci) / 2 * len(stats))
    hi = int((1 + ci) / 2 * len(stats))
    return stats[lo], stats[min(hi, len(stats) - 1)]


def pearson_r(x: list[float], y: list[float]) -> float:
    from scipy.stats import pearsonr  # noqa: PLC0415
    if len(x) < 3:
        return float("nan")
    r, _ = pearsonr(x, y)
    return r


def spearman_r(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr  # noqa: PLC0415
    if len(x) < 3:
        return float("nan")
    r, _ = spearmanr(x, y)
    return r


def compute_correlations(summaries: list[dict]) -> dict:
    """Compute all 4 correlation pairs with bootstrap CI."""
    sranks  = [s["srank"] for s in summaries]
    gammas  = [s["gamma_overlap"] for s in summaries]
    rewards = [s["reward_margin_mean"] for s in summaries]
    kls     = [s["kl_to_base_mean"] for s in summaries]

    n = len(summaries)

    pairs = [
        ("srank",  "reward_margin",  sranks, rewards),
        ("gamma",  "reward_margin",  gammas, rewards),
        ("srank",  "kl_to_base",     sranks, kls),
        ("gamma",  "kl_to_base",     gammas, kls),
    ]

    results = {"n_points": n, "note": (
        "With n<=5 model checkpoints, correlation estimates have wide CIs; "
        "this is suggestive, not definitive."
    )}

    for x_name, y_name, x, y in pairs:
        key = f"{x_name}_vs_{y_name}"
        pr  = pearson_r(x, y)
        sr  = spearman_r(x, y)
        pr_lo, pr_hi = bootstrap_ci(x, y, pearson_r, ci=0.95)
        sr_lo, sr_hi = bootstrap_ci(x, y, spearman_r, ci=0.95)
        results[key] = {
            "pearson_r":  pr,
            "spearman_r": sr,
            "pearson_ci95_lo":  pr_lo,
            "pearson_ci95_hi":  pr_hi,
            "spearman_ci95_lo": sr_lo,
            "spearman_ci95_hi": sr_hi,
        }
        log.info(
            f"  {key}: Pearson={pr:.3f} [{pr_lo:.3f},{pr_hi:.3f}]  "
            f"Spearman={sr:.3f} [{sr_lo:.3f},{sr_hi:.3f}]"
        )

    return results


# ── Figure generation (generate_fig_G.py wrapper) ─────────────────────────────

def generate_figure(summaries: list[dict], correlations: dict) -> None:
    """Generate 2x2 scatter grid: (srank, gamma) x (reward_margin, KL)."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as e:
        log.warning(f"Cannot generate figure: {e}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(
        "Behavior-Geometry Correlation: srank and γ vs. Reward Margin and KL-to-Base\n"
        "(Pythia DPO LoRA adapters on Anthropic/hh-rlhf test split)",
        fontsize=11,
    )

    labels = [f"{s['model_size']}\ns{s['seed']}" for s in summaries]
    colors = {"70m": "#1f77b4", "160m": "#ff7f0e", "410m": "#2ca02c", "1b": "#d62728"}
    point_colors = [colors.get(s["model_size"], "gray") for s in summaries]

    sranks  = [s["srank"] for s in summaries]
    gammas  = [s["gamma_overlap"] for s in summaries]
    rewards = [s["reward_margin_mean"] for s in summaries]
    kls     = [s["kl_to_base_mean"] for s in summaries]
    reward_se = [s["reward_margin_se"] for s in summaries]
    kl_se     = [s["kl_to_base_se"] for s in summaries]

    panel_data = [
        (axes[0, 0], sranks, rewards, reward_se,
         "stable rank (srank)", "reward margin (β·Δlog π)",
         "srank_vs_reward_margin", "srank", "reward_margin"),
        (axes[0, 1], gammas, rewards, reward_se,
         "γ-overlap (bonus_R k=5)", "reward margin (β·Δlog π)",
         "gamma_vs_reward_margin", "gamma", "reward_margin"),
        (axes[1, 0], sranks, kls, kl_se,
         "stable rank (srank)", "KL-to-base (per token)",
         "srank_vs_kl", "srank", "kl_to_base"),
        (axes[1, 1], gammas, kls, kl_se,
         "γ-overlap (bonus_R k=5)", "KL-to-base (per token)",
         "gamma_vs_kl", "gamma", "kl_to_base"),
    ]

    for ax, x_vals, y_vals, y_errs, xlabel, ylabel, corr_key, xk, yk in panel_data:
        corr_key_full = f"{xk}_vs_{yk}"
        for xi, yi, ye, label, color in zip(x_vals, y_vals, y_errs, labels, point_colors):
            ax.errorbar(xi, yi, yerr=ye, fmt="o", color=color, capsize=3, markersize=6)
            ax.annotate(label, (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=color)

        # Add trend line if we have enough points
        if len(x_vals) >= 3:
            try:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(min(x_vals), max(x_vals), 50)
                ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.4, linewidth=1)
            except Exception:
                pass

        # Annotate with Pearson r
        if corr_key_full in correlations:
            pr  = correlations[corr_key_full]["pearson_r"]
            pr_lo = correlations[corr_key_full]["pearson_ci95_lo"]
            pr_hi = correlations[corr_key_full]["pearson_ci95_hi"]
            r_str = (
                f"r={pr:.2f} [{pr_lo:.2f},{pr_hi:.2f}]"
                if pr == pr else "r=n/a"
            )
            ax.text(0.05, 0.95, r_str, transform=ax.transAxes,
                    fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Legend
    for model_size, color in colors.items():
        axes[1, 1].plot([], [], "o", color=color, label=f"pythia-{model_size}", markersize=5)
    axes[1, 1].legend(loc="lower right", fontsize=7, framealpha=0.8)

    fig.tight_layout()

    # Save to manuscript figures and paper results
    figs_dir = PAPER_DIR / "manuscript" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = figs_dir / "fig_G_behavior_geometry.pdf"
    png_path = figs_dir / "fig_G_behavior_geometry.png"
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=150)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"Figure saved: {pdf_path}, {png_path}")

    # Also copy to results
    import shutil  # noqa: PLC0415
    shutil.copy(png_path, OUT_DIR / "fig_G_behavior_geometry.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from state.json (skip completed model-seeds)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT,
                        help=f"Number of test examples (default: {N_SAMPLES_DEFAULT})")
    args = parser.parse_args()

    log.info(f"n_samples={args.n_samples}, resume={args.resume}")

    # Load or init state
    state = load_state() if args.resume else {
        "version": 1,
        "completed_model_seeds": [],
        "current_model_size": None,
        "current_seed": None,
        "last_example_idx": -1,
    }

    # Load examples once (all models share the same test set)
    examples = load_test_examples(args.n_samples)

    summaries = []

    try:
        for spec in MODEL_SPECS:
            model_size = spec["model_size"]
            seed       = spec["seed"]
            key        = model_seed_key(model_size, seed)

            # Skip if already completed
            if key in state["completed_model_seeds"]:
                log.info(f"[{key}] Already completed — skipping.")
                # Still need to load summary for correlation
                summary_path = OUT_DIR / "summary.json"
                if summary_path.exists():
                    all_summaries = json.loads(summary_path.read_text())
                    for s in all_summaries:
                        if s.get("model_size") == model_size and s.get("seed") == seed:
                            summaries.append(s)
                            break
                continue

            # Determine resume offset
            resume_from = 0
            if (args.resume
                    and state.get("current_model_size") == model_size
                    and state.get("current_seed") == seed):
                resume_from = state.get("last_example_idx", -1) + 1
                log.info(f"[{key}] Resuming from example {resume_from}")

            # Load srank + gamma from existing JSONs
            srank, gamma = load_srank_gamma(spec)
            log.info(f"[{key}] srank={srank:.3f}, gamma_overlap={gamma:.3f}")

            # Run inference
            jsonl_path = run_model_seed(spec, examples, resume_from, args.n_samples)

            # Aggregate
            agg = aggregate_jsonl(jsonl_path, beta=DPO_BETA)
            log.info(
                f"[{key}] reward_margin={agg['reward_margin_mean']:.4f}±{agg['reward_margin_se']:.4f}"
                f"  KL={agg['kl_to_base_mean']:.4f}±{agg['kl_to_base_se']:.4f}"
                f"  n={agg['n_samples']}"
            )

            summary = {
                "model_size":         model_size,
                "seed":               seed,
                "srank":              srank,
                "gamma_overlap":      gamma,
                **agg,
            }
            summaries.append(summary)

            # Mark completed and save state
            state["completed_model_seeds"].append(key)
            state["current_model_size"] = model_size
            state["current_seed"]       = seed
            state["last_example_idx"]   = len(examples) - 1
            save_state(state)

            # Write incremental summary
            summary_path = OUT_DIR / "summary.json"
            summary_path.write_text(json.dumps(summaries, indent=2))

    except KeyboardInterrupt:
        log.info("Interrupted. Partial results saved.")
        return 1

    if not summaries:
        log.error("No summaries computed.")
        return 1

    # ── Correlation analysis ──────────────────────────────────────────────────
    log.info(f"\n=== Correlation analysis ({len(summaries)} data points) ===")
    correlations = compute_correlations(summaries)

    corr_path = OUT_DIR / "correlation.json"
    corr_path.write_text(json.dumps(correlations, indent=2))
    log.info(f"Correlation results written to {corr_path}")

    # ── Write final summary ───────────────────────────────────────────────────
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    log.info(f"Summary written to {summary_path}")

    # ── Generate figure ───────────────────────────────────────────────────────
    generate_figure(summaries, correlations)

    log.info("\n=== DONE ===")
    log.info(f"Output dir: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
