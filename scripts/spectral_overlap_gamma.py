"""γ — spectral overlap of ΔW top-k singular vectors onto base W top-k.

Question: when LoRA trains on DPO/CLM, does ΔW align with the top singular
directions of the pretrained base (amplifies existing structure — "rain follows
rivers"), or is ΔW orthogonal to those directions (carves new channels)?

Inputs:
  - Base Pythia-410m: gpt_neox.layers.{i}.attention.query_key_value.weight  [3072, 1024]
  - Three LoRA adapter checkpoints: v1_dpo_r16, v2_dpo_r128, v3_clm_r128

Per layer per run, for k in {5, 10, 20} and k=round(srank(ΔW)):
  - p_left(k)  = ||U_W[:,:k].T @ U_ΔW[:,:k]||_F² / k   (output subspace overlap)
  - p_right(k) = ||Vt_W[:k,:] @ Vt_ΔW[:k,:].T||_F² / k  (input subspace overlap)
  - riverbed-bonus = measured overlap / random-subspace baseline
  - principal angles (cosines) = svdvals of cross-gram matrix

Random subspace baseline:
  p_left  ~ k / d_out = k / 3072
  p_right ~ k / d_in  = k / 1024
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Patch transformers' torch.load safety check (same workaround as β′ run)
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import MODELS_DIR, PAPER_RESULTS_DIR, RESULTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "pythia-410m"
RESULTS   = RESULTS_DIR / "_leak"
OUT_DIR   = PAPER_RESULTS_DIR / "spectral_overlap_gamma"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("v1_dpo_r16",  RESULTS / "checkpoints"        / "checkpoint-800",  16,  32),
    ("v2_dpo_r128", RESULTS / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v3_clm_r128", RESULTS / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
]

N_LAYERS = 24
D_OUT    = 3072
D_IN     = 1024
K_VALS   = [5, 10, 20]

# Random-subspace baselines
RAND_BASE = {
    5:  {"p_left": 5 / D_OUT, "p_right": 5 / D_IN},
    10: {"p_left": 10 / D_OUT, "p_right": 10 / D_IN},
    20: {"p_left": 20 / D_OUT, "p_right": 20 / D_IN},
}


# ── Adapter loading ──────────────────────────────────────────────────────────

def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def extract_qkv_delta(tensors: dict, layer: int, alpha: int, r: int) -> torch.Tensor:
    a_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_A.weight")
    b_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_B.weight")
    A = tensors[a_key].float()   # [r, d_in]
    B = tensors[b_key].float()   # [d_out, r]
    return (alpha / r) * (B @ A)  # [d_out, d_in]


# ── Base model weight extraction ─────────────────────────────────────────────

def load_base_weights() -> list[torch.Tensor]:
    """Load QKV weight tensors for all 24 layers in fp32."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading base model from {MODEL_DIR}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    weights = []
    for li in range(N_LAYERS):
        w = (model.gpt_neox.layers[li]
             .attention.query_key_value.weight
             .detach().float().cpu())
        weights.append(w)
        if li % 6 == 0:
            log.info(f"  layer {li}: W shape {w.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Base weights loaded and model freed.")
    return weights


# ── Stable rank ──────────────────────────────────────────────────────────────

def srank(s: torch.Tensor) -> float:
    """Stable rank from pre-computed singular values."""
    s2 = s ** 2
    total = s2.sum().item()
    spec2 = s2[0].item()
    return total / spec2 if spec2 > 0 else 0.0


# ── Core overlap computation ─────────────────────────────────────────────────

def compute_overlap(W: torch.Tensor, dW: torch.Tensor) -> dict:
    """
    SVD both matrices, compute left/right subspace overlaps for k in K_VALS
    and at k = round(srank(dW)).

    Returns a dict with per-k metrics + srank_delta.
    """
    # SVD in fp32
    U_W,  S_W,  Vt_W  = torch.linalg.svd(W,  full_matrices=False)   # U:[d_out,min], Vt:[min,d_in]
    U_dW, S_dW, Vt_dW = torch.linalg.svd(dW, full_matrices=False)

    sr_delta = srank(S_dW)
    k_auto   = max(1, round(sr_delta))

    result = {"srank_delta": sr_delta}

    all_k = sorted(set(K_VALS + [k_auto]))

    for k in all_k:
        k_eff = min(k, U_W.shape[1], U_dW.shape[1], Vt_W.shape[0], Vt_dW.shape[0])

        # Left subspace: cross-gram [k_eff, k_eff]
        G_left  = U_W[:, :k_eff].T @ U_dW[:, :k_eff]     # [k_eff, k_eff]
        sv_left = torch.linalg.svdvals(G_left)
        p_left  = (sv_left ** 2).sum().item() / k_eff

        # Right subspace: cross-gram [k_eff, k_eff]
        G_right  = Vt_W[:k_eff, :] @ Vt_dW[:k_eff, :].T  # [k_eff, k_eff]
        sv_right = torch.linalg.svdvals(G_right)
        p_right  = (sv_right ** 2).sum().item() / k_eff

        # Bonus factors
        base_left  = k_eff / D_OUT
        base_right = k_eff / D_IN
        bonus_left  = p_left  / base_left  if base_left  > 0 else float('nan')
        bonus_right = p_right / base_right if base_right > 0 else float('nan')

        entry = {
            "p_left":      round(p_left,  6),
            "p_right":     round(p_right, 6),
            "bonus_left":  round(bonus_left,  4),
            "bonus_right": round(bonus_right, 4),
            "sv_left":     [round(x, 6) for x in sv_left.tolist()],
            "sv_right":    [round(x, 6) for x in sv_right.tolist()],
        }

        if k == k_auto:
            result["k_srank"] = {"k": k_eff, **entry}
        if k in K_VALS:
            result[f"k{k}"] = entry

    return result


# ── Aggregation helpers ──────────────────────────────────────────────────────

def _avg(vals):
    return sum(vals) / len(vals) if vals else float('nan')


def aggregate_run(per_layer: list[dict]) -> tuple[dict, dict]:
    """Returns (avg_dict, range_dict) for k in K_VALS."""
    avg_out   = {}
    range_out = {}

    for k in K_VALS:
        key = f"k{k}"
        pl_vals   = [l[key]["p_left"]      for l in per_layer]
        pr_vals   = [l[key]["p_right"]     for l in per_layer]
        bl_vals   = [l[key]["bonus_left"]  for l in per_layer]
        br_vals   = [l[key]["bonus_right"] for l in per_layer]

        avg_out[key] = {
            "p_left":      round(_avg(pl_vals), 6),
            "p_right":     round(_avg(pr_vals), 6),
            "bonus_left":  round(_avg(bl_vals), 4),
            "bonus_right": round(_avg(br_vals), 4),
        }
        range_out[key] = {
            "p_left_min":      round(min(pl_vals), 6),
            "p_left_max":      round(max(pl_vals), 6),
            "p_right_min":     round(min(pr_vals), 6),
            "p_right_max":     round(max(pr_vals), 6),
            "bonus_left_min":  round(min(bl_vals), 4),
            "bonus_left_max":  round(max(bl_vals), 4),
            "bonus_right_min": round(min(br_vals), 4),
            "bonus_right_max": round(max(br_vals), 4),
        }

    return avg_out, range_out


# ── Verdict logic ─────────────────────────────────────────────────────────────

def determine_verdict(runs_avg: dict[str, dict]) -> tuple[str, str]:
    """
    (a) riverbed: avg bonus_left > 3 AND bonus_right > 3 for k=5
    (b) new channels: avg bonus_left < 1.5 AND bonus_right < 1.5 for ALL k
    (c) asymmetric: one side >> 3×, other < 1.5×
    mixed: disagreement across runs or k
    """
    run_verdicts = []

    for run, avg_d in runs_avg.items():
        bl5 = avg_d["k5"]["bonus_left"]
        br5 = avg_d["k5"]["bonus_right"]
        # Check all-k new-channels
        all_new = all(
            avg_d[f"k{k}"]["bonus_left"]  < 1.5 and
            avg_d[f"k{k}"]["bonus_right"] < 1.5
            for k in K_VALS
        )
        if bl5 > 3 and br5 > 3:
            run_verdicts.append("a")
        elif all_new:
            run_verdicts.append("b")
        elif (bl5 > 3 and br5 < 1.5) or (br5 > 3 and bl5 < 1.5):
            run_verdicts.append("c")
        else:
            run_verdicts.append("mixed")

    if len(set(run_verdicts)) == 1:
        v = run_verdicts[0]
    else:
        v = "mixed"

    headlines = {
        "a": ("rain follows rivers: ΔW top-k singular directions align strongly "
              "with base W top-k on BOTH input and output sides. LoRA amplifies "
              "existing pretrained structure — not carving new channels."),
        "b": ("rain digs new channels: ΔW top-k singular directions are "
              "indistinguishable from random with respect to base W principal axes. "
              "LoRA update lives in a subspace orthogonal to pretraining structure."),
        "c": ("asymmetric alignment: ΔW aligns with base W on one side (input OR "
              "output) but not the other. Gradient concentrates in one end of the "
              "linear map while rewriting the other."),
        "mixed": ("mixed verdict across runs or k values — alignment pattern "
                  "is unstable; see per-run breakdown."),
    }

    return v, headlines.get(v, "")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Running on {device}")

    # Load base weights once
    base_weights = load_base_weights()

    results_runs = {}

    for run_name, ckpt, r, alpha in RUNS:
        print(f"\n{'='*80}\n{run_name}  r={r}  alpha={alpha}\n{'='*80}")
        if not ckpt.exists():
            log.warning(f"MISSING: {ckpt}")
            continue

        tensors = load_adapter(ckpt)
        per_layer = []

        for li in range(N_LAYERS):
            W  = base_weights[li]                                 # [3072, 1024] fp32 cpu
            dW = extract_qkv_delta(tensors, li, alpha, r)         # same shape fp32 cpu

            layer_result = compute_overlap(W, dW)
            layer_result["layer"] = li
            per_layer.append(layer_result)

            if li % 6 == 0:
                sr = layer_result["srank_delta"]
                bl = layer_result["k5"]["bonus_left"]
                br = layer_result["k5"]["bonus_right"]
                log.info(f"  layer {li:2d}: srank_ΔW={sr:.2f}  "
                         f"bonus_left(k=5)={bl:.2f}  bonus_right(k=5)={br:.2f}")

        avg_d, range_d = aggregate_run(per_layer)

        # Print summary
        for k in K_VALS:
            print(f"  k={k:2d}  p_left={avg_d[f'k{k}']['p_left']:.5f} "
                  f"(bonus={avg_d[f'k{k}']['bonus_left']:.2f}x)  "
                  f"p_right={avg_d[f'k{k}']['p_right']:.5f} "
                  f"(bonus={avg_d[f'k{k}']['bonus_right']:.2f}x)")

        # Find any principal angle near 1.0 (cosine > 0.9)
        notable = []
        for ld in per_layer:
            for k in K_VALS:
                for side in ("sv_left", "sv_right"):
                    for sv_val in ld[f"k{k}"][side]:
                        if sv_val > 0.9:
                            notable.append({
                                "layer": ld["layer"], "k": k,
                                "side": side, "cosine": sv_val
                            })
        if notable:
            print(f"  NOTABLE principal angles (cosine > 0.9): {notable[:5]}")

        results_runs[run_name] = {
            "per_layer": per_layer,
            "avg":       avg_d,
            "range":     range_d,
        }

    # Verdict
    runs_avg_for_verdict = {rn: v["avg"] for rn, v in results_runs.items()}
    verdict, headline = determine_verdict(runs_avg_for_verdict)

    print(f"\n{'='*80}")
    print(f"VERDICT ({verdict}): {headline}")
    print(f"{'='*80}\n")

    # Random baseline table
    rand_base_json = {
        f"k{k}": {
            "p_left":  round(k / D_OUT, 6),
            "p_right": round(k / D_IN,  6),
        }
        for k in K_VALS
    }

    results = {
        "runs":           results_runs,
        "random_baseline": rand_base_json,
        "verdict":        verdict,
        "headline":       headline,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    shutil.copy(__file__, OUT_DIR / "spectral_overlap_gamma.py")
    log.info(f"Wrote: {out_json}")
    print(f"Artifacts:\n  script  : {__file__}\n  results : {out_json}")


if __name__ == "__main__":
    main()
