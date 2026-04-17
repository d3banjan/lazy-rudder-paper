"""γ₁ seed-117 replicate — spectral overlap at Pythia-1B with both seed=42 and seed=117.

Replicates spectral_overlap_gamma.py but at 1B scale:
  - Base: gpt_neox.layers.{i}.attention.query_key_value.weight  [6144, 2048]
  - 16 layers (not 24)
  - Random baseline: k/6144 (left), k/2048 (right)

Outputs:
  results/spectral_overlap_gamma_1b/results.json

Also appends comparison_vs_410m using the 410m results file if present.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

# Patch transformers' torch.load safety check
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "pythia-1b"
RESULTS      = ROOT / "results" / "_leak_1b"
RESULTS_S117 = ROOT / "results" / "_leak_1b_seed117"
OUT_DIR   = ROOT / "results" / "spectral_overlap_gamma_1b_seed117"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 410m reference results for comparison
REF_GAMMA_410M = ROOT / "results" / "spectral_overlap_gamma" / "results.json"

RUNS = [
    # Seed=42 baseline
    ("v2_dpo_r128_1b_s42",       RESULTS      / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v3_clm_r128_1b_s42",       RESULTS      / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
    # Seed=117 buggy (shuffle seed was 42 — same data draw as s42)
    ("v2_dpo_r128_1b_s117_bug",  RESULTS_S117 / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    # Seed=117 correct (independent draw — shuffle seed 117)
    ("v3_dpo_r128_1b_s117",      RESULTS_S117 / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v4_clm_r128_1b_s117",      RESULTS_S117 / "v4" / "checkpoints" / "checkpoint-800", 128, 256),
]

K_VALS = [5, 10, 20]

# Dimensions determined at runtime from the loaded model
D_OUT = None  # 6144 for Pythia-1B (3 * 2048)
D_IN  = None  # 2048 for Pythia-1B


# ── Adapter loading ──────────────────────────────────────────────────────────

def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    safetensor_path = path / "adapter_model.safetensors"
    if safetensor_path.exists():
        out = {}
        with safe_open(str(safetensor_path), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    # Fallback: pytorch bin
    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter_model.safetensors or .bin in {path}")


def extract_qkv_delta(tensors: dict, layer: int, alpha: int, r: int) -> torch.Tensor:
    a_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_A.weight")
    b_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_B.weight")
    A = tensors[a_key].float()   # [r, d_in]
    B = tensors[b_key].float()   # [d_out, r]
    return (alpha / r) * (B @ A)  # [d_out, d_in]


# ── Base model weight extraction ─────────────────────────────────────────────

def load_base_weights() -> tuple[list[torch.Tensor], int]:
    """Load QKV weight tensors for all layers in fp32. Returns (weights, n_layers)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading Pythia-1B base from {MODEL_DIR}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    n_layers = len(model.gpt_neox.layers)
    log.info(f"Pythia-1B: {n_layers} layers  hidden_size={model.config.hidden_size}")

    weights = []
    for li in range(n_layers):
        w = (model.gpt_neox.layers[li]
             .attention.query_key_value.weight
             .detach().float().cpu())
        weights.append(w)
        if li % 4 == 0:
            log.info(f"  layer {li}: W shape {w.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Base weights loaded and model freed.")
    return weights, n_layers


# ── Stable rank ──────────────────────────────────────────────────────────────

def srank(s: torch.Tensor) -> float:
    s2 = s ** 2
    total = s2.sum().item()
    spec2 = s2[0].item()
    return total / spec2 if spec2 > 0 else 0.0


# ── Core overlap computation ─────────────────────────────────────────────────

def compute_overlap(W: torch.Tensor, dW: torch.Tensor) -> dict:
    """
    SVD both matrices, compute left/right subspace overlaps for k in K_VALS
    and at k = round(srank(dW)).

    Uses module-level D_OUT / D_IN for random baselines.
    """
    U_W,  S_W,  Vt_W  = torch.linalg.svd(W,  full_matrices=False)
    U_dW, S_dW, Vt_dW = torch.linalg.svd(dW, full_matrices=False)

    sr_delta = srank(S_dW)
    k_auto   = max(1, round(sr_delta))

    result = {"srank_delta": sr_delta}

    all_k = sorted(set(K_VALS + [k_auto]))

    for k in all_k:
        k_eff = min(k, U_W.shape[1], U_dW.shape[1], Vt_W.shape[0], Vt_dW.shape[0])

        # Left subspace cross-gram
        G_left  = U_W[:, :k_eff].T @ U_dW[:, :k_eff]
        sv_left = torch.linalg.svdvals(G_left)
        p_left  = (sv_left ** 2).sum().item() / k_eff

        # Right subspace cross-gram
        G_right  = Vt_W[:k_eff, :] @ Vt_dW[:k_eff, :].T
        sv_right = torch.linalg.svdvals(G_right)
        p_right  = (sv_right ** 2).sum().item() / k_eff

        # Bonus factors vs random baseline
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

    # k_srank aggregation
    srank_vals   = [l["srank_delta"] for l in per_layer]
    br_srank     = [l["k_srank"]["bonus_right"] for l in per_layer]
    bl_srank     = [l["k_srank"]["bonus_left"]  for l in per_layer]
    avg_out["k_srank"] = {
        "srank_mean":    round(_avg(srank_vals), 4),
        "bonus_left":    round(_avg(bl_srank),   4),
        "bonus_right":   round(_avg(br_srank),   4),
    }

    return avg_out, range_out


# ── Layer-depth pattern ───────────────────────────────────────────────────────

def layer_depth_summary(per_layer: list[dict]) -> dict:
    """Compare early (layer 0) vs late (last layer) bonus_right at k=5."""
    if len(per_layer) < 2:
        return {}
    early = per_layer[0]["k5"]["bonus_right"]
    late  = per_layer[-1]["k5"]["bonus_right"]
    return {
        "layer_0_bonus_right_k5":  round(early, 4),
        "layer_last_bonus_right_k5": round(late, 4),
        "early_to_late_ratio":     round(late / early, 4) if early > 0 else float('nan'),
    }


# ── Verdict logic ─────────────────────────────────────────────────────────────

def determine_verdict_1b(runs_avg: dict[str, dict]) -> dict:
    """
    For each run, compute scaling verdict vs 410m baseline:
      (a) universal:      bonus_R_k5 in 4-10x AND srank <= 8
      (b) width-scaling:  bonus_R_k5 > 12x  OR  srank > 12
      (c) inverse:        bonus_R_k5 < 3x
      mixed:              DPO and CLM diverge
    """
    per_run = {}
    for run, avg_d in runs_avg.items():
        br5   = avg_d["k5"]["bonus_right"]
        sr    = avg_d["k_srank"]["srank_mean"]
        if br5 >= 4 and br5 <= 10 and sr <= 8:
            verdict = "universal"
        elif br5 > 12 or sr > 12:
            verdict = "width-scaling"
        elif br5 < 3:
            verdict = "inverse"
        else:
            verdict = "mixed"
        per_run[run] = {"verdict": verdict, "bonus_right_k5": br5, "srank_mean": sr}

    verdicts = [v["verdict"] for v in per_run.values()]
    if len(set(verdicts)) == 1:
        overall = verdicts[0]
    else:
        overall = "mixed"

    return {"overall": overall, "per_run": per_run}


# ── 410m comparison block ─────────────────────────────────────────────────────

KNOWN_410M = {
    "v2_dpo_r128": {
        "bonus_L_k5":   1.47,
        "bonus_R_k5":   5.06,
        "bonus_R_srank": 7.07,
        "srank":         3.92,
    },
    "v3_clm_r128": {
        "bonus_L_k5":   1.47,
        "bonus_R_k5":   5.22,
        "bonus_R_srank": 7.07,
        "srank":         3.92,
    },
}

# Map 1B run names to their 410m counterparts
RUN_MAP_410M = {
    "v2_dpo_r128_1b_s42":  "v2_dpo_r128",
    "v3_clm_r128_1b_s42":  "v3_clm_r128",
    "v2_dpo_r128_1b_s117": "v2_dpo_r128",
    "v3_clm_r128_1b_s117": "v3_clm_r128",
}


def build_comparison(runs_avg: dict[str, dict], verdict_info: dict) -> dict:
    """Build comparison_vs_410m block."""
    out = {}
    for run_1b, avg_d in runs_avg.items():
        ref_key = RUN_MAP_410M.get(run_1b)
        ref_410m = KNOWN_410M.get(ref_key, {})

        br5    = avg_d["k5"]["bonus_right"]
        bl5    = avg_d["k5"]["bonus_left"]
        sr     = avg_d["k_srank"]["srank_mean"]
        br_sr  = avg_d["k_srank"]["bonus_right"]

        per_run_v = verdict_info["per_run"].get(run_1b, {}).get("verdict", "unknown")

        out[run_1b] = {
            "410m": ref_410m,
            "1b": {
                "bonus_L_k5":    round(bl5, 4),
                "bonus_R_k5":    round(br5, 4),
                "bonus_R_srank": round(br_sr, 4),
                "srank":         round(sr, 4),
            },
            "scaling_law": per_run_v,
        }
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global D_OUT, D_IN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Running on {device}")

    # Load base weights; n_layers discovered dynamically
    base_weights, n_layers = load_base_weights()

    # Set dimensions from actual weights
    D_OUT = base_weights[0].shape[0]   # 6144 for 1B
    D_IN  = base_weights[0].shape[1]   # 2048 for 1B
    log.info(f"D_OUT={D_OUT}  D_IN={D_IN}  n_layers={n_layers}")

    results_runs = {}
    runs_avg_for_verdict = {}

    for run_name, ckpt, r, alpha in RUNS:
        print(f"\n{'='*80}\n{run_name}  r={r}  alpha={alpha}\n{'='*80}")
        if not ckpt.exists():
            log.warning(f"MISSING checkpoint: {ckpt}")
            continue

        tensors = load_adapter(ckpt)
        per_layer = []

        for li in range(n_layers):
            W  = base_weights[li]
            dW = extract_qkv_delta(tensors, li, alpha, r)

            layer_result = compute_overlap(W, dW)
            layer_result["layer"] = li
            per_layer.append(layer_result)

            if li % 4 == 0:
                sr  = layer_result["srank_delta"]
                bl  = layer_result["k5"]["bonus_left"]
                br  = layer_result["k5"]["bonus_right"]
                log.info(f"  layer {li:2d}: srank_ΔW={sr:.2f}  "
                         f"bonus_left(k=5)={bl:.2f}  bonus_right(k=5)={br:.2f}")

        avg_d, range_d = aggregate_run(per_layer)
        depth_d = layer_depth_summary(per_layer)

        # Print summary
        for k in K_VALS:
            print(f"  k={k:2d}  p_left={avg_d[f'k{k}']['p_left']:.5f} "
                  f"(bonus={avg_d[f'k{k}']['bonus_left']:.2f}x)  "
                  f"p_right={avg_d[f'k{k}']['p_right']:.5f} "
                  f"(bonus={avg_d[f'k{k}']['bonus_right']:.2f}x)")
        print(f"  k=srank:  srank_mean={avg_d['k_srank']['srank_mean']:.2f}  "
              f"bonus_right={avg_d['k_srank']['bonus_right']:.2f}x")
        print(f"  layer depth: {depth_d}")

        # Notable principal angles (cosine > 0.9)
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
            "per_layer":   per_layer,
            "avg":         avg_d,
            "range":       range_d,
            "depth":       depth_d,
        }
        runs_avg_for_verdict[run_name] = avg_d

    # Verdict
    verdict_info = determine_verdict_1b(runs_avg_for_verdict)
    comparison   = build_comparison(runs_avg_for_verdict, verdict_info)

    print(f"\n{'='*80}")
    print(f"VERDICT: {verdict_info['overall']}")
    for run, info in verdict_info["per_run"].items():
        print(f"  {run}: {info['verdict']}  bonus_R_k5={info['bonus_right_k5']:.2f}x  srank={info['srank_mean']:.2f}")
    print(f"{'='*80}\n")

    # Random baseline table
    rand_base_json = {
        f"k{k}": {
            "p_left":  round(k / D_OUT, 6),  # k/6144
            "p_right": round(k / D_IN,  6),  # k/2048
        }
        for k in K_VALS
    }
    rand_base_json["notes"] = {
        "D_OUT": D_OUT,
        "D_IN":  D_IN,
        "random_p_left_k5":  round(5 / D_OUT, 6),
        "random_p_right_k5": round(5 / D_IN,  6),
        "vs_410m_p_left_k5":  round(5 / 3072, 6),
        "vs_410m_p_right_k5": round(5 / 1024, 6),
    }

    results = {
        "model":             "pythia-1b",
        "n_layers":          n_layers,
        "d_out":             D_OUT,
        "d_in":              D_IN,
        "runs":              results_runs,
        "random_baseline":   rand_base_json,
        "verdict":           verdict_info,
        "comparison_vs_410m": comparison,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    shutil.copy(__file__, OUT_DIR / Path(__file__).name)
    log.info(f"Wrote: {out_json}")
    print(f"Artifacts:\n  script  : {__file__}\n  results : {out_json}")


if __name__ == "__main__":
    main()
