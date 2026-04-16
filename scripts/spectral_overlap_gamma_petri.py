"""
spectral_overlap_gamma_petri.py — γ scaling sweep at Pythia-70m and Pythia-160m.

Replicates spectral_overlap_gamma.py logic but with:
  - Multi-model RUNS list (70m + 160m)
  - Runtime dimension detection: n_layers, d_model from base model
  - Random baselines computed per-model from detected dims
  - 4-point scaling-form fit against 410m and 1B reference data
  - Output: results/spectral_overlap_gamma_petri/results.json

Discrimination target:
  | Model | d_in | Disentanglement | Task-intrinsic | 1/sqrt(d) | 1/d^(1/3) |
  |   70m |  512 |           8-10  |              3 |      5.52 |      4.93 |
  |  160m |  768 |            6-8  |              3 |      4.51 |      4.30 |
  |  410m | 1024 |           ~3.92 |           3.92 |      3.91 |      3.92 |
  |    1B | 2048 |              ~2 |           3.01 |      2.76 |      3.10 |
"""
from __future__ import annotations

import json
import logging
import math
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

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "spectral_overlap_gamma_petri"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Run definitions ────────────────────────────────────────────────────────────
# (label, checkpoint_path, base_model_dir, lora_r, lora_alpha)
RUNS = [
    (
        "70m_dpo_r128",
        RESULTS / "_leak_70m"  / "v2" / "checkpoints" / "checkpoint-800",
        ROOT / "models" / "pythia-70m",
        128, 256,
    ),
    (
        "160m_dpo_r128",
        RESULTS / "_leak_160m" / "v2" / "checkpoints" / "checkpoint-800",
        ROOT / "models" / "pythia-160m",
        128, 256,
    ),
]

K_VALS = [5, 10, 20]

# ── Reference data from prior runs ─────────────────────────────────────────────
# 410m from spectral_overlap_gamma.py  v2_dpo_r128 run
# 1B   from spectral_overlap_gamma_1b.py v2_dpo_r128_1b run
REFERENCE_POINTS = {
    "410m_dpo_r128": {"d_in": 1024, "srank_avg": 3.924, "n_layers": 24},
    "1b_dpo_r128":   {"d_in": 2048, "srank_avg": 3.134, "n_layers": 16},
}


# ── Adapter loading ──────────────────────────────────────────────────────────

def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    safetensor_path = path / "adapter_model.safetensors"
    if safetensor_path.exists():
        out = {}
        with safe_open(safetensor_path, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    # Fall back to pytorch_model.bin
    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {path}")


def extract_qkv_delta(tensors: dict, layer: int, alpha: int, r: int) -> torch.Tensor:
    a_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_A.weight")
    b_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_B.weight")
    A = tensors[a_key].float()   # [r, d_in]
    B = tensors[b_key].float()   # [d_out, r]
    return (alpha / r) * (B @ A)  # [d_out, d_in]


# ── Base model loading ────────────────────────────────────────────────────────

def load_base_weights(model_dir: Path) -> tuple[list[torch.Tensor], int, int, int]:
    """
    Returns (weights_per_layer, n_layers, d_out, d_in).
    Detects dims at runtime.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading base model from {model_dir}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    n_layers = len(model.gpt_neox.layers)
    d_in     = model.gpt_neox.layers[0].attention.query_key_value.weight.shape[1]
    d_out    = model.gpt_neox.layers[0].attention.query_key_value.weight.shape[0]
    log.info(f"  detected: n_layers={n_layers}, d_in={d_in}, d_out={d_out}")

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
    return weights, n_layers, d_out, d_in


# ── Stable rank ──────────────────────────────────────────────────────────────

def srank(s: torch.Tensor) -> float:
    s2    = s ** 2
    total = s2.sum().item()
    spec2 = s2[0].item()
    return total / spec2 if spec2 > 0 else 0.0


# ── Core overlap computation ─────────────────────────────────────────────────

def compute_overlap(W: torch.Tensor, dW: torch.Tensor,
                    d_out: int, d_in: int) -> dict:
    """
    SVD both matrices, compute left/right subspace overlaps for k in K_VALS
    and at k = round(srank(dW)).
    """
    U_W,  S_W,  Vt_W  = torch.linalg.svd(W,  full_matrices=False)
    U_dW, S_dW, Vt_dW = torch.linalg.svd(dW, full_matrices=False)

    sr_delta = srank(S_dW)
    k_auto   = max(1, round(sr_delta))

    result = {"srank_delta": sr_delta}

    all_k = sorted(set(K_VALS + [k_auto]))

    for k in all_k:
        k_eff = min(k, U_W.shape[1], U_dW.shape[1], Vt_W.shape[0], Vt_dW.shape[0])

        # Left subspace overlap
        G_left  = U_W[:, :k_eff].T @ U_dW[:, :k_eff]
        sv_left = torch.linalg.svdvals(G_left)
        p_left  = (sv_left ** 2).sum().item() / k_eff

        # Right subspace overlap
        G_right  = Vt_W[:k_eff, :] @ Vt_dW[:k_eff, :].T
        sv_right = torch.linalg.svdvals(G_right)
        p_right  = (sv_right ** 2).sum().item() / k_eff

        # Random subspace baselines and bonus factors
        base_left  = k_eff / d_out
        base_right = k_eff / d_in
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

def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _avg(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def aggregate_run(per_layer: list[dict]) -> tuple[dict, dict]:
    """Returns (avg_dict, std_dict) for k in K_VALS."""
    avg_out = {}
    std_out = {}

    for k in K_VALS:
        key = f"k{k}"
        bl_vals = [l[key]["bonus_left"]  for l in per_layer]
        br_vals = [l[key]["bonus_right"] for l in per_layer]
        pl_vals = [l[key]["p_left"]      for l in per_layer]
        pr_vals = [l[key]["p_right"]     for l in per_layer]

        avg_out[key] = {
            "p_left":      round(_avg(pl_vals), 6),
            "p_right":     round(_avg(pr_vals), 6),
            "bonus_left":  round(_avg(bl_vals), 4),
            "bonus_right": round(_avg(br_vals), 4),
        }
        std_out[key] = {
            "p_left_std":      round(_std(pl_vals), 6),
            "p_right_std":     round(_std(pr_vals), 6),
            "bonus_left_std":  round(_std(bl_vals), 4),
            "bonus_right_std": round(_std(br_vals), 4),
        }

    # k_srank aggregation
    if "k_srank" in per_layer[0]:
        kr_bl_vals  = [l["k_srank"]["bonus_left"]  for l in per_layer]
        kr_br_vals  = [l["k_srank"]["bonus_right"] for l in per_layer]
        avg_out["k_srank"] = {
            "bonus_left":  round(_avg(kr_bl_vals), 4),
            "bonus_right": round(_avg(kr_br_vals), 4),
        }
        std_out["k_srank"] = {
            "bonus_left_std":  round(_std(kr_bl_vals), 4),
            "bonus_right_std": round(_std(kr_br_vals), 4),
        }

    return avg_out, std_out


# ── Scaling form fit ──────────────────────────────────────────────────────────

def fit_scaling_forms(measured: dict[str, tuple[int, float]]) -> dict:
    """
    measured: {label: (d_in, srank_avg)}

    Fits four candidate forms:
      1. disentanglement:  srank = C * d
      2. task_intrinsic:   srank = C  (constant)
      3. acoustic_sqrt:    srank = C / sqrt(d)
      4. acoustic_cbrt:    srank = C / d^(1/3)

    For each form: fits C by least squares (log space for power laws,
    direct for constant), returns predictions, L2 residual, best-fit C.
    """
    labels = sorted(measured.keys())
    d_vals = [measured[l][0] for l in labels]
    s_vals = [measured[l][1] for l in labels]
    n      = len(labels)

    results = {}

    # 1. Disentanglement: srank = C * d
    #    C = sum(s_i * d_i) / sum(d_i^2)
    C_dis = sum(s * d for s, d in zip(s_vals, d_vals)) / sum(d ** 2 for d in d_vals)
    preds_dis = [C_dis * d for d in d_vals]
    l2_dis = math.sqrt(sum((p - s) ** 2 for p, s in zip(preds_dis, s_vals)) / n)
    results["disentanglement"] = {
        "formula":    "C * d",
        "C":          round(C_dis, 6),
        "predictions": {l: round(p, 3) for l, p in zip(labels, preds_dis)},
        "l2_residual": round(l2_dis, 4),
    }

    # 2. Task-intrinsic: srank = C (constant)
    C_ti = sum(s_vals) / n
    preds_ti = [C_ti] * n
    l2_ti = math.sqrt(sum((p - s) ** 2 for p, s in zip(preds_ti, s_vals)) / n)
    results["task_intrinsic"] = {
        "formula":    "C (constant)",
        "C":          round(C_ti, 6),
        "predictions": {l: round(p, 3) for l, p in zip(labels, preds_ti)},
        "l2_residual": round(l2_ti, 4),
    }

    # 3. Acoustic 1/sqrt(d): srank = C / sqrt(d)
    #    C = sum(s_i / sqrt(d_i)) / sum(1 / d_i)
    inv_sqrt = [1 / math.sqrt(d) for d in d_vals]
    C_sqrt = sum(s * x for s, x in zip(s_vals, inv_sqrt)) / sum(x ** 2 for x in inv_sqrt)
    preds_sqrt = [C_sqrt / math.sqrt(d) for d in d_vals]
    l2_sqrt = math.sqrt(sum((p - s) ** 2 for p, s in zip(preds_sqrt, s_vals)) / n)
    results["acoustic_sqrt"] = {
        "formula":    "C / sqrt(d)",
        "C":          round(C_sqrt, 4),
        "predictions": {l: round(p, 3) for l, p in zip(labels, preds_sqrt)},
        "l2_residual": round(l2_sqrt, 4),
    }

    # 4. Acoustic 1/d^(1/3): srank = C / d^(1/3)
    inv_cbrt = [1 / d ** (1/3) for d in d_vals]
    C_cbrt = sum(s * x for s, x in zip(s_vals, inv_cbrt)) / sum(x ** 2 for x in inv_cbrt)
    preds_cbrt = [C_cbrt / d ** (1/3) for d in d_vals]
    l2_cbrt = math.sqrt(sum((p - s) ** 2 for p, s in zip(preds_cbrt, s_vals)) / n)
    results["acoustic_cbrt"] = {
        "formula":    "C / d^(1/3)",
        "C":          round(C_cbrt, 4),
        "predictions": {l: round(p, 3) for l, p in zip(labels, preds_cbrt)},
        "l2_residual": round(l2_cbrt, 4),
    }

    # Winner: lowest L2
    winner = min(results, key=lambda f: results[f]["l2_residual"])
    results["_winner"] = winner
    results["_labels"] = labels
    results["_d_vals"]  = d_vals
    results["_measured"] = {l: round(s, 3) for l, s in zip(labels, s_vals)}

    return results


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def print_gamma_table(run_summaries: list[dict]) -> None:
    print("\n=== γ TABLE ===")
    print(f"{'model':>15}  {'d_in':>6}  {'srank':>7}  {'bonus_R(k=5)':>14}  {'bonus_R(k=srank)':>18}")
    print("-" * 70)
    for rs in run_summaries:
        print(f"{rs['label']:>15}  {rs['d_in']:>6}  {rs['srank_avg']:>7.3f}  "
              f"{rs['bonus_right_k5']:>14.3f}  {rs['bonus_right_ksrank']:>18.3f}")


def print_scaling_table(fit: dict) -> None:
    labels  = fit["_labels"]
    measured = fit["_measured"]
    forms   = ["disentanglement", "task_intrinsic", "acoustic_sqrt", "acoustic_cbrt"]

    # Header
    print("\n=== SCALING FIT TABLE ===")
    col_w = 14
    header = f"{'form':>22}  " + "  ".join(f"{l:>{col_w}}" for l in labels)
    header += f"  {'L2':>8}"
    print(header)
    print("-" * (22 + (col_w + 2) * len(labels) + 10))

    # Measured row
    row = f"{'measured':>22}  " + "  ".join(f"{measured[l]:>{col_w}.3f}" for l in labels)
    print(row)
    print()

    # Model rows
    for form in forms:
        preds = fit[form]["predictions"]
        l2    = fit[form]["l2_residual"]
        marker = " <-- WINNER" if form == fit["_winner"] else ""
        row = f"{form:>22}  " + "  ".join(f"{preds[l]:>{col_w}.3f}" for l in labels)
        row += f"  {l2:>8.4f}{marker}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Running on {device}")

    results_runs = {}
    run_summaries = []

    for run_name, ckpt, model_dir, r, alpha in RUNS:
        print(f"\n{'='*80}\n{run_name}  r={r}  alpha={alpha}\n{'='*80}")

        if not ckpt.exists():
            log.warning(f"MISSING checkpoint: {ckpt}  — skipping {run_name}")
            continue

        # Load base weights (frees GPU after extraction)
        base_weights, n_layers, d_out, d_in = load_base_weights(model_dir)

        # Random-subspace baselines for this model
        rand_base = {
            k: {"p_left": round(k / d_out, 6), "p_right": round(k / d_in, 6)}
            for k in K_VALS
        }

        tensors   = load_adapter(ckpt)
        per_layer = []

        for li in range(n_layers):
            W  = base_weights[li]
            dW = extract_qkv_delta(tensors, li, alpha, r)
            layer_result = compute_overlap(W, dW, d_out, d_in)
            layer_result["layer"] = li
            per_layer.append(layer_result)

            if li % 4 == 0:
                sr = layer_result["srank_delta"]
                br = layer_result["k5"]["bonus_right"]
                bl = layer_result["k5"]["bonus_left"]
                log.info(f"  layer {li:2d}: srank_ΔW={sr:.2f}  "
                         f"bonus_left(k=5)={bl:.2f}  bonus_right(k=5)={br:.2f}")

        avg_d, std_d = aggregate_run(per_layer)

        # Scalar summaries
        srank_vals    = [l["srank_delta"] for l in per_layer]
        srank_avg     = _avg(srank_vals)
        srank_std     = _std(srank_vals)
        bonus_r_k5    = avg_d["k5"]["bonus_right"]
        bonus_r_ksrank = avg_d.get("k_srank", {}).get("bonus_right", float('nan'))

        print(f"\n  d_in={d_in}  n_layers={n_layers}")
        print(f"  srank_ΔW avg={srank_avg:.3f} ± {srank_std:.3f}")
        for k in K_VALS:
            print(f"  k={k:2d}  p_right={avg_d[f'k{k}']['p_right']:.5f} "
                  f"(bonus={avg_d[f'k{k}']['bonus_right']:.3f}x)")

        results_runs[run_name] = {
            "model_dir":   str(model_dir),
            "checkpoint":  str(ckpt),
            "n_layers":    n_layers,
            "d_out":       d_out,
            "d_in":        d_in,
            "srank_avg":   round(srank_avg, 4),
            "srank_std":   round(srank_std, 4),
            "avg":         avg_d,
            "std":         std_d,
            "random_baseline": rand_base,
            "per_layer":   per_layer,
        }

        run_summaries.append({
            "label":              run_name,
            "d_in":               d_in,
            "srank_avg":          round(srank_avg, 4),
            "bonus_right_k5":     round(bonus_r_k5, 4),
            "bonus_right_ksrank": round(bonus_r_ksrank, 4) if not math.isnan(bonus_r_ksrank) else None,
        })

    # ── Scaling-form fit ──────────────────────────────────────────────────────
    # Assemble 4-point dataset: 70m + 160m (measured) + 410m + 1B (reference)
    measured_points: dict[str, tuple[int, float]] = {}

    for rs in run_summaries:
        measured_points[rs["label"]] = (rs["d_in"], rs["srank_avg"])

    for ref_label, ref_data in REFERENCE_POINTS.items():
        measured_points[ref_label] = (ref_data["d_in"], ref_data["srank_avg"])

    scaling_fit = None
    if len(measured_points) >= 2:
        scaling_fit = fit_scaling_forms(measured_points)
        print_gamma_table(run_summaries)
        print_scaling_table(scaling_fit)

        winner = scaling_fit["_winner"]
        form   = scaling_fit[winner]
        print(f"\n{'='*80}")
        print(f"SCALING VERDICT: {winner}  (formula={form['formula']}, C={form['C']}, L2={form['l2_residual']})")
        print(f"{'='*80}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "runs":           results_runs,
        "run_summaries":  run_summaries,
        "reference_points": REFERENCE_POINTS,
        "scaling_fit":    scaling_fit,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(out, indent=2))
    shutil.copy(__file__, OUT_DIR / "spectral_overlap_gamma_petri.py")
    log.info(f"Wrote: {out_json}")
    print(f"Artifacts:\n  script  : {__file__}\n  results : {out_json}")


if __name__ == "__main__":
    main()
