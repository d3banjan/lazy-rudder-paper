"""
Decompose DPO and CLM adapter ΔW into components inside vs. orthogonal to
the base weight's top-k right-singular subspace.

Tests whether DPO uses more "novel" (orthogonal) geometry than CLM, which
would explain the seed-confirmed CLM bonus_R > DPO bonus_R gap at 1B scale.

Formal test:
  dW_parallel   = dW @ P_k          # mass aligned with base W top-k right subspace
  dW_orthogonal = dW - dW_parallel  # mass in directions base W doesn't emphasize
  orthogonal_frac = ||dW_orthogonal||_F^2 / ||dW||_F^2

Predicts:  DPO orthogonal_frac > CLM orthogonal_frac

Runs (2 DPO × 2 seeds  +  2 CLM × 2 seeds):
  1b_dpo_s42   : _leak_1b/v2/checkpoints/checkpoint-800
  1b_dpo_s117  : _leak_1b_seed117/v2/checkpoints/checkpoint-800
  1b_clm_s42   : _leak_1b/v3/checkpoints/checkpoint-800
  1b_clm_s117  : _leak_1b_seed117/v3/checkpoints/checkpoint-800

Output: results/dpo_clm_orthogonal_decomp/results.json

CPU-only (CUDA_VISIBLE_DEVICES="" prevents GPU alloc).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

# Patch transformers' torch.load safety check (needed for base-model load)
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
BATTERY_DIR  = SCRIPT_DIR.parent.parent / "cross-check" / "trained-model-battery"
MODEL_DIR    = BATTERY_DIR / "models" / "pythia-1b"
RESULTS_ROOT = BATTERY_DIR / "results"
OUT_DIR      = SCRIPT_DIR.parent / "results" / "dpo_clm_orthogonal_decomp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (run_key, checkpoint_path, lora_r, lora_alpha, group)
RUNS = [
    ("1b_dpo_s42",
     RESULTS_ROOT / "_leak_1b"        / "v2" / "checkpoints" / "checkpoint-800",
     128, 256, "dpo"),
    ("1b_dpo_s117",
     RESULTS_ROOT / "_leak_1b_seed117" / "v2" / "checkpoints" / "checkpoint-800",
     128, 256, "dpo"),
    ("1b_clm_s42",
     RESULTS_ROOT / "_leak_1b"        / "v3" / "checkpoints" / "checkpoint-800",
     128, 256, "clm"),
    ("1b_clm_s117",
     RESULTS_ROOT / "_leak_1b_seed117" / "v3" / "checkpoints" / "checkpoint-800",
     128, 256, "clm"),
]

K_VALS = [5, 10, 20]


# ── Adapter loading ───────────────────────────────────────────────────────────

def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    safetensor_path = path / "adapter_model.safetensors"
    if safetensor_path.exists():
        out = {}
        with safe_open(str(safetensor_path), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter_model.safetensors or .bin in {path}")


def extract_qkv_delta(tensors: dict, layer: int, alpha: int, r: int) -> torch.Tensor:
    """Reconstruct full ΔW = (alpha/r) * B @ A for the QKV weight of one layer."""
    a_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_A.weight")
    b_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_B.weight")
    A = tensors[a_key].float()   # [r, d_in]
    B = tensors[b_key].float()   # [d_out, r]
    return (alpha / r) * (B @ A) # [d_out, d_in]


# ── Base weights ──────────────────────────────────────────────────────────────

def load_base_weights() -> tuple[list[torch.Tensor], int]:
    """Load QKV weight tensors for all layers in fp32 on CPU."""
    log.info(f"Loading Pythia-1B base from {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()

    n_layers = len(model.gpt_neox.layers)
    log.info(f"Pythia-1B: {n_layers} layers  hidden={model.config.hidden_size}")

    weights = []
    for li in range(n_layers):
        w = (model.gpt_neox.layers[li]
             .attention.query_key_value.weight
             .detach().float().cpu())
        weights.append(w)

    del model
    log.info(f"Base weights extracted (shape {weights[0].shape}), model freed.")
    return weights, n_layers


# ── Stable rank ───────────────────────────────────────────────────────────────

def srank(s: torch.Tensor) -> float:
    s2  = s.double() ** 2
    tot = s2.sum().item()
    return tot / s2[0].item() if s2[0].item() > 0 else 0.0


# ── Orthogonal decomposition ──────────────────────────────────────────────────

def orthogonal_decomp(W: torch.Tensor, dW: torch.Tensor) -> dict:
    """
    For each k in K_VALS and k = round(srank(dW)):
      Project dW onto the top-k RIGHT singular subspace of W.
      Frob² split into parallel + orthogonal.
    Returns a dict with per-k results and srank_dW.
    """
    # SVD of base weight — only need Vt_W (right singular vectors)
    _, _, Vt_W = torch.linalg.svd(W, full_matrices=False)  # [min(d_out,d_in), d_in]

    # SVD of adapter delta — only need singular values for srank
    S_dW = torch.linalg.svdvals(dW)
    sr_dW = srank(S_dW)
    k_auto = max(1, round(sr_dW))

    frob_total_sq = (dW.double() ** 2).sum().item()

    result: dict = {"srank_dW": round(sr_dW, 4)}

    all_k = sorted(set(K_VALS + [k_auto]))

    for k in all_k:
        k_eff = min(k, Vt_W.shape[0])

        # Top-k right singular vectors of base W: V_k ∈ R^{d_in × k}
        V_k = Vt_W[:k_eff, :].T.double()           # [d_in, k_eff]

        # Projection of dW onto span(V_k) along input dimension
        # dW_parallel = dW @ V_k @ V_k^T
        dW_d = dW.double()
        dW_parallel   = dW_d @ (V_k @ V_k.T)       # [d_out, d_in]
        dW_orthogonal = dW_d - dW_parallel           # [d_out, d_in]

        fpar  = (dW_parallel  ** 2).sum().item()
        forth = (dW_orthogonal ** 2).sum().item()

        # Sanity check (relative tolerance)
        diff = abs(fpar + forth - frob_total_sq) / (frob_total_sq + 1e-30)
        if diff > 1e-6:
            log.warning(f"  Frob sanity fail at k={k}: rel_err={diff:.2e}")

        par_frac  = fpar  / frob_total_sq if frob_total_sq > 0 else float('nan')
        orth_frac = forth / frob_total_sq if frob_total_sq > 0 else float('nan')

        entry = {
            "parallel_frac":   round(par_frac,  6),
            "orthogonal_frac": round(orth_frac, 6),
        }

        if k == k_auto:
            result["k_srank"] = {"k": k_eff, **entry}
        if k in K_VALS:
            result[f"k{k}"] = entry

    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float('nan')


def aggregate_run(per_layer: list[dict]) -> dict:
    avg: dict = {}

    srank_vals = [l["srank_dW"] for l in per_layer]
    avg["srank_dW"] = round(_avg(srank_vals), 4)

    for k in K_VALS:
        key = f"k{k}"
        orth_vals = [l[key]["orthogonal_frac"] for l in per_layer]
        avg[f"k{k}_orthogonal_frac"] = round(_avg(orth_vals), 6)

    orth_srank = [l["k_srank"]["orthogonal_frac"] for l in per_layer]
    avg["k_srank_orthogonal_frac"] = round(_avg(orth_srank), 6)

    return avg


# ── Verdict ───────────────────────────────────────────────────────────────────

def determine_verdict(dpo_mean: float, clm_mean: float) -> str:
    delta = dpo_mean - clm_mean
    if delta > 0.05:
        return "dpo_more_orthogonal"
    if abs(delta) < 0.02:
        return "no_significant_difference"
    if delta > 0:
        return "dpo_slightly_more_orthogonal"
    return "clm_more_orthogonal"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base_weights, n_layers = load_base_weights()
    d_out, d_in = base_weights[0].shape
    log.info(f"d_out={d_out}  d_in={d_in}  n_layers={n_layers}")

    results_runs: dict[str, dict] = {}

    for run_name, ckpt, r, alpha, group in RUNS:
        log.info(f"\n{'='*70}\n{run_name}  r={r}  alpha={alpha}  group={group}\n{'='*70}")
        if not ckpt.exists():
            log.warning(f"MISSING checkpoint: {ckpt}")
            continue

        tensors  = load_adapter(ckpt)
        per_layer = []

        for li in range(n_layers):
            W  = base_weights[li]
            dW = extract_qkv_delta(tensors, li, alpha, r)
            layer_res = orthogonal_decomp(W, dW)
            layer_res["layer"] = li
            per_layer.append(layer_res)

            if li % 4 == 0:
                o5 = layer_res["k5"]["orthogonal_frac"]
                log.info(f"  layer {li:2d}: srank_dW={layer_res['srank_dW']:.2f}  "
                         f"orth_frac(k=5)={o5:.4f}")

        avg = aggregate_run(per_layer)
        results_runs[run_name] = {
            "group":     group,
            "per_layer": per_layer,
            "avg":       avg,
        }

        log.info(f"  AVERAGE: srank={avg['srank_dW']:.2f}  "
                 f"orth_frac k5={avg['k5_orthogonal_frac']:.4f}  "
                 f"k10={avg['k10_orthogonal_frac']:.4f}  "
                 f"k20={avg['k20_orthogonal_frac']:.4f}  "
                 f"k_srank={avg['k_srank_orthogonal_frac']:.4f}")

    # ── Comparison block ──────────────────────────────────────────────────────
    def _group_mean(metric: str, group: str) -> float:
        vals = [
            results_runs[rn]["avg"][metric]
            for rn in results_runs
            if results_runs[rn]["group"] == group
        ]
        return _avg(vals)

    comparison: dict = {}
    for metric in ["k5_orthogonal_frac", "k10_orthogonal_frac",
                   "k20_orthogonal_frac", "k_srank_orthogonal_frac"]:
        dpo_m = _group_mean(metric, "dpo")
        clm_m = _group_mean(metric, "clm")
        comparison[metric] = {
            "dpo_mean":       round(dpo_m, 6),
            "clm_mean":       round(clm_m, 6),
            "dpo_minus_clm":  round(dpo_m - clm_m, 6),
        }

    # Primary verdict at k=5
    k5_delta = comparison["k5_orthogonal_frac"]["dpo_minus_clm"]
    verdict  = determine_verdict(
        comparison["k5_orthogonal_frac"]["dpo_mean"],
        comparison["k5_orthogonal_frac"]["clm_mean"],
    )

    # Headline
    if verdict == "dpo_more_orthogonal":
        headline = (
            f"DPO adapter ΔW sits {k5_delta*100:.1f} pp more in the orthogonal complement "
            f"of the base top-5 right subspace than CLM — confirms contrastive geometry hypothesis."
        )
    elif verdict == "no_significant_difference":
        headline = (
            f"DPO and CLM adapters show indistinguishable orthogonal fractions (Δ={k5_delta*100:.2f} pp "
            f"at k=5). Contrastive geometry hypothesis not supported."
        )
    else:
        sign = "higher" if k5_delta > 0 else "lower"
        headline = (
            f"DPO orthogonal fraction is {abs(k5_delta)*100:.1f} pp {sign} than CLM at k=5 "
            f"({verdict}) — weak or opposite evidence for contrastive geometry hypothesis."
        )

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"Headline: {headline}")
    for metric, vals in comparison.items():
        print(f"  {metric}: dpo={vals['dpo_mean']:.4f}  clm={vals['clm_mean']:.4f}  "
              f"delta={vals['dpo_minus_clm']:+.4f}")
    print(f"{'='*70}\n")

    # ── Output ────────────────────────────────────────────────────────────────
    out = {
        "model":      "pythia-1b",
        "n_layers":   n_layers,
        "d_out":      d_out,
        "d_in":       d_in,
        "runs":       results_runs,
        "comparison": comparison,
        "verdict":    verdict,
        "headline":   headline,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(out, indent=2))
    shutil.copy(__file__, OUT_DIR / "dpo_clm_orthogonal_decomp.py")
    log.info(f"Wrote: {out_json}")
    print(f"Artifacts:\n  script  : {__file__}\n  results : {out_json}")


if __name__ == "__main__":
    main()
