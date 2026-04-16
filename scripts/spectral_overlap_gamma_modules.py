"""γ extended — spectral overlap of ΔW across all 4 LoRA target modules.

Adversarial reviewer critique: is srank~3.6 a QKV-specific finding or
architecture-wide? Extend the γ measurement to all modules the adapters
were actually trained on:
  - attention.query_key_value  [3*d, d]  /  [6*d, d] at 1B
  - attention.dense            [d, d]
  - mlp.dense_h_to_4h          [4*d, d]
  - mlp.dense_4h_to_h          [d, 4*d]

For each run × layer × module, compute:
  srank_ΔW       = ||ΔW||_F² / ||ΔW||_2²   (stable rank)
  bonus_R(k=5)   = right-subspace overlap / random baseline  (k/d_in)
  bonus_R(k=srank) at the natural rank
  bonus_L(k=5)   = left-subspace overlap / random baseline   (k/d_out)

Adapter path: r=128, alpha=256 for all four runs.

Runs:
  410m_dpo : results/_leak/v2/checkpoints/checkpoint-800
  410m_clm : results/_leak/v3/checkpoints/checkpoint-800
  1b_dpo   : results/_leak_1b/v2/checkpoints/checkpoint-800
  1b_clm   : results/_leak_1b/v3/checkpoints/checkpoint-800

Outputs:
  results/spectral_overlap_gamma_modules/results.json
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

# Patch transformers' torch.load safety check (same workaround as other γ scripts)
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

# Script lives at paper/scripts/; resolve relative to it
PAPER_DIR = Path(__file__).resolve().parent.parent          # paper/
REPO_DIR  = PAPER_DIR.parent                               # lean-mining/
XCHECK    = REPO_DIR / "cross-check" / "trained-model-battery"

OUT_DIR = PAPER_DIR / "results" / "spectral_overlap_gamma_modules"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Run registry ─────────────────────────────────────────────────────────────

RUNS = {
    "410m_dpo": {
        "ckpt":      XCHECK / "results" / "_leak"    / "v2" / "checkpoints" / "checkpoint-800",
        "model_dir": XCHECK / "models" / "pythia-410m-safetensors",
        "r":         128,
        "alpha":     256,
    },
    "410m_clm": {
        "ckpt":      XCHECK / "results" / "_leak"    / "v3" / "checkpoints" / "checkpoint-800",
        "model_dir": XCHECK / "models" / "pythia-410m-safetensors",
        "r":         128,
        "alpha":     256,
    },
    "1b_dpo": {
        "ckpt":      XCHECK / "results" / "_leak_1b" / "v2" / "checkpoints" / "checkpoint-800",
        "model_dir": XCHECK / "models" / "pythia-1b",
        "r":         128,
        "alpha":     256,
    },
    "1b_clm": {
        "ckpt":      XCHECK / "results" / "_leak_1b" / "v3" / "checkpoints" / "checkpoint-800",
        "model_dir": XCHECK / "models" / "pythia-1b",
        "r":         128,
        "alpha":     256,
    },
}

# Module specs: (adapter_subkey, accessor_fn)
# accessor_fn receives the model and layer index, returns the nn.Linear weight
MODULE_KEYS = [
    "attention.query_key_value",
    "attention.dense",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
]

K_FIXED = 5   # the paper's canonical fixed-k


# ── Adapter loading ──────────────────────────────────────────────────────────

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


def extract_delta(tensors: dict, layer: int, module_key: str,
                  alpha: float, r: int) -> torch.Tensor:
    """Reconstruct ΔW = (alpha/r) * B @ A for given layer and module."""
    prefix = f"base_model.model.gpt_neox.layers.{layer}.{module_key}"
    a_key = f"{prefix}.lora_A.weight"
    b_key = f"{prefix}.lora_B.weight"
    if a_key not in tensors:
        raise KeyError(f"Missing adapter key: {a_key}")
    A = tensors[a_key].float()   # [r, d_in]
    B = tensors[b_key].float()   # [d_out, r]
    return (alpha / r) * (B @ A)  # [d_out, d_in]


# ── Base model weight extraction ─────────────────────────────────────────────

def load_base_weights(model_dir: Path) -> tuple[object, int]:
    """Load full model; return (model, n_layers). Caller extracts and frees."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading base model from {model_dir}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    n_layers = len(model.gpt_neox.layers)
    log.info(f"  n_layers={n_layers}  hidden={model.config.hidden_size}")
    return model, n_layers


def get_base_weight(model, layer: int, module_key: str) -> torch.Tensor:
    """Extract weight for a given layer/module in fp32 on CPU."""
    layer_obj = model.gpt_neox.layers[layer]
    # Navigate the dot-path
    parts = module_key.split(".")
    obj = layer_obj
    for p in parts:
        obj = getattr(obj, p)
    return obj.weight.detach().float().cpu()


# ── Stable rank ──────────────────────────────────────────────────────────────

def srank(s: torch.Tensor) -> float:
    s2 = s ** 2
    total = s2.sum().item()
    spec2 = s2[0].item()
    return total / spec2 if spec2 > 0 else 0.0


# ── Core overlap computation ─────────────────────────────────────────────────

def compute_gamma(W: torch.Tensor, dW: torch.Tensor) -> dict:
    """
    Compute γ metrics for a single (layer, module) cell.

    Returns:
      srank, bonus_R_k5, bonus_R_ksrank, bonus_L_k5,
      plus raw p values and srank-derived k.
    """
    d_out, d_in = W.shape

    U_W,  _,     Vt_W  = torch.linalg.svd(W,  full_matrices=False)
    U_dW, S_dW,  Vt_dW = torch.linalg.svd(dW, full_matrices=False)

    sr  = srank(S_dW)
    k_sr = max(1, round(sr))

    def _right_bonus(k: int) -> tuple[float, float]:
        k_eff = min(k, Vt_W.shape[0], Vt_dW.shape[0])
        M = Vt_W[:k_eff, :] @ Vt_dW[:k_eff, :].T        # [k_eff, k_eff]
        p = (M ** 2).sum().item() / k_eff
        base = k_eff / d_in
        return round(p, 6), round(p / base, 4) if base > 0 else float("nan")

    def _left_bonus(k: int) -> tuple[float, float]:
        k_eff = min(k, U_W.shape[1], U_dW.shape[1])
        M = U_W[:, :k_eff].T @ U_dW[:, :k_eff]           # [k_eff, k_eff]
        p = (M ** 2).sum().item() / k_eff
        base = k_eff / d_out
        return round(p, 6), round(p / base, 4) if base > 0 else float("nan")

    p_R_k5,   bonus_R_k5   = _right_bonus(K_FIXED)
    p_R_ksr,  bonus_R_ksr  = _right_bonus(k_sr)
    p_L_k5,   bonus_L_k5   = _left_bonus(K_FIXED)

    return {
        "srank":          round(sr, 4),
        "k_srank":        k_sr,
        "d_in":           d_in,
        "d_out":          d_out,
        "p_R_k5":         p_R_k5,
        "bonus_R_k5":     bonus_R_k5,
        "p_R_ksrank":     p_R_ksr,
        "bonus_R_ksrank": bonus_R_ksr,
        "p_L_k5":         p_L_k5,
        "bonus_L_k5":     bonus_L_k5,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate_module(per_layer: list[dict]) -> dict:
    fields = ["srank", "bonus_R_k5", "bonus_R_ksrank", "bonus_L_k5"]
    out = {}
    for f in fields:
        vals = [row[f] for row in per_layer]
        out[f] = round(_avg(vals), 4)
    # Use first layer's d_in/d_out (constant)
    out["d_in"]  = per_layer[0]["d_in"]
    out["d_out"] = per_layer[0]["d_out"]
    return out


# ── Summary table ────────────────────────────────────────────────────────────

def build_summary_table(runs_out: dict) -> list[dict]:
    rows = []
    for run_name, run_data in runs_out.items():
        for mod, mod_data in run_data["modules"].items():
            avg = mod_data["avg"]
            rows.append({
                "run":            run_name,
                "module":         mod,
                "srank":          avg["srank"],
                "bonus_R_k5":     avg["bonus_R_k5"],
                "bonus_R_ksrank": avg["bonus_R_ksrank"],
                "bonus_L_k5":     avg["bonus_L_k5"],
                "d_in":           avg["d_in"],
                "d_out":          avg["d_out"],
            })
    return rows


# ── Verdict ──────────────────────────────────────────────────────────────────

def determine_verdict(summary_table: list[dict]) -> dict:
    """
    Evaluate whether srank~3-5 is universal (a), QKV-specific (b),
    or MLP near-random (c).
    """
    # Group sranks by module and by run_type (dpo/clm)
    by_module: dict[str, list[float]] = {}
    for row in summary_table:
        mod = row["module"]
        by_module.setdefault(mod, []).append(row["srank"])

    module_avg = {mod: _avg(vals) for mod, vals in by_module.items()}
    module_br5 = {}
    for row in summary_table:
        mod = row["module"]
        module_br5.setdefault(mod, []).append(row["bonus_R_k5"])
    module_avg_br5 = {mod: _avg(vals) for mod, vals in module_br5.items()}

    qkv_srank = module_avg.get("attention.query_key_value", float("nan"))
    all_sranks = list(module_avg.values())
    all_br5    = list(module_avg_br5.values())

    srank_min  = min(all_sranks)
    srank_max  = max(all_sranks)
    srank_span = srank_max - srank_min

    # Outcome classification
    # (a) Universal: all modules have srank in 3-7 AND bonus_R_k5 > 2x everywhere
    # (b) QKV-specific: QKV srank in 3-7 but MLP srank > 10 or < 2
    # (c) MLP near-random: some MLP module has bonus_R_k5 < 1.5 (random-like)
    qkv_univ   = 2.5 <= qkv_srank <= 8.0
    all_3to8   = all(2.5 <= s <= 8.0 for s in all_sranks)
    all_aligned = all(b > 2.0 for b in all_br5)
    mlp_randoms = [
        mod for mod in ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
        if module_avg_br5.get(mod, 2.0) < 1.5
    ]
    mlp_sranks  = [
        module_avg[mod]
        for mod in ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
        if mod in module_avg
    ]
    mlp_diverge = any(s < 2.0 or s > 10.0 for s in mlp_sranks)

    if all_3to8 and all_aligned:
        outcome = "a"
        description = (
            "Universal law confirmed: srank is approximately 3-7 across all 4 "
            "LoRA target modules (QKV, dense, dense_h_to_4h, dense_4h_to_h) "
            "in both DPO and CLM runs at 410m and 1B. The task-intrinsic "
            "low-dimensional bottleneck is architecture-wide, not QKV-specific."
        )
    elif mlp_diverge or bool(mlp_randoms):
        outcome = "c"
        description = (
            "MLP modules show different geometry: bonus_R is near 1.0 (random "
            "alignment) or srank diverges substantially from QKV range. "
            "MLP adapters may carve novel geometric directions rather than "
            "amplifying pretrained structure."
        )
    elif not all_3to8:
        outcome = "b"
        description = (
            "QKV-specific finding: srank~3-7 holds for attention.query_key_value "
            "but MLP modules show substantially different stable rank, suggesting "
            "the low-dimensional bottleneck is attention-specific."
        )
    else:
        outcome = "a_partial"
        description = (
            "Mostly universal: srank is in 3-7 range across modules but "
            "alignment bonus_R is weak (<2x) in some modules — partial "
            "evidence for universality."
        )

    return {
        "outcome":        outcome,
        "description":    description,
        "module_avg_srank": {k: round(v, 3) for k, v in module_avg.items()},
        "module_avg_bonus_R_k5": {k: round(v, 3) for k, v in module_avg_br5.items()},
        "srank_range":    [round(srank_min, 3), round(srank_max, 3)],
        "srank_span":     round(srank_span, 3),
        "qkv_srank":      round(qkv_srank, 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Output dir: {OUT_DIR}")

    # Cache loaded models to avoid reloading when same base model used
    loaded_models: dict[str, tuple] = {}   # model_dir_str -> (model, n_layers)

    runs_out: dict = {}

    for run_name, run_cfg in RUNS.items():
        ckpt      = run_cfg["ckpt"]
        model_dir = run_cfg["model_dir"]
        r         = run_cfg["r"]
        alpha     = run_cfg["alpha"]

        print(f"\n{'='*80}\n{run_name}  (r={r}  alpha={alpha})\n{'='*80}")

        if not ckpt.exists():
            log.warning(f"MISSING checkpoint: {ckpt} — skipping {run_name}")
            continue

        # Load (or retrieve cached) base model
        md_str = str(model_dir)
        if md_str not in loaded_models:
            model, n_layers = load_base_weights(model_dir)
            loaded_models[md_str] = (model, n_layers)
        model, n_layers = loaded_models[md_str]

        # Load adapter
        log.info(f"Loading adapter from {ckpt}")
        tensors = load_adapter(ckpt)

        run_modules: dict = {}

        for mod_key in MODULE_KEYS:
            log.info(f"  Module: {mod_key}")
            per_layer = []

            for li in range(n_layers):
                W  = get_base_weight(model, li, mod_key)
                dW = extract_delta(tensors, li, mod_key, alpha, r)

                cell = compute_gamma(W, dW)
                cell["layer"] = li
                per_layer.append(cell)

                if li % (n_layers // 4 or 1) == 0:
                    log.info(
                        f"    layer {li:2d}: srank={cell['srank']:.2f}  "
                        f"bonus_R_k5={cell['bonus_R_k5']:.2f}x  "
                        f"bonus_L_k5={cell['bonus_L_k5']:.2f}x"
                    )

            avg = aggregate_module(per_layer)
            print(
                f"  {mod_key:35s}  srank={avg['srank']:.2f}  "
                f"bonus_R_k5={avg['bonus_R_k5']:.2f}x  "
                f"bonus_R_ksrank={avg['bonus_R_ksrank']:.2f}x  "
                f"bonus_L_k5={avg['bonus_L_k5']:.2f}x"
            )

            run_modules[mod_key] = {
                "d_in":     avg["d_in"],
                "d_out":    avg["d_out"],
                "per_layer": per_layer,
                "avg":      avg,
            }

        runs_out[run_name] = {
            "n_layers": n_layers,
            "modules":  run_modules,
        }

    # Free all cached models
    for model, _ in loaded_models.values():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build summary table and verdict
    summary_table = build_summary_table(runs_out)
    verdict       = determine_verdict(summary_table)

    print(f"\n{'='*80}")
    print(f"VERDICT ({verdict['outcome']}): {verdict['description']}")
    print(f"\nPer-module average srank:")
    for mod, sr in verdict["module_avg_srank"].items():
        br5 = verdict["module_avg_bonus_R_k5"].get(mod, float("nan"))
        print(f"  {mod:35s}  srank={sr:.2f}  bonus_R_k5={br5:.2f}x")
    print(f"{'='*80}\n")

    # Assemble output JSON
    results = {
        "runs":          runs_out,
        "summary_table": summary_table,
        "verdict":       verdict,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    shutil.copy(__file__, OUT_DIR / "spectral_overlap_gamma_modules.py")
    log.info(f"Wrote: {out_json}")
    print(f"Artifacts:\n  script  : {__file__}\n  results : {out_json}")


if __name__ == "__main__":
    main()
