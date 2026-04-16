"""
Bias-theory autopsy via decomposition.

Tests whether LoRA ΔW on attention.query_key_value can be reconstructed as a
LayerNorm γ-gain modulation of the base weight matrix:

    ΔW_pred_right = W_base @ diag(g)        (per-input-channel scaling,
                                              matches LN-γ before QKV)
    ΔW_pred_left  = diag(g) @ W_base         (per-output-channel scaling,
                                              for comparison)

Residual Frobenius fraction = ||ΔW - ΔW_pred||² / ||ΔW||².

Verdict:
  < 0.20 → gain-theory resurrected (ΔW ~ LN-γ only)
  > 0.80 → matrix rewiring genuine (γ's 5-dim rudder is real geometry)
  else   → mixed (partial gain + partial rewire)

Runs on existing adapters (no training needed, ~2 min CPU).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import MODELS_DIR, RESULTS_DIR, BASE_DIR

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM


# HF repo: EleutherAI/pythia-410m (safetensors native)
BASE = MODELS_DIR / "pythia-410m"
OUT = RESULTS_DIR / "bias_theory_autopsy"
OUT.mkdir(exist_ok=True)

RUNS = [
    ("v1_dpo_r16",  RESULTS_DIR / "_leak" / "checkpoints" / "checkpoint-800",     16,  32),
    ("v2_dpo_r128", RESULTS_DIR / "_leak" / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v3_clm_r128", RESULTS_DIR / "_leak" / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
]


def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def decompose_layer(dW: torch.Tensor, W: torch.Tensor) -> dict:
    """Fit ΔW ≈ W @ diag(g_R)  and  ΔW ≈ diag(g_L) @ W  separately.
    Report residual Frobenius fraction for each."""
    frob_dW_sq = (dW * dW).sum().item()
    if frob_dW_sq == 0.0:
        return {"frob_dW_sq": 0.0, "res_frac_right": 0.0, "res_frac_left": 0.0,
                "g_right_mean": 0.0, "g_right_std": 0.0,
                "g_left_mean": 0.0, "g_left_std": 0.0}

    # Right-mult: per-column scalar g_R[j] such that dW[:,j] ≈ g_R[j] * W[:,j]
    col_dots = (dW * W).sum(dim=0)              # [d_in]
    col_sq   = (W * W).sum(dim=0).clamp_min(1e-12)
    g_R      = col_dots / col_sq                # [d_in]
    dW_pred_R = W * g_R.unsqueeze(0)            # broadcast [d_out, d_in]
    res_R_sq  = ((dW - dW_pred_R) ** 2).sum().item()

    # Left-mult: per-row scalar g_L[i] such that dW[i,:] ≈ g_L[i] * W[i,:]
    row_dots = (dW * W).sum(dim=1)              # [d_out]
    row_sq   = (W * W).sum(dim=1).clamp_min(1e-12)
    g_L      = row_dots / row_sq                # [d_out]
    dW_pred_L = W * g_L.unsqueeze(1)            # broadcast
    res_L_sq  = ((dW - dW_pred_L) ** 2).sum().item()

    return {
        "frob_dW_sq":     frob_dW_sq,
        "res_frac_right": res_R_sq / frob_dW_sq,
        "res_frac_left":  res_L_sq / frob_dW_sq,
        "g_right_mean":   g_R.mean().item(),
        "g_right_std":    g_R.std().item(),
        "g_left_mean":    g_L.mean().item(),
        "g_left_std":     g_L.std().item(),
    }


def verdict_from_frac(frac: float) -> str:
    if frac < 0.20:
        return "GAIN_THEORY_RESURRECTED"
    if frac > 0.80:
        return "MATRIX_REWIRE_REAL"
    return "MIXED"


def main():
    print(f"Loading base Pythia-410m from {BASE}")
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE), torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    W_base = [
        base.gpt_neox.layers[i].attention.query_key_value.weight.detach().float().cpu()
        for i in range(24)
    ]
    print(f"Base weights cached: {len(W_base)} layers, shape {W_base[0].shape}")
    del base

    results = {}
    for name, ckpt, r, alpha in RUNS:
        if not ckpt.exists():
            print(f"SKIP {name}: {ckpt} missing")
            continue
        print(f"\n{name}  r={r}  α={alpha}")
        tensors = load_adapter(ckpt)
        per_layer = []
        for i in range(24):
            a_key = f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_A.weight"
            b_key = f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_B.weight"
            A = tensors[a_key].float()
            B = tensors[b_key].float()
            dW = (alpha / r) * (B @ A)           # [3072, 1024]
            stats = decompose_layer(dW, W_base[i])
            stats["layer"] = i
            per_layer.append(stats)

        avg_R = sum(p["res_frac_right"] for p in per_layer) / len(per_layer)
        avg_L = sum(p["res_frac_left"]  for p in per_layer) / len(per_layer)
        verdict_R = verdict_from_frac(avg_R)
        verdict_L = verdict_from_frac(avg_L)
        results[name] = {
            "r":            r,
            "alpha":        alpha,
            "per_layer":    per_layer,
            "avg_res_frac_right": avg_R,
            "avg_res_frac_left":  avg_L,
            "verdict_right": verdict_R,
            "verdict_left":  verdict_L,
        }
        print(f"  avg residual frac (right-mult / LN-γ hypothesis): {avg_R:.4f}  →  {verdict_R}")
        print(f"  avg residual frac (left-mult  / output-gain):      {avg_L:.4f}  →  {verdict_L}")
        print(f"  range right: [{min(p['res_frac_right'] for p in per_layer):.4f}, "
              f"{max(p['res_frac_right'] for p in per_layer):.4f}]")
        print(f"  range left : [{min(p['res_frac_left']  for p in per_layer):.4f}, "
              f"{max(p['res_frac_left']  for p in per_layer):.4f}]")

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT / 'results.json'}")

    print("\n" + "=" * 70)
    print("HEADLINE TABLE")
    print("=" * 70)
    print(f"{'run':14s}  {'res_frac_R':>10s}  {'res_frac_L':>10s}  {'verdict (right)':>28s}")
    for name, r_data in results.items():
        print(f"{name:14s}  {r_data['avg_res_frac_right']:10.4f}  "
              f"{r_data['avg_res_frac_left']:10.4f}  {r_data['verdict_right']:>28s}")


if __name__ == "__main__":
    main()
