"""Quick seed-variance check: γ metric on 1B DPO at seeds 42 and 117.

Computes per-layer stable rank and right-subspace overlap (bonus_R) for
both adapters at k=5 and k=round(srank). Reports side-by-side.

CPU-only (base model in fp32; SVD on 16 layers × [6144, 2048]).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import BASE_DIR, MODELS_DIR, PAPER_RESULTS_DIR, RESULTS_DIR

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

BASE = MODELS_DIR / "pythia-1b"

RUNS = [
    ("seed42_DPO",  RESULTS_DIR / "_leak_1b"         / "v2" / "checkpoints" / "checkpoint-800"),
    ("seed117_DPO", RESULTS_DIR / "_leak_1b_seed117" / "v2" / "checkpoints" / "checkpoint-800"),
]
R, ALPHA = 128, 256
N_LAYERS = 16


def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def srank_and_bonus(dW: torch.Tensor, W: torch.Tensor, k_fixed: int = 5) -> dict:
    """Return stable rank of dW + right-subspace bonus at k_fixed and k=srank."""
    # Stable rank
    s_dW = torch.linalg.svdvals(dW)
    frob_sq = (s_dW ** 2).sum().item()
    spec_sq = (s_dW[0] ** 2).item()
    srank = frob_sq / spec_sq if spec_sq > 0 else 0.0

    # Right-subspace of both (full SVD for alignment)
    _, _, Vt_dW = torch.linalg.svd(dW, full_matrices=False)   # [min, d_in]
    _, _, Vt_W  = torch.linalg.svd(W,  full_matrices=False)   # [min, d_in]

    d_in = W.shape[1]  # 2048

    def bonus_at_k(k: int) -> float:
        V_dW_k = Vt_dW[:k, :]
        V_W_k  = Vt_W[:k, :]
        # p_right = ||V_W_k @ V_dW_k^T||_F^2 / k
        M = V_W_k @ V_dW_k.T   # [k, k]
        p = (M ** 2).sum().item() / k
        random_baseline = k / d_in
        return p / random_baseline

    k_nat = max(1, round(srank))
    return {
        "srank":        srank,
        "k_nat":        k_nat,
        "bonus_k5":     bonus_at_k(k_fixed),
        "bonus_k_nat":  bonus_at_k(k_nat),
    }


def main():
    print(f"Loading base Pythia-1B from {BASE}")
    base = AutoModelForCausalLM.from_pretrained(str(BASE), torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    W_base = [base.gpt_neox.layers[i].attention.query_key_value.weight.detach().float().cpu()
              for i in range(N_LAYERS)]
    print(f"Base weights cached: {len(W_base)} layers, shape {W_base[0].shape}")
    del base

    all_results = {}
    for run_name, ckpt in RUNS:
        if not ckpt.exists():
            print(f"SKIP {run_name}: {ckpt} missing")
            continue
        print(f"\n{run_name}")
        tensors = load_adapter(ckpt)
        per_layer = []
        for i in range(N_LAYERS):
            a_key = f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_A.weight"
            b_key = f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_B.weight"
            A = tensors[a_key].float()
            B = tensors[b_key].float()
            dW = (ALPHA / R) * (B @ A)
            stats = srank_and_bonus(dW, W_base[i])
            stats["layer"] = i
            per_layer.append(stats)
        avg = {
            "srank":       sum(p["srank"] for p in per_layer) / N_LAYERS,
            "bonus_k5":    sum(p["bonus_k5"] for p in per_layer) / N_LAYERS,
            "bonus_k_nat": sum(p["bonus_k_nat"] for p in per_layer) / N_LAYERS,
        }
        all_results[run_name] = {"per_layer": per_layer, "avg": avg}
        print(f"  avg srank       = {avg['srank']:.3f}")
        print(f"  avg bonus_k5    = {avg['bonus_k5']:.3f}×")
        print(f"  avg bonus_k_nat = {avg['bonus_k_nat']:.3f}×")

    print("\n" + "=" * 70)
    print("SEED-VARIANCE COMPARISON — 1B DPO r=128")
    print("=" * 70)
    print(f"{'run':18s} {'srank':>8s} {'bonus_k5':>10s} {'bonus_k_nat':>12s}")
    for name, d in all_results.items():
        a = d["avg"]
        print(f"{name:18s} {a['srank']:>8.3f} {a['bonus_k5']:>10.3f} {a['bonus_k_nat']:>12.3f}")

    # Delta
    if len(all_results) == 2:
        names = list(all_results.keys())
        a0, a1 = all_results[names[0]]["avg"], all_results[names[1]]["avg"]
        print(f"\nΔsrank       = {a1['srank'] - a0['srank']:+.3f}")
        print(f"Δbonus_k5    = {a1['bonus_k5'] - a0['bonus_k5']:+.3f}×")
        print(f"Δbonus_k_nat = {a1['bonus_k_nat'] - a0['bonus_k_nat']:+.3f}×")

    out = PAPER_RESULTS_DIR / "seed_variance_quick"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
