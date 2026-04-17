"""Spectral autopsy: stable-rank of LoRA ΔW across runs.

For each run (v1/v2/v3), per LoRA-targeted layer, compute the singular
spectrum of ΔW = (alpha/r) · B @ A and report:

  - Frobenius norm ‖ΔW‖_F
  - spectral norm σ_max
  - stable rank = ‖ΔW‖_F² / σ_max²  (effective dimension of the update)
  - k90, k99 = smallest k such that Σ_{i≤k} σ_i² / Σ σ_i² ≥ 0.9, 0.99

Thesis under test: if r=128 runs use stable rank ≪ 128, the "LoRA
straitjacket" narrative fails — the contrastive gradient is itself
voluntarily low-rank and L-conservation is a structural property of
the objective, not an artifact of the rank cap.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PAPER_RESULTS_DIR, RESULTS_DIR  # noqa: E402


RESULTS = RESULTS_DIR / "_leak"
OUT_DIR = PAPER_RESULTS_DIR / "spectral_autopsy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("v1_dpo_r16",  RESULTS / "checkpoints" / "checkpoint-800", 16,  32),
    ("v2_dpo_r128", RESULTS / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v3_clm_r128", RESULTS / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
]


@dataclass
class LayerStats:
    layer: int
    module: str
    shape: tuple
    frob: float
    spec: float
    stable_rank: float
    k90: int
    k99: int
    configured_r: int


def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def extract_delta_w(tensors: dict, layer: int, module: str,
                     alpha: int, r: int) -> torch.Tensor | None:
    a_key = f"base_model.model.gpt_neox.layers.{layer}.{module}.lora_A.weight"
    b_key = f"base_model.model.gpt_neox.layers.{layer}.{module}.lora_B.weight"
    if a_key not in tensors or b_key not in tensors:
        return None
    A = tensors[a_key].float()  # [r, in]
    B = tensors[b_key].float()  # [out, r]
    return (alpha / r) * (B @ A)  # [out, in]


def analyze(delta_w: torch.Tensor, configured_r: int,
            layer: int, module: str) -> LayerStats:
    # SVD — full matrix, take singular values only.
    s = torch.linalg.svdvals(delta_w)
    s_sq = s ** 2
    total = s_sq.sum().item()
    frob = total ** 0.5
    spec = s[0].item()
    stable = total / (spec ** 2) if spec > 0 else 0.0
    cumsum = torch.cumsum(s_sq, 0) / total
    k90 = int((cumsum >= 0.90).nonzero()[0].item()) + 1
    k99 = int((cumsum >= 0.99).nonzero()[0].item()) + 1
    return LayerStats(
        layer=layer, module=module, shape=tuple(delta_w.shape),
        frob=frob, spec=spec, stable_rank=stable, k90=k90, k99=k99,
        configured_r=configured_r,
    )


MODULES = ["attention.query_key_value", "attention.dense",
           "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
N_LAYERS = 24  # Pythia-410m


def run_autopsy():
    all_results = {}
    for run_name, ckpt_path, r, alpha in RUNS:
        print(f"\n{'='*78}\n{run_name}  (r={r}, alpha={alpha})  {ckpt_path}\n{'='*78}")
        if not ckpt_path.exists():
            print(f"  MISSING: {ckpt_path}")
            continue
        tensors = load_adapter(ckpt_path)
        run_stats = []
        for layer in range(N_LAYERS):
            for module in MODULES:
                dw = extract_delta_w(tensors, layer, module, alpha, r)
                if dw is None:
                    continue
                s = analyze(dw, r, layer, module)
                run_stats.append(s)
        all_results[run_name] = run_stats
        per_module_agg(run_stats, r)
    # Save JSON
    out = {}
    for run_name, stats in all_results.items():
        out[run_name] = [s.__dict__ for s in stats]
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    shutil.copy(__file__, OUT_DIR / "spectral_autopsy.py")
    print(f"\nWrote: {OUT_DIR / 'results.json'}")
    # Headline
    print("\n" + "="*78)
    print("HEADLINE — stable rank / configured r, averaged across all layers × modules")
    print("="*78)
    for run_name, stats in all_results.items():
        if not stats: continue
        avg_stable = sum(s.stable_rank for s in stats) / len(stats)
        avg_k90 = sum(s.k90 for s in stats) / len(stats)
        avg_k99 = sum(s.k99 for s in stats) / len(stats)
        r = stats[0].configured_r
        print(f"  {run_name:14s}  srank={avg_stable:6.2f}/{r}  "
              f"k90={avg_k90:5.2f}  k99={avg_k99:5.2f}  "
              f"ratio={avg_stable/r:.3f}")


def per_module_agg(stats: list[LayerStats], r: int):
    by_mod = {}
    for s in stats:
        by_mod.setdefault(s.module, []).append(s)
    print(f"  {'module':20s}  {'srank_avg':>10s}  {'k90_avg':>8s}  {'k99_avg':>8s}  srank/r")
    for mod, xs in by_mod.items():
        avg_sr = sum(x.stable_rank for x in xs) / len(xs)
        avg_k90 = sum(x.k90 for x in xs) / len(xs)
        avg_k99 = sum(x.k99 for x in xs) / len(xs)
        print(f"  {mod:20s}  {avg_sr:10.2f}  {avg_k90:8.2f}  {avg_k99:8.2f}  {avg_sr/r:.3f}")


if __name__ == "__main__":
    run_autopsy()
