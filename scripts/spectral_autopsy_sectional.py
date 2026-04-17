"""Sectional spectral autopsy: split ΔW into on-block (variance-quartile
block-diagonal) and off-block (cross-block) components, SVD each.

Prediction under test (Move β): if the conservation of L is a structural
property of the DPO/CLM gradient rather than a rank artifact, then

  - ΔW_on  has most of the Frobenius energy and a small stable rank (~5).
  - ΔW_off has near-zero Frobenius energy AND near-zero stable rank —
    the gradient is directionally orthogonal to the leak operator.

Uses the 4-quartile channel partition from
    results/_leak/v2/channel_partition.json
(derived from activation variance on the base Pythia-410m; applies to
all three runs since they share the same base.)

Scope: attention.query_key_value only. Matches the original L metric.
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
OUT_DIR = PAPER_RESULTS_DIR / "spectral_autopsy_sectional"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITION_JSON = RESULTS / "v2" / "channel_partition.json"

RUNS = [
    ("v1_dpo_r16",  RESULTS / "checkpoints" / "checkpoint-800", 16,  32),
    ("v2_dpo_r128", RESULTS / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v3_clm_r128", RESULTS / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
]


@dataclass
class SectionalStats:
    layer: int
    frob_on: float
    frob_off: float
    frob_total: float
    srank_on: float
    srank_off: float
    k90_on: int
    k90_off: int
    leak_fraction: float   # frob_off^2 / frob_total^2
    rank_on_over_r: float  # srank_on / configured_r


def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def extract_qkv_delta(tensors: dict, layer: int, alpha: int, r: int) -> torch.Tensor:
    a_key = f"base_model.model.gpt_neox.layers.{layer}.attention.query_key_value.lora_A.weight"
    b_key = f"base_model.model.gpt_neox.layers.{layer}.attention.query_key_value.lora_B.weight"
    A = tensors[a_key].float()
    B = tensors[b_key].float()
    return (alpha / r) * (B @ A)  # [3072, 1024]


def build_block_masks(quartiles: torch.Tensor, d_rows: int) -> tuple[torch.Tensor, list]:
    """Build an on-block mask for a [d_rows, 1024] matrix given per-column
    quartile labels. Rows split proportionally to column-block widths."""
    d_cols = quartiles.numel()
    mask = torch.zeros((d_rows, d_cols), dtype=torch.bool)
    col_blocks = [torch.where(quartiles == q)[0] for q in range(4)]
    # Row blocks: proportional split of d_rows by col-block widths.
    row_blocks = []
    running = 0
    for cb in col_blocks:
        chunk = int(round(len(cb) / d_cols * d_rows))
        row_blocks.append(torch.arange(running, min(running + chunk, d_rows)))
        running += chunk
    # Dump any remainder into the last block.
    if running < d_rows:
        row_blocks[-1] = torch.cat([row_blocks[-1], torch.arange(running, d_rows)])
    for rb, cb in zip(row_blocks, col_blocks):
        mask[rb.unsqueeze(1), cb.unsqueeze(0)] = True
    return mask, col_blocks


def svd_metrics(W: torch.Tensor) -> tuple[float, float, int]:
    """Return (frobenius, stable_rank, k90)."""
    if W.abs().sum().item() == 0.0:
        return 0.0, 0.0, 0
    s = torch.linalg.svdvals(W)
    s_sq = s ** 2
    total = s_sq.sum().item()
    frob = total ** 0.5
    spec = s[0].item()
    stable = total / (spec ** 2) if spec > 0 else 0.0
    cumsum = torch.cumsum(s_sq, 0) / total
    k90_idx = (cumsum >= 0.90).nonzero()
    k90 = int(k90_idx[0].item()) + 1 if len(k90_idx) > 0 else int(s.numel())
    return frob, stable, k90


def analyze_layer(dw: torch.Tensor, quartiles: torch.Tensor,
                   r: int, layer: int) -> SectionalStats:
    mask, _ = build_block_masks(quartiles, dw.shape[0])
    dw_on = dw * mask
    dw_off = dw * (~mask)
    frob_on, sr_on, k90_on = svd_metrics(dw_on)
    frob_off, sr_off, k90_off = svd_metrics(dw_off)
    frob_total = (frob_on ** 2 + frob_off ** 2) ** 0.5
    leak = (frob_off ** 2) / (frob_total ** 2) if frob_total > 0 else 0.0
    return SectionalStats(
        layer=layer, frob_on=frob_on, frob_off=frob_off, frob_total=frob_total,
        srank_on=sr_on, srank_off=sr_off, k90_on=k90_on, k90_off=k90_off,
        leak_fraction=leak, rank_on_over_r=sr_on / r,
    )


def run_sectional():
    with open(PARTITION_JSON) as f:
        partition = json.load(f)["partition"]

    all_out = {}
    for run_name, ckpt, r, alpha in RUNS:
        print(f"\n{'=' * 80}\n{run_name}  r={r}  alpha={alpha}\n{'=' * 80}")
        if not ckpt.exists():
            print(f"  MISSING: {ckpt}")
            continue
        tensors = load_adapter(ckpt)
        per_layer = []
        for li in range(24):
            quartiles = torch.tensor(partition[str(li)]["input_quartiles"])
            dw = extract_qkv_delta(tensors, li, alpha, r)
            s = analyze_layer(dw, quartiles, r, li)
            per_layer.append(s)
        all_out[run_name] = per_layer

        # Aggregate per run.
        avg_leak = sum(x.leak_fraction for x in per_layer) / len(per_layer)
        avg_sr_on = sum(x.srank_on for x in per_layer) / len(per_layer)
        avg_sr_off = sum(x.srank_off for x in per_layer) / len(per_layer)
        avg_k90_on = sum(x.k90_on for x in per_layer) / len(per_layer)
        avg_k90_off = sum(x.k90_off for x in per_layer) / len(per_layer)
        avg_frob_on = sum(x.frob_on for x in per_layer) / len(per_layer)
        avg_frob_off = sum(x.frob_off for x in per_layer) / len(per_layer)
        print(f"  leak_fraction (frob_off²/total²): {avg_leak:.4f}  "
              f"(iso baseline w/ 4 equal blocks = {1 - 4*(0.25**2):.3f} = 0.750)")
        print(f"  srank_on  = {avg_sr_on:6.2f}/{r}   srank_off = {avg_sr_off:6.2f}")
        print(f"  k90_on    = {avg_k90_on:6.2f}     k90_off   = {avg_k90_off:6.2f}")
        print(f"  frob_on   = {avg_frob_on:.4f}   frob_off   = {avg_frob_off:.4f}   "
              f"ratio off/on = {avg_frob_off/avg_frob_on:.3f}")

    out_json = {
        run: [s.__dict__ for s in stats] for run, stats in all_out.items()
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    shutil.copy(__file__, OUT_DIR / "spectral_autopsy_sectional.py")
    print(f"\nWrote: {OUT_DIR / 'results.json'}")

    # Headline comparison: does the update "know" about the block structure?
    print("\n" + "=" * 80)
    print("HEADLINE — does ΔW respect the block-diagonal variance partition?")
    print("=" * 80)
    print("Isotropic baseline: leak_fraction = 0.750  (4 equal blocks ⇒ 12/16 off-block)")
    for run_name, stats in all_out.items():
        if not stats: continue
        avg_leak = sum(x.leak_fraction for x in stats) / len(stats)
        avg_sr_on = sum(x.srank_on for x in stats) / len(stats)
        avg_sr_off = sum(x.srank_off for x in stats) / len(stats)
        r = stats[0].rank_on_over_r and (stats[0].srank_on / stats[0].rank_on_over_r) or 0
        print(f"  {run_name:14s}  "
              f"leak={avg_leak:.4f}  Δ_iso={avg_leak-0.750:+.4f}  "
              f"srank_on={avg_sr_on:.2f}  srank_off={avg_sr_off:.2f}  "
              f"ratio_off/on={avg_sr_off/avg_sr_on:.3f}")


if __name__ == "__main__":
    run_sectional()
