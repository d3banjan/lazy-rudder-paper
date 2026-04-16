"""Task #9 — re-measure L across #8's DPO checkpoints using a non-uniform,
orbit-fraction-matched partition to break the geometric k=4 isotropic baseline.

Partition widths match Pythia-410m DPO orbit fractions:
    gauge = 0.5%   → 5 channels
    SRN   = 66.2%  → 678 channels
    PS    = 33.3%  → 341 channels
    (d_model = 1024)

Channels assigned to blocks by ascending variance:
    bottom 33.3% by variance → PS (dormant)
    next   66.2%              → SRN (active)
    top    0.5%               → gauge

Isotropic baseline for this partition (random matrix, any partition):
    L_iso = 1 − Σ(p_i²)
          = 1 − (0.005² + 0.662² + 0.333²)
          = 1 − 0.549158
          ≈ 0.4508

Outcomes:
  • L flat at ~0.4508: conservation is just "preserves isotropy" (weak).
  • L flat at value clearly ≠ 0.4508: DPO conserves a structural quantity,
    distinct from isotropic baseline. Noether-like. Strong claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


BASE_PATH = "models/pythia-410m"
CKPT_DIR = Path("results/_leak/v2/checkpoints")
PARTITION_PATH = Path("results/_leak/v2/channel_partition.json")  # from #8
OUT_PATH = Path("results/_leak/v2/l_trajectory_orbit_fraction.json")

# Pythia-410m DPO orbit fractions (from task #1 multi-scale sweep)
P_GAUGE = 0.005
P_SRN   = 0.662
P_PS    = 0.333
# widths sum to 1.000 exactly

L_ISO = 1.0 - (P_GAUGE ** 2 + P_SRN ** 2 + P_PS ** 2)


def build_block_assignment(variance: torch.Tensor,
                           frac_bottom: float,
                           frac_middle: float,
                           frac_top: float) -> list[torch.Tensor]:
    """
    Given per-channel variance, return 3 disjoint index tensors:
      block 0 = bottom `frac_bottom` by variance  (PS-like)
      block 1 = next   `frac_middle`              (SRN-like)
      block 2 = top    `frac_top`                 (gauge-like)
    """
    assert abs(frac_bottom + frac_middle + frac_top - 1.0) < 1e-6
    n = variance.numel()
    order = torch.argsort(variance)   # ascending
    n_bot = int(round(frac_bottom * n))
    n_top = int(round(frac_top * n))
    # guard: ensure at least 1 channel per block
    n_bot = max(1, n_bot)
    n_top = max(1, n_top)
    n_mid = n - n_bot - n_top
    return [
        order[:n_bot],
        order[n_bot:n_bot + n_mid],
        order[n_bot + n_mid:],
    ]


def block_off_mass_nonuniform(W: torch.Tensor,
                              row_blocks: list[torch.Tensor],
                              col_blocks: list[torch.Tensor]) -> float:
    """
    L = (sum of OFF-diagonal-block Frobenius mass) / (total Frobenius mass).
    Diagonal blocks: (row_blocks[i], col_blocks[i]) pairs.
    """
    total_sq = (W ** 2).sum().item()
    if total_sq == 0.0:
        return 0.0
    diag_sq = 0.0
    assert len(row_blocks) == len(col_blocks)
    for i in range(len(row_blocks)):
        r_idx = row_blocks[i]
        c_idx = col_blocks[i]
        # gather rows then cols
        sub = W[r_idx][:, c_idx]
        diag_sq += (sub ** 2).sum().item()
    return (total_sq - diag_sq) / total_sq


def compute_L(model, per_layer_partitions: dict[int, list[torch.Tensor]]) -> dict:
    """For each transformer layer, use layer-specific row/col block assignment.
    Apply to Q, K, V packed in query_key_value.weight (shape: 3*d × d).
    Same block assignment for rows (output space) and cols (input space) — they
    both index the d_model residual-variance sort.
    """
    layers = model.gpt_neox.layers
    per_layer = []
    for i, layer in enumerate(layers):
        blocks = per_layer_partitions.get(i)
        if blocks is None:
            per_layer.append(float('nan'))
            continue
        qkv = layer.attention.query_key_value.weight.detach()
        d = qkv.shape[1]
        q_w = qkv[:d, :]
        k_w = qkv[d:2 * d, :]
        v_w = qkv[2 * d:, :]

        # For QKV matrices: rows are (head_dim × num_heads), cols are d_model.
        # The "residual variance" ordering is on d_model (cols). Rows of the
        # packed matrix don't index residual — we still want COL-direction blocks.
        # Use block assignment on cols only; for rows use 3 equal splits of rows
        # so the diagonal structure is well-defined.
        d_rows = q_w.shape[0]
        # equal row splits sized to match col block proportions — keeps diag well-defined
        row_cuts = []
        running = 0
        for b in blocks:
            chunk = int(round(len(b) / d * d_rows))
            row_cuts.append(torch.arange(running, min(running + chunk, d_rows)))
            running += chunk
        # pad last
        if running < d_rows:
            row_cuts[-1] = torch.cat([row_cuts[-1], torch.arange(running, d_rows)])

        vals = [block_off_mass_nonuniform(w, row_cuts, blocks)
                for w in (q_w, k_w, v_w)]
        per_layer.append(sum(vals) / len(vals))

    valid = [v for v in per_layer if not math.isnan(v)]
    return {'L_mean': sum(valid) / len(valid) if valid else float('nan'),
            'per_layer': per_layer}


def main() -> None:
    # Load base model + recompute variance (use cached #8 partition data as sanity)
    print(f"[base] loading {BASE_PATH}")
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.float32)
    n_layers = len(base.gpt_neox.layers)

    # Reuse #8's channel variance to build NEW non-uniform partition
    if not PARTITION_PATH.exists():
        raise FileNotFoundError(f"need #8's partition: {PARTITION_PATH}")

    partition_info = json.loads(PARTITION_PATH.read_text())
    # Expected shape: {"<layer_idx>": {"variance": [...] or "quartiles": [...]}}
    # We want raw per-layer per-channel variance; fall back to re-computing if absent.

    per_layer_blocks = {}
    for i in range(n_layers):
        key = str(i)
        layer_info = partition_info.get(key)
        if layer_info is None or 'variance' not in layer_info:
            print(f"  layer {i}: no stored variance, falling back to quartile assignment")
            # fallback: uniform spread
            d = base.gpt_neox.layers[i].attention.query_key_value.weight.shape[1]
            order = torch.arange(d)
        else:
            var = torch.tensor(layer_info['variance'])
            order = torch.argsort(var)
        # Build blocks from the ordering
        d = len(order)
        n_bot = max(1, int(round(P_PS * d)))       # PS = bottom
        n_top = max(1, int(round(P_GAUGE * d)))    # gauge = top
        n_mid = d - n_bot - n_top                  # SRN = middle
        per_layer_blocks[i] = [
            order[:n_bot],
            order[n_bot:n_bot + n_mid],
            order[n_bot + n_mid:],
        ]

    trajectory = []

    # Step 0: base model
    L = compute_L(base, per_layer_blocks)
    trajectory.append({'step': 0, 'L_mean': L['L_mean'], 'per_layer': L['per_layer']})
    print(f"[step 0] L = {L['L_mean']:.10f}")
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ckpt_dirs = sorted(CKPT_DIR.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split('-')[1]))
    for ckpt in ckpt_dirs:
        step = int(ckpt.name.split('-')[1])
        base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.float32)
        peft = PeftModel.from_pretrained(base, ckpt)
        merged = peft.merge_and_unload()
        L = compute_L(merged, per_layer_blocks)
        trajectory.append({'step': step, 'L_mean': L['L_mean'], 'per_layer': L['per_layer']})
        print(f"[step {step}] L = {L['L_mean']:.10f}")
        del base, peft, merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_PATH.write_text(json.dumps(trajectory, indent=2))
    print(f"\nWrote {len(trajectory)} entries → {OUT_PATH}")

    L_vals = [e['L_mean'] for e in trajectory]
    mean = sum(L_vals) / len(L_vals)
    rng = max(L_vals) - min(L_vals)
    print(f"\nPartition widths: gauge={P_GAUGE}, SRN={P_SRN}, PS={P_PS}")
    print(f"Isotropic baseline L_iso = {L_ISO:.6f}")
    print(f"Measured L_mean = {mean:.10f}   (range = {rng:.4e}, rel = {rng/mean:.4e})")
    delta_iso = mean - L_ISO
    print(f"|L_measured − L_iso| = {abs(delta_iso):.6f}")
    print()
    if abs(delta_iso) < 0.01:
        print("VERDICT: L ≈ isotropic baseline. Conservation = preserves isotropy (weak).")
    elif rng / mean < 0.001:
        print(f"VERDICT: L flat at {mean:.6f} ≠ {L_ISO:.6f}. STRUCTURAL CONSERVATION (Noether-like).")
    else:
        print(f"VERDICT: L drifts (range {rng:.4e}). No clean conservation.")


if __name__ == "__main__":
    main()
