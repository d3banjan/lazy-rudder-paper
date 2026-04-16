"""Re-measure L across LoRA checkpoints by merging adapter into base before reading weights.

Fixes the dpo_leak_train.py bug: original compute_L read query_key_value.weight which,
under PEFT, is the frozen base weight — not the merged effective weight. This script
loads each adapter checkpoint, merges it, and computes L on the true effective weights.
"""

from __future__ import annotations
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import MODELS_DIR, RESULTS_DIR, BASE_DIR

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


BASE_PATH = str(MODELS_DIR / "pythia-410m")
CKPT_DIR = RESULTS_DIR / "_leak" / "checkpoints"
OUT_PATH = RESULTS_DIR / "_leak" / "l_trajectory_merged.json"


def block_off_mass(W: torch.Tensor, k: int = 4) -> float:
    rows, cols = W.shape
    r_bs = max(1, rows // k)
    c_bs = max(1, cols // k)
    total_sq = (W ** 2).sum().item()
    if total_sq == 0.0:
        return 0.0
    diag_sq = 0.0
    for i in range(k):
        r0, r1 = i * r_bs, min((i + 1) * r_bs, rows)
        c0, c1 = i * c_bs, min((i + 1) * c_bs, cols)
        diag_sq += (W[r0:r1, c0:c1] ** 2).sum().item()
    return (total_sq - diag_sq) / total_sq


def compute_L_from_merged(model, k: int = 4):
    """Expects a merged-and-unloaded model (plain transformer, no PEFT wrapper)."""
    layers = model.gpt_neox.layers
    per_layer = []
    for layer in layers:
        qkv = layer.attention.query_key_value.weight.detach()
        d = qkv.shape[1]
        q_w, k_w, v_w = qkv[:d, :], qkv[d:2*d, :], qkv[2*d:, :]
        vals = [block_off_mass(w, k) for w in (q_w, k_w, v_w)]
        vals = [v for v in vals if not math.isnan(v)]
        per_layer.append(sum(vals) / len(vals) if vals else float('nan'))
    valid = [v for v in per_layer if not math.isnan(v)]
    return {'L_mean': sum(valid) / len(valid) if valid else float('nan'), 'per_layer': per_layer}


def main() -> None:
    trajectory = []

    # Step 0: base model, no adapter
    print(f"[step 0] loading base from {BASE_PATH}")
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.float32)
    L0 = compute_L_from_merged(base)
    trajectory.append({'step': 0, 'L_mean': L0['L_mean'], 'per_layer': L0['per_layer']})
    print(f"  L_mean = {L0['L_mean']:.10f}")
    del base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ckpt_dirs = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[1]))
    for ckpt in ckpt_dirs:
        step = int(ckpt.name.split('-')[1])
        print(f"[step {step}] loading base + adapter from {ckpt}")
        # Load fresh base, apply adapter, merge
        base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.float32)
        peft = PeftModel.from_pretrained(base, ckpt)
        merged = peft.merge_and_unload()
        L = compute_L_from_merged(merged)
        trajectory.append({'step': step, 'L_mean': L['L_mean'], 'per_layer': L['per_layer']})
        print(f"  L_mean = {L['L_mean']:.10f}")
        del base, peft, merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_PATH.write_text(json.dumps(trajectory, indent=2))
    print(f"\nWrote {len(trajectory)} entries to {OUT_PATH}")
    print("\nSummary:")
    for entry in trajectory:
        print(f"  step {entry['step']:4d}: L = {entry['L_mean']:.10f}")

    L_vals = [e['L_mean'] for e in trajectory]
    rng = max(L_vals) - min(L_vals)
    rel = rng / (sum(L_vals) / len(L_vals))
    print(f"\nrange = {rng:.4e}  |  relative = {rel:.4e}  ({rel*100:.4f}%)")


if __name__ == "__main__":
    main()
