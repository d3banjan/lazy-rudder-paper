"""Task #10 — clean re-run of #9 with fresh per-layer variance on ALL 24 layers.

Fixes #9's contamination: #8's channel_partition.json only had variance data
for layers 0-9; layers 10-23 silently fell back to torch.arange (coord blocks).
58% of layers were wrong. Here we compute variance on the full stack ourselves,
then apply the non-uniform (0.5/66.2/33.3) orbit-fraction partition to all
layers uniformly.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


BASE_PATH = str(MODELS_DIR / "pythia-410m")
CKPT_DIR = RESULTS_DIR / "_leak" / "v2" / "checkpoints"
OUT_PATH = RESULTS_DIR / "_leak" / "v2" / "l_trajectory_orbit_fraction_clean.json"

P_GAUGE = 0.005
P_SRN = 0.662
P_PS = 0.333
L_ISO = 1.0 - (P_GAUGE ** 2 + P_SRN ** 2 + P_PS ** 2)  # ≈ 0.4508

N_PROMPTS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_per_layer_variance(model, prompts: list[torch.Tensor]) -> dict[int, torch.Tensor]:
    """Forward-hook each layer output; accumulate per-channel variance per layer."""
    layers = model.gpt_neox.layers
    d = model.config.hidden_size
    sums = [torch.zeros(d, dtype=torch.float64) for _ in layers]
    sqs = [torch.zeros(d, dtype=torch.float64) for _ in layers]
    counts = [0 for _ in layers]

    def mk_hook(i):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            flat = h.detach().float().cpu().reshape(-1, h.shape[-1])
            sums[i].add_(flat.sum(dim=0).double())
            sqs[i].add_((flat.double() ** 2).sum(dim=0))
            counts[i] += flat.shape[0]
        return hook

    handles = [layer.register_forward_hook(mk_hook(i)) for i, layer in enumerate(layers)]
    try:
        model.eval()
        with torch.no_grad():
            for p in prompts:
                model(p)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for i in range(len(layers)):
        if counts[i] == 0:
            continue
        mean = sums[i] / counts[i]
        var = sqs[i] / counts[i] - mean ** 2
        out[i] = var.float()
    return out


def build_blocks(variance: torch.Tensor) -> list[torch.Tensor]:
    d = variance.numel()
    order = torch.argsort(variance)  # ascending
    n_bot = max(1, int(round(P_PS * d)))       # PS at bottom
    n_top = max(1, int(round(P_GAUGE * d)))    # gauge at top
    n_mid = d - n_bot - n_top                   # SRN middle
    return [order[:n_bot], order[n_bot:n_bot + n_mid], order[n_bot + n_mid:]]


def block_off_mass_nonuniform(W: torch.Tensor,
                              row_blocks: list[torch.Tensor],
                              col_blocks: list[torch.Tensor]) -> float:
    total_sq = (W ** 2).sum().item()
    if total_sq == 0.0:
        return 0.0
    diag_sq = 0.0
    for rb, cb in zip(row_blocks, col_blocks):
        sub = W[rb][:, cb]
        diag_sq += (sub ** 2).sum().item()
    return (total_sq - diag_sq) / total_sq


def compute_L(model, per_layer_blocks: dict[int, list[torch.Tensor]]) -> dict:
    layers = model.gpt_neox.layers
    per_layer = []
    for i, layer in enumerate(layers):
        blocks = per_layer_blocks.get(i)
        if blocks is None:
            per_layer.append(float('nan'))
            continue
        qkv = layer.attention.query_key_value.weight.detach()
        d = qkv.shape[1]
        q_w, k_w, v_w = qkv[:d, :], qkv[d:2*d, :], qkv[2*d:, :]
        # Row blocks: split rows proportionally to col-block widths
        d_rows = q_w.shape[0]
        row_blocks = []
        running = 0
        for b in blocks:
            chunk = int(round(len(b) / d * d_rows))
            row_blocks.append(torch.arange(running, min(running + chunk, d_rows)))
            running += chunk
        if running < d_rows:
            row_blocks[-1] = torch.cat([row_blocks[-1], torch.arange(running, d_rows)])

        vals = [block_off_mass_nonuniform(w, row_blocks, blocks) for w in (q_w, k_w, v_w)]
        per_layer.append(sum(vals) / len(vals))

    valid = [v for v in per_layer if not math.isnan(v)]
    return {'L_mean': sum(valid) / len(valid) if valid else float('nan'),
            'per_layer': per_layer}


def load_prompts(tokenizer, n: int = N_PROMPTS) -> list[torch.Tensor]:
    ds = load_dataset("Anthropic/hh-rlhf", split="test", streaming=False)
    prompts = []
    for ex in ds:
        text = ex["chosen"].strip()
        if len(text) < 50:
            continue
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        prompts.append(toks.input_ids.to(DEVICE))
        if len(prompts) >= n:
            break
    return prompts


def main() -> None:
    print(f"[base] loading {BASE_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.float32).to(DEVICE)

    print("[variance] computing per-layer variance on hh-rlhf prompts")
    prompts = load_prompts(tokenizer)
    variances = compute_per_layer_variance(base, prompts)
    print(f"  got variance for {len(variances)}/{len(base.gpt_neox.layers)} layers")
    # per-layer variance summary
    for i, v in variances.items():
        if i < 3 or i > len(base.gpt_neox.layers) - 3:
            print(f"    layer {i}: var min={v.min():.4e} med={v.median():.4e} max={v.max():.4e}")

    per_layer_blocks = {i: build_blocks(v) for i, v in variances.items()}

    # step 0 on base (no adapter)
    base_cpu = base.float().cpu()
    L = compute_L(base_cpu, per_layer_blocks)
    trajectory = [{'step': 0, 'L_mean': L['L_mean'], 'per_layer': L['per_layer']}]
    print(f"[step 0] L = {L['L_mean']:.10f}")
    del base, base_cpu
    torch.cuda.empty_cache()

    ckpts = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[1]))
    for ckpt in ckpts:
        step = int(ckpt.name.split('-')[1])
        b = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.float32)
        peft = PeftModel.from_pretrained(b, ckpt)
        merged = peft.merge_and_unload()
        L = compute_L(merged, per_layer_blocks)
        trajectory.append({'step': step, 'L_mean': L['L_mean'], 'per_layer': L['per_layer']})
        print(f"[step {step}] L = {L['L_mean']:.10f}")
        del b, peft, merged
        torch.cuda.empty_cache()

    OUT_PATH.write_text(json.dumps(trajectory, indent=2))
    print(f"\nWrote {len(trajectory)} entries → {OUT_PATH}")

    L_vals = [e['L_mean'] for e in trajectory]
    mean = sum(L_vals) / len(L_vals)
    rng = max(L_vals) - min(L_vals)
    delta_iso = mean - L_ISO
    print(f"\nPartition: gauge={P_GAUGE} SRN={P_SRN} PS={P_PS}")
    print(f"L_iso = {L_ISO:.6f}")
    print(f"L_mean = {mean:.10f}  range = {rng:.4e}  rel = {rng/mean:.4e}")
    print(f"L_mean − L_iso = {delta_iso:+.6f} ({delta_iso/L_ISO*100:+.3f}%)")
    print()
    if rng / mean > 0.001:
        print(f"VERDICT: L drifts. Not conserved.")
    elif abs(delta_iso) < 0.005:
        print(f"VERDICT: L flat AND within 0.5% of isotropic baseline. "
              f"Conservation ≈ isotropy preservation (weak).")
    else:
        print(f"VERDICT: L flat at {mean:.6f} ≠ {L_ISO:.6f} (|Δ| = {abs(delta_iso):.4f}). "
              f"STRUCTURAL CONSERVATION.")


if __name__ == "__main__":
    main()
