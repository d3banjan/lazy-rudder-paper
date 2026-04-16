"""
Compute orbit-aligned channel partition for attention-leak L measurement.

Procedure:
  1. Load base pythia-410m
  2. Forward 100 hh-rlhf prompts through the model
  3. Capture per-layer hidden-state activations (residual stream after each layer)
  4. Per layer: compute per-channel variance across (tokens × prompts)
  5. Per layer: sort channels by variance, split into 4 quartiles
  6. Save channel_partition.json with {layer_idx: {input_quartiles: [...], output_quartiles: [...]}}

Fallback (--weight-variance flag): use per-column variance of QKV weight matrices
instead of activation capture. No forward pass needed.

Usage:
    uv run compute_channel_partition.py [--weight-variance] [--n-samples 100]
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import MODELS_DIR, RESULTS_DIR, BASE_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "pythia-410m"
OUT_DIR   = RESULTS_DIR / "_leak" / "v2"
OUT_PATH  = OUT_DIR / "channel_partition.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── activation-based partition ─────────────────────────────────────────────────

def load_hh_rlhf_prompts(tokenizer, n: int = 100, max_length: int = 256):
    """Load n prompts from hh-rlhf (cached). Returns list of tokenized tensors."""
    from datasets import load_dataset
    log.info("Loading hh-rlhf prompts for activation capture...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
    except Exception as e:
        log.warning(f"hh-rlhf failed ({e}), trying trl-internal-testing fallback")
        ds = load_dataset("trl-internal-testing/hh-rlhf-helpful-base", split="train")

    ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))

    def extract_prompt(ex):
        text = ex.get("chosen", ex.get("prompt", ""))
        sep = "\n\nAssistant:"
        if sep in text:
            text = text.rsplit(sep, 1)[0] + sep
        return text[:512]

    prompts = [extract_prompt(ex) for ex in ds]
    log.info(f"Extracted {len(prompts)} prompts")

    tokenized = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", max_length=max_length,
                        truncation=True, padding=False)
        tokenized.append(ids["input_ids"])  # shape (1, seq_len)
    return tokenized


def compute_activation_partition(model, tokenizer, n_samples: int = 100, k: int = 4):
    """
    Capture hidden states after each transformer layer, compute per-channel variance,
    return quartile assignments per layer.

    Returns: dict {layer_idx: {'input_quartiles': List[int], 'output_quartiles': List[int]}}
    where each list maps channel_idx → quartile (0=lowest variance, 3=highest).
    input_quartiles and output_quartiles are the same (we use residual stream channels),
    so both dims of each W matrix share the same partition.
    """
    model.eval()
    device = next(model.parameters()).device

    # Determine architecture
    base = model
    layers = base.gpt_neox.layers
    n_layers = len(layers)
    d_model = layers[0].attention.query_key_value.weight.shape[1]
    log.info(f"Model: {n_layers} layers, d_model={d_model}")

    # We'll accumulate running statistics (mean, M2 for Welford) per layer per channel
    # Shape: [n_layers, d_model] — residual stream *after* each layer (output of layer)
    # Also capture the *input* to each layer (residual before) = output of previous layer
    # Since input to layer 0 = embedding output, and input to layer i = output of layer i-1,
    # We capture hidden_states before each layer and after last layer.

    # For simplicity: use the output hidden state at each layer as the representative.
    # This gives "output_quartiles"; we set input_quartiles = previous layer's output_quartiles
    # (layer 0 input = embedding output).

    # Accumulate sum and sum-of-squares for variance: shape [n_layers, d_model]
    sum_acts  = [torch.zeros(d_model) for _ in range(n_layers)]
    sum2_acts = [torch.zeros(d_model) for _ in range(n_layers)]
    n_tokens  = [0] * n_layers

    prompts = load_hh_rlhf_prompts(tokenizer, n=n_samples)
    log.info(f"Running forward passes on {len(prompts)} prompts to capture activations...")

    hooks = []
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output can be a tuple; first element is the hidden state tensor
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # hs shape: (batch=1, seq_len, d_model)
            hs_cpu = hs.detach().float().squeeze(0).cpu()  # (seq_len, d_model)
            T = hs_cpu.shape[0]
            sum_acts[layer_idx]  += hs_cpu.sum(dim=0)
            sum2_acts[layer_idx] += (hs_cpu ** 2).sum(dim=0)
            n_tokens[layer_idx]  += T
        return hook

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for idx, input_ids in enumerate(prompts):
            if idx % 20 == 0:
                log.info(f"  Prompt {idx+1}/{len(prompts)}, seq_len={input_ids.shape[1]}")
            input_ids = input_ids.to(device)
            try:
                model(input_ids, use_cache=False)
            except Exception as e:
                log.warning(f"Forward pass failed on prompt {idx}: {e}")

    for h in hooks:
        h.remove()

    log.info("Computing per-channel variance per layer...")

    # Compute variance per channel per layer and assign quartiles
    partition = {}
    var_stats = []  # for summary printing

    for i in range(n_layers):
        N = n_tokens[i]
        if N == 0:
            log.warning(f"Layer {i}: no tokens captured, using uniform partition")
            partition[i] = {
                "input_quartiles":  list(range(d_model)),  # degenerate
                "output_quartiles": list(range(d_model)),
                "method": "uniform_fallback",
            }
            continue

        mean_ch = sum_acts[i] / N
        var_ch  = sum2_acts[i] / N - mean_ch ** 2  # (d_model,)
        var_ch  = var_ch.clamp(min=0.0)  # numerical safety

        var_stats.append({
            "layer": i,
            "var_min":    var_ch.min().item(),
            "var_median": var_ch.median().item(),
            "var_max":    var_ch.max().item(),
            "var_mean":   var_ch.mean().item(),
        })

        # Rank channels by variance, assign quartile 0..3
        sorted_idx = torch.argsort(var_ch)  # ascending
        q_labels = torch.zeros(d_model, dtype=torch.long)
        q_size = d_model // k
        for q in range(k):
            start = q * q_size
            end   = (q + 1) * q_size if q < k - 1 else d_model
            q_labels[sorted_idx[start:end]] = q

        q_list = q_labels.tolist()
        partition[i] = {
            "input_quartiles":  q_list,   # rows of W (output dim) use output quartile
            "output_quartiles": q_list,   # cols of W (input dim) same — residual stream
            "n_tokens":         N,
            "var_min":          var_ch.min().item(),
            "var_median":       var_ch.median().item(),
            "var_max":          var_ch.max().item(),
            "method":           "activation_variance",
        }

    # Print variance summary
    log.info("=== Per-layer activation variance summary ===")
    log.info(f"{'layer':>5}  {'var_min':>12}  {'var_median':>12}  {'var_max':>12}  {'var_mean':>12}")
    for s in var_stats[:5]:
        log.info(f"{s['layer']:>5}  {s['var_min']:>12.6f}  {s['var_median']:>12.6f}  "
                 f"{s['var_max']:>12.6f}  {s['var_mean']:>12.6f}")
    if len(var_stats) > 10:
        log.info("  ...")
        for s in var_stats[-5:]:
            log.info(f"{s['layer']:>5}  {s['var_min']:>12.6f}  {s['var_median']:>12.6f}  "
                     f"{s['var_max']:>12.6f}  {s['var_mean']:>12.6f}")

    return partition, var_stats


# ── weight-variance fallback partition ────────────────────────────────────────

def compute_weight_partition(model, k: int = 4):
    """
    Fallback: use per-column variance of each layer's QKV weight matrix as
    channel importance signal. No forward pass needed.
    Columns of W_qkv correspond to input (residual stream) channels.
    """
    base = model
    layers = base.gpt_neox.layers
    n_layers = len(layers)
    partition = {}
    var_stats = []

    log.info("Computing weight-column-variance partition (fallback mode)...")

    for i, layer in enumerate(layers):
        qkv = layer.attention.query_key_value.weight.detach().float()  # (3*d, d)
        d = qkv.shape[1]
        # Per-column variance across all 3*d rows
        col_var = qkv.var(dim=0)  # (d,) — variance across output channels per input channel

        var_stats.append({
            "layer":      i,
            "var_min":    col_var.min().item(),
            "var_median": col_var.median().item(),
            "var_max":    col_var.max().item(),
            "var_mean":   col_var.mean().item(),
        })

        sorted_idx = torch.argsort(col_var)
        q_labels = torch.zeros(d, dtype=torch.long)
        q_size = d // k
        for q in range(k):
            start = q * q_size
            end   = (q + 1) * q_size if q < k - 1 else d
            q_labels[sorted_idx[start:end]] = q

        q_list = q_labels.tolist()
        partition[i] = {
            "input_quartiles":  q_list,
            "output_quartiles": q_list,
            "var_min":    col_var.min().item(),
            "var_median": col_var.median().item(),
            "var_max":    col_var.max().item(),
            "method":     "weight_col_variance",
        }

    log.info("=== Per-layer weight-column variance summary ===")
    log.info(f"{'layer':>5}  {'var_min':>12}  {'var_median':>12}  {'var_max':>12}  {'var_mean':>12}")
    for s in var_stats[:5]:
        log.info(f"{s['layer']:>5}  {s['var_min']:>12.6f}  {s['var_median']:>12.6f}  "
                 f"{s['var_max']:>12.6f}  {s['var_mean']:>12.6f}")
    if len(var_stats) > 10:
        log.info("  ...")
        for s in var_stats[-5:]:
            log.info(f"{s['layer']:>5}  {s['var_min']:>12.6f}  {s['var_median']:>12.6f}  "
                     f"{s['var_max']:>12.6f}  {s['var_mean']:>12.6f}")

    return partition, var_stats


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-variance", action="store_true",
                        help="Use weight column variance instead of activation variance")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of prompts to forward for activation capture")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Loading base model from {MODEL_DIR}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if args.weight_variance:
        partition, var_stats = compute_weight_partition(model)
        method = "weight_col_variance"
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            partition, var_stats = compute_activation_partition(
                model, tokenizer, n_samples=args.n_samples
            )
            method = "activation_variance"
        except Exception as e:
            log.warning(f"Activation capture failed ({e}), falling back to weight variance")
            partition, var_stats = compute_weight_partition(model)
            method = "weight_col_variance_fallback"

    output = {
        "method": method,
        "n_layers": len(partition),
        "k": 4,
        "var_stats": var_stats,
        "partition": {str(k): v for k, v in partition.items()},
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    log.info(f"Saved partition to {OUT_PATH}")
    log.info(f"Method: {method}, layers: {len(partition)}")

    # Print brief summary
    print("\n=== PARTITION SUMMARY ===")
    print(f"Method: {method}")
    print(f"Layers: {len(partition)}, k=4 quartiles")
    print(f"{'layer':>5}  {'var_min':>10}  {'var_median':>12}  {'var_max':>10}")
    for s in var_stats[:5]:
        print(f"{s['layer']:>5}  {s['var_min']:>10.5f}  {s['var_median']:>12.5f}  {s['var_max']:>10.5f}")
    if len(var_stats) > 5:
        print(f"  ... ({len(var_stats)-5} more layers)")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
