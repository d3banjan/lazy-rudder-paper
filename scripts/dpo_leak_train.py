"""
Attention-leak conservation law test: mini-DPO on pythia-410m.
Measures L (off-block Frobenius mass fraction) at each checkpoint.
Task #3 — block_rg_plan.md [P3] line 245.
"""
import os, sys, json, time, math, logging
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/home/debanjan/Code/Research/lean-mining/cross-check/trained-model-battery")
MODEL_DIR = BASE_DIR / "models" / "pythia-410m"
OUT_DIR   = BASE_DIR / "results" / "_leak"
CKPT_DIR  = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUT_DIR / "l_trajectory.json"
LOG_FILE    = OUT_DIR / "training_log.txt"

# ── L-metric ───────────────────────────────────────────────────────────────────
# For each weight matrix W ∈ R^{d_model × d_model}:
#   Partition rows into k=4 blocks of 256; cols into k=4 blocks of 256.
#   L = ||W_off||_F^2 / ||W||_F^2
#   where W_off = W with diagonal blocks zeroed.
# L(checkpoint) = mean across {W_Q, W_K, W_V} across all 24 layers.

def block_off_mass(W: torch.Tensor, k: int = 4) -> float:
    """
    W: 2-D tensor (out, in) or any shape; we use first two dims.
    Returns off-diagonal-block Frobenius mass fraction.
    """
    if W.dim() < 2:
        return float('nan')
    W = W.float()
    rows, cols = W.shape[0], W.shape[1]
    # Only square or near-square matrices make sense for block partition.
    # For rectangular (e.g. qkv packed), we work on each head slice separately
    # but here we use the simpler row/col partition approach.
    r_bs = rows // k   # row block size (may leave remainder)
    c_bs = cols // k

    if r_bs == 0 or c_bs == 0:
        return float('nan')

    total_sq = (W ** 2).sum().item()
    if total_sq == 0.0:
        return 0.0

    # Sum of diagonal blocks
    diag_sq = 0.0
    for i in range(k):
        r0, r1 = i * r_bs, min((i + 1) * r_bs, rows)
        c0, c1 = i * c_bs, min((i + 1) * c_bs, cols)
        diag_sq += (W[r0:r1, c0:c1] ** 2).sum().item()

    off_sq = total_sq - diag_sq
    return off_sq / total_sq


def compute_L(model, k: int = 4) -> dict:
    """
    Compute L for the model's attention Q/K/V projections.
    Returns {'L_mean': float, 'per_layer': [float, ...]}
    For GPTNeoX: layers are in model.gpt_neox.layers[i].attention
    Weight matrices: query_key_value (combined QKV packed) and dense (out proj).
    We split QKV: shape is (3*d_model, d_model) → split into 3 × (d_model, d_model).
    """
    per_layer = []
    base = model
    # Handle PEFT model wrapper (PeftModelForCausalLM has base_model.model)
    # Plain nn.Module also has .base_model → itself, so check for peft specifically
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model

    # Try multiple attribute paths for GPTNeoX / GPT-style models
    layers = None
    for attr_path in ['gpt_neox.layers', 'model.layers', 'transformer.h']:
        obj = base
        try:
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        log.error("Cannot find transformer layers in model")
        return {'L_mean': float('nan'), 'per_layer': []}

    for layer in layers:
        try:
            attn = layer.attention
            # GPTNeoX uses query_key_value: (3*hidden, hidden)
            qkv = attn.query_key_value.weight.detach()  # (3*d, d)
            d = qkv.shape[1]
            # Split into Q, K, V each (d, d)
            q_w = qkv[:d, :]
            k_w = qkv[d:2*d, :]
            v_w = qkv[2*d:, :]

            vals = [block_off_mass(w, k) for w in [q_w, k_w, v_w]]
            vals = [v for v in vals if not math.isnan(v)]
            layer_L = sum(vals) / len(vals) if vals else float('nan')
            per_layer.append(layer_L)
        except AttributeError as e:
            log.warning(f"Layer missing attention.query_key_value: {e}")
            per_layer.append(float('nan'))

    valid = [v for v in per_layer if not math.isnan(v)]
    L_mean = sum(valid) / len(valid) if valid else float('nan')
    return {'L_mean': L_mean, 'per_layer': per_layer}


# ── dataset preparation ────────────────────────────────────────────────────────
def load_hh_rlhf_dpo(tokenizer, max_samples: int = 2000):
    """
    Load Anthropic/hh-rlhf and format for TRL DPO.
    TRL DPO expects columns: prompt, chosen, rejected (as strings or chat lists).
    hh-rlhf has: chosen, rejected (full conversation strings).
    We extract the last human turn as prompt and model response as chosen/rejected.
    """
    from datasets import load_dataset
    log.info("Loading Anthropic/hh-rlhf dataset...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
        log.info(f"Loaded hh-rlhf: {len(ds)} examples")
    except Exception as e:
        log.warning(f"hh-rlhf failed ({e}), trying trl-internal-testing/hh-rlhf-helpful-base")
        try:
            ds = load_dataset("trl-internal-testing/hh-rlhf-helpful-base", split="train")
            log.info(f"Loaded trl-internal-testing/hh-rlhf: {len(ds)} examples")
        except Exception as e2:
            log.warning(f"Fallback 1 failed ({e2}), trying HuggingFaceH4/ultrafeedback_binarized")
            ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
            log.info(f"Loaded ultrafeedback: {len(ds)} examples")
            # ultrafeedback already has prompt/chosen/rejected
            ds = ds.select(range(min(max_samples, len(ds))))
            return ds

    # Shuffle and select
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    def parse_hh(example):
        """Parse hh-rlhf: extract prompt + chosen/rejected responses."""
        chosen_text   = example["chosen"]
        rejected_text = example["rejected"]

        # Split on last "\n\nAssistant:" to separate prompt from response
        sep = "\n\nAssistant:"
        if sep in chosen_text:
            parts = chosen_text.rsplit(sep, 1)
            prompt   = parts[0] + sep
            chosen_r = parts[1].strip()
        else:
            # fallback: whole text is chosen, use first 256 chars as prompt
            prompt   = chosen_text[:256]
            chosen_r = chosen_text[256:]

        if sep in rejected_text:
            parts    = rejected_text.rsplit(sep, 1)
            rejected_r = parts[1].strip()
        else:
            rejected_r = rejected_text[256:]

        return {
            "prompt":   prompt,
            "chosen":   chosen_r,
            "rejected": rejected_r,
        }

    ds = ds.map(parse_hh, remove_columns=ds.column_names)
    # Filter out empty responses
    ds = ds.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
    log.info(f"After parsing: {len(ds)} examples")
    return ds


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    import transformers.trainer_callback as tc

    t_start = time.time()
    trajectory = []

    # ── tokenizer ──────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer pad_token: {tokenizer.pad_token!r}")

    # ── model (load in fp16 to save memory) ───────────────────────────────────
    log.info("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # ── LoRA config ────────────────────────────────────────────────────────────
    # For GPTNeoX, target query_key_value and dense (attn out) + mlp layers
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy_model = get_peft_model(policy_model, lora_cfg)
    policy_model.print_trainable_parameters()

    # ── reference model (frozen base) ─────────────────────────────────────────
    log.info("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── dataset ────────────────────────────────────────────────────────────────
    dataset = load_hh_rlhf_dpo(tokenizer, max_samples=2000)

    # ── DPO config ─────────────────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir                  = str(CKPT_DIR),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        max_steps                   = 800,
        learning_rate               = 5e-6,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 50,
        max_grad_norm               = 1.0,
        fp16                        = True,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        beta                        = 0.1,
        max_length                  = 512,
        save_strategy               = "steps",
        save_steps                  = 100,
        save_total_limit            = 10,
        logging_steps               = 10,
        logging_first_step          = True,
        report_to                   = "none",
        seed                        = 42,
        disable_tqdm                = False,
        dataset_num_proc            = 2,
        remove_unused_columns       = False,
        use_cache                   = False,       # must be False w/ gradient checkpointing
        dataloader_num_workers      = 0,
    )

    log.info(f"DPO Training config: {training_args.to_dict()}")

    # ── step-0 L measurement (base, no LoRA adaptation yet) ───────────────────
    log.info("Measuring L at step 0 (base weights)...")
    with torch.no_grad():
        step0_L = compute_L(policy_model)
    log.info(f"Step 0: L_mean = {step0_L['L_mean']:.6f}")
    trajectory.append({
        "step":      0,
        "L":         step0_L["L_mean"],
        "per_layer": step0_L["per_layer"],
    })

    # ── custom callback to measure L at each saved checkpoint ─────────────────
    class LeakCallback(tc.TrainerCallback):
        def __init__(self):
            self.last_loss = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.last_loss = logs.get("loss", self.last_loss)
                log.info(f"Step {state.global_step}: loss={self.last_loss}")

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            log.info(f"Checkpoint saved at step {step}, measuring L...")
            with torch.no_grad():
                ldata = compute_L(trainer.model)
            log.info(f"Step {step}: L_mean = {ldata['L_mean']:.6f}")
            trajectory.append({
                "step":      step,
                "L":         ldata["L_mean"],
                "per_layer": ldata["per_layer"],
            })
            # Save trajectory incrementally
            with open(RESULT_JSON, "w") as f:
                json.dump(trajectory, f, indent=2)

    leak_cb = LeakCallback()

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model             = policy_model,
        ref_model         = ref_model,
        args              = training_args,
        train_dataset     = dataset,
        processing_class  = tokenizer,   # TRL 1.1.0: renamed from tokenizer
        callbacks         = [leak_cb],
    )

    log.info("Starting DPO training...")
    train_result = trainer.train()

    # ── final L measurement ────────────────────────────────────────────────────
    final_step = train_result.global_step
    log.info(f"Training complete at step {final_step}. Measuring final L...")
    with torch.no_grad():
        final_L = compute_L(policy_model)

    # Only append final if not already captured by on_save
    if not trajectory or trajectory[-1]["step"] != final_step:
        trajectory.append({
            "step":      final_step,
            "L":         final_L["L_mean"],
            "per_layer": final_L["per_layer"],
        })

    # ── verdict ────────────────────────────────────────────────────────────────
    L_vals = [e["L"] for e in trajectory if not math.isnan(e["L"])]
    L_min  = min(L_vals)
    L_max  = max(L_vals)
    L_mean = sum(L_vals) / len(L_vals)
    L_var  = sum((v - L_mean)**2 for v in L_vals) / len(L_vals)
    L_range_frac = (L_max - L_min) / L_mean if L_mean > 0 else float('nan')

    L_first = L_vals[0]
    L_last  = L_vals[-1]
    delta_frac = (L_last - L_first) / L_first if L_first > 0 else float('nan')

    if L_range_frac < 0.05:
        verdict = "conserved"
    elif delta_frac <= -0.10:
        verdict = "decay"
    elif delta_frac >= 0.10:
        verdict = "growth"
    else:
        verdict = "mixed"

    summary = {
        "verdict":       verdict,
        "n_checkpoints": len(trajectory),
        "L_min":         L_min,
        "L_max":         L_max,
        "L_mean":        L_mean,
        "L_var":         L_var,
        "L_range_frac":  L_range_frac,
        "delta_frac":    delta_frac,
        "train_loss":    train_result.training_loss,
        "total_steps":   final_step,
        "wall_sec":      time.time() - t_start,
    }

    log.info(f"\n{'='*60}")
    log.info(f"VERDICT: {verdict}")
    log.info(f"L range_frac={L_range_frac:.4f}, delta_frac={delta_frac:.4f}")
    log.info(f"min={L_min:.6f}, max={L_max:.6f}, var={L_var:.8f}")
    log.info(f"Train loss: {train_result.training_loss:.6f}")
    log.info(f"Wall time: {summary['wall_sec']/3600:.2f}h")
    log.info(f"{'='*60}")

    # ── save full results ──────────────────────────────────────────────────────
    with open(RESULT_JSON, "w") as f:
        json.dump(trajectory, f, indent=2)

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Results saved to {RESULT_JSON}")
    log.info(f"Summary saved to {summary_path}")

    # Print final trajectory as table
    print("\n=== L TRAJECTORY TABLE ===")
    print(f"{'step':>6}  {'L':>10}")
    print("-" * 20)
    for entry in trajectory:
        print(f"{entry['step']:>6}  {entry['L']:>10.6f}")
    print(f"\nVerdict: {verdict}")
    print(f"min={L_min:.6f}  max={L_max:.6f}  var={L_var:.8f}")

    return summary


if __name__ == "__main__":
    main()
