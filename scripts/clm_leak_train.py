"""
Task #11 — SGD control for leak conservation.

Identical setup to #10 DPO (pythia-410m, LoRA r=128), but uses standard
causal-LM next-token-prediction (CLM) training instead of DPO.

If L stays flat → conservation is generic (low-rank LoRA artifact).
If L drifts    → conservation is DPO-specific (Noether-like framing validated).

Output: results/_leak/v3/l_trajectory_clm.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/home/debanjan/Code/Research/lean-mining/cross-check/trained-model-battery")
MODEL_DIR = BASE_DIR / "models" / "pythia-410m"
OUT_DIR   = BASE_DIR / "results" / "_leak" / "v3"
CKPT_DIR  = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUT_DIR / "l_trajectory_clm.json"
LOG_FILE    = OUT_DIR / "training_log.txt"

# ── same partition fractions as #10 ───────────────────────────────────────────
P_GAUGE = 0.005
P_SRN   = 0.662
P_PS    = 0.333
L_ISO   = 1.0 - (P_GAUGE**2 + P_SRN**2 + P_PS**2)  # ≈ 0.4508

N_PROMPTS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── #10-identical L measurement ───────────────────────────────────────────────

def compute_per_layer_variance(model, prompts: list) -> dict:
    """Forward-hook each layer; accumulate per-channel variance. Same as #10."""
    layers = model.gpt_neox.layers
    d = model.config.hidden_size
    sums  = [torch.zeros(d, dtype=torch.float64) for _ in layers]
    sqs   = [torch.zeros(d, dtype=torch.float64) for _ in layers]
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
        var  = sqs[i] / counts[i] - mean**2
        out[i] = var.float()
    return out


def build_blocks(variance: torch.Tensor) -> list:
    """Same non-uniform split as #10: PS=0.333 (bottom), SRN=0.662 (mid), gauge=0.005 (top)."""
    d = variance.numel()
    order  = torch.argsort(variance)
    n_bot  = max(1, int(round(P_PS * d)))
    n_top  = max(1, int(round(P_GAUGE * d)))
    n_mid  = d - n_bot - n_top
    return [order[:n_bot], order[n_bot:n_bot + n_mid], order[n_bot + n_mid:]]


def block_off_mass_nonuniform(W: torch.Tensor,
                               row_blocks: list,
                               col_blocks: list) -> float:
    total_sq = (W**2).sum().item()
    if total_sq == 0.0:
        return 0.0
    diag_sq = 0.0
    for rb, cb in zip(row_blocks, col_blocks):
        sub = W[rb][:, cb]
        diag_sq += (sub**2).sum().item()
    return (total_sq - diag_sq) / total_sq


def compute_L(model, per_layer_blocks: dict) -> dict:
    """Identical to #10's compute_L."""
    layers = model.gpt_neox.layers
    per_layer = []
    for i, layer in enumerate(layers):
        blocks = per_layer_blocks.get(i)
        if blocks is None:
            per_layer.append(float('nan'))
            continue
        qkv = layer.attention.query_key_value.weight.detach()
        d   = qkv.shape[1]
        q_w, k_w, v_w = qkv[:d, :], qkv[d:2*d, :], qkv[2*d:, :]
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


def load_prompts(tokenizer, n: int = N_PROMPTS) -> list:
    from datasets import load_dataset
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


# ── dataset (CLM: only 'chosen' text, no preference pairs) ───────────────────

def load_hh_rlhf_clm(tokenizer, max_samples: int = 2000, max_length: int = 512):
    from datasets import load_dataset
    log.info("Loading Anthropic/hh-rlhf for CLM (chosen text only)...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
    log.info(f"Loaded {len(ds)} examples")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    def tokenize(example):
        text = example["chosen"]
        enc  = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=2)
    ds.set_format("torch")
    log.info(f"Tokenized CLM dataset: {len(ds)} examples")
    return ds


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import transformers
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        Trainer, TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    import transformers.trainer_callback as tc
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    t_start = time.time()
    trajectory = []

    # Training config — identical to #10 except CLM replaces DPO
    TRAIN_CONFIG = {
        "model":            str(MODEL_DIR),
        "lora_r":           128,
        "lora_alpha":       256,
        "target_modules":   ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "lora_dropout":     0.05,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps":        800,
        "learning_rate":    5e-6,
        "lr_scheduler_type": "cosine",
        "warmup_steps":     50,
        "max_grad_norm":    1.0,
        "fp16":             True,
        "max_length":       512,
        "dataset":          "Anthropic/hh-rlhf",
        "dataset_field":    "chosen",
        "n_samples":        2000,
        "save_steps":       100,
        "seed":             42,
        "objective":        "CLM_next_token_prediction",
        "partition":        {"PS": P_PS, "SRN": P_SRN, "gauge": P_GAUGE},
        "L_iso":            L_ISO,
    }
    log.info(f"Training config:\n{json.dumps(TRAIN_CONFIG, indent=2)}")
    (OUT_DIR / "train_config.json").write_text(json.dumps(TRAIN_CONFIG, indent=2))

    # ── tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── base model in fp32 for step-0 measurement ────────────────────────────
    log.info("Loading base model fp32 for step-0 L measurement...")
    base_for_step0 = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(DEVICE)
    log.info("Computing per-layer variance on hh-rlhf test prompts (same as #10)...")
    prompts = load_prompts(tokenizer, N_PROMPTS)
    variances = compute_per_layer_variance(base_for_step0, prompts)
    log.info(f"Got variance for {len(variances)}/{len(base_for_step0.gpt_neox.layers)} layers")
    per_layer_blocks = {i: build_blocks(v) for i, v in variances.items()}

    base_cpu = base_for_step0.float().cpu()
    L0 = compute_L(base_cpu, per_layer_blocks)
    log.info(f"Step 0: L_mean = {L0['L_mean']:.10f}")
    trajectory.append({'step': 0, 'L_mean': L0['L_mean'], 'per_layer': L0['per_layer']})
    RESULT_JSON.write_text(json.dumps(trajectory, indent=2))
    del base_for_step0, base_cpu
    torch.cuda.empty_cache()

    # ── policy model (fp16) for training ─────────────────────────────────────
    log.info("Loading policy model fp16 for CLM training...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    lora_cfg = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy_model = get_peft_model(policy_model, lora_cfg)
    policy_model.print_trainable_parameters()

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = load_hh_rlhf_clm(tokenizer, max_samples=2000, max_length=512)

    # Use DataCollatorForLanguageModeling with mlm=False (standard CLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ── training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                    = str(CKPT_DIR),
        per_device_train_batch_size   = 1,
        gradient_accumulation_steps   = 8,
        max_steps                     = 800,
        learning_rate                 = 5e-6,
        lr_scheduler_type             = "cosine",
        warmup_steps                  = 50,
        max_grad_norm                 = 1.0,
        fp16                          = True,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        save_strategy                 = "steps",
        save_steps                    = 100,
        save_total_limit              = 10,
        logging_steps                 = 10,
        logging_first_step            = True,
        report_to                     = "none",
        seed                          = 42,
        disable_tqdm                  = False,
        dataloader_num_workers        = 0,
        remove_unused_columns         = False,
        use_cache                     = False,
    )

    # ── L-measurement callback (same as dpo_leak_train_v2) ───────────────────
    class LeakCallbackCLM(tc.TrainerCallback):
        def __init__(self):
            self.step_losses: list = []   # (step, loss) for log

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.step_losses.append((state.global_step, logs["loss"]))
                log.info(f"[train] step={state.global_step} loss={logs['loss']:.6f}")

        def on_save(self, args, state, control, **kwargs):
            step      = state.global_step
            ckpt_path = Path(args.output_dir) / f"checkpoint-{step}"
            log.info(f"Checkpoint step {step}: merging + measuring L...")
            try:
                b = AutoModelForCausalLM.from_pretrained(
                    str(MODEL_DIR), torch_dtype=torch.float32
                )
                peft = PeftModel.from_pretrained(b, str(ckpt_path))
                merged = peft.merge_and_unload()
                L = compute_L(merged, per_layer_blocks)
                log.info(f"Step {step}: L_mean = {L['L_mean']:.10f}")
                trajectory.append({'step': step, 'L_mean': L['L_mean'],
                                   'per_layer': L['per_layer']})
                del b, peft, merged
            except Exception as e:
                log.error(f"Step {step}: merge/measure failed: {e}")
                trajectory.append({'step': step, 'L_mean': float('nan'), 'per_layer': []})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            RESULT_JSON.write_text(json.dumps(trajectory, indent=2))

    leak_cb = LeakCallbackCLM()

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model         = policy_model,
        args          = training_args,
        train_dataset = dataset,
        data_collator = data_collator,
        callbacks     = [leak_cb],
    )

    log.info("Starting CLM training (r=128, 800 steps, lr=5e-6)...")
    train_result = trainer.train()

    # ── final measurement ─────────────────────────────────────────────────────
    final_step = train_result.global_step
    log.info(f"Training complete at step {final_step}.")

    final_ckpt = CKPT_DIR / f"checkpoint-{final_step}"
    if not final_ckpt.exists():
        trainer.save_model(str(final_ckpt))

    if final_ckpt.exists() and (not trajectory or trajectory[-1]["step"] != final_step):
        try:
            b = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
            peft = PeftModel.from_pretrained(b, str(final_ckpt))
            merged = peft.merge_and_unload()
            L = compute_L(merged, per_layer_blocks)
            trajectory.append({'step': final_step, 'L_mean': L['L_mean'],
                               'per_layer': L['per_layer']})
            log.info(f"Final step {final_step}: L_mean = {L['L_mean']:.10f}")
            del b, peft, merged
            torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"Final merge failed: {e}")

    RESULT_JSON.write_text(json.dumps(trajectory, indent=2))

    # ── summary ───────────────────────────────────────────────────────────────
    L_vals   = [e['L_mean'] for e in trajectory if not math.isnan(e['L_mean'])]
    L_min    = min(L_vals)
    L_max    = max(L_vals)
    L_mean   = sum(L_vals) / len(L_vals)
    L_range  = L_max - L_min
    rel_range = L_range / L_mean if L_mean > 0 else float('nan')
    delta_iso = L_mean - L_ISO

    summary = {
        "objective":     "CLM",
        "n_checkpoints": len(trajectory),
        "L_min":         L_min,
        "L_max":         L_max,
        "L_mean":        L_mean,
        "L_range":       L_range,
        "rel_range":     rel_range,
        "delta_iso":     delta_iso,
        "delta_iso_pct": delta_iso / L_ISO * 100,
        "train_loss":    train_result.training_loss,
        "total_steps":   final_step,
        "wall_sec":      time.time() - t_start,
        "lora_r":        128,
        "step_losses":   leak_cb.step_losses,
    }
    (OUT_DIR / "summary_v3_clm.json").write_text(json.dumps(summary, indent=2))

    log.info(f"\n{'='*60}")
    log.info(f"CLM SUMMARY")
    log.info(f"  L_mean  = {L_mean:.10f}")
    log.info(f"  L_range = {L_range:.4e}  rel = {rel_range:.4e}")
    log.info(f"  Δ_iso   = {delta_iso:+.6f} ({delta_iso/L_ISO*100:+.3f}%)")
    log.info(f"  Wall    = {(time.time()-t_start)/60:.1f} min")
    log.info(f"{'='*60}")

    print("\n=== CLM L-TRAJECTORY (#11 SGD CONTROL) ===")
    print(f"{'step':>6}  {'L_mean':>14}")
    print("-" * 24)
    for e in trajectory:
        print(f"{e['step']:>6}  {e['L_mean']:>14.10f}")
    print(f"\nL_mean={L_mean:.10f}  range={L_range:.4e}  rel={rel_range:.4e}")
    print(f"Δ_iso = {delta_iso:+.6f} ({delta_iso/L_ISO*100:+.3f}%)")
    print(f"Wall  = {(time.time()-t_start)/60:.1f} min")

    return summary


if __name__ == "__main__":
    main()
