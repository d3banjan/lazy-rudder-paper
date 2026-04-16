"""
BitFit-DPO Strike — gauge theory test for alignment.

Freeze all weight matrices; train only .bias tensors under DPO on Pythia-410m.
Decisive test: can alignment be achieved with ZERO new weight directions,
only additive bias shifts?

Physics prediction:
- final_loss <= 0.35 → gauge_alive (bias floodgates suffice)
- final_loss >= 0.60 → gauge_dead (weight rewiring necessary)
- else → partial

Reference: LoRA-DPO v2 (r=128) final train_loss = 0.487
"""
import os, sys, json, time, math, logging
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import MODELS_DIR, RESULTS_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = MODELS_DIR / "pythia-410m"
OUT_DIR   = RESULTS_DIR / "bitfit_dpo_strike"
CKPT_DIR  = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON    = OUT_DIR / "loss_trajectory.json"
SUMMARY_JSON   = OUT_DIR / "summary.json"

# Reference from LoRA-DPO v2 run
LORA_DPO_V2_FINAL_LOSS = 0.487


# ── dataset ─────────────────────────────────────────────────────────────────────

def load_hh_rlhf_dpo(tokenizer, max_samples: int = 2000):
    from datasets import load_dataset
    log.info("Loading Anthropic/hh-rlhf dataset...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
        log.info(f"Loaded hh-rlhf: {len(ds)} examples")
    except Exception as e:
        log.warning(f"hh-rlhf failed ({e}), trying trl-internal-testing fallback")
        try:
            ds = load_dataset("trl-internal-testing/hh-rlhf-helpful-base", split="train")
            log.info(f"Loaded trl-internal-testing/hh-rlhf: {len(ds)} examples")
        except Exception as e2:
            log.warning(f"Fallback 1 failed ({e2}), trying ultrafeedback")
            ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
            log.info(f"Loaded ultrafeedback: {len(ds)} examples")
            ds = ds.select(range(min(max_samples, len(ds))))
            return ds

    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    def parse_hh(example):
        chosen_text   = example["chosen"]
        rejected_text = example["rejected"]
        sep = "\n\nAssistant:"
        if sep in chosen_text:
            parts    = chosen_text.rsplit(sep, 1)
            prompt   = parts[0] + sep
            chosen_r = parts[1].strip()
        else:
            prompt   = chosen_text[:256]
            chosen_r = chosen_text[256:]
        if sep in rejected_text:
            parts      = rejected_text.rsplit(sep, 1)
            rejected_r = parts[1].strip()
        else:
            rejected_r = rejected_text[256:]
        return {"prompt": prompt, "chosen": chosen_r, "rejected": rejected_r}

    ds = ds.map(parse_hh, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
    log.info(f"After parsing: {len(ds)} examples")
    return ds


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    import transformers.trainer_callback as tc

    t_start = time.time()
    trajectory = []  # [{step, loss}]

    # ── tokenizer ──────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer pad_token: {tokenizer.pad_token!r}")

    # ── policy model (fp16) ────────────────────────────────────────────────────
    log.info("Loading policy model in fp16...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # ── BitFit: freeze all weights, unfreeze only biases ──────────────────────
    # Cast bias params to fp32 so that fp16 grad scaling doesn't error.
    # (fp16 amp scaler requires all parameters with gradients to be fp32.)
    log.info("Applying BitFit freeze: weights frozen, biases trainable (fp32)...")
    total_params     = 0
    trainable_params = 0
    for name, p in policy_model.named_parameters():
        total_params += p.numel()
        if name.endswith(".bias"):
            p.data = p.data.to(torch.float32)   # upgrade to fp32 for gradient stability
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False

    trainable_frac = trainable_params / total_params
    log.info(
        f"BitFit config confirmed: {trainable_params:,} / {total_params:,} params trainable "
        f"({trainable_frac * 100:.4f}%)"
    )
    print(f"\n[BitFit] trainable: {trainable_params:,} / {total_params:,} = {trainable_frac*100:.4f}%\n")

    # ── reference model (CPU, frozen) ─────────────────────────────────────────
    # Keep ref on CPU to save GPU memory (policy model + optimiser is ~3-4 GB).
    # DPOTrainer will move tensors to CPU for ref log-prob computation.
    log.info("Loading reference model onto CPU (frozen, to save GPU memory)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    ref_model = ref_model.to("cpu")
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── dataset ────────────────────────────────────────────────────────────────
    dataset = load_hh_rlhf_dpo(tokenizer, max_samples=2000)

    # ── DPO config ─────────────────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir                    = str(CKPT_DIR),
        per_device_train_batch_size   = 1,
        gradient_accumulation_steps   = 8,
        max_steps                     = 800,
        learning_rate                 = 1e-4,           # BitFit: higher LR needed
        lr_scheduler_type             = "cosine",
        warmup_steps                  = 50,
        max_grad_norm                 = 1.0,
        fp16                          = False,   # biases are fp32; no fp16 scaler needed
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        beta                          = 0.1,
        max_length                    = 512,
        save_strategy                 = "steps",
        save_steps                    = 100,
        save_total_limit              = 2,
        logging_steps                 = 10,
        logging_first_step            = True,
        report_to                     = "none",
        seed                          = 42,
        disable_tqdm                  = False,
        dataset_num_proc              = 2,
        remove_unused_columns         = False,
        use_cache                     = False,
        dataloader_num_workers        = 0,
    )

    # ── loss-trajectory callback ───────────────────────────────────────────────
    class LossCallback(tc.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                step = state.global_step
                loss = logs["loss"]
                trajectory.append({"step": step, "loss": loss})
                log.info(f"Step {step}: train_loss={loss:.6f}")
                # Incremental save
                RESULT_JSON.write_text(json.dumps(trajectory, indent=2))

    loss_cb = LossCallback()

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model            = policy_model,
        ref_model        = ref_model,
        args             = training_args,
        train_dataset    = dataset,
        processing_class = tokenizer,
        callbacks        = [loss_cb],
    )

    log.info("Starting BitFit-DPO training (biases only, weights frozen)...")
    train_result = trainer.train()

    wall_sec = time.time() - t_start

    # ── final trajectory write ─────────────────────────────────────────────────
    RESULT_JSON.write_text(json.dumps(trajectory, indent=2))
    log.info(f"Loss trajectory saved: {RESULT_JSON} ({len(trajectory)} entries)")

    # ── verdict ────────────────────────────────────────────────────────────────
    final_loss = train_result.training_loss

    loss_vals = [e["loss"] for e in trajectory if e["loss"] is not None]
    initial_loss = loss_vals[0]  if loss_vals else float("nan")
    min_loss     = min(loss_vals) if loss_vals else float("nan")

    if final_loss <= 0.35:
        verdict = "gauge_alive"
        headline = (
            f"BitFit alone drove DPO loss to {final_loss:.4f}, beating the 0.35 threshold. "
            "Bias floodgates suffice for alignment; weight rewiring is not strictly necessary. "
            "gamma's 3-4 dim rudder is a convenience, not the irreducible mechanism."
        )
    elif final_loss >= 0.60:
        verdict = "gauge_dead"
        headline = (
            f"BitFit stalled at {final_loss:.4f} (near base DPO starting ~0.69). "
            "Additive bias shifts cannot produce DPO alignment. "
            "Weight rewiring is strictly necessary; gamma's 3-4 dim rudder is the irreducible mechanism."
        )
    else:
        verdict = "partial"
        headline = (
            f"BitFit achieved partial alignment ({final_loss:.4f}), below the 0.60 stall threshold "
            f"but above the 0.35 full-alignment threshold. "
            "Some of the alignment signal is bias-accessible; the remainder requires weight rewiring."
        )

    summary = {
        "objective":              "DPO",
        "method":                 "BitFit (biases-only, weights frozen)",
        "trainable_params":       trainable_params,
        "total_params":           total_params,
        "trainable_frac":         trainable_frac,
        "initial_loss":           initial_loss,
        "final_loss":             final_loss,
        "min_loss":               min_loss,
        "wall_sec":               wall_sec,
        "learning_rate":          1e-4,
        "lora_dpo_v2_final_loss": LORA_DPO_V2_FINAL_LOSS,
        "verdict":                verdict,
        "headline":               headline,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved: {SUMMARY_JSON}")

    # ── report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BITFIT-DPO STRIKE — RESULTS")
    print("=" * 60)
    print(f"BitFit config: {trainable_params:,} / {total_params:,} trainable ({trainable_frac*100:.4f}%)")
    print(f"\nLoss trajectory:")
    print(f"  Initial loss (step {trajectory[0]['step'] if trajectory else '?'}): {initial_loss:.4f}")
    # Milestone steps
    for milestone in [100, 200, 400, 800]:
        entries = [e for e in trajectory if e["step"] <= milestone]
        if entries:
            val = entries[-1]["loss"]
            print(f"  Step ~{milestone}: {val:.4f}")
    print(f"  Min loss: {min_loss:.4f}")
    print(f"  Final loss (step 800): {final_loss:.4f}")
    print(f"\nReference: LoRA-DPO v2 (r=128) final loss = {LORA_DPO_V2_FINAL_LOSS:.4f}")
    print(f"\nWall time: {wall_sec/60:.1f} min")
    print(f"\nVerdict: {verdict.upper()}")
    print(f"{headline}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
