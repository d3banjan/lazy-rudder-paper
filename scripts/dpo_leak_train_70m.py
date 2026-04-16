"""
Petri-dish DPO train — Pythia-70m, r=128.

Used to measure ΔW srank at small d_in=512, to discriminate:
  - Disentanglement hypothesis (srank ∝ d)
  - Task-intrinsic hypothesis (srank ≈ 3 independent of d)
  - Acoustic 1/√d and 1/d^(1/3) scaling

Mirrors dpo_leak_train_v2.py at 410m with minimal changes:
  - MODEL_DIR  → pythia-70m
  - OUT_DIR    → results/_leak_70m/v2
  - save_total_limit = 2 (disk hygiene; we only need checkpoint-800)

Wall estimate: ~10-15 min.
"""
import os, sys, json, time, math, logging
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/debanjan/Code/Research/lean-mining/cross-check/trained-model-battery")
MODEL_DIR  = BASE_DIR / "models" / "pythia-70m"
OUT_DIR    = BASE_DIR / "results" / "_leak_70m" / "v2"
CKPT_DIR   = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE   = OUT_DIR / "training_log.txt"


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
            parts = chosen_text.rsplit(sep, 1)
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    import transformers.trainer_callback as tc

    t_start = time.time()

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

    # Detect architecture dims
    n_layers = len(policy_model.gpt_neox.layers)
    d_model  = policy_model.gpt_neox.layers[0].attention.query_key_value.weight.shape[1]
    log.info(f"Pythia-70m: n_layers={n_layers}, d_model={d_model}")

    # ── LoRA config r=128 ─────────────────────────────────────────────────────
    LORA_R = 128
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,   # 256
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy_model = get_peft_model(policy_model, lora_cfg)
    policy_model.print_trainable_parameters()

    # ── reference model (frozen) ───────────────────────────────────────────────
    log.info("Loading reference model (frozen base)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
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
        save_steps                  = 400,       # only mid + final; combined with save_total_limit=2
        save_total_limit            = 2,         # disk hygiene: keep only 2 checkpoints
        logging_steps               = 10,
        logging_first_step          = True,
        report_to                   = "none",
        seed                        = 42,
        disable_tqdm                = False,
        dataset_num_proc            = 2,
        remove_unused_columns       = False,
        use_cache                   = False,
        dataloader_num_workers      = 0,
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model            = policy_model,
        ref_model        = ref_model,
        args             = training_args,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )

    log.info("Starting DPO training (Pythia-70m, r=128)...")
    train_result = trainer.train()

    # ── save final checkpoint explicitly at step 800 ───────────────────────────
    final_step = train_result.global_step
    final_ckpt = CKPT_DIR / f"checkpoint-{final_step}"
    if not final_ckpt.exists():
        log.info(f"Saving final checkpoint to {final_ckpt}")
        trainer.save_model(str(final_ckpt))

    wall = time.time() - t_start
    summary = {
        "model":         "pythia-70m",
        "n_layers":      n_layers,
        "d_model":       d_model,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_R * 2,
        "train_loss":    train_result.training_loss,
        "total_steps":   final_step,
        "wall_sec":      wall,
        "checkpoint_dir": str(final_ckpt),
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Train loss: {train_result.training_loss:.6f}")
    log.info(f"Wall time: {wall/60:.1f}min")
    log.info(f"Final checkpoint: {final_ckpt}")
    log.info(f"Summary: {OUT_DIR / 'summary.json'}")

    return summary


if __name__ == "__main__":
    main()
