"""
BitFit-DPO Strike — EXTENDED (steps 800 → 1600)

Resumes from checkpoint-800 to test punctuated-equilibrium hypothesis:
does the accelerating trend at steps 720-800 (25% → 39% ratio) continue
into a real phase transition, or was the original GAUGE_DEAD verdict correct?

Reference losses:
  - LoRA-DPO v2 (r=128): 0.487
  - BitFit step-800 endpoint (logged): 0.612
  - BitFit step-800 endpoint (training_loss avg): 0.6601

Updated verdict thresholds (applied at step 1600):
  <= 0.35  → gauge_alive_at_1600   (punctuated equilibrium confirmed)
  0.35-0.50 → gauge_partial_at_1600 (reaches but doesn't beat LoRA)
  0.50-0.60 → gauge_progressive     (still improving, needs even longer)
  >= 0.60  → gauge_confirmed_dead   (plateau validated)
"""
import os, sys, json, time, math, logging
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import MODELS_DIR, RESULTS_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR  = MODELS_DIR / "pythia-410m"
# Resume checkpoint from the ORIGINAL run (DO NOT overwrite it)
RESUME_CKPT = RESULTS_DIR / "bitfit_dpo_strike" / "checkpoints" / "checkpoint-800"
# New output directory — original preserved intact
OUT_DIR    = RESULTS_DIR / "bitfit_dpo_strike_extended"
CKPT_DIR   = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON  = OUT_DIR / "loss_trajectory.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

# Prior run data for merged trajectory
PRIOR_TRAJ_JSON = RESULTS_DIR / "bitfit_dpo_strike" / "loss_trajectory.json"

# Reference values
LORA_DPO_V2_FINAL_LOSS = 0.487
BITFIT_STEP800_LOGGED  = 0.612   # last logged step value from prior run
BITFIT_STEP800_AVG     = 0.6601  # training_loss (averaged) from prior run
LORA_INITIAL_DROP      = 0.206   # LoRA's total drop: 0.693 - 0.487


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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    import transformers.trainer_callback as tc

    t_start = time.time()
    trajectory_new = []  # Only steps 800–1600 from this run

    # ── tokenizer ──────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer pad_token: {tokenizer.pad_token!r}")

    # ── policy model (fp16 base; biases will be cast to fp32) ──────────────────
    # We load the BASE model; checkpoint-800 restores weights via resume_from_checkpoint.
    log.info("Loading policy model base in fp16...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # ── BitFit: freeze all weights, unfreeze only biases ──────────────────────
    # IMPORTANT: PyTorch freezing does NOT persist in checkpoints (requires_grad
    # is not saved). We must re-apply here, after model load, before training.
    # The optimizer state in the checkpoint was saved for the bias params only,
    # and will be correctly matched when resume_from_checkpoint is used.
    log.info("Re-applying BitFit freeze: weights frozen, biases trainable (fp32)...")
    total_params     = 0
    trainable_params = 0
    for name, p in policy_model.named_parameters():
        total_params += p.numel()
        if name.endswith(".bias"):
            p.data = p.data.to(torch.float32)   # fp32 for gradient stability
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
    log.info("Loading reference model onto CPU (frozen, to save GPU memory)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    ref_model = ref_model.to("cpu")
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── dataset (same seed=42 shuffle → identical sample order) ───────────────
    dataset = load_hh_rlhf_dpo(tokenizer, max_samples=2000)

    # ── DPO config ─────────────────────────────────────────────────────────────
    # max_steps=1600 extends the schedule. Cosine curve is recomputed from this
    # new total: LR at step 800 (mid-point of 1600-step cosine) is ~LR_max/2.
    # This is acknowledged: it's a different LR profile than "train 1600 from scratch"
    # but correct for an extension run.
    training_args = DPOConfig(
        output_dir                    = str(CKPT_DIR),
        per_device_train_batch_size   = 1,
        gradient_accumulation_steps   = 8,
        max_steps                     = 1600,
        learning_rate                 = 1e-4,
        lr_scheduler_type             = "cosine",
        warmup_steps                  = 50,
        max_grad_norm                 = 1.0,
        fp16                          = False,   # biases are fp32
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        beta                          = 0.1,
        max_length                    = 512,
        save_strategy                 = "steps",
        save_steps                    = 100,
        save_total_limit              = 3,
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
                trajectory_new.append({"step": step, "loss": loss})
                log.info(f"Step {step}: train_loss={loss:.6f}")
                # Incremental save of new-steps-only trajectory
                RESULT_JSON.write_text(json.dumps(trajectory_new, indent=2))

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

    log.info(f"Resuming from checkpoint: {RESUME_CKPT}")
    log.info("Starting BitFit-DPO extended training (steps 800 → 1600)...")
    train_result = trainer.train(resume_from_checkpoint=str(RESUME_CKPT))

    wall_sec = time.time() - t_start

    # ── merge trajectories ─────────────────────────────────────────────────────
    # Load the original step 1-800 trajectory, append new steps > 800
    prior_trajectory = []
    if PRIOR_TRAJ_JSON.exists():
        try:
            prior_trajectory = json.loads(PRIOR_TRAJ_JSON.read_text())
            log.info(f"Loaded prior trajectory: {len(prior_trajectory)} entries (steps 1–800)")
        except Exception as e:
            log.warning(f"Could not load prior trajectory: {e}")

    # new entries from this run are steps > 800 (resume replays step 800 log too)
    new_entries = [e for e in trajectory_new if e["step"] > 800]
    merged = prior_trajectory + new_entries

    MERGED_JSON = OUT_DIR / "loss_trajectory_merged.json"
    MERGED_JSON.write_text(json.dumps(merged, indent=2))
    # Also write the new-steps-only trajectory (already written incrementally)
    RESULT_JSON.write_text(json.dumps(trajectory_new, indent=2))

    log.info(f"New trajectory saved: {RESULT_JSON} ({len(trajectory_new)} entries)")
    log.info(f"Merged trajectory saved: {MERGED_JSON} ({len(merged)} entries)")

    # ── analysis ──────────────────────────────────────────────────────────────
    final_loss = train_result.training_loss

    all_loss_vals = [e["loss"] for e in merged if e["loss"] is not None]
    new_loss_vals = [e["loss"] for e in new_entries if e["loss"] is not None]

    min_loss_overall  = min(all_loss_vals)  if all_loss_vals else float("nan")
    min_loss_new      = min(new_loss_vals)  if new_loss_vals else float("nan")

    # Last-10-window mean at step 1600 (entries closest to step 1600)
    last_10_new = [e["loss"] for e in sorted(new_entries, key=lambda x: x["step"])[-10:]
                   if e["loss"] is not None]
    last_10_mean = sum(last_10_new) / len(last_10_new) if last_10_new else float("nan")

    # Last-100-step mean (entries with step > 1500)
    last_100_entries = [e["loss"] for e in new_entries if e["step"] > 1500 and e["loss"] is not None]
    last_100_mean = sum(last_100_entries) / len(last_100_entries) if last_100_entries else float("nan")

    # Learning ratio using last-10 window as the "final" metric
    # Total LoRA drop: 0.693 - 0.487 = 0.206
    initial_loss    = all_loss_vals[0] if all_loss_vals else 0.693
    total_drop_new  = initial_loss - last_10_mean
    learning_ratio  = total_drop_new / LORA_INITIAL_DROP  if LORA_INITIAL_DROP > 0 else float("nan")

    # Step-800 vs step-1600 improvement
    improvement_800_to_1600 = BITFIT_STEP800_LOGGED - last_10_mean

    # Shape analysis: compare first-half (800-1200) vs second-half (1200-1600) of new run
    first_half  = [e["loss"] for e in new_entries if 800  < e["step"] <= 1200 and e["loss"] is not None]
    second_half = [e["loss"] for e in new_entries if 1200 < e["step"] <= 1600 and e["loss"] is not None]
    mean_first  = sum(first_half)  / len(first_half)  if first_half  else float("nan")
    mean_second = sum(second_half) / len(second_half) if second_half else float("nan")

    # ── verdict ────────────────────────────────────────────────────────────────
    eval_loss = last_10_mean  # prefer windowed average over noisy single-step value

    if eval_loss <= 0.35:
        verdict = "gauge_alive_at_1600"
        headline = (
            f"BitFit drove DPO loss to {eval_loss:.4f} (last-10 window) by step 1600, "
            "clearing the 0.35 threshold. Punctuated-equilibrium confirmed: "
            "the accelerating trend at steps 720-800 was a genuine precursor. "
            "Additive bias shifts suffice for alignment; the GAUGE_DEAD verdict is revised."
        )
    elif eval_loss <= 0.50:
        verdict = "gauge_partial_at_1600"
        headline = (
            f"BitFit reached {eval_loss:.4f} (last-10 window) by step 1600, "
            f"closing in on LoRA's {LORA_DPO_V2_FINAL_LOSS} but not beating it. "
            "Punctuated dynamics partially confirmed: biases carry significant alignment signal "
            "but weight rewiring still provides a non-trivial residual advantage."
        )
    elif eval_loss <= 0.60:
        verdict = "gauge_progressive"
        headline = (
            f"BitFit reached {eval_loss:.4f} (last-10 window) by step 1600. "
            "Still improving from the step-800 endpoint (0.612) but slowly. "
            "Would require substantially more steps to approach LoRA parity; "
            "the asymptotic capacity may still be below LoRA's final 0.487."
        )
    else:
        verdict = "gauge_confirmed_dead"
        headline = (
            f"BitFit plateaued at {eval_loss:.4f} (last-10 window) by step 1600, "
            f"above the 0.60 threshold despite 800 additional steps. "
            "The accelerating trend at steps 720-800 was transient noise, not a phase transition. "
            "GAUGE_DEAD verdict confirmed: weight rewiring is strictly necessary for alignment."
        )

    summary = {
        "objective":                   "DPO",
        "method":                      "BitFit extended (biases-only, resume from step-800)",
        "trainable_params":            trainable_params,
        "total_params":                total_params,
        "trainable_frac":              trainable_frac,
        "resume_from":                 str(RESUME_CKPT),
        # Prior run anchors
        "bitfit_step800_logged":       BITFIT_STEP800_LOGGED,
        "bitfit_step800_avg":          BITFIT_STEP800_AVG,
        # Extended run results
        "final_loss_training_avg":     final_loss,
        "last_10_window_mean":         last_10_mean,
        "last_100_step_mean":          last_100_mean,
        "min_loss_new_steps":          min_loss_new,
        "min_loss_overall":            min_loss_overall,
        # Comparison
        "lora_dpo_v2_final_loss":      LORA_DPO_V2_FINAL_LOSS,
        "improvement_step800_to_1600": improvement_800_to_1600,
        "learning_ratio_at_1600":      learning_ratio,
        # Shape
        "mean_loss_steps_800_1200":    mean_first,
        "mean_loss_steps_1200_1600":   mean_second,
        "half_to_half_drop":           mean_first - mean_second if (first_half and second_half) else float("nan"),
        # Verdict
        "verdict":                     verdict,
        "headline":                    headline,
        "wall_sec":                    wall_sec,
        "cosine_note":                 (
            "max_steps=1600 recomputes cosine curve; steps 800-1600 see lower LR "
            "(mid-cosine ~LR/2) vs. a fresh 1600-step run. Acknowledged trade-off."
        ),
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved: {SUMMARY_JSON}")

    # ── terminal report ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("BITFIT-DPO STRIKE EXTENDED — RESULTS (steps 800 → 1600)")
    print("=" * 68)
    print(f"BitFit config: {trainable_params:,} / {total_params:,} trainable ({trainable_frac*100:.4f}%)")
    print(f"\nTrajectory milestones:")
    print(f"  Step 800 (prior run, logged):  {BITFIT_STEP800_LOGGED:.4f}")
    for milestone in [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]:
        entries = [e for e in new_entries if e["step"] <= milestone]
        if entries:
            val = entries[-1]["loss"]
            print(f"  Step ~{milestone}:  {val:.4f}")
    print(f"\nAnalysis:")
    print(f"  Min loss (steps 800–1600):     {min_loss_new:.4f}")
    print(f"  Last-10-window mean (≈1600):   {last_10_mean:.4f}")
    print(f"  Last-100-step mean (>1500):    {last_100_mean:.4f}")
    print(f"  Mean steps 800–1200:           {mean_first:.4f}")
    print(f"  Mean steps 1200–1600:          {mean_second:.4f}")
    print(f"  Improvement step 800→1600:     {improvement_800_to_1600:+.4f}")
    print(f"\nComparison:")
    print(f"  LoRA-DPO v2 final:             {LORA_DPO_V2_FINAL_LOSS:.4f}")
    print(f"  Updated learning ratio:        {learning_ratio:.1%}  (BitFit drop / LoRA drop)")
    print(f"\nWall time: {wall_sec/60:.1f} min")
    print(f"\nVerdict: {verdict.upper()}")
    print(f"{headline}")
    print("=" * 68)

    return summary


if __name__ == "__main__":
    main()
