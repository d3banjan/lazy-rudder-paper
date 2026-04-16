"""
γ₁ Step 1 — DPO LoRA r=128 on Pythia-1B (SEED=117 replicate).

Identical to dpo_leak_train_v2.py except:
  - MODEL_DIR → models/pythia-1b
  - OUT_DIR   → results/_leak_1b_seed117/v2
  - save_total_limit = 2 (saves disk; only need checkpoint-800 for γ)

PARTITION_PATH will not exist; script falls back to coord-block for L
measurement (acceptable — we only need the final adapter checkpoint for γ₁).
"""
import os, sys, json, time, math, logging
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _paths import MODELS_DIR, RESULTS_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR  = MODELS_DIR / "pythia-1b"
OUT_DIR    = RESULTS_DIR / "_leak_1b_seed117" / "v2"
CKPT_DIR   = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON    = OUT_DIR / "l_trajectory_orbit_aligned.json"
LOG_FILE       = OUT_DIR / "training_log.txt"
PARTITION_PATH = OUT_DIR / "channel_partition.json"  # will not exist; fallback OK


# ── orbit-aligned L-metric (identical to v2) ──────────────────────────────────

def load_partition(path: Path) -> dict:
    if not path.exists():
        log.warning(f"Partition file not found: {path} — using coord-block fallback")
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw["partition"].items()}


def coord_block_off_mass(W: torch.Tensor, k: int = 4) -> float:
    if W.dim() < 2:
        return float('nan')
    W = W.float()
    rows, cols = W.shape[0], W.shape[1]
    r_bs = rows // k
    c_bs = cols // k
    if r_bs == 0 or c_bs == 0:
        return float('nan')
    total_sq = (W ** 2).sum().item()
    if total_sq == 0.0:
        return 0.0
    diag_sq = 0.0
    for i in range(k):
        r0, r1 = i * r_bs, min((i + 1) * r_bs, rows)
        c0, c1 = i * c_bs, min((i + 1) * c_bs, cols)
        diag_sq += (W[r0:r1, c0:c1] ** 2).sum().item()
    return (total_sq - diag_sq) / total_sq


def orbit_aligned_off_mass(W: torch.Tensor, row_quartiles: list, col_quartiles: list,
                            k: int = 4) -> float:
    if W.dim() < 2:
        return float('nan')
    W = W.float()
    out_dim, in_dim = W.shape[0], W.shape[1]
    rq = row_quartiles[:out_dim]
    cq = col_quartiles[:in_dim]
    if len(rq) != out_dim or len(cq) != in_dim:
        return float('nan')
    rq_t = torch.tensor(rq, dtype=torch.long)
    cq_t = torch.tensor(cq, dtype=torch.long)
    total_sq = (W ** 2).sum().item()
    if total_sq == 0.0:
        return 0.0
    diag_sq = 0.0
    for q in range(k):
        row_mask = (rq_t == q).nonzero(as_tuple=True)[0]
        col_mask = (cq_t == q).nonzero(as_tuple=True)[0]
        if len(row_mask) == 0 or len(col_mask) == 0:
            continue
        block = W[row_mask][:, col_mask]
        diag_sq += (block ** 2).sum().item()
    return (total_sq - diag_sq) / total_sq


def compute_L_orbit_aligned(model, partition: dict, k: int = 4) -> dict:
    base = model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model

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
        return {'L_mean': float('nan'), 'per_layer': [], 'method': 'error'}

    use_orbit = bool(partition)
    method = "orbit_aligned" if use_orbit else "coord_block_fallback"
    per_layer = []

    for i, layer in enumerate(layers):
        try:
            attn = layer.attention
            qkv = attn.query_key_value.weight.detach()  # (3*d, d)
            d = qkv.shape[1]
            q_w = qkv[:d, :]
            k_w = qkv[d:2*d, :]
            v_w = qkv[2*d:, :]

            if use_orbit and i in partition:
                rq = partition[i]["output_quartiles"]
                cq = partition[i]["input_quartiles"]
                vals = [orbit_aligned_off_mass(w, rq, cq, k) for w in [q_w, k_w, v_w]]
            else:
                vals = [coord_block_off_mass(w, k) for w in [q_w, k_w, v_w]]

            vals = [v for v in vals if not math.isnan(v)]
            layer_L = sum(vals) / len(vals) if vals else float('nan')
            per_layer.append(layer_L)
        except AttributeError as e:
            log.warning(f"Layer {i} missing attention.query_key_value: {e}")
            per_layer.append(float('nan'))

    valid = [v for v in per_layer if not math.isnan(v)]
    L_mean = sum(valid) / len(valid) if valid else float('nan')
    return {'L_mean': L_mean, 'per_layer': per_layer, 'method': method}


def compute_L_merged(base_path: str, ckpt_path: str, partition: dict, k: int = 4) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM as AMCL
    base = AMCL.from_pretrained(base_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    peft = PeftModel.from_pretrained(base, ckpt_path)
    merged = peft.merge_and_unload()
    merged.eval()
    result = compute_L_orbit_aligned(merged, partition, k)
    del base, peft, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ── dataset ───────────────────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    import transformers.trainer_callback as tc

    t_start = time.time()
    torch.manual_seed(117)
    torch.cuda.manual_seed_all(117)
    trajectory = []

    # ── partition ─────────────────────────────────────────────────────────────
    partition = load_partition(PARTITION_PATH)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer pad_token: {tokenizer.pad_token!r}")

    # ── policy model (fp16) ───────────────────────────────────────────────────
    log.info("Loading policy model (Pythia-1B) in fp16...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # ── LoRA config r=128 — same as 410m v2 ──────────────────────────────────
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

    # ── reference model (frozen) ──────────────────────────────────────────────
    log.info("Loading reference model (frozen Pythia-1B base)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = load_hh_rlhf_dpo(tokenizer, max_samples=2000)

    # ── DPO config ────────────────────────────────────────────────────────────
    training_args = DPOConfig(
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
        beta                          = 0.1,
        max_length                    = 512,
        save_strategy                 = "steps",
        save_steps                    = 100,
        save_total_limit              = 2,   # only keep last 2; we need checkpoint-800
        logging_steps                 = 10,
        logging_first_step            = True,
        report_to                     = "none",
        seed                          = 117,
        disable_tqdm                  = False,
        dataset_num_proc              = 2,
        remove_unused_columns         = False,
        use_cache                     = False,
        dataloader_num_workers        = 0,
    )

    # ── step 0: base weights (adapter zero at init) ───────────────────────────
    log.info("Measuring L at step 0 (base weights)...")
    base_for_step0 = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    with torch.no_grad():
        step0_L = compute_L_orbit_aligned(base_for_step0, partition)
    del base_for_step0
    log.info(f"Step 0: L_mean = {step0_L['L_mean']:.6f} (method={step0_L['method']})")
    trajectory.append({"step": 0, "L": step0_L["L_mean"], "per_layer": step0_L["per_layer"],
                       "method": step0_L["method"]})

    # ── callback ──────────────────────────────────────────────────────────────
    class LeakCallback1B(tc.TrainerCallback):
        def __init__(self):
            self.last_loss = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.last_loss = logs.get("loss", self.last_loss)
                log.info(f"Step {state.global_step}: loss={self.last_loss}")

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            ckpt_path = Path(args.output_dir) / f"checkpoint-{step}"
            log.info(f"Checkpoint at step {step}: {ckpt_path} — merging and measuring L...")
            try:
                ldata = compute_L_merged(str(MODEL_DIR), str(ckpt_path), partition)
                log.info(f"Step {step}: L_mean = {ldata['L_mean']:.6f} (method={ldata['method']})")
                trajectory.append({
                    "step":      step,
                    "L":         ldata["L_mean"],
                    "per_layer": ldata["per_layer"],
                    "method":    ldata["method"],
                })
            except Exception as e:
                log.error(f"Step {step}: merge+measure failed: {e}")
                trajectory.append({"step": step, "L": float('nan'), "per_layer": [], "method": "error"})
            RESULT_JSON.write_text(json.dumps(trajectory, indent=2))

    leak_cb = LeakCallback1B()

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model            = policy_model,
        ref_model        = ref_model,
        args             = training_args,
        train_dataset    = dataset,
        processing_class = tokenizer,
        callbacks        = [leak_cb],
    )

    log.info("Starting DPO training (Pythia-1B, r=128, 800 steps)...")
    train_result = trainer.train()

    # ── final measurement ─────────────────────────────────────────────────────
    final_step = train_result.global_step
    log.info(f"Training complete at step {final_step}.")

    final_ckpt = CKPT_DIR / f"checkpoint-{final_step}"
    if not final_ckpt.exists():
        trainer.save_model(str(final_ckpt))

    if final_ckpt.exists() and (not trajectory or trajectory[-1]["step"] != final_step):
        try:
            final_L = compute_L_merged(str(MODEL_DIR), str(final_ckpt), partition)
            trajectory.append({
                "step":      final_step,
                "L":         final_L["L_mean"],
                "per_layer": final_L["per_layer"],
                "method":    final_L.get("method", "?"),
            })
        except Exception as e:
            log.error(f"Final merge failed: {e}")
            trajectory.append({"step": final_step, "L": float('nan'), "per_layer": [], "method": "error"})

    L_vals = [e["L"] for e in trajectory if not math.isnan(e["L"])]
    L_min  = min(L_vals) if L_vals else float('nan')
    L_max  = max(L_vals) if L_vals else float('nan')
    L_mean = sum(L_vals) / len(L_vals) if L_vals else float('nan')

    summary = {
        "model":           "pythia-1b",
        "objective":       "DPO",
        "lora_r":          LORA_R,
        "train_loss":      train_result.training_loss,
        "total_steps":     final_step,
        "wall_sec":        time.time() - t_start,
        "L_mean":          L_mean,
        "n_checkpoints":   len(trajectory),
    }

    RESULT_JSON.write_text(json.dumps(trajectory, indent=2))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info(f"Wall time: {summary['wall_sec']/3600:.2f}h  train_loss={train_result.training_loss:.6f}")
    log.info(f"Checkpoint saved at: {final_ckpt}")
    print(f"\nDPO 1B done. Wall={summary['wall_sec']/60:.1f}min  loss={train_result.training_loss:.4f}")
    print(f"Final checkpoint: {final_ckpt}")

    return summary


if __name__ == "__main__":
    main()
