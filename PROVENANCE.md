# Provenance Trace — lazy-rudder-paper results

Every `paper/results/` JSON has a companion `*.config.json` (Option A sidecar).
This table gives the one-line trace: result file → script → training seed → model → LoRA(r,α) → LR → batch → steps → dataset.

All hyperparameter values were verified by reading the source script directly.

---

## Analysis results (no training in the script itself)

| Result file | Source script | Checkpoint seeds used | Model probed | LoRA(r,α) in checkpoint | Notes |
|---|---|---|---|---|---|
| `angular_fourier_delta_prime/results.json` | `scripts/angular_fourier_delta_prime.py` | N/A (activation .pt files) | pythia-410m (base/sft/dpo) | N/A | Reads `_orbit/*.pt` activation tensors |
| `bias_theory_autopsy/results.json` | `scripts/bias_theory_autopsy.py` | 42 (v1,v2,v3 ckpts) | pythia-410m | v1: r=16,α=32 / v2,v3: r=128,α=256 | Gain-subspace residual test |
| `dpo_clm_orthogonal_decomp/results.json` | `scripts/dpo_clm_orthogonal_decomp.py` | 42 (s42 ckpts), 117 (s117 ckpts) | pythia-1b | r=128, α=256 | 4-way seed×objective decomposition |
| `seed_variance_quick/results.json` | `scripts/seed_variance_quick.py` | 42, 117 (both 1B DPO ckpts) | pythia-1b | r=128, α=256 | CPU-only; srank+bonus_R per layer |
| `spectral_autopsy/results.json` | `scripts/spectral_autopsy.py` | 42 (v1,v2,v3 ckpts) | pythia-410m | v1: r=16,α=32 / v2,v3: r=128,α=256 | Full srank + k90/k99 per layer×module |
| `spectral_autopsy_sectional/results.json` | `scripts/spectral_autopsy_sectional.py` | 42 (v1,v2,v3 ckpts) | pythia-410m | v1: r=16,α=32 / v2,v3: r=128,α=256 | Soft-partition frob_on vs frob_off |
| `spectral_autopsy_sectional_3tier/results.json` | `scripts/spectral_autopsy_sectional_3tier.py` | 42 (v1,v2,v3 ckpts) | pythia-410m | v1: r=16,α=32 / v2,v3: r=128,α=256 | 3-tier: P_PS=0.333, P_SRN=0.662, P_GAUGE=0.005 |
| `spectral_overlap_gamma/results.json` | `scripts/spectral_overlap_gamma.py` | 42 (v1,v2,v3 ckpts) | pythia-410m | v1: r=16,α=32 / v2,v3: r=128,α=256 | γ bonus_R at k=5,10,20 and k=srank |
| `spectral_overlap_gamma_1b/results.json` | `scripts/spectral_overlap_gamma_1b.py` | 42 (1b v2,v3 ckpts) | pythia-1b | r=128, α=256 | Reference run for seed comparison |
| `spectral_overlap_gamma_1b_seed117/results.json` | `scripts/spectral_overlap_gamma_1b_seed117.py` | 42 and 117 (4 ckpts) | pythia-1b | r=128, α=256 | 4-way: seed×objective at 1B |
| `spectral_overlap_gamma_modules/results.json` | `scripts/spectral_overlap_gamma_modules.py` | 42 (410m+1b ckpts) | pythia-410m + pythia-1b | r=128, α=256 | Module-level universality check |
| `spectral_overlap_gamma_petri/results.json` | `scripts/spectral_overlap_gamma_petri.py` | 42 (70m, 160m ckpts) | pythia-70m + pythia-160m | r=128, α=256 | Acoustic vs task-intrinsic scaling test |
| `two_point_correlator_delta/results.json` | `scripts/two_point_correlator_delta.py` | N/A (activation .pt files) | pythia-410m (base/sft/dpo) | N/A | Reads `_orbit/*.pt`; FFT falsification |
| `_leak/v2/channel_partition.json` | `scripts/compute_channel_partition.py` | 0 (activation shuffle seed) | pythia-410m | N/A | 128 samples from Anthropic/hh-rlhf |

---

## Training results (checkpoints produced by these scripts)

These are the underlying trained adapter checkpoints that the analysis scripts above consume.
The paper does not include the checkpoint binaries; only sidecar provenance is tracked here.

| Result dir (in cross-check/trained-model-battery/results/) | Training script | Seed | Model | LoRA(r,α) | LR | Effective batch | Steps | Dataset | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `_leak/checkpoints/` (v1) | `dpo_leak_train.py` | 42 | pythia-410m | r=16, α=32 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `_leak/v2/checkpoints/` | `dpo_leak_train_v2.py` | 42 | pythia-410m | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `_leak/v3/checkpoints/` | `clm_leak_train.py` | 42 | pythia-410m | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (chosen only) | positive |
| `_leak_70m/v2/checkpoints/` | `dpo_leak_train_70m.py` | 42 | pythia-70m | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `_leak_160m/v2/checkpoints/` | `dpo_leak_train_160m.py` | 42 | pythia-160m | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `_leak_1b/v2/checkpoints/` | `dpo_leak_train_1b.py` | 42 | pythia-1b | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `_leak_1b/v3/checkpoints/` | `clm_leak_train_1b.py` | 42 | pythia-1b | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (chosen only) | positive |
| `_leak_1b_seed117/v3/checkpoints/` | `dpo_leak_train_1b_seed117.py` | 117 (independent draw; shuffle seed 117) | pythia-1b | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `_leak_1b_seed117/v4/checkpoints/` | `clm_leak_train_1b_seed117.py` | 117 (independent draw; shuffle seed 117) | pythia-1b | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (chosen only) | positive |
| `_leak_1b_seed117/v2/checkpoints/` (archived) | `dpo_leak_train_1b_seed117.py` (pre-fix; shuffle seed 42) | 117 model-init, shared data draw | pythia-1b | r=128, α=256 | 5e-6 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | superseded by v3 — shuffle-seed bug shared the seed-42 draw |
| `bitfit_dpo_strike/checkpoints/` | `bitfit_dpo_strike.py` | 42 | pythia-410m | N/A (biases only) | 1e-4 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive |
| `bitfit_dpo_strike_extended/checkpoints/` | `bitfit_dpo_strike_extended.py` | 42 | pythia-410m | N/A (biases only) | 1e-4 (cosine restart) | 8 (1×8 accum) | 1600 (800→1600) | Anthropic/hh-rlhf (2000 samples) | positive |

**Common training config** (all LoRA runs): `warmup_steps=50`, `lr_scheduler=cosine`, `fp16=True`, `gradient_checkpointing=True`, `lora_dropout=0.05`, `target_modules=["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]`.

---

## BitFit reference loss 0.487 — provenance

The macro `\loraLoss` / value `0.487` cited in the manuscript as the LoRA-DPO v2 reference loss is traceable:

- **Source**: `cross-check/trained-model-battery/results/_leak/v2/summary_v2.json` → field `train_loss: 0.48728475034236907`
- **Training script**: `cross-check/trained-model-battery/dpo_leak_train_v2.py`
- **Run config**: pythia-410m, r=128, α=256, LR=5e-6, seed=42, 800 steps, Anthropic/hh-rlhf (2000 samples)
- **Checkpoint**: `_leak/v2/checkpoints/checkpoint-800`
- **Verdict**: fully traceable

The value is hardcoded as `LORA_DPO_V2_FINAL_LOSS = 0.487` in `bitfit_dpo_strike.py` and `bitfit_dpo_strike_extended.py` (truncated to 3 decimal places from the full value 0.48728…).

---

## Unrecoverable results

None. All 14 result JSON files (16 counting `_leak/v2/channel_partition.json` and `bitfit_dpo_strike/loss_trajectory.json`) are traceable to named scripts in `paper/scripts/` or `cross-check/trained-model-battery/`.

---

## Caveats

1. **Non-bit-identical reruns**: Re-running training scripts will not produce bit-identical checkpoints due to non-deterministic CUDA operations. Seeds control data shuffling and torch initialization but not all GPU kernel ordering.
2. **Dataset fallbacks**: Training scripts try `Anthropic/hh-rlhf` first, then `trl-internal-testing/hh-rlhf-helpful-base`, then `HuggingFaceH4/ultrafeedback_binarized`. If the primary source is unavailable, the fallback may produce a different dataset and non-comparable results.
3. **Analysis scripts are deterministic** once the checkpoint adapters are fixed. Re-running them against the same checkpoints produces identical output.
4. **Activation .pt files**: `angular_fourier_delta_prime.py` and `two_point_correlator_delta.py` depend on pre-recorded activation tensors in `cross-check/trained-model-battery/results/_orbit/`. These are not included in the paper repository but can be regenerated from the named checkpoints.
