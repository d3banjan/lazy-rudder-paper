# Provenance Trace — lazy-rudder-paper results

Every `paper/results/` JSON has a companion `*.config.json` (Option A sidecar).
This table gives the one-line trace: result file → script → training seed → model → LoRA(r,α) → LR → batch → steps → dataset.

All hyperparameter values were verified by reading the source script directly.

## Checkpoint mirror

All adapter weights cited in the analysis tables below are publicly mirrored at:

> <https://huggingface.co/d3banjan/lazy-rudder-checkpoints>

Repo layout mirrors the on-disk `cross-check/trained-model-battery/results/`
tree, so `make fetch-checkpoints` (= `python scripts/fetch_checkpoints.py`)
is a drop-in for `make training` when only analysis is needed (no GPU).

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
| `bitfit_dpo_strike/checkpoints/` | `bitfit_dpo_strike.py` | 42 | pythia-410m | N/A (biases only) | 1e-4 | 8 (1×8 accum) | 800 | Anthropic/hh-rlhf (2000 samples) | positive (trajectory only — weights **not mirrored**; rerun script to regenerate) |
| `bitfit_dpo_strike_extended/checkpoints/` | `bitfit_dpo_strike_extended.py` | 42 | pythia-410m | N/A (biases only) | 1e-4 (cosine restart) | 8 (1×8 accum) | 1500 (800→1500) | Anthropic/hh-rlhf (2000 samples) | positive (trajectory only — weights **not mirrored**; rerun script to regenerate) |

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

## External checkpoints (T2.1 — opportunistic cross-arch)

T2.1 opportunistic run reuses **publicly cached** HF-hub checkpoints (no local training).
Frozen revisions for reproducibility:

| Role | HF repo | Revision SHA |
|---|---|---|
| base | `Qwen/Qwen2-1.5B` | `8a16abf2848eda07cc5253dec660bf1ce007ad7a` |
| SFT (Instruct) | `Qwen/Qwen2-1.5B-Instruct` | `ba1cf1846d7df0a0591d6c00649f57e798519da8` |
| DPO | `lewtun/qwen2-1.5B-ultrafeedback-online-dpo` | `6750e4e383493166cdd8f47a6ebb7a3b79c0a7c6` |

| Result file | Source script | Notes |
|---|---|---|
| `results/t21_qwen_fullweight/per_layer.json` | `scripts/t21_qwen_fullweight_delta.py` | 196 matched params × 3 deltas (Δ_SFT, Δ_DPO, Δ_total) |
| `results/t21_qwen_fullweight/summary.json` | same | mean/median srank + γ per delta, attn/MLP split |
| `results/t21_qwen_fullweight/figH_qwen_delta.png` | same | violin: srank + γ distributions per delta type |

**Caveats (echoed in summary.json `notes`):**
- Full-weight (not LoRA): different inductive bias than the Pythia LoRA chain.
- Dataset: UltraFeedback (not hh-rlhf) — DPO signal is complementary, not strict replication.
- Scale: Qwen2-1.5B (~1.5B params), between Pythia-410m and Pythia-1B in size.
- DPO trainer: TRL OnlineDPOConfig (reward model `CIR-AMS/BTRM_Qwen2_7B_0613`), not offline DPO.
- Strict T2.1 (fresh LoRA-DPO on Qwen + hh-rlhf, r=128 α=256) remains queued separately if needed.

---

## T2.1b cross-probe scoring

**RETRACTED (length-bias artifact, see Appendix `\ref{app:cross-probe-retraction}`) — infrastructure and data retained for post-hoc inspection.**

The original claim (commit `46633a6`) was that srank collapse is decoupled from
in-distribution reward (Pythia-hh Pearson ≈ +0.22) but becomes a strong negative
predictor of OOD transfer (Pythia-UF Pearson ≈ −0.92), reframing the geometric floor as
representational overfitting.  This claim is retracted: the UF Pearson of −0.92 was
produced by summing log-probabilities over all tokens in each response, which introduces
a length-proportional bias when chosen/rejected response lengths are asymmetric across
pairs within a probe (as they systematically are in UltraFeedback).  Per-token
normalization collapses the UF Pearson to +0.12; length-invariant margin_win_rate yields
hh rates in [0.515, 0.546] and UF rates in [0.475, 0.527] — both within ±0.03 of chance.
There is no statistically meaningful cross-probe asymmetry at n=5 once length bias is removed.

The scoring infrastructure, JSONL files, and correlation JSON are retained below for
post-hoc inspection.

**Scoring JSONL files** (500 examples per checkpoint × probe, β=0.1):

| JSONL path | Checkpoint | Probe | Notes |
|---|---|---|---|
| `results/cross_probe/pythia_lora_42_v2_70m__uf.jsonl` | Pythia-70m LoRA-DPO s42 | UltraFeedback | cross-probe (trained on hh-rlhf) |
| `results/cross_probe/pythia_lora_42_v2_160m__uf.jsonl` | Pythia-160m LoRA-DPO s42 | UltraFeedback | cross-probe |
| `results/cross_probe/pythia_lora_42_v2_410m__uf.jsonl` | Pythia-410m LoRA-DPO s42 | UltraFeedback | cross-probe |
| `results/cross_probe/pythia_lora_42_v2_1b__uf.jsonl` | Pythia-1B LoRA-DPO s42 | UltraFeedback | cross-probe |
| `results/cross_probe/pythia_lora_42_v2_1b_s117__uf.jsonl` | Pythia-1B LoRA-DPO s117 | UltraFeedback | cross-probe |
| `results/cross_probe/qwen_fullweight__hh.jsonl` | Qwen2-1.5B full-weight TRL-online-DPO | hh-rlhf | cross-probe (trained on UF) |
| `results/cross_probe/qwen_fullweight__uf.jsonl` | Qwen2-1.5B full-weight TRL-online-DPO | UltraFeedback | in-distribution check |

**Note:** Pythia-on-hh scores reuse T1.2 data via symlinks in `results/cross_probe/`
pointing to `results/behavior_geometry/checkpoint_*.jsonl` — not re-scored.

**Scripts:**
- `scripts/cross_probe_score.py` — GPU scoring (not re-run for this step)
- `scripts/cross_probe_correlate.py` — CPU correlation + Fig I (bugs A+B fixed 2026-04-18)
- `scripts/cross_probe_manifest.json` — job manifest

**Config:** β=0.1, n_examples=500 per (checkpoint, probe).
**Combined correlations** use within-family z-scored srank/gamma/reward_margin to avoid
Simpson's-paradox artifacts (Qwen srank ≈28, Pythia srank ≈3.5).

**Verdict:** RETRACTED — see header above.  Original claim (OOD overfitting penalty) falsified by length-bias analysis.

---

## Unrecoverable results

None. All 14 result JSON files (16 counting `_leak/v2/channel_partition.json` and `bitfit_dpo_strike/loss_trajectory.json`) are traceable to named scripts in `paper/scripts/` or `cross-check/trained-model-battery/`.

---

## Caveats

1. **Non-bit-identical reruns**: Re-running training scripts will not produce bit-identical checkpoints due to non-deterministic CUDA operations. Seeds control data shuffling and torch initialization but not all GPU kernel ordering.
2. **Dataset fallbacks**: Training scripts try `Anthropic/hh-rlhf` first, then `trl-internal-testing/hh-rlhf-helpful-base`, then `HuggingFaceH4/ultrafeedback_binarized`. If the primary source is unavailable, the fallback may produce a different dataset and non-comparable results.
3. **Analysis scripts are deterministic** once the checkpoint adapters are fixed. Re-running them against the same checkpoints produces identical output.
4. **Activation .pt files**: `angular_fourier_delta_prime.py` and `two_point_correlator_delta.py` depend on pre-recorded activation tensors in `cross-check/trained-model-battery/results/_orbit/`. These are not included in the paper repository but can be regenerated from the named checkpoints.
