# lazy-rudder-paper

Source artifacts for the manuscript on LoRA alignment geometry across Pythia 70M–1B.

## Summary

This paper presents an axiomatic framework and empirical scaling study decoupling the energy from the geometry of LoRA adapters under Direct Preference Optimization (DPO). Using Lean 4 formal verification, we prove that $\alpha$-scaling bounds Frobenius energy (validating RsLoRA's $\alpha/\sqrt{r}$ invariance) but leaves stable rank unchanged. Empirically, across three orders of model width (Pythia 70M → 1B), we observe a **task-intrinsic stable rank** of $\text{srank} \approx 3.6 \pm 0.5$—independent of model capacity. The alignment manifold is shaped by preference-learning complexity, not parameter count.

## Repository Structure

```
paper/
├── lean/                    Lean 4 theorems (Mathlib v4.28.0)
│   ├── SubspaceOverlap.lean             -- 19 proven theorems/lemmas, 18 sorry stubs
│   └── NeuralGeometry_aggregator.lean   -- module container
├── manuscript/
│   ├── main.tex                         -- main document (auto-generated macros from scripts)
│   ├── values.tex, tables.tex, lean_status.tex -- generated from scripts/results/
│   └── Makefile                         -- orchestrates paper build
├── scripts/                 28 Python drivers (training, analysis, ablations)
├── results/                 16 results.json files (experiments only, no checkpoints)
└── Makefile                 top-level task orchestrator
```

## Reproducibility — analysis-only path (no GPU)

All adapter checkpoints cited in the paper are mirrored on HuggingFace at
<https://huggingface.co/d3banjan/lazy-rudder-checkpoints> (~2.5 GB, public).
With them you can rebuild every JSON in `results/`, every figure, and the
final PDF without re-running any training:

```bash
git clone https://github.com/d3banjan/lazy-rudder-paper
cd lazy-rudder-paper

# 1. Tell the scripts where to put the checkpoints (any writeable dir).
cp config.example.toml config.toml
$EDITOR config.toml         # set results_dir = "/path/to/results"

# 2. Pull adapters from HuggingFace (idempotent, ~2.5 GB).
uv sync                     # or pip install huggingface_hub
make fetch-checkpoints      # = python scripts/fetch_checkpoints.py

# 3. Re-run analysis + rebuild paper.
make analysis               # all *.json under results/
make paper                  # manuscript/main.pdf
```

Hard reproducibility requires the same Pythia base weights from EleutherAI
(downloaded automatically into `models_dir` on first analysis run).

## Build Instructions

### Prerequisites

```bash
# Python ≥ 3.12
uv sync   # or: pip install torch transformers peft trl datasets safetensors scipy

# Lean (for lean_status.tex generation)
elan update  # Lean via elan; v4.28.0 pinned in lean-toolchain
```

### Building the PDF only

```bash
cd manuscript
make paper       # full rebuild (pdflatex + bibtex)
make values      # regenerate values.tex from results/*.json
make tables      # regenerate tables.tex
make lean-status # regenerate lean_status.tex from lean/*.lean
```

Or from repo root:
```bash
make paper       # delegates to manuscript/Makefile
```

## Lean Theorems (SubspaceOverlap.lean)

| Status | Count | Examples |
|--------|-------|----------|
| **Proven** | 19 | `frobeniusSq_nonneg`, `frobeniusSq_eq_zero_iff`, `rank_outer_product_le_one`, `rank_add_le`, `rank_sum_outer_products_le`, `outerProduct_eq_col_mul_row`, `frobeniusSq_smul`, `spectralSq_smul`, `ratio_smul_invariant_of_quadratic`, `stableRank_smul_invariant`, `loraUpdate_frob_decays`, `rsLoraUpdate_frob_bounded` |
| **Sorry Stubs** | 18 | `stableRank_le_rank`, `stableRank_mul_le_min`, `stable_rank_disentanglement`, `rank_kronecker_eq_mul`, `random_subspace_expected_overlap`, `subspace_dilution`, `gamma_right_alignment`, `bias_autopsy_separation`, `stable_rank_acoustic_scaling`, `lower_bound_of_intent`, `core_preservation_under_alignment`, `rsLoRA_scaling_invariance`, and 6 others |

Load-bearing theorem: `rsLoraUpdate_frob_bounded` (Frobenius² ≤ α²c, rank-invariant under r).

## Scripts & Experiments (28 total)

### Training (GPU, ~5 hours serial)

| Script | Model | Objective | Steps | Output | Verdict |
|--------|-------|-----------|-------|--------|---------|
| `dpo_leak_train_70m.py` | Pythia-70M | DPO | 800 | `_leak_70m/v2/` | positive |
| `dpo_leak_train_160m.py` | Pythia-160M | DPO | 800 | `_leak_160m/v2/` | positive |
| `dpo_leak_train_v2.py` | Pythia-410M | DPO | 800 | `_leak/v2/` | positive |
| `clm_leak_train.py` | Pythia-410M | CLM (no ref) | 800 | `_leak/v3/` | positive |
| `dpo_leak_train_1b.py` | Pythia-1B | DPO (s=42) | 800 | `_leak_1b/v2/` | positive |
| `dpo_leak_train_1b_seed117.py` | Pythia-1B | DPO (s=117) | 800 | `_leak_1b_seed117/v3/` | positive |
| `clm_leak_train_1b.py` | Pythia-1B | CLM (s=42) | 800 | `_leak_1b/v3/` | positive |
| `clm_leak_train_1b_seed117.py` | Pythia-1B | CLM (s=117) | 800 | `_leak_1b_seed117/v4/` | positive |
| `bitfit_dpo_strike.py` | Pythia-410M | bias-only DPO | 800 | `bitfit_dpo_strike/` | positive (gauge-theory falsification) |
| `bitfit_dpo_strike_extended.py` | Pythia-410M | bias-only DPO | 1600 | `bitfit_dpo_strike_extended/` | positive |

### Analysis (CPU-feasible)

| Script | Purpose | Output | Verdict |
|--------|---------|--------|---------|
| `spectral_overlap_gamma.py` | γ subspace overlap at 410M (DPO + CLM) | `spectral_overlap_gamma/results.json` | positive (5–8× random baseline) |
| `spectral_overlap_gamma_1b.py` | γ at 1B seed 42 | `spectral_overlap_gamma_1b/results.json` | positive |
| `spectral_overlap_gamma_1b_seed117.py` | γ at 1B seed 117 + 4-way seed comparison | `spectral_overlap_gamma_1b_seed117/results.json` | positive |
| `spectral_overlap_gamma_petri.py` | γ petri-dish at 70M + 160M | `spectral_overlap_gamma_petri/results.json` | positive (acoustic scaling signal) |
| `spectral_overlap_gamma_modules.py` | γ layer-by-layer module probe | `spectral_overlap_gamma_modules/results.json` | exploratory |
| `spectral_autopsy.py` | stable-rank + sectional SVD audit at 410M | `spectral_autopsy/results.json` | positive (srank ≈ 3.9) |
| `spectral_autopsy_sectional_3tier.py` | 3-tier orbit-aligned sectional decomposition | `spectral_autopsy_sectional_3tier/results.json` | positive (orbit fraction isolation) |
| `spectral_autopsy_sectional.py` | sectional SVD via soft partition | `spectral_autopsy_sectional/results.json` | exploratory |
| `bias_theory_autopsy.py` | diagonal-gain decomposition test (LN-γ) | `bias_theory_autopsy/results.json` | positive (99.97% residual outside gain subspace) |
| `seed_variance_quick.py` | γ across seeds (1B DPO s=42,117) | `seed_variance_quick/results.json` | positive (≤5% variance) |
| `two_point_correlator_delta.py` | δ: 2-point layer-depth correlator C(L,L+k) | `two_point_correlator_delta/results.json` | **negative** (no depth-wise structure; FFT falsification) |
| `angular_fourier_delta_prime.py` | δ′: angular-Fourier spectroscopy on γ basis | `angular_fourier_delta_prime/results.json` | **negative** (no angular modulation; δ′ falsified) |
| `compute_channel_partition.py` | auxiliary: soft channel partition for 3-tier | `_leak/v2/channel_partition.json` | required for sectional_3tier |
| `dpo_clm_orthogonal_decomp.py` | DPO vs CLM geometric decomposition | `dpo_clm_orthogonal_decomp/results.json` | exploratory |
| `remeasure_leak_trajectory.py` | loss trajectory re-audit | (included in checkpoints/) | exploratory |
| `remeasure_leak_orbit_fraction.py` | orbit fraction measurement v1 | exploratory | |
| `remeasure_leak_orbit_fraction_clean.py` | orbit fraction measurement (cleaned) | exploratory | |

**Verdict key**: positive = supports manuscript claim; negative = falsification result; exploratory = auxiliary/variant probe.

## Results Inventory

| File | Experiment | Key Numbers | Seeded? |
|------|-----------|-------------|---------|
| `spectral_overlap_gamma/results.json` | γ at 410M, per layer | srank_delta ≈ 4.07 (DPO), bonus_R(k=srank) ≈ 2.1× | no |
| `spectral_overlap_gamma_1b/results.json` | γ at 1B seed 42, per layer | srank ≈ 3.13, bonus_R(k=srank) ≈ 5.4× | hardcoded s=42 |
| `spectral_overlap_gamma_1b_seed117/results.json` | γ comparison 1B s=42 vs s=117 (independent draw) | CLM bonus_R(k=5)=4.15× (s=42), 4.02× (s=117); DPO=3.24×/3.49× | hardcoded s=42,117 |
| `spectral_overlap_gamma_petri/results.json` | γ at 70M + 160M | 70M srank=3.93, 160M=3.51 (acoustic trend) | no |
| `spectral_autopsy/results.json` | srank + sectional SVD at 410M | global srank 3.92 (DPO), 3.92 (CLM) | no |
| `spectral_autopsy_sectional_3tier/results.json` | 3-tier orbit decomposition | orbit fraction ≈ 68–72% at fixed-window | no |
| `bias_theory_autopsy/results.json` | gain-subspace projection test | 99.97% outside LN-gain (DPO), 99.97% (CLM) | no |
| `seed_variance_quick/results.json` | γ seed variance 1B | Δ(bonus_R) ≤ 5% across 2 seeds | hardcoded s=42,117 |
| `two_point_correlator_delta/results.json` | layer-depth correlator C(L,L+k) | Pearson ≈ 0.96–0.98 (no trend) | no |
| `angular_fourier_delta_prime/results.json` | angular-Fourier probe of γ basis | basis_energy_fraction ≈ 0.01–0.13 (flat) | no |
| `bitfit_dpo_strike/summary.json` | bias-only DPO loss reduction | ~0.23 reduction (gauge-accessible loss) | no |
| `bitfit_dpo_strike_extended/loss_trajectory.json` | extended BitFit to 1600 steps | convergent to checkpoint-800 signal | no |

**Seeding note**: Seeds are encoded in script filenames and sidecar `*.config.json` files alongside each result JSON. See `PROVENANCE.md` for the complete trace table.

## Reproducibility & Path Configuration

### Base Models

Scripts require Pythia base weights from HuggingFace (`EleutherAI/pythia-{70m,160m,410m,1b}`).
Path resolution is handled by `scripts/_paths.py` in this order:

1. **Environment variables** (highest priority):
   ```
   LAZY_RUDDER_MODELS_DIR   # Pythia base weight root
   LAZY_RUDDER_RESULTS_DIR  # training output root
   LAZY_RUDDER_BASE_DIR     # battery root (legacy alias)
   ```
2. **`config.toml`** in the repo root — copy from `config.example.toml` and edit:
   ```bash
   cp config.example.toml config.toml
   # edit models_dir, results_dir in config.toml
   ```
3. **Fallback** (prints a warning): `../cross-check/trained-model-battery/{models,results}` — the original in-place dev layout. Works if you have the parent `lean-mining` repo checked out.

To download base weights programmatically:
```bash
python -c "from scripts._paths import download_model; download_model('pythia-410m')"
```
Or from `paper/scripts/`:
```bash
python -c "from _paths import download_model; download_model('pythia-410m')"
```
Repeat for `pythia-70m`, `pythia-160m`, `pythia-1b`.

### Reproducibility — What Is and Isn't Recoverable

Seeds are encoded in script filenames (e.g. `dpo_leak_train_1b_seed117.py` → seed=117) and in sidecar `*.config.json` files alongside every result JSON. To reproduce a result, run `paper/scripts/<script>.py`, which will write `paper/results/<name>/results.json`. The sidecar documents the exact checkpoint path, model, LoRA(r,α), LR, batch, steps, and dataset used.

**See `paper/PROVENANCE.md`** for the full trace table: `result file | script | seed | model | LoRA(r,α) | LR | batch | steps | dataset | verdict`.

Limitations:
- Re-running training scripts will not produce bit-identical checkpoints. Seeds control data shuffling and torch initialization; CUDA kernel ordering is non-deterministic.
- The dataset loader tries `Anthropic/hh-rlhf` first, then two fallbacks. If the primary source changes, results may differ.
- Analysis scripts are deterministic once checkpoints are fixed. Re-running them against the same adapters produces identical JSON output.
- Two analysis scripts (`angular_fourier_delta_prime.py`, `two_point_correlator_delta.py`) depend on pre-recorded activation tensors (`_orbit/*.pt`) not included in this repository. They can be regenerated from the named base-model checkpoints.

The reference value `lora_dpo_v2_final_loss = 0.487` used in `bitfit_dpo_strike.py` and `bitfit_dpo_strike_extended.py` is the training loss from the Pythia-410M DPO r=128 run at step 800 (`_leak/v2/summary_v2.json`, full value: 0.48728475). See PROVENANCE.md for the complete trace.

### Running Experiments

```bash
# Training (requires GPU, ~10–35 min each on RTX 3060 12GB)
make training        # all training jobs (serial, ~5 hours)
make train-410m-dpo  # single job: Pythia-410M DPO at 800 steps

# Analysis (CPU-OK, depends on checkpoints)
make analysis       # all analysis jobs
make gamma-410m     # single job: γ overlap at 410M

# Paper (requires LaTeX)
make paper          # full PDF build
```

### Hardware

- **Training**: RTX 3060 12GB was sufficient for all runs with fp16 + gradient checkpointing. Models ≥ 2.8B require more VRAM.
- **Analysis**: CPU-only; no GPU needed for gamma / autopsy / correlator scripts.
- **Paper build**: ~60 s per full rebuild (pdflatex + bibtex + generators).

## Headline Results

| Model | d_model | Layers | srank (DPO) | srank (CLM) | bonus_R(k=5, DPO) |
|-------|---------|--------|-------------|-------------|-------------------|
| 70M | 512 | 6 | — | — | 5.12× |
| 160M | 768 | 12 | — | — | 5.41× |
| 410M | 1024 | 24 | 3.92 | 3.92 | 5.06× |
| 1B (s=42) | 2048 | 16 | 3.13 | 2.89 | 3.24× |
| 1B (s=117) | 2048 | 16 | 3.17 | 2.83 | 3.58× |

**Random baseline** (k=5): k / d_model. At 1B, 5/2048 ≈ 0.24%.

## Citation

```bibtex
@misc{basu2026lazyrugder,
  title={Axiomatic Bounds on {LoRA} Alignment Geometry: 
         A Task-Intrinsic Dimensional Floor Across {Pythia} 70M--1B},
  author={Basu, Debanjan},
  year={2026},
  howpublished={\url{https://github.com/d3banjan/lazy-rudder-paper}}
}
```

## License

Private mirror for manuscript preparation. License TBD on public release.
