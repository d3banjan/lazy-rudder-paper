# lazy-rudder-paper

Source artifacts for the manuscript on LoRA alignment geometry at Pythia 70M–1B.

## Headline result

LoRA DPO adapters exhibit a **task-intrinsic dimensional constant** of
`srank ≈ 3.6 ± 0.5` across three orders of magnitude of model width
(Pythia 70M → 1B, `d_model` 512 → 2048). The alignment manifold's shape
is set by preference-learning complexity, not by model capacity.

## Layout

```
paper/
├── lean/                   Lean 4 theorems (Mathlib-based)
│   ├── SubspaceOverlap.lean             -- 12 real proofs, 0 sorry in scaling branch
│   └── NeuralGeometry_aggregator.lean    -- module aggregator
├── scripts/                26 Python experiment drivers
│   ├── spectral_overlap_gamma*.py       -- γ: ΔW right-subspace vs base W top-k
│   ├── spectral_autopsy*.py             -- ΔW stable-rank + sectional SVD
│   ├── bias_theory_autopsy.py           -- diagonal-gain decomposition test
│   ├── bitfit_dpo_strike*.py            -- gauge-theory falsification at 410m
│   ├── dpo_leak_train_*.py              -- DPO LoRA training per scale
│   ├── clm_leak_train*.py               -- CLM control training per scale
│   └── (seed_variance, two_point_correlator, angular_fourier, compute_channel_partition)
└── results/                results JSONs only (no checkpoints)
```

## Key theorems (Lean 4)

All in `lean/SubspaceOverlap.lean`. Mathlib pin `v4.28.0`.

| Theorem | Proven | Meaning |
|---|---|---|
| `frobeniusSq_nonneg` | ✓ | Frobenius² ≥ 0 |
| `frobeniusSq_eq_zero_iff` | ✓ | Frobenius² = 0 iff M = 0 |
| `rank_outer_product_le_one` | ✓ | rank(u ⊗ vᵀ) ≤ 1 |
| `rank_add_le` | ✓ | rank(A + B) ≤ rank A + rank B |
| `rank_sum_outer_products_le` | ✓ | rank(Σ uᵢ ⊗ vᵢᵀ) ≤ N |
| `frobeniusSq_smul` | ✓ | ‖cM‖²_F = c²‖M‖²_F |
| `spectralSq_smul` | ✓ | ‖cM‖²_2 = c²‖M‖²_2 |
| `ratio_smul_invariant_of_quadratic` | ✓ | generic ratio-under-scaling algebra |
| `stableRank_smul_invariant` | ✓ | stableRank(cM) = stableRank(M) |
| `loraUpdate_frob_decays` | ✓ | naive LoRA Frobenius² ≤ α²c/r (decays) |
| `rsLoraUpdate_frob_bounded` | ✓ | RsLoRA Frobenius² ≤ α²c (invariant in r) |

## Empirical headline table (see `results/`)

| Model | d | n_layers | srank | bonus_R(k=5) | bonus_R(k=srank) |
|---|---|---|---|---|---|
| 70M | 512 | 6 | 3.93 | 5.12× | 5.50× |
| 160M | 768 | 12 | 3.51 | 5.41× | 9.20× |
| 410M DPO | 1024 | 24 | 3.92 | 5.06× | 7.07× |
| 410M CLM | 1024 | 24 | 3.92 | 5.22× | 7.07× |
| 1B DPO s42 | 2048 | 16 | 3.13 | 3.24× | 5.38× |
| 1B DPO s117 | 2048 | 16 | 3.17 | 3.58× | 5.26× |
| 1B CLM s42 | 2048 | 16 | 2.89 | 4.15× | 8.23× |
| 1B CLM s117 | 2048 | 16 | 2.83 | 4.04× | 7.59× |

Random baseline at k=5 for bonus_R: 5/d_model (e.g. 1/205 at 1B).

## Reproducing

This repository is a **self-contained snapshot** of theorems, experiment
drivers, and analysis results. Training checkpoints (adapter weights)
and base model weights are NOT included — they are reproducible by
running the training scripts after obtaining the Pythia base models
from HuggingFace (`EleutherAI/pythia-{70m,160m,410m,1b}`).

### Environment

```bash
# Python ≥ 3.12
uv sync   # reads pyproject.toml from parent project (see below) OR
pip install torch transformers peft trl datasets safetensors scipy
```

The scripts were run with:
- `torch` 2.x, `transformers` ≥ 4.44, `peft` 0.19, `trl` (DPO trainer)
- `safetensors` for adapter/model I/O
- `scipy` (for `curve_fit` in `two_point_correlator_delta.py` only)

### Absolute paths

The scripts under `scripts/` were authored in the parent project layout
where they reference `../cross-check/trained-model-battery/models/...`
for base weights. To reproduce from this snapshot alone:
1. Download Pythia base models to a directory of your choice.
2. Edit `BASE` / `MODEL_DIR` / `ROOT` constants near the top of each
   script to point at your local paths (grep for `models/pythia-` to
   find the relevant lines).
3. Edit `RESULTS` / `OUT_DIR` to write into `results/` under this repo.

A future refactor will replace hard-coded paths with environment
variables (`LAZY_RUDDER_MODELS_DIR`, `LAZY_RUDDER_RESULTS_DIR`).

### Training reproduction (GPU)

```bash
# Each command takes 10–35 min on an RTX 3060 12GB.
python scripts/dpo_leak_train_70m.py   # → adapter at _leak_70m/v2/checkpoints/checkpoint-800
python scripts/dpo_leak_train_160m.py  # → _leak_160m/v2/...
python scripts/dpo_leak_train_v2.py    # → _leak/v2/... (410m DPO)
python scripts/clm_leak_train.py       # → _leak/v3/... (410m CLM control)
python scripts/dpo_leak_train_1b.py    # → _leak_1b/v2/... (1B DPO seed 42)
python scripts/dpo_leak_train_1b_seed117.py
python scripts/clm_leak_train_1b.py
python scripts/clm_leak_train_1b_seed117.py
python scripts/bitfit_dpo_strike.py           # bias-only ablation, 800 steps
python scripts/bitfit_dpo_strike_extended.py  # extension to 1600 steps
```

### Analysis reproduction (CPU-OK)

After training, run the γ / autopsy / correlator scripts in any order.
These load adapter checkpoints and write JSON files under `results/`:

```bash
python scripts/spectral_overlap_gamma.py         # γ at 410M (both objectives)
python scripts/spectral_overlap_gamma_1b.py      # γ at 1B
python scripts/spectral_overlap_gamma_1b_seed117.py
python scripts/spectral_overlap_gamma_petri.py   # γ at 70M + 160M
python scripts/spectral_autopsy.py               # rank / srank audit
python scripts/spectral_autopsy_sectional_3tier.py  # needs _leak/v2/channel_partition.json (included)
python scripts/bias_theory_autopsy.py            # LN-γ hidden-gain test
python scripts/seed_variance_quick.py            # γ across seeds at 1B
python scripts/two_point_correlator_delta.py     # δ FFT falsification
python scripts/angular_fourier_delta_prime.py    # δ′ angular-Fourier probe
```

### Building the paper

From `manuscript/`:
```bash
make paper       # runs all generators + 3× pdflatex + bibtex
make values      # just regenerate values.tex from results/*.json
make tables      # just regenerate tables.tex
make lean-status # just regenerate lean_status.tex from lean/*.lean
```

Or from this repo root (once `Makefile` is set up), `make paper`
delegates to `manuscript/`.

### Hardware

- Training: RTX 3060 12GB was sufficient for all runs with fp16 +
  gradient checkpointing. 2.8B+ models require more VRAM (see §6
  Limitations in the manuscript).
- Analysis: CPU is sufficient; no GPU needed for γ / autopsy / correlator.

## License

Private mirror for manuscript preparation. License TBD on public release.
