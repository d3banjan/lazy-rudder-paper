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

Training runs expect:
- Pythia base models under `cross-check/trained-model-battery/models/`
- `uv` or an equivalent Python env with `torch`, `transformers`, `peft`, `trl`, `datasets`, `safetensors`
- RTX 3060 12GB or equivalent for up to 1B DPO (ref+policy LoRA r=128, fp16, gradient_checkpointing, bs=1 grad_accum=8, seq_len=512)

Scripts assume the working directory is the parent project tree (paths reference `../cross-check/...`). Kept as-is for parent-repo consistency; see the parent's `lean-mining` repository for the full environment.

## License

Private mirror for manuscript preparation. License TBD on public release.
