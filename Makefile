# Top-level Makefile for lazy-rudder-paper.
#
# Hierarchical targets for: building the paper PDF, re-running experiments
# (training + analysis), and cleanup. Analysis targets are CPU-feasible
# and depend on training checkpoints; training targets require a GPU.
#
# Checkpoint files are gitignored (see .gitignore) — they can be
# regenerated via `make training`, or fetched from a future HuggingFace /
# GitHub LFS mirror for hard reproducibility.

.PHONY: help all paper figs experiments training analysis clean distclean \
        clean-checkpoints clean-results \
        fetch-checkpoints \
        train-70m train-160m train-410m-dpo train-410m-clm \
        train-1b-dpo train-1b-clm train-1b-dpo-s117 train-1b-clm-s117 \
        train-bitfit train-bitfit-ext \
        compute-channel-partition \
        gamma-410m gamma-1b gamma-1b-s117 gamma-petri \
        autopsy-bias autopsy-spectral autopsy-sectional \
        seed-variance delta delta-prime

PYTHON  ?= python3
SCRIPTS := scripts

# ─────────────────────────────────────────────────────────────────────────
# Default
# ─────────────────────────────────────────────────────────────────────────
all: paper

help:
	@echo "Targets (hierarchical):"
	@echo ""
	@echo "  PAPER"
	@echo "    make paper                — build manuscript/main.pdf from generated data + Lean"
	@echo ""
	@echo "  REPRODUCIBILITY (analysis-only path)"
	@echo "    make fetch-checkpoints    — pull adapter weights from HF (~2.5 GB, no GPU)"
	@echo "    make analysis             — all analysis jobs (CPU-OK, needs checkpoints)"
	@echo ""
	@echo "  EXPERIMENTS (two tiers)"
	@echo "    make experiments          — training + analysis (full re-run)"
	@echo "    make training             — all training jobs (GPU, ~5 hours serial)"
	@echo ""
	@echo "  TRAINING (individual, GPU required)"
	@echo "    make train-70m            — Pythia-70M  DPO r=128 800 steps   (~10 min)"
	@echo "    make train-160m           — Pythia-160M DPO r=128 800 steps   (~15 min)"
	@echo "    make train-410m-dpo       — Pythia-410M DPO r=128 800 steps   (~30 min)"
	@echo "    make train-410m-clm       — Pythia-410M CLM r=128 800 steps   (~25 min, no ref model)"
	@echo "    make train-1b-dpo         — Pythia-1B   DPO r=128 800 steps   (~35 min)"
	@echo "    make train-1b-clm         — Pythia-1B   CLM r=128 800 steps   (~30 min)"
	@echo "    make train-1b-dpo-s117    — seed=117 DPO replicate at 1B"
	@echo "    make train-1b-clm-s117    — seed=117 CLM replicate at 1B"
	@echo "    make train-bitfit         — bias-only DPO at 410M, 800 steps  (~35 min)"
	@echo "    make train-bitfit-ext     — extend BitFit from checkpoint-800 to 1600 (~35 min)"
	@echo ""
	@echo "  ANALYSIS (individual, CPU-OK)"
	@echo "    make gamma-410m           — γ subspace overlap at Pythia-410M"
	@echo "    make gamma-1b             — γ at Pythia-1B (seed 42)"
	@echo "    make gamma-1b-s117        — γ at Pythia-1B seed 117 + 4-way comparison"
	@echo "    make gamma-petri          — γ at Pythia-{70M, 160M} (petri-dish sweep)"
	@echo "    make autopsy-bias         — diagonal-gain decomposition test"
	@echo "    make autopsy-spectral     — stable-rank + sectional SVD audit"
	@echo "    make autopsy-sectional    — 3-tier orbit-aligned sectional SVD"
	@echo "    make seed-variance        — quick γ cross-seed check at 1B DPO"
	@echo "    make delta                — 2-point correlator C(L,L+k) across depth"
	@echo "    make delta-prime          — angular-Fourier spectroscopy on γ's basis"
	@echo ""
	@echo "  CLEANUP"
	@echo "    make clean                — remove LaTeX build artefacts"
	@echo "    make clean-checkpoints    — remove all adapter checkpoints (~1 GB)"
	@echo "    make clean-results        — remove all results/*.json (DANGEROUS)"
	@echo "    make distclean            — clean + clean-checkpoints"

# ─────────────────────────────────────────────────────────────────────────
# Paper build (delegates to manuscript/Makefile)
# ─────────────────────────────────────────────────────────────────────────
paper:
	$(MAKE) -C manuscript paper

# Pass-through for manuscript-level targets so `make values` from the
# top-level works equivalently to `cd manuscript && make values`.
values tables lean-status figs:
	$(MAKE) -C manuscript $@

# ─────────────────────────────────────────────────────────────────────────
# Aggregate experiment groups
# ─────────────────────────────────────────────────────────────────────────
training: train-70m train-160m train-410m-dpo train-410m-clm \
          train-1b-dpo train-1b-dpo-s117 train-1b-clm train-1b-clm-s117 \
          train-bitfit train-bitfit-ext

analysis: fetch-checkpoints \
          gamma-410m gamma-1b gamma-1b-s117 gamma-petri \
          autopsy-spectral autopsy-sectional autopsy-bias \
          seed-variance delta delta-prime

experiments: training analysis

# ─────────────────────────────────────────────────────────────────────────
# Fetch checkpoints from HuggingFace
#
# Pulls all paper-cited adapter weights (≈ 2.5 GB) from
# https://huggingface.co/d3banjan/lazy-rudder-checkpoints
# into RESULTS_DIR (resolved by scripts/_paths.py — env var
# LAZY_RUDDER_RESULTS_DIR or paper/config.toml). Idempotent.
# ─────────────────────────────────────────────────────────────────────────

fetch-checkpoints:
	$(PYTHON) $(SCRIPTS)/fetch_checkpoints.py

# ─────────────────────────────────────────────────────────────────────────
# Training targets
# Each writes adapter_model.safetensors to its checkpoint dir. We use the
# existence of that file as the completion sentinel so `make` skips rerun
# when the artefact already exists.
# ─────────────────────────────────────────────────────────────────────────

CKPT_70M         := results/_leak_70m/v2/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_160M        := results/_leak_160m/v2/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_410M_DPO    := results/_leak/v2/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_410M_CLM    := results/_leak/v3/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_1B_DPO      := results/_leak_1b/v2/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_1B_CLM      := results/_leak_1b/v3/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_1B_DPO_S117 := results/_leak_1b_seed117/v3/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_1B_CLM_S117 := results/_leak_1b_seed117/v4/checkpoints/checkpoint-800/adapter_model.safetensors
CKPT_BITFIT      := results/bitfit_dpo_strike/checkpoints/checkpoint-800/model.safetensors
CKPT_BITFIT_EXT  := results/bitfit_dpo_strike_extended/checkpoints/checkpoint-1600/model.safetensors

train-70m: $(CKPT_70M)
$(CKPT_70M):
	$(PYTHON) $(SCRIPTS)/dpo_leak_train_70m.py

train-160m: $(CKPT_160M)
$(CKPT_160M):
	$(PYTHON) $(SCRIPTS)/dpo_leak_train_160m.py

train-410m-dpo: $(CKPT_410M_DPO)
$(CKPT_410M_DPO):
	$(PYTHON) $(SCRIPTS)/dpo_leak_train_v2.py

train-410m-clm: $(CKPT_410M_CLM)
$(CKPT_410M_CLM):
	$(PYTHON) $(SCRIPTS)/clm_leak_train.py

train-1b-dpo: $(CKPT_1B_DPO)
$(CKPT_1B_DPO):
	$(PYTHON) $(SCRIPTS)/dpo_leak_train_1b.py

train-1b-clm: $(CKPT_1B_CLM)
$(CKPT_1B_CLM):
	$(PYTHON) $(SCRIPTS)/clm_leak_train_1b.py

train-1b-dpo-s117: $(CKPT_1B_DPO_S117)
$(CKPT_1B_DPO_S117):
	$(PYTHON) $(SCRIPTS)/dpo_leak_train_1b_seed117.py

train-1b-clm-s117: $(CKPT_1B_CLM_S117)
$(CKPT_1B_CLM_S117):
	$(PYTHON) $(SCRIPTS)/clm_leak_train_1b_seed117.py

train-bitfit: $(CKPT_BITFIT)
$(CKPT_BITFIT):
	$(PYTHON) $(SCRIPTS)/bitfit_dpo_strike.py

# BitFit extension depends on BitFit checkpoint-800
train-bitfit-ext: $(CKPT_BITFIT_EXT)
$(CKPT_BITFIT_EXT): $(CKPT_BITFIT)
	$(PYTHON) $(SCRIPTS)/bitfit_dpo_strike_extended.py

# ─────────────────────────────────────────────────────────────────────────
# Channel partition (auxiliary data needed by sectional 3-tier autopsy)
# ─────────────────────────────────────────────────────────────────────────
CHANNEL_PARTITION := results/_leak/v2/channel_partition.json

compute-channel-partition: $(CHANNEL_PARTITION)
$(CHANNEL_PARTITION):
	$(PYTHON) $(SCRIPTS)/compute_channel_partition.py

# ─────────────────────────────────────────────────────────────────────────
# Analysis targets
# Each writes a JSON file(s) to results/<experiment>/.
# ─────────────────────────────────────────────────────────────────────────

gamma-410m: results/spectral_overlap_gamma/results.json
results/spectral_overlap_gamma/results.json: $(CKPT_410M_DPO) $(CKPT_410M_CLM) $(SCRIPTS)/spectral_overlap_gamma.py
	$(PYTHON) $(SCRIPTS)/spectral_overlap_gamma.py

gamma-1b: results/spectral_overlap_gamma_1b/results.json
results/spectral_overlap_gamma_1b/results.json: $(CKPT_1B_DPO) $(CKPT_1B_CLM) $(SCRIPTS)/spectral_overlap_gamma_1b.py
	$(PYTHON) $(SCRIPTS)/spectral_overlap_gamma_1b.py

gamma-1b-s117: results/spectral_overlap_gamma_1b_seed117/results.json
results/spectral_overlap_gamma_1b_seed117/results.json: $(CKPT_1B_DPO) $(CKPT_1B_CLM) $(CKPT_1B_DPO_S117) $(CKPT_1B_CLM_S117) $(SCRIPTS)/spectral_overlap_gamma_1b_seed117.py
	$(PYTHON) $(SCRIPTS)/spectral_overlap_gamma_1b_seed117.py

gamma-petri: results/spectral_overlap_gamma_petri/results.json
results/spectral_overlap_gamma_petri/results.json: $(CKPT_70M) $(CKPT_160M) $(SCRIPTS)/spectral_overlap_gamma_petri.py
	$(PYTHON) $(SCRIPTS)/spectral_overlap_gamma_petri.py

autopsy-spectral: results/spectral_autopsy/results.json
results/spectral_autopsy/results.json: $(CKPT_410M_DPO) $(CKPT_410M_CLM) $(SCRIPTS)/spectral_autopsy.py
	$(PYTHON) $(SCRIPTS)/spectral_autopsy.py

autopsy-sectional: results/spectral_autopsy_sectional_3tier/results.json
results/spectral_autopsy_sectional_3tier/results.json: $(CKPT_410M_DPO) $(CKPT_410M_CLM) $(CHANNEL_PARTITION) $(SCRIPTS)/spectral_autopsy_sectional_3tier.py
	$(PYTHON) $(SCRIPTS)/spectral_autopsy_sectional_3tier.py

autopsy-bias: results/bias_theory_autopsy/results.json
results/bias_theory_autopsy/results.json: $(CKPT_410M_DPO) $(CKPT_410M_CLM) $(SCRIPTS)/bias_theory_autopsy.py
	$(PYTHON) $(SCRIPTS)/bias_theory_autopsy.py

seed-variance: results/seed_variance_quick/results.json
results/seed_variance_quick/results.json: $(CKPT_1B_DPO) $(CKPT_1B_DPO_S117) $(SCRIPTS)/seed_variance_quick.py
	$(PYTHON) $(SCRIPTS)/seed_variance_quick.py

delta: results/two_point_correlator_delta/results.json
results/two_point_correlator_delta/results.json: $(SCRIPTS)/two_point_correlator_delta.py
	$(PYTHON) $(SCRIPTS)/two_point_correlator_delta.py

delta-prime: results/angular_fourier_delta_prime/results.json
results/angular_fourier_delta_prime/results.json: $(SCRIPTS)/angular_fourier_delta_prime.py
	$(PYTHON) $(SCRIPTS)/angular_fourier_delta_prime.py

# ─────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────

clean:
	$(MAKE) -C manuscript clean

clean-checkpoints:
	find results -type d -name checkpoints -exec rm -rf {} + 2>/dev/null || true
	@echo "Checkpoint directories removed. Re-run 'make training' or fetch from HF/LFS mirror."

clean-results:
	@echo "About to remove ALL results/ JSONs + auxiliary data."
	@echo "Press Ctrl-C within 5 seconds to abort..."
	@sleep 5
	find results -name "*.json" -delete
	find results -name "*.pt" -delete

distclean: clean clean-checkpoints
	$(MAKE) -C manuscript distclean
