r"""generate_fig_A.py — Srank floor scatter: 4 models, constant vs power-law fits.

Data sources:
  - results/spectral_overlap_gamma_petri/results.json  (70m, 160m avg srank)
  - results/spectral_overlap_gamma/results.json         (410m, runs v2_dpo_r128 / v3_clm_r128)
  - results/spectral_overlap_gamma_1b/results.json      (1b, run v2_dpo_r128_1b)
  - Delta-AIC and fit constants from manuscript/values.tex macros (verified below)

srank values (authoritative, cross-checked against values.tex):
  70m:  3.930  (\srankSeventyM)
  160m: 3.510  (\srankOneSixtyM)
  410m: 3.924  (\srankFourTenM)
  1b:   3.134  (\srankOneBDPOsFortyTwo)

Fit constants (from values.tex):
  constant fit:   c0  = 3.629  (\srankFitTaskConst)
  cbrt fit:       c1  = 34.682 (\srankFitCbrtC)
  Delta-AIC sqrt: 5.547 (\srankFitDeltaAICSqrt)
  Delta-AIC cbrt: 1.747 (\srankFitDeltaAICCbrt)
"""

import sys
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
# RESULTS_DIR: prefer paper/results/ (authoritative data store).
# Falls back to _paths.RESULTS_DIR for environments using the original dev layout.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PAPER_DIR   = _SCRIPTS_DIR.parent

_PAPER_RESULTS = _PAPER_DIR / "results"
if _PAPER_RESULTS.exists():
    RESULTS_DIR = _PAPER_RESULTS
else:
    sys.path.insert(0, str(_SCRIPTS_DIR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from _paths import RESULTS_DIR

FIGURES_DIR = _PAPER_DIR / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_srank_values():
    """Load and verify srank values from source JSONs."""
    # 70m + 160m from petri dish
    petri_path = RESULTS_DIR / "spectral_overlap_gamma_petri" / "results.json"
    with open(petri_path) as f:
        petri = json.load(f)
    srank_70m  = petri["pythia-70m"]["avg"]["srank"]
    srank_160m = petri["pythia-160m"]["avg"]["srank"]

    # 410m from spectral_overlap_gamma, run v2_dpo_r128
    gamma_path = RESULTS_DIR / "spectral_overlap_gamma" / "results.json"
    with open(gamma_path) as f:
        gamma = json.load(f)
    # v1_dpo_r16 avg srank per_layer
    per_layer_v2 = gamma["runs"]["v2_dpo_r128"]["per_layer"]
    srank_410m = float(np.mean([x["srank_delta"] for x in per_layer_v2]))

    # 1b from spectral_overlap_gamma_1b, run v2_dpo_r128_1b
    gamma_1b_path = RESULTS_DIR / "spectral_overlap_gamma_1b" / "results.json"
    with open(gamma_1b_path) as f:
        gamma_1b = json.load(f)
    per_layer_1b = gamma_1b["runs"]["v2_dpo_r128_1b"]["per_layer"]
    srank_1b = float(np.mean([x["srank_delta"] for x in per_layer_1b]))

    return {
        "d_models": [512, 768, 1024, 2048],
        "labels":   ["70M", "160M", "410M", "1B"],
        "sranks":   [srank_70m, srank_160m, srank_410m, srank_1b],
    }


def main():
    data = load_srank_values()
    d_models = np.array(data["d_models"], dtype=float)
    sranks   = np.array(data["sranks"])
    labels   = data["labels"]

    # Fit constants (cross-checked against values.tex)
    c0_const  = 3.629    # \srankFitTaskConst
    c1_cbrt   = 34.682   # \srankFitCbrtC
    delta_aic_sqrt = 5.547   # \srankFitDeltaAICSqrt
    delta_aic_cbrt = 1.747   # \srankFitDeltaAICCbrt

    d_fine = np.linspace(300, 2500, 500)
    y_const = np.full_like(d_fine, c0_const)
    y_cbrt  = c1_cbrt / d_fine ** (1.0 / 3.0)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(d_models, sranks, s=55, color='#1f4e79', zorder=5, label='Measured $srank$ (DPO)')

    ax.plot(d_fine, y_const, lw=1.2, color='#2e7d32', ls='-',
            label=f'Constant fit $c_0={c0_const}$\n($\\Delta$AIC$_{{\\mathrm{{sqrt}}}}$={delta_aic_sqrt:.1f})')
    ax.plot(d_fine, y_cbrt, lw=1.2, color='#b71c1c', ls='--',
            label=f'$c/d^{{1/3}}$ fit $c={c1_cbrt:.0f}$\n($\\Delta$AIC={delta_aic_cbrt:.1f})')

    for d, s, lab in zip(d_models, sranks, labels):
        ax.annotate(lab, xy=(d, s), xytext=(5, 5), textcoords='offset points', fontsize=7)

    ax.set_xlabel('$d_{\\mathrm{model}}$', fontsize=9)
    ax.set_ylabel('Stable rank ($srank$)', fontsize=9)
    ax.set_xlim(250, 2300)
    ax.set_ylim(2.5, 5.0)
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_A.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig_A] Written: {out}")
    print(f"[fig_A] srank values: 70m={sranks[0]:.3f}, 160m={sranks[1]:.3f}, "
          f"410m={sranks[2]:.3f}, 1b={sranks[3]:.3f}")


if __name__ == "__main__":
    main()
