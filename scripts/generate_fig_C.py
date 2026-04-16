r"""generate_fig_C.py — Per-layer srank bars for Pythia-70M and Pythia-160M.

Data source:
  - results/spectral_overlap_gamma_petri/results.json
    keys: pythia-70m.per_layer[i].srank, pythia-160m.per_layer[i].srank

Values cross-checked:
  70m  avg srank: 3.930  (\srankSeventyM)
  160m avg srank: 3.510  (\srankOneSixtyM)
  Network aggregate (4-point mean): 3.629 (\srankFitTaskConst) — shown as dashed reference.
"""

import sys
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    path = RESULTS_DIR / "spectral_overlap_gamma_petri" / "results.json"
    with open(path) as f:
        data = json.load(f)

    m70  = data["pythia-70m"]
    m160 = data["pythia-160m"]

    srank_70m  = [x["srank"] for x in m70["per_layer"]]
    srank_160m = [x["srank"] for x in m160["per_layer"]]

    n70  = len(srank_70m)   # 6
    n160 = len(srank_160m)  # 12
    n_max = max(n70, n160)

    avg_70m  = m70["avg"]["srank"]
    avg_160m = m160["avg"]["srank"]
    # 4-point aggregate constant (values.tex \srankFitTaskConst)
    aggregate_const = 3.629

    print(f"[fig_C] 70m  avg srank: {avg_70m:.3f}  (values.tex: 3.930)")
    print(f"[fig_C] 160m avg srank: {avg_160m:.3f}  (values.tex: 3.510)")

    x_70m  = np.arange(n70)
    x_160m = np.arange(n160)
    bar_w  = 0.38

    fig, ax = plt.subplots(figsize=(5, 3))

    # 160m bars first (more layers, so use as backdrop x-axis reference)
    ax.bar(x_160m - bar_w / 2, srank_160m, bar_w, color='#e65100', alpha=0.85, label='Pythia-160M')
    ax.bar(x_70m  + bar_w / 2, srank_70m,  bar_w, color='#1565c0', alpha=0.85, label='Pythia-70M')

    ax.axhline(aggregate_const, color='#2e7d32', lw=1.0, ls='--',
               label=f'4-model aggregate $\\approx{aggregate_const}$')

    ax.set_xlabel('Layer index', fontsize=9)
    ax.set_ylabel('Per-layer $srank$', fontsize=9)
    ax.set_xlim(-0.7, n_max - 0.3)
    ax.set_xticks(np.arange(n_max))
    ax.set_xticklabels([str(i) for i in range(n_max)], fontsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(0, 8)
    ax.legend(fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_C.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig_C] Written: {out}")


if __name__ == "__main__":
    main()
