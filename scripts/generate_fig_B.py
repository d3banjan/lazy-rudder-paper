"""generate_fig_B.py — Top-k subspace overlap bonus_R: DPO vs CLM at 1B, per layer.

Data source:
  - results/spectral_overlap_gamma_1b/results.json
    runs: v2_dpo_r128_1b, v3_clm_r128_1b
    key: per_layer[i]['k5']['bonus_right']  (bonus_R at k=5 per layer)

Values cross-checked:
  DPO avg bonus_R(k=5): 3.239  (\bonusRKfiveOneBDPOsFortyTwo)
  CLM avg bonus_R(k=5): 4.151  (\bonusRKfiveOneBCLMsFortyTwo)
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
    path = RESULTS_DIR / "spectral_overlap_gamma_1b" / "results.json"
    with open(path) as f:
        data = json.load(f)

    dpo_layers = data["runs"]["v2_dpo_r128_1b"]["per_layer"]
    clm_layers = data["runs"]["v3_clm_r128_1b"]["per_layer"]

    n_layers = len(dpo_layers)
    layer_idx = list(range(n_layers))

    dpo_k5  = [x["k5"]["bonus_right"]  for x in dpo_layers]
    clm_k5  = [x["k5"]["bonus_right"]  for x in clm_layers]
    dpo_k10 = [x["k10"]["bonus_right"] for x in dpo_layers]
    clm_k10 = [x["k10"]["bonus_right"] for x in clm_layers]

    # averages for annotation
    dpo_avg_k5 = float(np.mean(dpo_k5))
    clm_avg_k5 = float(np.mean(clm_k5))

    print(f"[fig_B] DPO avg bonus_R(k=5): {dpo_avg_k5:.3f}  (values.tex: 3.239)")
    print(f"[fig_B] CLM avg bonus_R(k=5): {clm_avg_k5:.3f}  (values.tex: 4.151)")

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(layer_idx, dpo_k5, marker='o', ms=4, lw=1.2, color='#1565c0',
            label=f'DPO (avg {dpo_avg_k5:.2f}×)')
    ax.plot(layer_idx, clm_k5, marker='s', ms=4, lw=1.2, color='#e65100',
            label=f'CLM (avg {clm_avg_k5:.2f}×)')
    ax.axhline(1.0, color='#555555', lw=0.8, ls=':', label='Random baseline')

    ax.set_xlabel('Layer index', fontsize=9)
    ax.set_ylabel('$\\mathrm{bonus}_R(k{=}5)$', fontsize=9)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xticks(layer_idx)
    ax.set_xticklabels([str(i) for i in layer_idx], fontsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_B.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig_B] Written: {out}")


if __name__ == "__main__":
    main()
