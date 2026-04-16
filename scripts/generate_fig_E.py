r"""generate_fig_E.py — gamma-rudder subspace overlap across all four adapter modules.

Data source:
  - results/spectral_overlap_gamma_modules/results.json
    key: summary_table — list of per-run/module records with srank, bonus_R_k5, bonus_R_ksrank

Metric shown: bonus_R_k5 (subspace overlap at k=5, normalised to random baseline)
  This metric best shows architecture-wide invariance because it is directly
  comparable across modules with different d_in / d_out.

Module labels mapped to short names:
  attention.query_key_value  → QKV
  attention.dense            → Attn-dense
  mlp.dense_h_to_4h          → MLP-up
  mlp.dense_4h_to_h          → MLP-down

Runs plotted: 410m_dpo, 410m_clm, 1b_dpo, 1b_clm (4 bars per module).

Values cross-checked against values.tex:
  \modAvgBonusRKfiveQKV      = 4.419
  \modAvgBonusRKfiveAttnDense= 2.082
  \modAvgBonusRKfiveMLPUp    = 6.146
  \modAvgBonusRKfiveMLPDown  = 5.616
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

MODULE_SHORT = {
    "attention.query_key_value": "QKV",
    "attention.dense":           "Attn-dense",
    "mlp.dense_h_to_4h":        "MLP-up",
    "mlp.dense_4h_to_h":        "MLP-down",
}
RUN_LABELS = {
    "410m_dpo": "410M DPO",
    "410m_clm": "410M CLM",
    "1b_dpo":   "1B DPO",
    "1b_clm":   "1B CLM",
}
RUN_COLORS = {
    "410m_dpo": "#1565c0",
    "410m_clm": "#0097a7",
    "1b_dpo":   "#e65100",
    "1b_clm":   "#c62828",
}

MODULE_ORDER = [
    "attention.query_key_value",
    "attention.dense",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
]
RUN_ORDER = ["410m_dpo", "410m_clm", "1b_dpo", "1b_clm"]


def main():
    path = RESULTS_DIR / "spectral_overlap_gamma_modules" / "results.json"
    with open(path) as f:
        data = json.load(f)

    table = data["summary_table"]

    # Build dict: (run, module) -> bonus_R_k5
    lookup = {}
    for row in table:
        lookup[(row["run"], row["module"])] = row["bonus_R_k5"]

    n_mods = len(MODULE_ORDER)
    n_runs = len(RUN_ORDER)
    bar_w  = 0.18
    x_base = np.arange(n_mods)

    # Verify averages
    for mod in MODULE_ORDER:
        vals = [lookup[(r, mod)] for r in RUN_ORDER]
        print(f"[fig_E] {MODULE_SHORT[mod]} avg bonus_R_k5: {np.mean(vals):.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.2))

    for i, run in enumerate(RUN_ORDER):
        vals = [lookup[(run, mod)] for mod in MODULE_ORDER]
        offsets = x_base + (i - (n_runs - 1) / 2) * bar_w
        ax.bar(offsets, vals, bar_w, color=RUN_COLORS[run], alpha=0.87,
               label=RUN_LABELS[run])

    ax.axhline(1.0, color='#555555', lw=0.8, ls=':', label='Random baseline')

    ax.set_xticks(x_base)
    ax.set_xticklabels([MODULE_SHORT[m] for m in MODULE_ORDER], fontsize=8)
    ax.set_ylabel('$\\mathrm{bonus}_R(k{=}5)$', fontsize=9)
    ax.set_ylim(0, 12)
    ax.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax.tick_params(axis='y', labelsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_E.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig_E] Written: {out}")


if __name__ == "__main__":
    main()
