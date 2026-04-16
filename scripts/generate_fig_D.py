"""generate_fig_D.py — BitFit vs LoRA loss trajectories.

Data sources:
  - results/bitfit_dpo_strike/loss_trajectory.json          (steps 1–800)
  - results/bitfit_dpo_strike_extended/loss_trajectory.json (steps 810–1560)
  LoRA reference: 0.487  (\bitfitLoRAReference, sourced from PROVENANCE.md
    → results/_leak/v2/summary_v2.json, Pythia-410M LoRA r=128, 800 steps, seed 42)
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

# Authoritative LoRA reference from values.tex \bitfitLoRAReference
LORA_REFERENCE = 0.487


def load_trajectory(path):
    with open(path) as f:
        data = json.load(f)
    steps  = [x["step"] for x in data]
    losses = [x["loss"] for x in data]
    return np.array(steps), np.array(losses)


def main():
    steps_bf,  losses_bf  = load_trajectory(
        RESULTS_DIR / "bitfit_dpo_strike" / "loss_trajectory.json")
    steps_ext, losses_ext = load_trajectory(
        RESULTS_DIR / "bitfit_dpo_strike_extended" / "loss_trajectory.json")

    # Smooth losses (window=10) for readability
    def smooth(arr, w=10):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='valid')

    s_steps_bf   = steps_bf[5:-4]    # trim to match smoothed length
    s_losses_bf  = smooth(losses_bf)
    s_steps_ext  = steps_ext[5:-4]
    s_losses_ext = smooth(losses_ext)

    # Training-summary endpoints from values.tex (authoritative, cited in §4 prose).
    # Smoothed-curve endpoints differ by up to ~0.03 due to window placement;
    # we plot both: curve = 10-step smoothed trajectory, dot = summary.json endpoint.
    bitfit_annot = 0.660   # \bitfitFinalLoss
    ext_annot    = 0.607   # \bitfitExtFinalLoss

    print(f"[fig_D] BitFit smoothed endpoint: {s_losses_bf[-1]:.3f}  vs summary {bitfit_annot}")
    print(f"[fig_D] Extended smoothed endpoint: {s_losses_ext[-1]:.3f}  vs summary {ext_annot}")
    print(f"[fig_D] LoRA reference: {LORA_REFERENCE}")

    all_steps = np.concatenate([steps_bf, steps_ext])
    x_max = all_steps[-1] + 20

    fig, ax = plt.subplots(figsize=(4.5, 3))

    ax.plot(s_steps_bf, s_losses_bf, lw=1.2, color='#1565c0',
            label='BitFit (10-step smoothed trajectory)')
    ax.plot(s_steps_ext, s_losses_ext, lw=1.2, color='#7b1fa2',
            label='BitFit extended (10-step smoothed)')
    ax.axhline(LORA_REFERENCE, color='#c62828', lw=1.1, ls='--',
               label=f'LoRA-DPO reference {LORA_REFERENCE} (PROVENANCE.md)')

    # Mark training-summary endpoints as dots (authoritative values cited in §4 prose).
    ax.scatter([steps_bf[-1]], [bitfit_annot], s=25, color='#1565c0', zorder=5,
               edgecolor='white', linewidth=0.8, label='summary.json endpoint')
    ax.scatter([steps_ext[-1]], [ext_annot], s=25, color='#7b1fa2', zorder=5,
               edgecolor='white', linewidth=0.8)

    ax.annotate(f'{bitfit_annot}', xy=(steps_bf[-1], bitfit_annot),
                xytext=(8, 2), textcoords='offset points', fontsize=7, color='#1565c0')
    ax.annotate(f'{ext_annot}', xy=(steps_ext[-1], ext_annot),
                xytext=(8, -8), textcoords='offset points', fontsize=7, color='#7b1fa2')
    ax.annotate(f'{LORA_REFERENCE}', xy=(x_max * 0.6, LORA_REFERENCE),
                xytext=(0, 4), textcoords='offset points', fontsize=7, color='#c62828')

    ax.set_xlabel('Training step', fontsize=9)
    ax.set_ylabel('DPO loss', fontsize=9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0.45, 0.72)
    ax.legend(fontsize=7, framealpha=0.9, loc='upper right')
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_D.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig_D] Written: {out}")


if __name__ == "__main__":
    main()
