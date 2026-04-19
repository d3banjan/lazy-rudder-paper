r"""generate_fig_F.py — srank checkpoint-progression plot (T1.4 lazy-regime rebuttal).

PURPOSE
-------
Reviewer M7 asks whether the low srank at step 800 reflects a lazy / early-training
regime rather than a terminal property.  The rebuttal requires showing srank vs.
training step for each model size on a single plot.

DATA AVAILABILITY — PENDING
----------------------------
The HF checkpoint repository `d3banjan/lazy-rudder-checkpoints` stores only the
final checkpoint (step 800) for every LoRA run.  No intermediate adapter weights
were uploaded during the original training runs.  The `trainer_state.json` files
log loss and reward metrics at every 10 steps but do NOT include srank.  There are
no wandb or tensorboard logs on disk.

CONSEQUENCE
-----------
This script CANNOT produce a real srank trajectory without one of:
  (a) Re-training with `save_strategy="steps"` and `save_steps` ≤ 100, then
      computing srank on each saved adapter; OR
  (b) Adding a `SrankCallback` to the training scripts and re-running.

The script currently emits a placeholder figure that makes the pending-data
status visible in the built PDF, and exits with rc=0 so `make figs` does not
break.  Replace the stub in `_load_trajectory()` once real data exists.

HOW TO FILL THIS IN
-------------------
1. Add `save_steps=100` to each `dpo_leak_train_*.py` DPOConfig.
2. Re-run training for at least 70m and 160m (fastest runs).
3. For each saved adapter at step S, compute:
       srank = (||BA||_F^2) / (sigma_max(BA)^2)
   where BA = lora_B @ lora_A.
4. Collect per-model trajectories into results/srank_trajectory.json with schema:
       {
         "<model_tag>": {
           "steps": [100, 200, ..., 800],
           "srank": [<float>, ...]
         },
         ...
       }
5. Remove the `DATA_AVAILABLE = False` guard below and re-run this script.

USAGE (once data exists)
--------
  uv run python scripts/generate_fig_F.py
  # emits manuscript/figures/fig_F_progression.pdf + fig_F_progression.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── path setup (mirrors other generate_fig_*.py scripts) ─────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PAPER_DIR   = _SCRIPTS_DIR.parent

_PAPER_RESULTS = _PAPER_DIR / "results"
if _PAPER_RESULTS.exists():
    RESULTS_DIR = _PAPER_RESULTS
else:
    sys.path.insert(0, str(_SCRIPTS_DIR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from _paths import RESULTS_DIR  # noqa: F401  (may warn; that's fine)

FIGURES_DIR = _PAPER_DIR / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── data availability guard ───────────────────────────────────────────────────
DATA_AVAILABLE = False  # flip to True once srank_trajectory.json exists

TRAJECTORY_JSON = RESULTS_DIR / "srank_trajectory.json"


def _load_trajectory() -> dict:
    """Return trajectory dict.  Raises FileNotFoundError if data is absent."""
    import json
    with open(TRAJECTORY_JSON) as fh:
        return json.load(fh)


# ── colour palette (consistent with other figures) ───────────────────────────
_PALETTE = {
    "70M":  "#1f4e79",
    "160M": "#2e7d32",
    "410M": "#b71c1c",
    "1B":   "#6a1499",
}

_STEPS_FINAL = 800   # total training steps used in all main runs


def _plot_real(ax: plt.Axes, traj: dict) -> None:
    """Plot real srank-vs-step trajectories."""
    for model_tag, colour in _PALETTE.items():
        if model_tag not in traj:
            continue
        steps = np.array(traj[model_tag]["steps"])
        srank = np.array(traj[model_tag]["srank"])
        ax.plot(steps, srank, lw=1.5, color=colour, marker="o", ms=3,
                label=f"Pythia-{model_tag}")

    ax.axvline(_STEPS_FINAL, lw=0.8, ls=":", color="grey",
               label=f"Final step ({_STEPS_FINAL})")
    ax.set_xscale("log")
    ax.set_xlabel("Training step", fontsize=9)
    ax.set_ylabel("Stable rank ($srank$)", fontsize=9)
    ax.set_title("srank vs. training step (LoRA-DPO)", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=8)


def _plot_stub(ax: plt.Axes) -> None:
    """Emit a clearly-labelled placeholder when trajectory data is absent."""
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Diagonal hatching to signal "no data"
    for x in np.arange(-1, 2, 0.12):
        ax.plot([x, x + 1], [0, 1], lw=0.4, color="#bbbbbb", zorder=0)

    ax.text(0.5, 0.65,
            "PENDING DATA",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#b71c1c", transform=ax.transAxes)
    ax.text(0.5, 0.50,
            "srank trajectory requires intermediate\nLoRA checkpoints or a SrankCallback.",
            ha="center", va="center", fontsize=8, color="#444444",
            transform=ax.transAxes)
    ax.text(0.5, 0.34,
            "Re-run training with save_steps ≤ 100\n"
            "or add SrankCallback (see script docstring).",
            ha="center", va="center", fontsize=7, color="#666666",
            transform=ax.transAxes)

    ax.set_xlabel("Training step (pending)", fontsize=9)
    ax.set_ylabel("Stable rank ($srank$) — pending", fontsize=9)
    ax.set_title("srank vs. training step — data unavailable", fontsize=9)

    patches = [mpatches.Patch(color=c, label=f"Pythia-{m}")
               for m, c in _PALETTE.items()]
    ax.legend(handles=patches, fontsize=7, framealpha=0.9,
              title="(trajectories pending)")


def main() -> int:
    fig, ax = plt.subplots(figsize=(5, 3.5))

    if DATA_AVAILABLE and TRAJECTORY_JSON.exists():
        traj = _load_trajectory()
        _plot_real(ax, traj)
    else:
        _plot_stub(ax)

    fig.tight_layout()

    out_pdf = FIGURES_DIR / "fig_F_progression.pdf"
    out_png = FIGURES_DIR / "fig_F_progression.png"
    fig.savefig(str(out_pdf), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not DATA_AVAILABLE:
        print("[generate_fig_F] WARNING: data unavailable — stub figure written to:")
        print(f"  {out_pdf}")
        print(f"  {out_png}")
        print("  See script docstring for how to fill this in.")
    else:
        print(f"[generate_fig_F] OK → {out_pdf}, {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
