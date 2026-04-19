#!/usr/bin/env python3
"""generate_fig_G.py — Generate Fig G: Behavior-Geometry Correlation scatter.

Reads:
  paper/results/behavior_geometry/summary.json
  paper/results/behavior_geometry/correlation.json

Writes:
  paper/manuscript/figures/fig_G_behavior_geometry.{pdf,png}

Sidecar config:
  paper/scripts/generate_fig_G.config.json

Usage:
    uv run python generate_fig_G.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE      = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
RES_DIR   = PAPER_DIR / "results" / "behavior_geometry"
FIGS_DIR  = PAPER_DIR / "manuscript" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary_path = RES_DIR / "summary.json"
    corr_path    = RES_DIR / "correlation.json"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run behavior_geometry_link.py first.")
        return 1
    if not corr_path.exists():
        print(f"ERROR: {corr_path} not found. Run behavior_geometry_link.py first.")
        return 1

    summaries    = json.loads(summary_path.read_text())
    correlations = json.loads(corr_path.read_text())

    if not summaries:
        print("ERROR: summary.json is empty.")
        return 1

    labels = [f"{s['model_size']}\ns{s['seed']}" for s in summaries]
    colors = {"70m": "#1f77b4", "160m": "#ff7f0e", "410m": "#2ca02c", "1b": "#d62728"}
    point_colors = [colors.get(s["model_size"], "gray") for s in summaries]

    sranks  = [s["srank"] for s in summaries]
    gammas  = [s["gamma_overlap"] for s in summaries]
    rewards = [s["reward_margin_mean"] for s in summaries]
    kls     = [s["kl_to_base_mean"] for s in summaries]
    reward_se = [s["reward_margin_se"] for s in summaries]
    kl_se     = [s["kl_to_base_se"] for s in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(
        "Fig G: Behavior-Geometry Correlation\n"
        "(Pythia 70M–1B DPO LoRA adapters, Anthropic/hh-rlhf test split, n=500)",
        fontsize=11,
    )

    panel_data = [
        (axes[0, 0], sranks, rewards, reward_se,
         "stable rank (srank)", "reward margin (β·Δlog π)",
         "srank_vs_reward_margin"),
        (axes[0, 1], gammas, rewards, reward_se,
         "γ-overlap (bonus_R k=5)", "reward margin (β·Δlog π)",
         "gamma_vs_reward_margin"),
        (axes[1, 0], sranks, kls, kl_se,
         "stable rank (srank)", "KL-to-base (per token)",
         "srank_vs_kl_to_base"),
        (axes[1, 1], gammas, kls, kl_se,
         "γ-overlap (bonus_R k=5)", "KL-to-base (per token)",
         "gamma_vs_kl_to_base"),
    ]

    corr_key_map = {
        "srank_vs_reward_margin": "srank_vs_reward_margin",
        "gamma_vs_reward_margin": "gamma_vs_reward_margin",
        "srank_vs_kl_to_base":    "srank_vs_kl_to_base",
        "gamma_vs_kl_to_base":    "gamma_vs_kl_to_base",
    }

    for ax, x_vals, y_vals, y_errs, xlabel, ylabel, corr_key in panel_data:
        for xi, yi, ye, label, color in zip(x_vals, y_vals, y_errs, labels, point_colors):
            ax.errorbar(xi, yi, yerr=ye, fmt="o", color=color, capsize=3, markersize=6)
            ax.annotate(label, (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=color)

        if len(x_vals) >= 3:
            try:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(min(x_vals), max(x_vals), 50)
                ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.4, linewidth=1)
            except Exception:
                pass

        ck = corr_key_map.get(corr_key, corr_key)
        if ck in correlations:
            pr    = correlations[ck]["pearson_r"]
            pr_lo = correlations[ck]["pearson_ci95_lo"]
            pr_hi = correlations[ck]["pearson_ci95_hi"]
            r_str = f"r={pr:.2f} [{pr_lo:.2f},{pr_hi:.2f}]" if pr == pr else "r=n/a"
            ax.text(0.05, 0.95, r_str, transform=ax.transAxes,
                    fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for model_size, color in colors.items():
        axes[1, 1].plot([], [], "o", color=color, label=f"pythia-{model_size}", markersize=5)
    axes[1, 1].legend(loc="lower right", fontsize=7, framealpha=0.8)

    # Caption note
    fig.text(
        0.5, 0.01,
        "Error bars = ±1 SE. Dashed line = OLS fit. r = Pearson with 95% bootstrap CI (n≤5 points; estimates are suggestive).",
        ha="center", fontsize=7.5, style="italic", color="#555",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    pdf_path = FIGS_DIR / "fig_G_behavior_geometry.pdf"
    png_path = FIGS_DIR / "fig_G_behavior_geometry.png"
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=150)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Copy to results dir
    shutil.copy(png_path, RES_DIR / "fig_G_behavior_geometry.png")
    print(f"Copied: {RES_DIR / 'fig_G_behavior_geometry.png'}")

    # Write config sidecar
    config = {
        "source_script": str(Path(__file__).name),
        "inputs": [str(summary_path), str(corr_path)],
        "outputs": [str(pdf_path), str(png_path)],
        "description": (
            "2x2 scatter grid: (srank, gamma_overlap) x (reward_margin, KL-to-base). "
            "Addresses reviewer M9: does geometry predict behavior?"
        ),
    }
    cfg_path = HERE / "generate_fig_G.config.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"Config: {cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
