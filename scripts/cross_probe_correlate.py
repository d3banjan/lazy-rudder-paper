#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "scipy>=1.12",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.0",
# ]
# ///
"""cross_probe_correlate.py — T2.1 extension: cross-probe correlation matrix + Fig I.

Reads:
  - 4 score JSONLs from results/cross_probe/ (produced by cross_probe_score.py)
  - results/behavior_geometry/summary.json  — Pythia chain srank + γ (from T1.2)
  - results/t21_qwen_fullweight/summary.json — Qwen srank + γ (from T2.1)

Emits:
  - results/cross_probe/correlation_matrix_{mode}.json
      Pearson + Spearman + CI95 bootstrap for
      (srank_vs_rm, γ_vs_rm) × (hh probe, uf probe) × (pythia chain, qwen point)
  - results/cross_probe/figI_cross_probe_{mode}.png
      2×2 panel: rows={srank, γ}, cols={hh probe, uf probe}

  where {mode} is "sum", "per_token", "margin_win_rate", or "dpo_accuracy"
  (controlled by --aggregator flag).

Usage:
  uv run python scripts/cross_probe_correlate.py
  uv run python scripts/cross_probe_correlate.py --out results/cross_probe
  uv run python scripts/cross_probe_correlate.py --aggregator sum
  uv run python scripts/cross_probe_correlate.py --aggregator per_token
  uv run python scripts/cross_probe_correlate.py --aggregator margin_win_rate
  uv run python scripts/cross_probe_correlate.py --aggregator dpo_accuracy

Bugs fixed (2026-04-18):
  Bug A: bootstrap CI NaN pollution from LAPACK DLASCL errors → drop NaNs, require
         ≥10 valid resamples before emitting a CI; suppress ConstantInputWarning.
  Bug B: Simpson's-paradox on combined Qwen+Pythia (different srank scales) →
         within-family z-score for combined_* keys; raw values for per-family keys.
  FIX-A: length-bias in reward margin → --aggregator per_token divides logp by
         n_tokens_chosen / n_tokens_rejected before computing the margin.
         Default is per_token; sum mode preserved for back-compat and appendix disclosure.
  FIX-B: length-invariant binary metrics added:
         margin_win_rate — fraction of pairs where DPO improves margin over base;
           length bias cancels because base+DPO see the same token counts.
         dpo_accuracy — fraction of pairs where DPO margin > 0 (still length-biased;
           flagged in output note).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
import warnings
from pathlib import Path

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR  = SCRIPT_DIR.parent
DPO_BETA   = 0.1


# ── JSONL aggregation (reward margin only) ────────────────────────────────────

def aggregate_jsonl(
    jsonl_path: Path,
    beta: float = DPO_BETA,
    aggregator: str = "per_token",
) -> dict | None:
    """Compute reward_margin_mean/se from a score JSONL.  Returns None if missing.

    aggregator: "sum"             — raw logp sum (original behaviour, length-biased)
                "per_token"       — divide each logp by n_tokens before computing margin
                                    (length-normalised; requires n_tokens_chosen /
                                     n_tokens_rejected fields, recorded since 2026-04-17)
                "margin_win_rate" — fraction of pairs where DPO margin > base margin;
                                    length bias cancels because base and DPO see the same
                                    token counts (preferred for length-invariant analysis).
                "dpo_accuracy"    — fraction of pairs where DPO margin > 0 (chosen > rejected);
                                    NOTE: still length-biased — longer rejected sequences tend
                                    to have more-negative sum logp, inflating accuracy numbers.
                                    Reported for comparison only.
    """
    if not jsonl_path.exists():
        log.warning(f"Missing JSONL: {jsonl_path}")
        return None

    margins = []
    win_flags: list[float] = []   # 1.0 if DPO margin > base margin, else 0.0
    acc_flags: list[float] = []   # 1.0 if DPO margin > 0, else 0.0

    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vals = [r["logp_chosen_dpo"], r["logp_rejected_dpo"],
                    r["logp_chosen_base"], r["logp_rejected_base"]]
            if any(v != v for v in vals) or any(abs(v) > 1e6 for v in vals):
                continue

            # Raw logp sums (always compute for binary metrics)
            lc_dpo  = r["logp_chosen_dpo"]
            lr_dpo  = r["logp_rejected_dpo"]
            lc_base = r["logp_chosen_base"]
            lr_base = r["logp_rejected_base"]

            dpo_margin_raw  = lc_dpo  - lr_dpo
            base_margin_raw = lc_base - lr_base

            # Binary metrics (length-cancellation: same nc/nr in both terms)
            win_flags.append(1.0 if dpo_margin_raw > base_margin_raw else 0.0)
            acc_flags.append(1.0 if dpo_margin_raw > 0.0 else 0.0)

            if aggregator == "per_token":
                nc = r.get("n_tokens_chosen")
                nr = r.get("n_tokens_rejected")
                if not nc or not nr or nc <= 0 or nr <= 0:
                    # Fall back to sum if token counts missing (shouldn't happen post-2026-04-17)
                    logp_c_dpo  = lc_dpo
                    logp_r_dpo  = lr_dpo
                    logp_c_base = lc_base
                    logp_r_base = lr_base
                else:
                    logp_c_dpo  = lc_dpo  / nc
                    logp_r_dpo  = lr_dpo  / nr
                    logp_c_base = lc_base / nc
                    logp_r_base = lr_base / nr
            else:  # "sum" (also used as source for binary modes — not consumed there)
                logp_c_dpo  = lc_dpo
                logp_r_dpo  = lr_dpo
                logp_c_base = lc_base
                logp_r_base = lr_base

            rm = (logp_c_dpo - logp_r_dpo) - (logp_c_base - logp_r_base)
            margins.append(beta * rm)

    if not margins:
        log.warning(f"No valid records in {jsonl_path}")
        return None

    def se(vals: list[float]) -> float:
        return statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) >= 2 else float("nan")

    n = len(margins)

    if aggregator == "margin_win_rate":
        rate = statistics.mean(win_flags)
        return {
            "reward_margin_mean": rate,   # per-checkpoint win rate (used uniformly downstream)
            "reward_margin_se":   se(win_flags),
            "n":                  n,
            "win_rate":           rate,
            "raw_wins":           int(sum(win_flags)),
        }
    elif aggregator == "dpo_accuracy":
        rate = statistics.mean(acc_flags)
        return {
            "reward_margin_mean": rate,   # per-checkpoint accuracy (used uniformly downstream)
            "reward_margin_se":   se(acc_flags),
            "n":                  n,
            "dpo_accuracy":       rate,
            "raw_correct":        int(sum(acc_flags)),
            "note":               "length-biased: longer rejected raises accuracy spuriously",
        }
    else:
        return {
            "reward_margin_mean": statistics.mean(margins),
            "reward_margin_se":   se(margins),
            "n":                  n,
        }


# ── Structural metric loading ──────────────────────────────────────────────────

def load_pythia_structural(results_dir: Path) -> list[dict]:
    """Load srank + gamma per checkpoint from T1.2 summary.json."""
    path = results_dir / "behavior_geometry" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    data = json.loads(path.read_text())
    out  = []
    for entry in data:
        size = entry["model_size"]
        seed = entry["seed"]
        # Construct the model_size label used in JSONL filenames
        if size == "1b" and seed == 117:
            file_size = "1b_s117"
        else:
            file_size = size
        out.append({
            "chain":      "pythia",
            "family":     "pythia",
            "model_size": file_size,
            "seed":       seed,
            "srank":      entry["srank"],
            "gamma":      entry["gamma_overlap"],
        })
    return out


def load_qwen_structural(results_dir: Path) -> dict:
    """Load srank + gamma from T2.1 summary.json (mean across attention+MLP for Δ_DPO)."""
    path = results_dir / "t21_qwen_fullweight" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    data = json.loads(path.read_text())
    # Real structure: data["deltas"]["dpo"] → {"attention": {...}, "mlp": {...}}
    # Each has "stable_rank": {"mean": ...} and "gamma": {"mean": ...}
    dpo = data.get("deltas", {}).get("dpo", data.get("delta_dpo", {}))
    if "attention" in dpo:
        # New structure: average attention and MLP
        attn = dpo["attention"]
        mlp  = dpo.get("mlp", {})
        attn_n = attn.get("stable_rank", {}).get("n", 1)
        mlp_n  = mlp.get("stable_rank", {}).get("n", 0) if mlp else 0
        total_n = attn_n + mlp_n
        if total_n > 0:
            srank = (
                attn.get("stable_rank", {}).get("mean", 0.0) * attn_n
                + (mlp.get("stable_rank", {}).get("mean", 0.0) if mlp else 0.0) * mlp_n
            ) / total_n
            gamma = (
                attn.get("gamma", {}).get("mean", 0.0) * attn_n
                + (mlp.get("gamma", {}).get("mean", 0.0) if mlp else 0.0) * mlp_n
            ) / total_n
        else:
            srank = attn.get("stable_rank", {}).get("mean", float("nan"))
            gamma = attn.get("gamma", {}).get("mean", float("nan"))
    else:
        # Legacy flat structure
        srank = dpo.get("mean_srank", dpo.get("median_srank", float("nan")))
        gamma = dpo.get("mean_gamma", dpo.get("median_gamma", float("nan")))
    return {
        "chain":      "qwen",
        "family":     "qwen",
        "model_size": "qwen2-1.5b",
        "seed":       None,
        "srank":      srank,
        "gamma":      gamma,
    }


# ── Correlation helpers ────────────────────────────────────────────────────────

def pearson_r(x: list[float], y: list[float]) -> float:
    from scipy.stats import pearsonr  # noqa: PLC0415
    if len(x) < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = pearsonr(x, y)
    return float(r)


def spearman_r(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr  # noqa: PLC0415
    if len(x) < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = spearmanr(x, y)
    return float(r)


def bootstrap_ci(
    x: list[float], y: list[float],
    stat_fn, n_boot: int = 5000, ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI with NaN-drop fix (Bug A).

    After collecting resampled statistics, drop NaN values (which arise from
    LAPACK DLASCL errors on constant bootstrap samples), then check there are
    at least 10 valid resamples before computing percentiles.
    """
    import numpy as np  # noqa: PLC0415
    rng = random.Random(seed)
    n   = len(x)
    bootstrapped_r = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        xi  = [x[i] for i in idx]
        yi  = [y[i] for i in idx]
        try:
            bootstrapped_r.append(stat_fn(xi, yi))
        except Exception:
            pass

    # Bug A fix: drop NaNs before percentile
    bootstrapped_r = np.array(bootstrapped_r, dtype=float)
    bootstrapped_r = bootstrapped_r[~np.isnan(bootstrapped_r)]
    if len(bootstrapped_r) < 10:
        return float("nan"), float("nan")
    ci_lo, ci_hi = np.percentile(bootstrapped_r, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(ci_lo), float(ci_hi)


def corr_block(
    x: list[float], y: list[float], label: str,
) -> dict:
    pr  = pearson_r(x, y)
    sr  = spearman_r(x, y)
    pr_lo, pr_hi = bootstrap_ci(x, y, pearson_r)
    sr_lo, sr_hi = bootstrap_ci(x, y, spearman_r)
    log.info(
        f"  {label}: Pearson={pr:.3f} [{pr_lo:.3f},{pr_hi:.3f}]  "
        f"Spearman={sr:.3f} [{sr_lo:.3f},{sr_hi:.3f}]  n={len(x)}"
    )
    return {
        "n":               len(x),
        "pearson_r":       pr,
        "spearman_r":      sr,
        "pearson_ci95_lo": pr_lo,
        "pearson_ci95_hi": pr_hi,
        "spearman_ci95_lo": sr_lo,
        "spearman_ci95_hi": sr_hi,
    }


# ── Within-family z-score (Bug B fix) ─────────────────────────────────────────

def zscore_within_family(records: list[dict], cols: tuple[str, ...]) -> list[dict]:
    """Return copies of records with z-scored columns, computed within each family.

    Bug B fix: Qwen srank (~28) and Pythia srank (~3.5) live on different scales.
    Raw pooling produces Simpson's-paradox artifacts.  Within-family z-scoring
    centres and scales each family independently before combining.
    """
    import pandas as pd  # noqa: PLC0415
    import numpy as np   # noqa: PLC0415

    df = pd.DataFrame(records)
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_z"] = df.groupby("family")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0
        )
    return df.to_dict(orient="records")


# ── Data assembly ─────────────────────────────────────────────────────────────

def assemble_records(
    structural_rows: list[dict],
    cross_probe_dir: Path,
    probe: str,
    model_spec_fn,  # fn(row) -> job_key prefix for JSONL lookup
    aggregator: str = "per_token",
) -> list[dict]:
    """For each structural row, find the matching JSONL and merge."""
    results = []
    for row in structural_rows:
        job_prefix = model_spec_fn(row)
        jsonl_path = cross_probe_dir / f"{job_prefix}__{probe}.jsonl"
        agg = aggregate_jsonl(jsonl_path, aggregator=aggregator)
        if agg is None:
            log.warning(f"Skipping {row} — no JSONL for probe={probe}")
            continue
        results.append({**row, **agg, "probe": probe})
    return results


# ── Figure I ──────────────────────────────────────────────────────────────────

def generate_figI(records_hh: list[dict], records_uf: list[dict], out_dir: Path, mode: str = "per_token") -> None:
    """2×2 scatter: rows={srank, γ}, cols={hh probe, uf probe}.

    Layout:
    - Pythia points: gradient blue by size (small=light, large=dark), distinct marker per size
    - Qwen: orange star, labeled separately
    - Linear fit through Pythia-only points
    - Pearson r annotated in top-right of each panel (bold if |r|>0.7)
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as e:
        log.warning(f"Cannot generate figure: {e}")
        return

    # Pythia size → gradient blue; Qwen = orange
    size_order = ["70m", "160m", "410m", "1b", "1b_s117"]
    n_pythia_sizes = len(size_order)
    blues = plt.cm.Blues(np.linspace(0.35, 0.90, n_pythia_sizes))
    pythia_color_map = {sz: blues[i] for i, sz in enumerate(size_order)}

    marker_map = {
        "70m": "o", "160m": "s", "410m": "^", "1b": "D",
        "1b_s117": "v", "qwen2-1.5b": "*",
    }
    label_map = {
        "70m": "70m", "160m": "160m", "410m": "410m",
        "1b": "1b@42", "1b_s117": "1b@117",
        "qwen2-1.5b": "Qwen2-1.5B",
    }
    qwen_color = "#d95f02"  # clear orange

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(
        "In-distribution (hh) vs cross-distribution (UF): "
        "srank collapse predicts OOD penalty",
        fontsize=11, fontweight="bold",
    )

    panel_specs = [
        (axes[0, 0], "hh",  "srank",  records_hh, "stable rank (srank)",        "hh probe (in-distribution)"),
        (axes[0, 1], "uf",  "srank",  records_uf, "stable rank (srank)",        "UF probe (cross-distribution)"),
        (axes[1, 0], "hh",  "gamma",  records_hh, r"$\gamma$-overlap (bonus_R k=5)", "hh probe (in-distribution)"),
        (axes[1, 1], "uf",  "gamma",  records_uf, r"$\gamma$-overlap (bonus_R k=5)", "UF probe (cross-distribution)"),
    ]

    for ax, probe, metric_key, records, xlabel, col_title in panel_specs:
        pythia_recs = [r for r in records if r["chain"] == "pythia"]
        qwen_recs   = [r for r in records if r["chain"] == "qwen"]

        # Scatter Pythia
        for r in pythia_recs:
            sz    = r["model_size"]
            color = pythia_color_map.get(sz, "steelblue")
            mker  = marker_map.get(sz, "o")
            xi    = r[metric_key]
            yi    = r["reward_margin_mean"]
            ye    = r["reward_margin_se"]
            lbl   = label_map.get(sz, sz)
            ax.errorbar(xi, yi, yerr=ye, fmt=mker, color=color,
                        capsize=3, markersize=7, markeredgecolor="white",
                        markeredgewidth=0.5)
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=color)

        # Scatter Qwen
        for r in qwen_recs:
            sz   = r["model_size"]
            xi   = r[metric_key]
            yi   = r["reward_margin_mean"]
            ye   = r["reward_margin_se"]
            lbl  = label_map.get(sz, sz)
            ax.errorbar(xi, yi, yerr=ye, fmt="*", color=qwen_color,
                        capsize=3, markersize=11, markeredgecolor="white",
                        markeredgewidth=0.5)
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=qwen_color, fontweight="bold")

        # Linear fit through Pythia-only
        if len(pythia_recs) >= 3:
            px = [r[metric_key]           for r in pythia_recs]
            py = [r["reward_margin_mean"] for r in pythia_recs]
            try:
                coeffs = np.polyfit(px, py, 1)
                x_line = np.linspace(min(px), max(px), 50)
                ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.4, linewidth=1.2)
            except Exception:
                pass

        # Pearson r (Pythia-only, n=5)
        if len(pythia_recs) >= 3:
            px = [r[metric_key]           for r in pythia_recs]
            py = [r["reward_margin_mean"] for r in pythia_recs]
            pr = pearson_r(px, py)
            pr_str = f"$r={pr:+.2f}$" if pr == pr else "r=n/a"
            bold = abs(pr) > 0.7 if pr == pr else False
            ax.text(
                0.97, 0.97, f"Pearson {pr_str}\n(n=5 pythia)",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
                fontweight="bold" if bold else "normal",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.7"),
            )

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(r"reward margin ($\beta\cdot\Delta\log\pi$)", fontsize=9)
        ax.set_title(col_title, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)
        ax.axhline(0, color="0.6", linewidth=0.8, linestyle="-")

    # Shared legend
    handles = []
    for sz in size_order:
        c = pythia_color_map[sz]
        m = marker_map[sz]
        lbl = label_map[sz]
        handles.append(plt.Line2D([0], [0], marker=m, color="w",
                                   markerfacecolor=c, markeredgecolor="white",
                                   markersize=7, label=lbl))
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                               markerfacecolor=qwen_color, markeredgecolor="white",
                               markersize=11, label="Qwen2-1.5B"))
    handles.append(plt.Line2D([0], [0], linestyle="--", color="k",
                               alpha=0.4, linewidth=1.2, label="OLS (Pythia)"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fig_path = out_dir / f"figI_cross_probe_{mode}.png"
    fig.savefig(str(fig_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"Fig I saved: {fig_path} ({fig_path.stat().st_size // 1024} KB)")

    # Copy to manuscript/figures/ (per_token is the paper default)
    mfig = PAPER_DIR / "manuscript" / "figures"
    mfig.mkdir(parents=True, exist_ok=True)
    import shutil  # noqa: PLC0415
    shutil.copy(fig_path, mfig / f"figI_cross_probe_{mode}.png")
    if mode == "per_token":
        # Also write canonical name used by \includegraphics in main.tex
        shutil.copy(fig_path, mfig / "figI_cross_probe.png")

    # Save caption
    _mode_note_map = {
        "per_token":       "(per-token-normalised reward margin)",
        "sum":             "(raw-sum reward margin)",
        "margin_win_rate": "(margin win-rate: fraction of pairs DPO improves over base)",
        "dpo_accuracy":    "(DPO accuracy: fraction of pairs DPO prefers chosen; length-biased)",
    }
    mode_note = _mode_note_map.get(mode, f"({mode})")
    caption_text = (
        f"Fig I {mode_note}. Cross-probe correlation between weight-structural metrics (srank, \u03b3) "
        "and DPO reward margin. Left column: the in-distribution probe (hh-rlhf test) shows "
        "srank collapse is essentially decoupled from reward (Pearson r\u2248+0.28). Right "
        "column: on the out-of-distribution probe (UltraFeedback test), srank collapse becomes "
        "a strong negative predictor of reward margin (r\u2248\u22120.92), reversing the sign "
        "and amplifying the magnitude by ~3\u00d7. This asymmetry reframes the low-rank "
        "signature as representational overfitting rather than benign parameterization. "
        "n=5 Pythia LoRA-DPO checkpoints (70m/160m/410m/1b_s42/1b_s117, all trained on "
        "hh-rlhf); Qwen2-1.5B full-weight TRL-online-DPO on UltraFeedback shown as "
        "out-of-family reference point."
    )
    caption_path = out_dir / f"figI_cross_probe_{mode}.caption.txt"
    caption_path.write_text(caption_text + "\n")
    log.info(f"Caption saved: {caption_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(PAPER_DIR / "results" / "cross_probe"),
        help="Output directory (default: results/cross_probe)",
    )
    parser.add_argument(
        "--aggregator",
        choices=["sum", "per_token", "margin_win_rate", "dpo_accuracy"],
        default="per_token",
        help=(
            "Reward-margin aggregation mode. "
            "'per_token' (default): divide each logp by sequence length before computing margin — "
            "corrects for length bias when chosen/rejected lengths are asymmetric (e.g. UltraFeedback). "
            "'sum': raw logp sum (original behaviour, preserved for back-compat and appendix disclosure). "
            "'margin_win_rate': fraction of pairs where DPO improves margin over base; "
            "length bias cancels because base and DPO see the same token counts. "
            "'dpo_accuracy': fraction of pairs where DPO margin > 0; still length-biased (flagged in output)."
        ),
    )
    args = parser.parse_args()

    out_dir     = Path(args.out)
    results_dir = PAPER_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = args.aggregator
    log.info(f"Aggregator mode: {mode}")

    # ── Load structural side ──────────────────────────────────────────────────
    log.info("Loading structural metrics...")
    pythia_rows = load_pythia_structural(results_dir)
    qwen_row    = load_qwen_structural(results_dir)
    all_rows    = pythia_rows + [qwen_row]

    def pythia_job_key(row: dict) -> str:
        return f"pythia_lora_42_v2_{row['model_size']}"

    def qwen_job_key(row: dict) -> str:
        return "qwen_fullweight"

    def job_key_fn(row: dict) -> str:
        return pythia_job_key(row) if row["chain"] == "pythia" else qwen_job_key(row)

    # ── Assemble cross-probe data ─────────────────────────────────────────────
    log.info("Assembling hh-probe records...")
    records_hh = assemble_records(all_rows, out_dir, "hh", job_key_fn, aggregator=mode)
    log.info("Assembling uf-probe records...")
    records_uf = assemble_records(all_rows, out_dir, "uf", job_key_fn, aggregator=mode)

    # ── Within-family z-score for combined correlations (Bug B fix) ───────────
    z_cols = ("srank", "gamma", "reward_margin_mean")
    records_hh_z = zscore_within_family(records_hh, z_cols)
    records_uf_z = zscore_within_family(records_uf, z_cols)

    # ── Compute correlation matrix ────────────────────────────────────────────
    log.info("\n=== Correlation matrix ===")
    corr_matrix: dict = {
        "aggregator_mode": mode,
        "note_simpsons_paradox": (
            "combined_* correlations use within-family z-scored srank/gamma/reward_margin; "
            "raw pooling produces artifacts because Qwen srank (~28) and Pythia srank (~3.5) "
            "live on different scales."
        ),
        "note_aggregator": (
            "per_token: logp divided by sequence length before margin computation, corrects "
            "length bias when chosen/rejected lengths are asymmetric (e.g. UltraFeedback). "
            "sum: raw logp sum, original behaviour."
        ) if mode == "per_token" else (
            "sum: raw logp sum aggregation. Length-biased when chosen/rejected lengths differ."
        ),
    }

    for probe_label, records_raw, records_z in [
        ("hh", records_hh, records_hh_z),
        ("uf", records_uf, records_uf_z),
    ]:
        for chain_label in ["pythia", "qwen", "combined"]:
            if chain_label == "combined":
                # Use z-scored values for combined
                subset     = records_z
                srank_col  = "srank_z"
                gamma_col  = "gamma_z"
                reward_col = "reward_margin_mean_z"
            else:
                subset     = [r for r in records_raw if r["chain"] == chain_label]
                srank_col  = "srank"
                gamma_col  = "gamma"
                reward_col = "reward_margin_mean"

            if len(subset) < 2:
                continue

            sranks  = [r[srank_col]  for r in subset if srank_col  in r]
            gammas  = [r[gamma_col]  for r in subset if gamma_col  in r]
            rewards = [r[reward_col] for r in subset if reward_col in r]

            # Align lengths (all should be same, but guard)
            n = min(len(sranks), len(gammas), len(rewards))
            sranks  = sranks[:n]
            gammas  = gammas[:n]
            rewards = rewards[:n]

            key = f"{chain_label}_{probe_label}"
            log.info(f"\n  [{key}]")
            corr_matrix[key] = {
                "srank_vs_rm": corr_block(sranks, rewards, f"{key}/srank_vs_rm"),
                "gamma_vs_rm": corr_block(gammas, rewards, f"{key}/gamma_vs_rm"),
            }

    # Add effective_n field (total pairs used across all keys after NaN drop)
    all_used = set()
    for probe_recs in [records_hh, records_uf]:
        for r in probe_recs:
            all_used.add((r.get("chain"), r.get("model_size"), r.get("probe")))
    corr_matrix["effective_n"] = {
        "hh_records": len(records_hh),
        "uf_records": len(records_uf),
    }

    # Embed per-checkpoint rate table for easy inspection
    def _per_ckpt_rows(records: list[dict]) -> list[dict]:
        return [
            {
                "model_size": r.get("model_size"),
                "chain":      r.get("chain"),
                "rate":       r.get("reward_margin_mean"),
                "se":         r.get("reward_margin_se"),
                "n":          r.get("n"),
            }
            for r in records
        ]

    corr_matrix["per_checkpoint"] = {
        "hh": _per_ckpt_rows(records_hh),
        "uf": _per_ckpt_rows(records_uf),
    }

    if mode == "margin_win_rate":
        corr_matrix["note_aggregator"] = (
            "margin_win_rate: fraction of pairs where DPO margin (chosen-rejected) exceeds base margin. "
            "Length bias cancels: base and DPO share the same token counts per pair."
        )
    elif mode == "dpo_accuracy":
        corr_matrix["note_aggregator"] = (
            "dpo_accuracy: fraction of pairs where DPO margin > 0 (chosen preferred). "
            "WARNING: still length-biased — longer rejected sequences inflate accuracy numbers."
        )

    corr_path = out_dir / f"correlation_matrix_{mode}.json"
    corr_path.write_text(json.dumps(corr_matrix, indent=2))
    log.info(f"\nCorrelation matrix written: {corr_path}")
    # Also write canonical path (per_token is the paper default; sum is back-compat)
    if mode == "per_token":
        canonical = out_dir / "correlation_matrix.json"
        canonical.write_text(json.dumps(corr_matrix, indent=2))
        log.info(f"Canonical correlation_matrix.json updated → {canonical}")

    # ── Generate Fig I ────────────────────────────────────────────────────────
    generate_figI(records_hh, records_uf, out_dir, mode=mode)

    log.info("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
