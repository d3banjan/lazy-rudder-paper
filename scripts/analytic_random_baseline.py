"""Analytic Haar-random subspace overlap baseline.

The paper's overlap statistic for a pair of rank-k right-singular-vector
matrices V_W, V_ΔW ∈ R^{d × k} is:

    p_right(k) = ||V_W^T V_ΔW||_F² / k
               = (sum of squared singular values of the k×k cross-gram G) / k

For two independent Haar-uniform k-frames in R^d, the closed-form
expectation is:

    E[||G||_F²] = E[tr(G^T G)]
               = E[tr(V_W^T V_ΔW V_ΔW^T V_W)]
               = tr(E[V_ΔW V_ΔW^T] · V_W V_W^T)   # V_W fixed, V_ΔW Haar-random
               = tr((k/d) I_d · V_W V_W^T)         # E[V_ΔW V_ΔW^T] = (k/d)I for Haar
               = (k/d) · tr(V_W V_W^T)
               = (k/d) · k
               = k²/d

    => E[p_right(k)] = E[||G||_F²] / k = k / d

This is the exact (finite-d) Haar expectation, without any 1/d² or
large-d correction needed.  The paper already uses k/d as the bonus
denominator, which is therefore not a sample approximation — it IS the
analytic Haar expectation.

Citation: equivalent to E[tr(P₁ P₂)] = k₁ k₂ / d for rank-k₁ and
rank-k₂ projectors drawn independently from the Haar measure on G(k,d)
(Chikuse 2003, "Statistics on Special Manifolds", Lecture Notes in
Statistics 174, Springer; also Collins & Śniady 2006 Weingarten calculus).

Usage
-----
    uv run python papers/lazy-rudder/scripts/analytic_random_baseline.py [--sanity-n N]

Outputs
-------
    papers/lazy-rudder/results/analytic_random_baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ── optional numpy for sanity-check sampling ─────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR   = SCRIPTS_DIR.parent
OUT_DIR     = PAPER_DIR / "results" / "analytic_random_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── (d, k) tuples used in the paper ──────────────────────────────────────────
# Each entry: (label, d_in, d_out, k_list)
# Pythia-70m: QKV weight [1536, 512] — 3*d_model × d_model
# Pythia-160m: QKV weight [2304, 768]
# Pythia-410m: QKV weight [3072, 1024]
# Pythia-1B:   QKV weight [6144, 2048]
PAPER_CONFIGS = [
    ("pythia-70m",  512,  1536, [5, 10, 20]),
    ("pythia-160m", 768,  2304, [5, 10, 20]),
    ("pythia-410m", 1024, 3072, [5, 10, 20]),
    ("pythia-1b",   2048, 6144, [5, 10, 20]),
]

K_VALS_DEFAULT = [5, 10, 20]


def analytic_p(k: int, d: int) -> float:
    """Exact Haar expectation of p_right(k) or p_left(k).

    E[p(k)] = k / d

    Derivation:
        E[||V_W^T V_ΔW||_F²] / k
        = E[tr(V_ΔW V_ΔW^T · V_W V_W^T)] / k
        = tr(E[V_ΔW V_ΔW^T] · V_W V_W^T) / k
        = tr((k/d) I · V_W V_W^T) / k
        = (k/d) · tr(V_W V_W^T) / k
        = (k/d) · k / k
        = k / d
    """
    return k / d


def analytic_bonus(k: int, d: int) -> float:
    """E[bonus(k)] = E[p(k)] / (k/d) = 1.0 exactly."""
    return 1.0


def sampled_p(k: int, d: int, n_trials: int, rng) -> tuple[float, float]:
    """Monte Carlo estimate of E[p(k)] with std for sanity check."""
    if not HAS_NUMPY:
        return float("nan"), float("nan")
    trials = []
    for _ in range(n_trials):
        Q1, _ = np.linalg.qr(rng.standard_normal((d, k)))
        Q2, _ = np.linalg.qr(rng.standard_normal((d, k)))
        G  = Q1.T @ Q2
        sv = np.linalg.svd(G, compute_uv=False)
        trials.append(float((sv ** 2).sum() / k))
    arr = np.array(trials)
    return float(arr.mean()), float(arr.std() / math.sqrt(n_trials))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytic Haar-random subspace baseline")
    parser.add_argument(
        "--sanity-n", type=int, default=5000,
        help="Number of Monte Carlo trials for sanity check (0 to skip). Default: 5000"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0) if HAS_NUMPY else None

    output: dict = {
        "derivation": (
            "E[p_right(k)] = k / d  (exact, finite-d Haar measure on G(k,d)). "
            "Reference: Chikuse (2003) Statistics on Special Manifolds, "
            "Lecture Notes in Statistics 174, Springer; also Collins & Sniady (2006) "
            "Weingarten calculus for the unitary group. "
            "The paper's denominator k/d_in in bonus_right is therefore already "
            "the analytic Haar expectation, not a sampled approximation."
        ),
        "formula": "E[p(k, d)] = k / d",
        "bonus_at_random": 1.0,
        "configs": {},
        "sanity_check": {},
    }

    print("Analytic Haar-random subspace overlap baseline")
    print("=" * 60)
    print(f"Formula: E[p(k)] = k / d   =>   E[bonus(k)] = 1.0 exactly")
    print()

    for label, d_in, d_out, k_list in PAPER_CONFIGS:
        cfg_entry: dict = {}
        for k in k_list:
            p_right_analytic = analytic_p(k, d_in)
            p_left_analytic  = analytic_p(k, d_out)
            cfg_entry[f"k{k}"] = {
                "k": k,
                "d_in":  d_in,
                "d_out": d_out,
                "p_right_analytic": round(p_right_analytic, 8),
                "p_left_analytic":  round(p_left_analytic,  8),
                "bonus_at_random":  1.0,
                "note": f"E[p_right] = {k}/{d_in} = {p_right_analytic:.6f}",
            }
            print(
                f"  {label:12s}  k={k:2d}  d_in={d_in:4d}  "
                f"E[p_right]={p_right_analytic:.6f}  "
                f"E[p_left]={p_left_analytic:.6f}"
            )
        output["configs"][label] = cfg_entry

    # ── Sanity check: one representative point ───────────────────────────────
    if args.sanity_n > 0 and HAS_NUMPY:
        print()
        print(f"Sanity check: Monte Carlo (N={args.sanity_n}) vs analytic")
        print("-" * 60)
        # Use 410m / k=5 as canonical comparison point
        d_check, k_check = 1024, 5
        analytic_val = analytic_p(k_check, d_check)
        mc_mean, mc_se = sampled_p(k_check, d_check, args.sanity_n, rng)
        ratio = mc_mean / analytic_val if analytic_val > 0 else float("nan")
        print(
            f"  d={d_check} k={k_check}: "
            f"analytic={analytic_val:.6f}  "
            f"sampled={mc_mean:.6f} ±{mc_se:.6f}  "
            f"ratio={ratio:.4f}"
        )
        assert abs(ratio - 1.0) < 0.02, f"Sanity check failed: ratio={ratio:.4f}"
        print("  [PASS] Monte Carlo agrees with analytic within 2%")
        output["sanity_check"] = {
            "d": d_check,
            "k": k_check,
            "n_trials": args.sanity_n,
            "analytic": analytic_val,
            "sampled_mean": round(mc_mean, 8),
            "sampled_se":   round(mc_se, 8),
            "ratio": round(ratio, 6),
            "status": "PASS",
        }
    elif args.sanity_n > 0 and not HAS_NUMPY:
        print("  [SKIP] numpy not available — install numpy for Monte Carlo sanity check")
        output["sanity_check"] = {"status": "SKIPPED", "reason": "numpy not available"}

    # ── Write JSON ───────────────────────────────────────────────────────────
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
