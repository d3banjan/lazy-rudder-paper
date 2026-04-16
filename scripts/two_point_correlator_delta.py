"""
two_point_correlator_delta.py

Gauge-invariant 2-point correlator across residual-stream layer depth.
Lattice-QCD analog: <O(tau) O(0)> decay → bound-state spectrum.

Hypotheses:
  - Base Pythia-410m: fast-decay-only (unbound transients).
  - SFT/DPO: double-exponential with one slow mode (m_1 ≈ 0) = quasiparticle.

Probes are gauge-invariant (permutation / orthogonal):
  - O_col(L) = sorted column norms  (permutation-invariant)
  - O_svd(L) = sorted singular values  (permutation + orthogonal invariant)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
ORBIT_DIR = RESULTS_DIR / "_orbit"
OUT_DIR = RESULTS_DIR / "two_point_correlator_delta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "410m_base": ORBIT_DIR / "410m_base_activations.pt",
    "410m_sft": ORBIT_DIR / "410m_sft_activations.pt",
    "410m_dpo": ORBIT_DIR / "410m_dpo_activations.pt",
}

UNIT_PREFERENCE = ["residual", "resid", "block_output"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(path: Path) -> dict:
    print(f"  Loading {path.name} ...", flush=True)
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def select_unit(units: dict, preference: list[str]) -> str:
    for u in preference:
        if u in units:
            return u
    return list(units.keys())[0]


def get_probe_matrices(data: dict, set_name: str, unit: str) -> list[np.ndarray]:
    """Return list of numpy arrays [layer0, layer1, ...], each shape (N, d_model)."""
    layers = data[set_name]["units"][unit]
    return [t.float().numpy() for t in layers]


# ---------------------------------------------------------------------------
# Step 1 — gauge-invariant probes
# ---------------------------------------------------------------------------

def probe_col(H: np.ndarray) -> np.ndarray:
    """Sorted column norms: shape (d_model,)."""
    col_norms = np.linalg.norm(H, axis=0)  # (d,)
    return np.sort(col_norms)


def probe_svd(H: np.ndarray) -> np.ndarray:
    """Sorted singular values descending: shape (min(N, d),)."""
    sv = np.linalg.svd(H, compute_uv=False)
    return sv  # already sorted descending by numpy


# ---------------------------------------------------------------------------
# Step 2 — 2-point correlator C(L, L+k)
# ---------------------------------------------------------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compute_correlator_arrays(
    probes: list[np.ndarray],
    n_layers: int,
) -> dict[str, np.ndarray]:
    """
    Given list of probe vectors (one per layer), compute C(k) averaged over L.
    Returns dict with keys: pearson, cosine.  Values: arrays of length n_layers-1.
    """
    max_k = n_layers - 1
    pearson_vals = np.full(max_k, np.nan)
    cosine_vals = np.full(max_k, np.nan)

    for k in range(1, max_k + 1):
        pc_list, cc_list = [], []
        for L in range(n_layers - k):
            a, b = probes[L], probes[L + k]
            # align dims (svd can differ if N < d)
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
            pc = pearson_corr(a, b)
            cc = cosine_sim(a, b)
            if not np.isnan(pc):
                pc_list.append(pc)
            if not np.isnan(cc):
                cc_list.append(cc)
        pearson_vals[k - 1] = np.mean(pc_list) if pc_list else np.nan
        cosine_vals[k - 1] = np.mean(cc_list) if cc_list else np.nan

    return {"pearson": pearson_vals, "cosine": cosine_vals}


# ---------------------------------------------------------------------------
# Step 3 — exponential fits
# ---------------------------------------------------------------------------

def _single_exp(k, A, m, c):
    return A * np.exp(-m * k) + c


def _double_exp(k, A, m1, B, m2, c):
    # enforce ordering m1 <= m2 inside the model
    return A * np.exp(-m1 * k) + B * np.exp(-m2 * k) + c


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _f_stat(rss1: float, rss2: float, n: int, p1: int, p2: int) -> float:
    """F-test: model2 is better than model1. p2 > p1."""
    if rss2 <= 0 or rss1 <= rss2:
        return 0.0
    num = (rss1 - rss2) / (p2 - p1)
    den = rss2 / (n - p2)
    if den <= 0:
        return float("inf")
    return float(num / den)


def fit_correlator(C: np.ndarray) -> dict[str, Any]:
    """
    Fit single- and double-exponential to C(k) array (length n_layers-1).
    k values are 1..n_layers-1.
    """
    n = len(C)
    k_vals = np.arange(1, n + 1, dtype=float)

    # Filter NaNs
    mask = ~np.isnan(C)
    k_fit = k_vals[mask]
    C_fit = C[mask]
    n_fit = len(k_fit)

    if n_fit < 5:
        return {
            "fit_single": None,
            "fit_double": None,
            "verdict": "insufficient_data",
        }

    # ---- Single exp ----
    p0_s = [C_fit[0] - C_fit[-1], 0.1, C_fit[-1]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt_s, _ = curve_fit(
                _single_exp,
                k_fit,
                C_fit,
                p0=p0_s,
                bounds=([0, 0, -1.0], [2.0, 10.0, 1.0]),
                maxfev=5000,
            )
        A_s, m_s, c_s = popt_s
        pred_s = _single_exp(k_fit, *popt_s)
        rss_s = float(np.sum((C_fit - pred_s) ** 2))
        r2_s = _r_squared(C_fit, pred_s)
        fit_single = {"A": float(A_s), "m": float(m_s), "c": float(c_s), "R2": r2_s, "RSS": rss_s}
    except Exception:
        fit_single = None
        rss_s = float(np.sum((C_fit - np.mean(C_fit)) ** 2))
        r2_s = 0.0

    # ---- Double exp ----
    # Two mass initializations: one near zero (slow), one larger (fast)
    p0_d = [C_fit[0] * 0.5, 0.05, C_fit[0] * 0.5, 0.5, C_fit[-1]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt_d, _ = curve_fit(
                _double_exp,
                k_fit,
                C_fit,
                p0=p0_d,
                bounds=([0, 0, 0, 0, -1.0], [2.0, 5.0, 2.0, 10.0, 1.0]),
                maxfev=10000,
            )
        A_d, m1_d, B_d, m2_d, c_d = popt_d
        # Enforce ordering m1 <= m2
        if m1_d > m2_d:
            A_d, m1_d, B_d, m2_d = B_d, m2_d, A_d, m1_d
        pred_d = _double_exp(k_fit, A_d, m1_d, B_d, m2_d, c_d)
        rss_d = float(np.sum((C_fit - pred_d) ** 2))
        r2_d = _r_squared(C_fit, pred_d)
        slow_frac = float(A_d / (A_d + B_d + 1e-12))
        fit_double = {
            "A": float(A_d), "m1": float(m1_d),
            "B": float(B_d), "m2": float(m2_d),
            "c": float(c_d),
            "R2": r2_d, "RSS": rss_d,
            "slow_frac": slow_frac,
        }
    except Exception:
        fit_double = None
        rss_d = rss_s  # fallback
        r2_d = r2_s

    # ---- Verdict ----
    if fit_single is None and fit_double is None:
        verdict = "fit_failed"
    elif fit_double is None:
        verdict = "single-mode"
    else:
        r2_ratio = r2_d / (r2_s + 1e-8) if r2_s > 0 else 1.0
        m1 = fit_double["m1"]
        m2 = fit_double["m2"]
        if r2_ratio < 1.01:
            verdict = "single-mode"
        elif r2_ratio > 1.1 and m2 > 0 and m1 / (m2 + 1e-8) < 0.3:
            verdict = "double-mode"
        elif abs(m1 - m2) < 0.05:
            verdict = "degenerate"
        else:
            verdict = "single-mode"

    # F-stat
    f_stat = None
    if fit_single is not None and fit_double is not None:
        f_stat = _f_stat(fit_single["RSS"], fit_double["RSS"], n_fit, 3, 5)

    return {
        "fit_single": fit_single,
        "fit_double": fit_double,
        "f_stat": f_stat,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Step 4 — DMD transfer operator
# ---------------------------------------------------------------------------

def dmd_spectral_gap(probes: list[np.ndarray], max_rank: int = 20) -> dict[str, Any]:
    """
    Dynamic Mode Decomposition of probe sequence.
    Returns eigenvalue magnitudes and spectral gap.
    """
    n_layers = len(probes)
    if n_layers < 4:
        return {"lambda": [], "lambda1": None, "gap": None, "lambda2_ratio": None, "n_persistent": 0}

    # Align probe dimensions
    min_dim = min(len(p) for p in probes)
    X = np.stack([p[:min_dim] for p in probes], axis=1)  # (d, n_layers)

    X_past = X[:, :-1]    # (d, n_layers-1)
    X_future = X[:, 1:]   # (d, n_layers-1)

    # Reduced SVD of X_past
    r = min(max_rank, X_past.shape[0], X_past.shape[1])
    try:
        U, S, Vt = np.linalg.svd(X_past, full_matrices=False)
        U = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        # Avoid division by near-zero singular values
        thresh = 1e-10 * S_r[0] if S_r[0] > 0 else 1e-10
        S_inv = np.where(S_r > thresh, 1.0 / S_r, 0.0)

        # Projected transfer operator: T_tilde = U^T X_future V S^{-1}
        T_tilde = U.T @ X_future @ Vt_r.T @ np.diag(S_inv)
        eigvals = np.linalg.eigvals(T_tilde)
        abs_eigs = np.sort(np.abs(eigvals))[::-1]  # descending

        lambda1 = float(abs_eigs[0]) if len(abs_eigs) > 0 else None
        lambda2 = float(abs_eigs[1]) if len(abs_eigs) > 1 else None
        gap = float(lambda1 - lambda2) if (lambda1 is not None and lambda2 is not None) else None
        ratio = float(lambda2 / lambda1) if (lambda1 is not None and lambda2 is not None and lambda1 > 1e-10) else None
        n_persistent = int(np.sum(abs_eigs > 0.95))

        return {
            "lambda": abs_eigs[:10].tolist(),  # top 10
            "lambda1": lambda1,
            "gap": gap,
            "lambda2_ratio": ratio,
            "n_persistent": n_persistent,
        }
    except Exception as e:
        return {"lambda": [], "lambda1": None, "gap": None, "lambda2_ratio": None, "n_persistent": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Step 5 — Verdict logic
# ---------------------------------------------------------------------------

def model_verdict(per_set_results: dict[str, dict]) -> str:
    """Aggregate verdict across sets. Majority vote on set verdicts."""
    verdicts = [v["verdict"] for v in per_set_results.values() if "verdict" in v]
    double = sum(1 for v in verdicts if v == "double-mode")
    single = sum(1 for v in verdicts if v == "single-mode")
    if double > single:
        return "double-mode"
    elif single > 0:
        return "single-mode"
    return "indeterminate"


def headline_verdict(per_model: dict) -> tuple[str, str]:
    """
    Determine headline verdict:
      a: SFT/DPO show double-mode, base does not
      b: base also shows double-mode
      c: all single-mode
      d: mixed
    """
    base_v = per_model.get("410m_base", {}).get("verdict_summary", "single-mode")
    sft_v = per_model.get("410m_sft", {}).get("verdict_summary", "single-mode")
    dpo_v = per_model.get("410m_dpo", {}).get("verdict_summary", "single-mode")

    base_double = "double" in base_v
    sft_double = "double" in sft_v
    dpo_double = "double" in dpo_v

    if not base_double and not sft_double and not dpo_double:
        return "c", "No bound state in any model — quasiparticle hypothesis falsified."
    if base_double:
        return "b", "Base model already shows double-mode structure — bound state is pretraining-level, not training-induced."
    if sft_double or dpo_double:
        aligned = [m for m, d in [("SFT", sft_double), ("DPO", dpo_double)] if d]
        not_aligned = [m for m, d in [("SFT", sft_double), ("DPO", dpo_double)] if not d]
        if not_aligned:
            return "d", f"Mixed: {', '.join(aligned)} show double-mode, {', '.join(not_aligned)} do not."
        return "a", "Quasiparticle confirmed: SFT and DPO show double-mode slow bound state; base does not."
    return "d", "Mixed or inconclusive."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== two_point_correlator_delta.py ===")
    print(f"Output: {OUT_DIR / 'results.json'}\n")

    results: dict[str, Any] = {
        "models": list(MODEL_FILES.keys()),
        "sets": [],
        "unit_selected": None,
        "n_layers": None,
        "per_model": {},
    }

    # Discover sets and units from first model
    first_data = load_activations(list(MODEL_FILES.values())[0])
    sets_found = list(first_data.keys())
    results["sets"] = sets_found
    results["n_layers"] = 24  # known from inspection

    # Select unit
    sample_units = list(first_data[sets_found[0]]["units"].keys())
    unit_selected = select_unit(first_data[sets_found[0]]["units"], UNIT_PREFERENCE)
    results["unit_selected"] = unit_selected

    print(f"Sets: {sets_found}")
    print(f"Units available: {sample_units}")
    print(f"Unit selected: {unit_selected}")
    for s in sets_found:
        n_ex = len(first_data[s]["rows"])
        layer0 = first_data[s]["units"][unit_selected][0]
        print(f"  {s}: {n_ex} examples, shape per layer = {tuple(layer0.shape)}")

    print()

    # Process each model
    for model_name, model_path in MODEL_FILES.items():
        print(f"--- Model: {model_name} ---")
        data = load_activations(model_path)

        per_set: dict[str, Any] = {}

        for set_name in sets_found:
            print(f"  Set: {set_name}", flush=True)
            layers_np = get_probe_matrices(data, set_name, unit_selected)
            n_layers = len(layers_np)

            # Build probe sequences
            col_probes = [probe_col(H) for H in layers_np]
            svd_probes = [probe_svd(H) for H in layers_np]

            # Step 2: correlator arrays
            corr_col = compute_correlator_arrays(col_probes, n_layers)
            corr_svd = compute_correlator_arrays(svd_probes, n_layers)

            # Step 3: fits
            fit_col = fit_correlator(corr_col["pearson"])
            fit_col_cos = fit_correlator(corr_col["cosine"])
            fit_svd = fit_correlator(corr_svd["pearson"])
            fit_svd_cos = fit_correlator(corr_svd["cosine"])

            # Step 4: DMD
            dmd_col = dmd_spectral_gap(col_probes)
            dmd_svd = dmd_spectral_gap(svd_probes)

            # Verdict: use pearson_col as primary, fall back to svd
            primary_fit = fit_col
            secondary_fit = fit_svd
            if primary_fit["verdict"] == "insufficient_data":
                primary_fit = secondary_fit

            set_result = {
                "C_pearson_col": corr_col["pearson"].tolist(),
                "C_cos_col": corr_col["cosine"].tolist(),
                "C_pearson_svd": corr_svd["pearson"].tolist(),
                "C_cos_svd": corr_svd["cosine"].tolist(),
                "fit_single_col": primary_fit["fit_single"],
                "fit_double_col": primary_fit["fit_double"],
                "f_stat_col": primary_fit["f_stat"],
                "fit_single_svd": secondary_fit["fit_single"],
                "fit_double_svd": secondary_fit["fit_double"],
                "f_stat_svd": secondary_fit["f_stat"],
                "dmd_col": dmd_col,
                "dmd_svd": dmd_svd,
                "verdict": primary_fit["verdict"],
            }

            # Print summary for set
            fdc = primary_fit["fit_double"]
            fsc = primary_fit["fit_single"]
            if fdc:
                r2s_str = f"{fsc['R2']:.3f}" if fsc else "N/A"
                print(f"    col: verdict={primary_fit['verdict']}  m1={fdc['m1']:.3f} m2={fdc['m2']:.3f} slow_frac={fdc['slow_frac']:.2f}  R2_single={r2s_str}  R2_double={fdc['R2']:.3f}")
            elif fsc:
                print(f"    col: verdict={primary_fit['verdict']}  m={fsc['m']:.3f}  R2={fsc['R2']:.3f}")
            else:
                print(f"    col: verdict={primary_fit['verdict']}")
            lam1_str = f"{dmd_col['lambda1']:.4f}" if dmd_col["lambda1"] is not None else "N/A"
            gap_str = f"{dmd_col['gap']:.4f}" if dmd_col["gap"] is not None else "N/A"
            print(f"    dmd_col: λ1={lam1_str}  gap={gap_str}  n_persistent={dmd_col['n_persistent']}")

            per_set[set_name] = set_result

        # Model-level verdict
        mv = model_verdict(per_set)
        results["per_model"][model_name] = {
            "per_set": per_set,
            "verdict_summary": mv,
        }
        print(f"  => Model verdict: {mv}\n")

    # Headline
    hv_code, hv_text = headline_verdict(results["per_model"])
    results["headline_verdict"] = hv_code
    results["headline"] = hv_text

    print(f"HEADLINE VERDICT ({hv_code}): {hv_text}\n")

    # Write results
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
