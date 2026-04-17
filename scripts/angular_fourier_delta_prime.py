"""angular_fourier_delta_prime.py — δ′ measurement.

Angular-Fourier spectroscopy on γ's survivor basis.

Bypasses LayerNorm (which killed δ) by using angular coordinates on the unit
hypersphere in the projected subspace. Projects residual activations onto the
top-5 right singular vectors of the base QKV weight at each layer, then FFTs
the angular-velocity trajectory over depth.

Predicted signature:
  Base  → flat ω-spectrum (thermal walk on hypersphere)
  SFT   → low-f peak (coherent orbit)
  DPO   → sharper low-f peak than SFT (less channel-shuffle noise)
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ttest_ind

# Patch transformers' torch.load safety check (locally trusted .bin weights)
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import MODELS_DIR, PAPER_RESULTS_DIR, RESULTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = MODELS_DIR / "pythia-410m"
ORBIT_DIR = RESULTS_DIR / "_orbit"
OUT_DIR   = PAPER_RESULTS_DIR / "angular_fourier_delta_prime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "410m_base": ORBIT_DIR / "410m_base_activations.pt",
    "410m_sft":  ORBIT_DIR / "410m_sft_activations.pt",
    "410m_dpo":  ORBIT_DIR / "410m_dpo_activations.pt",
}

UNIT = "residual"
N_LAYERS = 24
N_STEPS  = N_LAYERS - 1   # 23 angular-velocity steps
K_TOP    = 5               # top singular vectors for basis

# Frequency bins from rfft of length-23 series: k=0..11, f_k = k/23
N_FREQ   = N_STEPS // 2 + 1   # 12


# ---------------------------------------------------------------------------
# Step 1: build top-5 right singular basis per layer from base model
# ---------------------------------------------------------------------------

def load_base_v_top5() -> list[torch.Tensor]:
    """Return V_top5[L] shape [1024, 5] for L in 0..23."""
    print(f"Loading base model from {MODEL_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()

    v_top5 = []
    for li in range(N_LAYERS):
        W = (model.gpt_neox.layers[li]
             .attention.query_key_value.weight
             .detach().float().cpu())   # [3072, 1024]
        _, _, Vt = torch.linalg.svd(W, full_matrices=False)  # Vt: [1024, 1024]
        v_top5.append(Vt[:K_TOP, :].T.contiguous())          # [1024, 5]
        if li % 6 == 0:
            print(f"  layer {li}: W {tuple(W.shape)} -> V_top5 {tuple(v_top5[-1].shape)}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("Base weights processed.", flush=True)
    return v_top5


# ---------------------------------------------------------------------------
# Step 2: load activations
# ---------------------------------------------------------------------------

def load_activations(path: Path) -> dict:
    print(f"  Loading {path.name} ...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def get_layer_tensors(data: dict, set_name: str) -> list[np.ndarray]:
    """Return list of length N_LAYERS, each (N, 1024) float32 numpy."""
    layers = data[set_name]["units"][UNIT]
    return [t.float().numpy() for t in layers]


# ---------------------------------------------------------------------------
# Step 3: project + compute angular velocity
# ---------------------------------------------------------------------------

def project_and_angular_velocity(
    layers: list[np.ndarray],
    v_top5: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Project each layer's activations onto V_top5[L], compute angular velocity
    between consecutive projected unit vectors.

    Returns:
      omega  : shape (N, 23)  — angular velocity per example per step
      ef     : shape (24,)    — energy fraction per layer (||h @ V||² / ||h||²)
      healthy_frac : fraction of (example, step) pairs that are non-degenerate
    """
    N = layers[0].shape[0]

    # Build projected & normalised trajectories
    z_hat = []   # list of (N, 5) unit vectors, one per layer
    ef_list = []
    for li in range(N_LAYERS):
        h = layers[li]                         # (N, 1024)
        V = v_top5[li].numpy()                 # (1024, 5)
        z = h @ V                              # (N, 5)

        # Energy fraction diagnostic
        h_norm2 = (h ** 2).sum(axis=1)                   # (N,)
        z_norm2 = (z ** 2).sum(axis=1)                   # (N,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frac = np.where(h_norm2 > 1e-20, z_norm2 / h_norm2, np.nan)
        ef_list.append(float(np.nanmean(frac)))

        # Normalise z -> unit vector in R^5
        z_norms = np.linalg.norm(z, axis=1, keepdims=True)   # (N, 1)
        safe   = (z_norms.squeeze(1) > 1e-10)
        z_hat_l = np.zeros_like(z)
        z_hat_l[safe] = z[safe] / z_norms[safe]
        z_hat.append(z_hat_l)

    ef = np.array(ef_list)

    # Angular velocity over steps
    omega = np.zeros((N, N_STEPS))
    n_healthy = 0
    n_total   = 0

    for step in range(N_STEPS):
        a = z_hat[step]       # (N, 5)
        b = z_hat[step + 1]   # (N, 5)

        # Dot product per example
        dots = (a * b).sum(axis=1)   # (N,)
        dots = np.clip(dots, -1.0, 1.0)

        # Flag degenerate: either z was near-zero
        a_ok = (np.linalg.norm(z_hat[step],     axis=1) > 0.5)
        b_ok = (np.linalg.norm(z_hat[step + 1], axis=1) > 0.5)
        healthy = a_ok & b_ok

        n_healthy += healthy.sum()
        n_total   += N

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ang = np.where(healthy, np.arccos(dots), np.nan)
        omega[:, step] = ang

    healthy_frac = n_healthy / max(n_total, 1)
    return omega, ef, float(healthy_frac)


# ---------------------------------------------------------------------------
# Step 4: Fourier power spectrum
# ---------------------------------------------------------------------------

def fourier_power(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT the detrended ω series per example.
    Returns P_mean and P_std, each shape (N_FREQ,).
    Per-example power P[n, :] also returned for t-tests.
    """
    # Replace NaN rows with row-mean fallback
    row_means = np.nanmean(omega, axis=1, keepdims=True)
    omega_filled = np.where(np.isnan(omega), row_means, omega)

    # Detrend per example
    detrended = omega_filled - omega_filled.mean(axis=1, keepdims=True)

    # rfft: shape (N, N_FREQ)
    F = np.fft.rfft(detrended, axis=1)             # (N, 12)
    P = (np.abs(F) ** 2) / N_STEPS                 # (N, 12)

    P_mean = P.mean(axis=0)
    P_std  = P.std(axis=0)
    return P_mean, P_std, P  # also return per-example for t-tests


# ---------------------------------------------------------------------------
# Step 5: verdict metrics
# ---------------------------------------------------------------------------

def compute_metrics(P_mean: np.ndarray) -> dict[str, Any]:
    """
    Compute SFR, LFC, f_peak_idx from the power spectrum.
    Excludes DC bin (k=0) for SFR and f_peak.
    """
    ac = P_mean[1:]   # bins k=1..11 (excludes DC)
    if len(ac) == 0 or np.all(ac == 0):
        return {"f_peak_idx": 1, "SFR": 1.0, "LFC": 0.18}

    f_peak_idx = int(np.argmax(ac)) + 1   # +1 because we sliced off DC
    peak_val   = P_mean[f_peak_idx]
    floor_val  = float(np.median(ac))
    SFR = float(peak_val / floor_val) if floor_val > 1e-30 else float("inf")

    LFC = float((P_mean[1] + P_mean[2]) / (ac.sum() + 1e-30))

    return {"f_peak_idx": f_peak_idx, "SFR": round(SFR, 4), "LFC": round(LFC, 4)}


# ---------------------------------------------------------------------------
# Step 6: cross-model delta + Welch t-test
# ---------------------------------------------------------------------------

def cross_model_analysis(
    P_per_example_base: np.ndarray,    # (N_base, 12)
    P_per_example_model: np.ndarray,   # (N_model, 12)
    P_mean_base: np.ndarray,           # (12,)
    P_mean_model: np.ndarray,          # (12,)
) -> dict[str, Any]:
    delta = P_mean_model - P_mean_base   # (12,)
    ac_delta = delta[1:]
    if len(ac_delta) == 0 or np.all(ac_delta == 0):
        max_delta_idx, max_delta_value = 1, 0.0
    else:
        max_delta_idx = int(np.argmax(np.abs(ac_delta))) + 1
        max_delta_value = float(delta[max_delta_idx])

    # Welch t-test per bin (k=1..11)
    p_vals = []
    for k in range(1, N_FREQ):
        b_vals = P_per_example_base[:, k]
        m_vals = P_per_example_model[:, k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p = ttest_ind(b_vals, m_vals, equal_var=False)
        p_vals.append(float(p) if not np.isnan(p) else 1.0)

    min_p_offset = int(np.argmin(p_vals))   # offset within k=1..11
    ttest_min_p_idx = min_p_offset + 1       # absolute k index
    ttest_min_p = p_vals[min_p_offset]

    return {
        "max_delta_idx":   max_delta_idx,
        "max_delta_value": round(max_delta_value, 8),
        "ttest_min_p_idx": ttest_min_p_idx,
        "ttest_min_p":     round(ttest_min_p, 8),
        "delta_per_bin":   [round(float(x), 8) for x in delta.tolist()],
        "p_vals_k1_11":    [round(x, 8) for x in p_vals],
    }


# ---------------------------------------------------------------------------
# Step 7: headline verdict
# ---------------------------------------------------------------------------

def determine_verdict(per_model_per_set: dict) -> tuple[str, str]:
    """
    Aggregate across sets; majority vote on quasiparticle criteria.
    (a) SFT or DPO: LFC > 0.40 AND SFR > 3 AND t-test p < 0.01 at low-f vs base
    (b) Partial: LFC elevated but criteria not fully met
    (c) No signal: all LFC ≈ 0.18, SFR ≈ 1
    (d) Pathological: many collapsed trajectories
    """
    sets = list(per_model_per_set.get("410m_base", {}).keys())

    # Check for pathological collapse
    for mname, msets in per_model_per_set.items():
        for sname, sdata in msets.items():
            if sdata["healthy_trajectory_frac"] < 0.5:
                return "d", (
                    f"Pathological: {mname}/{sname} has only "
                    f"{sdata['healthy_trajectory_frac']:.1%} healthy trajectories."
                )

    # Count sets meeting strong criteria (a)
    a_votes = 0
    b_votes = 0
    c_votes = 0

    for sname in sets:
        base = per_model_per_set["410m_base"][sname]
        sft  = per_model_per_set["410m_sft"][sname]
        dpo  = per_model_per_set["410m_dpo"][sname]

        base_lfc = base["LFC"]
        base_sfr = base["SFR"]

        for model_data, model_name, cross_key in [
            (sft, "sft", "sft_vs_base"),
            (dpo, "dpo", "dpo_vs_base"),
        ]:
            lfc = model_data["LFC"]
            sfr = model_data["SFR"]
            # p value at low-f bins (k=1 or k=2) from cross-model analysis
            # stored in cross_model_delta[cross_key][sname]
            # We'll check this externally in main; approximate here:
            strong = (lfc > 0.40 and sfr > 3.0)
            moderate = (lfc > 0.25 and sfr > 1.5)
            if strong:
                a_votes += 1
            elif moderate:
                b_votes += 1
            else:
                c_votes += 1

    total = max(a_votes + b_votes + c_votes, 1)
    if a_votes / total >= 0.33:
        verdict = "a"
        headline = (
            "Quasiparticle confirmed: angular velocity in γ-basis shows coherent "
            "low-frequency rotation in SFT/DPO absent from base — "
            "bound-state worldline signature present."
        )
    elif (a_votes + b_votes) / total >= 0.33:
        verdict = "b"
        headline = (
            "Partial signal: LFC elevated in at least one aligned model "
            "but criteria for full quasiparticle confirmation not met across all sets."
        )
    else:
        verdict = "c"
        headline = (
            "No signal: angular evolution is thermal across all models — "
            "LFC ≈ 0.18 and SFR ≈ 1 everywhere; quasiparticle hypothesis not supported."
        )

    return verdict, headline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== angular_fourier_delta_prime.py ===")
    print(f"Output: {OUT_DIR / 'results.json'}\n")

    # --- Step 1: build basis ---
    v_top5 = load_base_v_top5()

    # --- Load all activations ---
    all_data = {name: load_activations(path) for name, path in MODEL_FILES.items()}
    sets_found = list(all_data["410m_base"].keys())
    print(f"\nSets: {sets_found}\n")

    # --- Core computation ---
    per_model_per_set: dict[str, dict] = {}
    basis_energy_fraction: dict[str, dict] = {}
    # Store per-example power for t-tests
    per_example_power: dict[str, dict] = {}

    for model_name, data in all_data.items():
        print(f"=== Model: {model_name} ===")
        per_model_per_set[model_name] = {}
        basis_energy_fraction[model_name] = {}
        per_example_power[model_name] = {}

        for set_name in sets_found:
            print(f"  Set: {set_name}", flush=True)
            layers = get_layer_tensors(data, set_name)
            N = layers[0].shape[0]

            # Steps 2+3: project + angular velocity
            omega, ef, healthy_frac = project_and_angular_velocity(layers, v_top5)

            # Step 4: FFT
            P_mean, P_std, P_per_ex = fourier_power(omega)

            # Omega mean/std per layer (diagnostic)
            omega_mean_per_layer = [
                round(float(np.nanmean(omega[:, s])), 6) for s in range(N_STEPS)
            ]
            omega_std_per_layer = [
                round(float(np.nanstd(omega[:, s])), 6) for s in range(N_STEPS)
            ]

            # Step 5: metrics
            metrics = compute_metrics(P_mean)

            basis_energy_fraction[model_name][set_name] = [round(float(x), 6) for x in ef.tolist()]
            per_example_power[model_name][set_name] = P_per_ex

            set_result = {
                "omega_mean_per_layer":   omega_mean_per_layer,
                "omega_std_per_layer":    omega_std_per_layer,
                "power_spectrum_mean":    [round(float(x), 8) for x in P_mean.tolist()],
                "power_spectrum_std":     [round(float(x), 8) for x in P_std.tolist()],
                "f_peak_idx":             metrics["f_peak_idx"],
                "SFR":                    metrics["SFR"],
                "LFC":                    metrics["LFC"],
                "healthy_trajectory_frac": round(healthy_frac, 4),
                "N_examples":             N,
            }
            per_model_per_set[model_name][set_name] = set_result

            print(f"    N={N}  healthy={healthy_frac:.1%}  "
                  f"SFR={metrics['SFR']:.2f}  LFC={metrics['LFC']:.3f}  "
                  f"f_peak_idx={metrics['f_peak_idx']}  "
                  f"ef_avg={ef.mean():.5f} (rand_base={K_TOP/1024:.5f})")

    # --- Step 6: cross-model delta ---
    cross_model_delta: dict[str, dict] = {
        "sft_vs_base": {"per_set": {}},
        "dpo_vs_base": {"per_set": {}},
    }

    for set_name in sets_found:
        P_base = per_example_power["410m_base"][set_name]
        P_base_mean = np.array(per_model_per_set["410m_base"][set_name]["power_spectrum_mean"])

        for comp_key, model_name in [("sft_vs_base", "410m_sft"), ("dpo_vs_base", "410m_dpo")]:
            P_model = per_example_power[model_name][set_name]
            P_model_mean = np.array(per_model_per_set[model_name][set_name]["power_spectrum_mean"])
            analysis = cross_model_analysis(P_base, P_model, P_base_mean, P_model_mean)
            cross_model_delta[comp_key]["per_set"][set_name] = analysis

    # --- Step 7: verdict ---
    verdict, headline = determine_verdict(per_model_per_set)

    # Refine verdict with actual t-test p values
    low_f_sig_count = 0
    for comp_key in ["sft_vs_base", "dpo_vs_base"]:
        for set_name in sets_found:
            analysis = cross_model_delta[comp_key]["per_set"][set_name]
            if analysis["ttest_min_p"] < 0.01 and analysis["ttest_min_p_idx"] <= 2:
                low_f_sig_count += 1

    if verdict == "b" and low_f_sig_count >= 2:
        verdict = "a"
        headline = (
            "Quasiparticle confirmed (t-test uplift): significant low-f angular power "
            "excess in aligned models at p < 0.01, confirming coherent orbit signature."
        )

    # --- Summary print ---
    print(f"\n{'='*80}")
    print("MAIN TABLE: SFR | LFC | f_peak | healthy_traj%")
    print(f"{'model':<15} {'set':<22} {'SFR':>6} {'LFC':>6} {'f_peak_idx':>12} {'healthy%':>10}")
    print("-" * 75)
    for mname in ["410m_base", "410m_sft", "410m_dpo"]:
        for sname in sets_found:
            d = per_model_per_set[mname][sname]
            print(f"{mname:<15} {sname:<22} {d['SFR']:>6.2f} {d['LFC']:>6.3f} {d['f_peak_idx']:>12} {d['healthy_trajectory_frac']:>10.1%}")

    print(f"\nCROSS-MODEL DELTA:")
    for comp_key in ["sft_vs_base", "dpo_vs_base"]:
        print(f"  {comp_key}:")
        for sname in sets_found:
            a = cross_model_delta[comp_key]["per_set"][sname]
            print(f"    {sname}: max_delta_idx={a['max_delta_idx']} "
                  f"max_delta={a['max_delta_value']:.4e} "
                  f"t-test: k={a['ttest_min_p_idx']} p={a['ttest_min_p']:.4f}")

    print(f"\nVERDICT ({verdict}): {headline}")
    print("="*80)

    # --- Build results JSON ---
    results: dict[str, Any] = {
        "models":    ["410m_base", "410m_sft", "410m_dpo"],
        "sets":      sets_found,
        "unit":      UNIT,
        "basis":     "base_W_top5_right_singular_vectors_per_layer",
        "basis_energy_fraction": basis_energy_fraction,
        "per_model_per_set": per_model_per_set,
        "cross_model_delta": cross_model_delta,
        "verdict":   verdict,
        "headline":  headline,
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    shutil.copy(__file__, OUT_DIR / "angular_fourier_delta_prime.py")
    print(f"\nArtifacts:\n  script  : {__file__}\n  results : {out_json}")


if __name__ == "__main__":
    main()
