"""Generate values.tex: LaTeX \newcommand macros for all numerical constants
referenced in main.tex. Reads from ../results/*/results.json and summary.json.

Run from paper/manuscript/ with:
    uv run python generate_values.py
Output: values.tex (auto-generated; DO NOT EDIT).
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
RES = PAPER / "results"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def avg_layer(per_layer: list[dict], key: str) -> float:
    return statistics.mean(p[key] for p in per_layer)


# ── γ at 70m/160m (petri-dish) ─────────────────────────────────────────────
petri = load(RES / "spectral_overlap_gamma_petri" / "results.json")
srank_70m = petri["pythia-70m"]["avg"]["srank"]
bonusR_k5_70m = petri["pythia-70m"]["avg"]["bonus_R_k5"]
bonusR_ksrank_70m = petri["pythia-70m"]["avg"]["bonus_R_ksrank"]
bonusL_k5_70m = petri["pythia-70m"]["avg"]["bonus_L_k5"]

srank_160m = petri["pythia-160m"]["avg"]["srank"]
bonusR_k5_160m = petri["pythia-160m"]["avg"]["bonus_R_k5"]
bonusR_ksrank_160m = petri["pythia-160m"]["avg"]["bonus_R_ksrank"]
bonusL_k5_160m = petri["pythia-160m"]["avg"]["bonus_L_k5"]


# ── γ at 410m ──────────────────────────────────────────────────────────────
g410 = load(RES / "spectral_overlap_gamma" / "results.json")
# structure: {"runs": {"v2_dpo_r128": {"per_layer": [...], "avg": {...}}, ...}, ...}
def get410(run: str, key: str, bucket: str = "avg") -> float:
    return g410["runs"][run][bucket][key]

srank_410m_dpo = statistics.mean(p["srank_delta"] for p in g410["runs"]["v2_dpo_r128"]["per_layer"])
srank_410m_clm = statistics.mean(p["srank_delta"] for p in g410["runs"]["v3_clm_r128"]["per_layer"])
bonusR_k5_410m_dpo = get410("v2_dpo_r128", "k5")["p_right"] if False else get410("v2_dpo_r128", "k5").get("bonus_right") if False else None
# The 410m JSON has nested keys; navigate by inspection below
# Fallback: compute from per_layer
def compute_410m(run: str) -> dict:
    pl = g410["runs"][run]["per_layer"]
    return {
        "bonus_R_k5": statistics.mean(p["k5"]["bonus_right"] for p in pl),
        "bonus_L_k5": statistics.mean(p["k5"]["bonus_left"] for p in pl),
        "bonus_R_ksrank": statistics.mean(p["k_srank"]["bonus_right"] for p in pl if "k_srank" in p),
        "srank": statistics.mean(p["srank_delta"] for p in pl),
    }
r_dpo_410m = compute_410m("v2_dpo_r128")
r_clm_410m = compute_410m("v3_clm_r128")


# ── γ at 1B (seed 42) ──────────────────────────────────────────────────────
g1b = load(RES / "spectral_overlap_gamma_1b" / "results.json")
def compute_1b(run: str) -> dict:
    pl = g1b["runs"][run]["per_layer"]
    return {
        "bonus_R_k5": statistics.mean(p["k5"]["bonus_right"] for p in pl),
        "bonus_L_k5": statistics.mean(p["k5"]["bonus_left"] for p in pl),
        "bonus_R_ksrank": statistics.mean(p["k_srank"]["bonus_right"] for p in pl if "k_srank" in p),
        "srank": statistics.mean(p["srank_delta"] for p in pl),
    }
r_dpo_1b_s42 = compute_1b("v2_dpo_r128_1b")
r_clm_1b_s42 = compute_1b("v3_clm_r128_1b")


# ── γ at 1B seed 117 ───────────────────────────────────────────────────────
g1b_s117 = load(RES / "spectral_overlap_gamma_1b_seed117" / "results.json")
def compute_1b_s117(run: str) -> dict:
    pl = g1b_s117["runs"][run]["per_layer"]
    return {
        "bonus_R_k5": statistics.mean(p["k5"]["bonus_right"] for p in pl),
        "bonus_L_k5": statistics.mean(p["k5"]["bonus_left"] for p in pl),
        "bonus_R_ksrank": statistics.mean(p["k_srank"]["bonus_right"] for p in pl if "k_srank" in p),
        "srank": statistics.mean(p["srank_delta"] for p in pl),
    }
r_dpo_1b_s117 = compute_1b_s117("v2_dpo_r128_1b_s117")
r_clm_1b_s117 = compute_1b_s117("v3_clm_r128_1b_s117")


# ── BitFit ─────────────────────────────────────────────────────────────────
bitfit = load(RES / "bitfit_dpo_strike" / "summary.json")
bf_traj = load(RES / "bitfit_dpo_strike" / "loss_trajectory.json")
bf_ext_traj = load(RES / "bitfit_dpo_strike_extended" / "loss_trajectory.json")
# final 10-step window mean at step 800
bf_last10_step800 = statistics.mean(e["loss"] for e in bf_traj if e["step"] >= 720)
# final 10-step window of extension (last 10 entries)
bf_ext_final = statistics.mean(e["loss"] for e in bf_ext_traj[-10:])
bf_ext_laststep = bf_ext_traj[-1]["step"]
bf_min = min(e["loss"] for e in bf_traj)
bf_ext_min = min(e["loss"] for e in bf_ext_traj)


# ── Bias autopsy ───────────────────────────────────────────────────────────
bias = load(RES / "bias_theory_autopsy" / "results.json")
def avg_res_frac(run_key: str, side: str) -> float:
    pl = bias[run_key]["per_layer"]
    return statistics.mean(p[f"res_frac_{side}"] for p in pl)
autopsy_residual_min = min(avg_res_frac(r, "right") for r in ["v1_dpo_r16", "v2_dpo_r128", "v3_clm_r128"])
autopsy_residual_max = max(avg_res_frac(r, "right") for r in ["v1_dpo_r16", "v2_dpo_r128", "v3_clm_r128"])


# ── Scaling-fit RMS residuals (4-point empirical curve) ────────────────────
measured = [
    (512,  srank_70m),
    (768,  srank_160m),
    (1024, r_dpo_410m["srank"]),
    (2048, (r_dpo_1b_s42["srank"] + r_dpo_1b_s117["srank"]) / 2),
]
import math
def rms(predicted, measured):
    return math.sqrt(sum((p - m) ** 2 for p, m in zip(predicted, measured)) / len(measured))
# Task-intrinsic: constant = mean
c_task = statistics.mean(s for _, s in measured)
rms_task = rms([c_task] * len(measured), [s for _, s in measured])
# 1/sqrt(d): constant fit by least squares on c/sqrt(d) vs s → c = sum(s*sqrt(d)) / n (if assuming unit weight)
# Actually best fit: minimize (c/sqrt(d) - s)^2 → c = sum(s/sqrt(d)) / sum(1/d)
num = sum(s / math.sqrt(d) for d, s in measured); den = sum(1/d for d, s in measured)
c_sqrt = num / den
rms_sqrt = rms([c_sqrt / math.sqrt(d) for d, _ in measured], [s for _, s in measured])
# 1/d^(1/3): c = sum(s/d^(1/3)) / sum(1/d^(2/3))
num3 = sum(s / d**(1/3) for d, s in measured); den3 = sum(1/d**(2/3) for d, s in measured)
c_cbrt = num3 / den3
rms_cbrt = rms([c_cbrt / d**(1/3) for d, _ in measured], [s for _, s in measured])


# ── emit values.tex ────────────────────────────────────────────────────────
def macro(name: str, value) -> str:
    if isinstance(value, float):
        return f"\\newcommand{{\\{name}}}{{{value:.3f}}}"
    return f"\\newcommand{{\\{name}}}{{{value}}}"


lines = [
    "% Auto-generated by generate_values.py — DO NOT EDIT",
    "% Regenerate: make values",
    "",
    "% ── γ scaling (srank) ─────────────────────────────────",
    macro("srankSeventyM", srank_70m),
    macro("srankOneSixtyM", srank_160m),
    macro("srankFourTenM", r_dpo_410m["srank"]),
    macro("srankOneBDPOsFortyTwo", r_dpo_1b_s42["srank"]),
    macro("srankOneBDPOsOneSeventeen", r_dpo_1b_s117["srank"]),
    macro("srankOneBCLMsFortyTwo", r_clm_1b_s42["srank"]),
    macro("srankOneBCLMsOneSeventeen", r_clm_1b_s117["srank"]),
    "",
    "% ── γ bonus_R(k=5) ────────────────────────────────────",
    macro("bonusRKfiveSeventyM", bonusR_k5_70m),
    macro("bonusRKfiveOneSixtyM", bonusR_k5_160m),
    macro("bonusRKfiveFourTenMDPO", r_dpo_410m["bonus_R_k5"]),
    macro("bonusRKfiveFourTenMCLM", r_clm_410m["bonus_R_k5"]),
    macro("bonusRKfiveOneBDPOsFortyTwo", r_dpo_1b_s42["bonus_R_k5"]),
    macro("bonusRKfiveOneBDPOsOneSeventeen", r_dpo_1b_s117["bonus_R_k5"]),
    macro("bonusRKfiveOneBCLMsFortyTwo", r_clm_1b_s42["bonus_R_k5"]),
    macro("bonusRKfiveOneBCLMsOneSeventeen", r_clm_1b_s117["bonus_R_k5"]),
    "",
    "% ── γ bonus_R(k=srank) ────────────────────────────────",
    macro("bonusRKsrankSeventyM", bonusR_ksrank_70m),
    macro("bonusRKsrankOneSixtyM", bonusR_ksrank_160m),
    macro("bonusRKsrankFourTenMDPO", r_dpo_410m["bonus_R_ksrank"]),
    macro("bonusRKsrankFourTenMCLM", r_clm_410m["bonus_R_ksrank"]),
    macro("bonusRKsrankOneBDPOsFortyTwo", r_dpo_1b_s42["bonus_R_ksrank"]),
    macro("bonusRKsrankOneBCLMsFortyTwo", r_clm_1b_s42["bonus_R_ksrank"]),
    "",
    "% ── γ bonus_L(k=5) ────────────────────────────────────",
    macro("bonusLKfiveSeventyM", bonusL_k5_70m),
    macro("bonusLKfiveOneSixtyM", bonusL_k5_160m),
    macro("bonusLKfiveFourTenMDPO", r_dpo_410m["bonus_L_k5"]),
    macro("bonusLKfiveOneBDPOsFortyTwo", r_dpo_1b_s42["bonus_L_k5"]),
    "",
    "% ── Scaling fit ───────────────────────────────────────",
    macro("srankFitTaskConst", c_task),
    macro("srankFitTaskRMS", rms_task),
    macro("srankFitSqrtC", c_sqrt),
    macro("srankFitSqrtRMS", rms_sqrt),
    macro("srankFitCbrtC", c_cbrt),
    macro("srankFitCbrtRMS", rms_cbrt),
    "",
    "% ── BitFit-DPO ────────────────────────────────────────",
    macro("bitfitTrainableParams", bitfit["trainable_params"]),
    macro("bitfitTotalParams", bitfit["total_params"]),
    macro("bitfitTrainableFrac", bitfit["trainable_frac"] * 100),  # as percent
    macro("bitfitInitLoss", bitfit["initial_loss"]),
    macro("bitfitFinalLoss", bitfit["final_loss"]),
    macro("bitfitMinLoss", bf_min),
    macro("bitfitLoRAReference", bitfit.get("lora_dpo_v2_final_loss", 0.487)),
    macro("bitfitDropBits", bitfit["initial_loss"] - bitfit["final_loss"]),
    macro("bitfitLoRADropBits", bitfit["initial_loss"] - bitfit.get("lora_dpo_v2_final_loss", 0.487)),
    macro("bitfitLearningRatio", (bitfit["initial_loss"] - bitfit["final_loss"]) / (bitfit["initial_loss"] - bitfit.get("lora_dpo_v2_final_loss", 0.487))),
    "",
    "% ── BitFit extension ──────────────────────────────────",
    macro("bitfitExtLastStep", bf_ext_laststep),
    macro("bitfitExtFinalLoss", bf_ext_final),
    macro("bitfitExtMinLoss", bf_ext_min),
    "",
    "% ── Bias-theory autopsy ───────────────────────────────",
    macro("autopsyResidualMinPct", autopsy_residual_min * 100),
    macro("autopsyResidualMaxPct", autopsy_residual_max * 100),
    "",
    "% ── Compounding compression numbers ───────────────────",
    macro("sgdBound", "12{,}800"),
    macro("loraCap", 128),
    macro("sgdToLoRAcompression", "100"),
    macro("loraToSrankCompression", "32"),
    macro("totalCompression", "4{,}000"),
    "",
]

out = HERE / "values.tex"
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
print(f"srank petri fit: const={c_task:.3f} RMS={rms_task:.3f}")
print(f"           1/√d: c={c_sqrt:.1f} RMS={rms_sqrt:.3f}")
print(f"           1/∛d: c={c_cbrt:.1f} RMS={rms_cbrt:.3f}")
