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
    macro("bonusRKfiveOneBDPOavgSeeds", (r_dpo_1b_s42["bonus_R_k5"] + r_dpo_1b_s117["bonus_R_k5"]) / 2),
    macro("bonusRKfiveOneBCLMavgSeeds", (r_clm_1b_s42["bonus_R_k5"] + r_clm_1b_s117["bonus_R_k5"]) / 2),
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
    "% ── Cross-module γ (attention + MLP) ──────────────────",
]

mod_path = RES / "spectral_overlap_gamma_modules" / "results.json"
if mod_path.exists():
    mod = load(mod_path)
    # Per-run, per-module macros
    run_labels = {"410m_dpo": "FourTenMDPO", "410m_clm": "FourTenMCLM",
                  "1b_dpo": "OneBDPO", "1b_clm": "OneBCLM"}
    mod_labels = {"attention.query_key_value": "QKV",
                  "attention.dense": "AttnDense",
                  "mlp.dense_h_to_4h": "MLPUp",
                  "mlp.dense_4h_to_h": "MLPDown"}
    for run_key, run_tag in run_labels.items():
        if run_key not in mod["runs"]:
            continue
        for mod_key, mod_tag in mod_labels.items():
            if mod_key not in mod["runs"][run_key]["modules"]:
                continue
            avg = mod["runs"][run_key]["modules"][mod_key]["avg"]
            lines += [
                macro(f"modSrank{run_tag}{mod_tag}",      avg["srank"]),
                macro(f"modBonusRKfive{run_tag}{mod_tag}", avg["bonus_R_k5"]),
                macro(f"modBonusRKsrank{run_tag}{mod_tag}", avg["bonus_R_ksrank"]),
                macro(f"modBonusLKfive{run_tag}{mod_tag}", avg["bonus_L_k5"]),
            ]
    # Cross-run per-module averages
    for mod_key, mod_tag in mod_labels.items():
        sranks = [mod["runs"][r]["modules"][mod_key]["avg"]["srank"]
                  for r in mod["runs"] if mod_key in mod["runs"][r]["modules"]]
        bRs = [mod["runs"][r]["modules"][mod_key]["avg"]["bonus_R_k5"]
               for r in mod["runs"] if mod_key in mod["runs"][r]["modules"]]
        if sranks:
            lines += [
                macro(f"modAvgSrank{mod_tag}", sum(sranks)/len(sranks)),
                macro(f"modAvgBonusRKfive{mod_tag}", sum(bRs)/len(bRs)),
            ]
    # Peak bonus_R across all cells
    all_cells = [(r, m, mod["runs"][r]["modules"][m]["avg"]["bonus_R_ksrank"])
                 for r in mod["runs"] for m in mod["runs"][r]["modules"]]
    peak_run, peak_mod, peak_val = max(all_cells, key=lambda x: x[2])
    lines += [
        macro("modPeakBonusRKsrank", peak_val),
        macro("modPeakRun", peak_run.replace("_", " ")),
        macro("modPeakModule", peak_mod.replace("_", "\\_")),
    ]
    # srank range
    all_sranks = [c[2] for c in [(r, m, mod["runs"][r]["modules"][m]["avg"]["srank"])
                                  for r in mod["runs"] for m in mod["runs"][r]["modules"]]]
    lines += [
        macro("modSrankMin", min(all_sranks)),
        macro("modSrankMax", max(all_sranks)),
    ]
else:
    for label in ["FourTenMDPO", "FourTenMCLM", "OneBDPO", "OneBCLM"]:
        for m in ["QKV", "AttnDense", "MLPUp", "MLPDown"]:
            lines += [macro(f"modSrank{label}{m}", 0.0),
                      macro(f"modBonusRKfive{label}{m}", 0.0),
                      macro(f"modBonusRKsrank{label}{m}", 0.0),
                      macro(f"modBonusLKfive{label}{m}", 0.0)]

lines += [
    "",
    "% ── DPO-CLM orthogonal complement ─────────────────────",
]

# DPO-CLM orthogonal decomposition
orth_path = RES / "dpo_clm_orthogonal_decomp" / "results.json"
if orth_path.exists():
    orth = load(orth_path)
    comp = orth["comparison"]
    # Convert fractions to percentage points
    for k_name in ["k5", "k10", "k20", "k_srank"]:
        entry = comp.get(f"{k_name}_orthogonal_frac", comp.get(k_name, {}))
        if isinstance(entry, dict):
            delta = entry.get("dpo_minus_clm", 0.0)
            dpo_mean = entry.get("dpo_mean", 0.0)
            clm_mean = entry.get("clm_mean", 0.0)
            label_map = {"k5": "Kfive", "k10": "Kten", "k20": "Ktwenty", "k_srank": "Ksrank"}
            lines += [
                macro(f"orthDPOMean{label_map[k_name]}",  dpo_mean * 100),  # percentage
                macro(f"orthCLMMean{label_map[k_name]}",  clm_mean * 100),
                macro(f"orthDPOminusCLM{label_map[k_name]}", delta * 100),  # percentage points
            ]
    verdict = orth.get("verdict", "unknown")
    lines += [macro("orthVerdict", verdict.replace("_", " "))]
else:
    # Stubs if orthogonal decomp hasn't run yet — avoid undefined-macro LaTeX errors
    for label in ["Kfive", "Kten", "Ktwenty", "Ksrank"]:
        lines += [macro(f"orthDPOMean{label}", 0.0),
                  macro(f"orthCLMMean{label}", 0.0),
                  macro(f"orthDPOminusCLM{label}", 0.0)]
    lines += [macro("orthVerdict", "pending")]

lines += [
    "",
    "% ── Bias-theory autopsy ───────────────────────────────",
    macro("autopsyResidualMinPct", autopsy_residual_min * 100),
    macro("autopsyResidualMaxPct", autopsy_residual_max * 100),
    "",
    "% ── Early vs late layer split (per-scale) ────────────",
]

# Compute early/late-quarter averages per scale
def el_split_petri(model_key: str) -> dict:
    pl = petri[model_key]["per_layer"]
    n = len(pl); q = max(1, n // 4)
    return {
        "early_sr": statistics.mean(p["srank"] for p in pl[:q]),
        "late_sr":  statistics.mean(p["srank"] for p in pl[-q:]),
        "early_bR": statistics.mean(p["bonus_R_k5"] for p in pl[:q]),
        "late_bR":  statistics.mean(p["bonus_R_k5"] for p in pl[-q:]),
    }

def el_split_gamma(runs: dict, run_key: str) -> dict:
    pl = runs[run_key]["per_layer"]
    n = len(pl); q = max(1, n // 4)
    return {
        "early_sr": statistics.mean(p["srank_delta"] for p in pl[:q]),
        "late_sr":  statistics.mean(p["srank_delta"] for p in pl[-q:]),
        "early_bR": statistics.mean(p["k5"]["bonus_right"] for p in pl[:q]),
        "late_bR":  statistics.mean(p["k5"]["bonus_right"] for p in pl[-q:]),
    }

el_70m  = el_split_petri("pythia-70m")
el_160m = el_split_petri("pythia-160m")
el_410m = el_split_gamma(g410["runs"], "v2_dpo_r128")
el_1b   = el_split_gamma(g1b["runs"],  "v2_dpo_r128_1b")

for label, d in [("SeventyM", el_70m), ("OneSixtyM", el_160m),
                  ("FourTenM", el_410m), ("OneB", el_1b)]:
    lines += [
        macro(f"earlyLayerSrank{label}", d["early_sr"]),
        macro(f"lateLayerSrank{label}",  d["late_sr"]),
        macro(f"earlyLayerBonusR{label}", d["early_bR"]),
        macro(f"lateLayerBonusR{label}",  d["late_bR"]),
    ]

lines += [
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
