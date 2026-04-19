#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.40",
#   "safetensors>=0.4",
#   "numpy>=1.26",
#   "matplotlib>=3.8",
#   "huggingface_hub>=0.23",
# ]
# ///
"""T2.1 — Qwen2-1.5B full-weight delta signatures (SFT / DPO / total).

Models:
  base     : Qwen/Qwen2-1.5B
  sft      : Qwen/Qwen2-1.5B-Instruct
  dpo      : lewtun/qwen2-1.5B-ultrafeedback-online-dpo

Deltas:
  Δ_SFT   = W_Instruct − W_base
  Δ_DPO   = W_DPO − W_Instruct   (primary T2.1 target)
  Δ_total = W_DPO − W_base

Per matched param (q/k/v/o/gate/up/down proj):
  - stable_rank = ‖Δ‖_F² / ‖Δ‖_2²
  - gamma       = right-sv overlap (top-k Δ vs top-k W_base), k = min(128, d/4)
  - rel_fro     = ‖Δ‖_F / ‖W_base‖_F

Resume-safe: per-layer results flushed to per_layer.jsonl + state.json.
Final outputs: per_layer.json, summary.json, figH_qwen_delta.png.
"""
from __future__ import annotations

import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR  = SCRIPT_DIR.parent
OUT_DIR    = PAPER_DIR / "results" / "t21_qwen_fullweight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH   = OUT_DIR / "per_layer.jsonl"
STATE_PATH   = OUT_DIR / "state.json"
PER_LAYER_PATH = OUT_DIR / "per_layer.json"
SUMMARY_PATH = OUT_DIR / "summary.json"
FIG_PATH     = OUT_DIR / "figH_qwen_delta.png"

# ── Model IDs ─────────────────────────────────────────────────────────────────

MODEL_BASE    = "Qwen/Qwen2-1.5B"
MODEL_SFT     = "Qwen/Qwen2-1.5B-Instruct"
MODEL_DPO     = "lewtun/qwen2-1.5B-ultrafeedback-online-dpo"

# ── Target param patterns ──────────────────────────────────────────────────────

PARAM_RE = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    """Load model on CPU in fp32 and return state_dict with model freed."""
    from transformers import AutoModelForCausalLM  # lazy import for uv script
    log.info(f"Loading {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    log.info(f"Loaded {model_id}, {len(sd)} tensors")
    return sd


def stable_rank(t: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(t.float())
    fro_sq = (sv ** 2).sum().item()
    spec_sq = (sv[0].item()) ** 2
    return fro_sq / spec_sq if spec_sq > 0 else float("nan")


def gamma_overlap(delta: torch.Tensor, base: torch.Tensor, k: int) -> float:
    """Right-singular-subspace overlap: cosine² sum between top-k cols of V."""
    # SVD: delta → U_d S_d Vt_d,  base → U_b S_b Vt_b
    # We want: ‖Vt_b[:k] @ Vt_d[:k].T‖_F² / k
    d = min(delta.shape)
    k = min(k, d)
    _, _, Vt_d = torch.linalg.svd(delta.float(), full_matrices=False)
    _, _, Vt_b = torch.linalg.svd(base.float(), full_matrices=False)
    # Vt rows are right singular vectors; top-k right svs = Vt_b[:k], Vt_d[:k]
    cross = Vt_b[:k] @ Vt_d[:k].T  # (k, k)
    return (cross ** 2).sum().item() / k


def rel_fro(delta: torch.Tensor, base: torch.Tensor) -> float:
    return (delta.norm("fro") / base.norm("fro")).item()


def load_state() -> set[str]:
    if STATE_PATH.exists():
        s = json.loads(STATE_PATH.read_text())
        return set(s.get("done", []))
    return set()


def save_state(done: set[str]) -> None:
    STATE_PATH.write_text(json.dumps({"done": sorted(done)}, indent=2))


def append_jsonl(record: dict) -> None:
    with JSONL_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Main computation ──────────────────────────────────────────────────────────


def compute_all() -> None:
    done = load_state()

    # ── Load all three state dicts ─────────────────────────────────────────────
    log.info("=== Loading all three model state dicts ===")
    sd_base = load_state_dict(MODEL_BASE)
    sd_sft  = load_state_dict(MODEL_SFT)
    sd_dpo  = load_state_dict(MODEL_DPO)

    # ── Identify matched params ────────────────────────────────────────────────
    base_params = {k for k in sd_base if PARAM_RE.search(k)}
    sft_params  = {k for k in sd_sft  if PARAM_RE.search(k)}
    dpo_params  = {k for k in sd_dpo  if PARAM_RE.search(k)}
    matched = sorted(base_params & sft_params & dpo_params)
    log.info(f"Matched params: {len(matched)}")

    # ── Per-param loop ─────────────────────────────────────────────────────────
    for param_name in matched:
        if param_name in done:
            log.info(f"  SKIP (already done): {param_name}")
            continue

        W_base = sd_base[param_name]
        W_sft  = sd_sft[param_name]
        W_dpo  = sd_dpo[param_name]

        # Shape check
        if not (W_base.shape == W_sft.shape == W_dpo.shape):
            log.warning(f"  Shape mismatch for {param_name}: "
                        f"{W_base.shape} vs {W_sft.shape} vs {W_dpo.shape}")
            done.add(param_name)
            save_state(done)
            continue

        d_min = min(W_base.shape)
        k = min(128, max(1, d_min // 4))

        log.info(f"  {param_name}  shape={tuple(W_base.shape)}  k={k}")

        record: dict = {"param": param_name, "shape": list(W_base.shape), "k": k}

        for delta_name, W_a, W_b_for_ref in [
            ("sft",   W_sft,  W_base),   # Δ_SFT   = W_sft  − W_base
            ("dpo",   W_dpo,  W_sft),    # Δ_DPO   = W_dpo  − W_sft
            ("total", W_dpo,  W_base),   # Δ_total = W_dpo  − W_base
        ]:
            # Reference for rel_fro and gamma is always W_base
            delta = W_a.float() - W_b_for_ref.float()

            sr  = stable_rank(delta)
            gam = gamma_overlap(delta, W_base.float(), k)
            rf  = rel_fro(delta, W_base.float())

            record[delta_name] = {
                "stable_rank": round(sr, 4),
                "gamma":       round(gam, 6),
                "rel_fro":     round(rf, 6),
            }

            log.info(f"    Δ_{delta_name}: srank={sr:.1f}  γ={gam:.4f}  rel_fro={rf:.4f}")

            # Free delta immediately
            del delta
            gc.collect()

        append_jsonl(record)
        done.add(param_name)
        save_state(done)

    log.info("=== All params computed ===")

    # ── Free state dicts ───────────────────────────────────────────────────────
    del sd_base, sd_sft, sd_dpo
    gc.collect()


# ── Consolidate JSONL → per_layer.json ───────────────────────────────────────


def consolidate() -> list[dict]:
    records = []
    with JSONL_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    PER_LAYER_PATH.write_text(json.dumps(records, indent=2))
    log.info(f"Wrote {PER_LAYER_PATH} ({len(records)} records)")
    return records


# ── Summary stats ──────────────────────────────────────────────────────────────

ATT_RE = re.compile(r"(q_proj|k_proj|v_proj|o_proj)")
MLP_RE = re.compile(r"(gate_proj|up_proj|down_proj)")


def build_summary(records: list[dict]) -> None:
    from collections import defaultdict

    stats: dict[str, dict[str, list]] = {
        "sft":   {"attn_srank": [], "mlp_srank": [], "attn_gamma": [], "mlp_gamma": []},
        "dpo":   {"attn_srank": [], "mlp_srank": [], "attn_gamma": [], "mlp_gamma": []},
        "total": {"attn_srank": [], "mlp_srank": [], "attn_gamma": [], "mlp_gamma": []},
    }

    for rec in records:
        p = rec["param"]
        is_attn = bool(ATT_RE.search(p))
        is_mlp  = bool(MLP_RE.search(p))
        for delta_name in ("sft", "dpo", "total"):
            if delta_name not in rec:
                continue
            sr  = rec[delta_name]["stable_rank"]
            gam = rec[delta_name]["gamma"]
            if is_attn:
                stats[delta_name]["attn_srank"].append(sr)
                stats[delta_name]["attn_gamma"].append(gam)
            if is_mlp:
                stats[delta_name]["mlp_srank"].append(sr)
                stats[delta_name]["mlp_gamma"].append(gam)

    def agg(vals: list[float]) -> dict:
        if not vals:
            return {}
        a = np.array(vals)
        return {
            "mean":   round(float(np.mean(a)), 4),
            "median": round(float(np.median(a)), 4),
            "std":    round(float(np.std(a)), 4),
            "min":    round(float(np.min(a)), 4),
            "max":    round(float(np.max(a)), 4),
            "n":      len(vals),
        }

    summary = {
        "model_base": MODEL_BASE,
        "model_sft":  MODEL_SFT,
        "model_dpo":  MODEL_DPO,
        "n_params_matched": len(records),
        "notes": [
            "Full-weight (not LoRA); UltraFeedback (not hh-rlhf); Qwen2-1.5B (~1.5B, vs Pythia 70m–1B).",
            "Complementary evidence, not strict replication. Strict T2.1 (fresh LoRA-DPO on Qwen + hh-rlhf) queued separately.",
        ],
        "deltas": {},
    }

    for delta_name in ("sft", "dpo", "total"):
        s = stats[delta_name]
        summary["deltas"][delta_name] = {
            "attention": {
                "stable_rank": agg(s["attn_srank"]),
                "gamma":       agg(s["attn_gamma"]),
            },
            "mlp": {
                "stable_rank": agg(s["mlp_srank"]),
                "gamma":       agg(s["mlp_gamma"]),
            },
        }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {SUMMARY_PATH}")

    # ── Print key stats ──────────────────────────────────────────────────────
    log.info("=== KEY SUMMARY ===")
    for delta_name in ("sft", "dpo", "total"):
        d = summary["deltas"][delta_name]
        log.info(
            f"Δ_{delta_name}  attn srank={d['attention']['stable_rank'].get('mean','?'):.1f}"
            f"  mlp srank={d['mlp']['stable_rank'].get('mean','?'):.1f}"
            f"  attn γ={d['attention']['gamma'].get('mean','?'):.4f}"
            f"  mlp γ={d['mlp']['gamma'].get('mean','?'):.4f}"
        )


# ── Figure H ──────────────────────────────────────────────────────────────────


def build_figure(records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    delta_names  = ["sft", "dpo", "total"]
    delta_labels = ["Δ_SFT", "Δ_DPO", "Δ_total"]
    colors       = ["#4C72B0", "#DD8452", "#55A868"]

    def collect(key: str, sub: str) -> list[list[float]]:
        """Return list of lists (one per delta) of `sub` values from `key` group."""
        out = []
        for dn in delta_names:
            vals = []
            for rec in records:
                if dn not in rec:
                    continue
                p = rec["param"]
                if key == "attn" and not ATT_RE.search(p):
                    continue
                if key == "mlp" and not MLP_RE.search(p):
                    continue
                if key == "all":
                    pass
                vals.append(rec[dn][sub])
            out.append(vals)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Qwen2-1.5B Full-Weight Delta Signatures (T2.1)\n"
        "Base → SFT (Δ_SFT), SFT → DPO (Δ_DPO), Base → DPO (Δ_total)",
        fontsize=12,
    )

    panel_specs = [
        (axes[0, 0], "all",  "stable_rank", "Stable Rank — All Params"),
        (axes[0, 1], "all",  "gamma",       "γ (Right-SV Overlap) — All Params"),
        (axes[1, 0], "attn", "stable_rank", "Stable Rank — Attention (q/k/v/o)"),
        (axes[1, 1], "mlp",  "stable_rank", "Stable Rank — MLP (gate/up/down)"),
    ]

    for ax, key, sub, title in panel_specs:
        groups = collect(key, sub)
        parts = ax.violinplot(
            groups,
            positions=range(len(delta_names)),
            showmedians=True,
            showextrema=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(delta_names)))
        ax.set_xticklabels(delta_labels)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(sub.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=l, alpha=0.7)
        for c, l in zip(colors, delta_labels)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {FIG_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    compute_all()
    records = consolidate()
    build_summary(records)
    build_figure(records)
    log.info("Done.")
