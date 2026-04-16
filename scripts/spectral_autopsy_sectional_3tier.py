"""Sectional spectral autopsy — 3-tier unequal partition (β′ measurement).

Re-measures ΔW sectional leak on the operative partition that matches the
orbit-fraction grid used in the L-conservation claim:
    PS    33.3% (lowest variance, 341 channels)
    SRN   66.2% (middle,         678 channels)
    gauge  0.5% (highest,          5 channels)

Isotropic baseline for this partition:
    L_iso = 1 − (0.333² + 0.662² + 0.005²) ≈ 0.4508

Contrasts with β (4-equal-quartile, L_iso = 0.750).

Step 1: forward 100 hh-rlhf prompts through base Pythia-410m to get
        per-layer per-channel activation variance. Cached in
        results/spectral_autopsy_sectional_3tier/variance.pt.
Step 2: build 3-tier col/row blocks per layer from variance sort order.
Step 3: for each of 3 runs, compute per-layer sectional leak + SVD metrics.
Step 4: aggregate, print verdict, write results.json.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from safetensors import safe_open

# Patch transformers' torch.load safety check: the base Pythia-410m weights are
# stored as a .bin file, which triggers a torch>=2.6 guard in transformers 5.5+.
# These are locally trusted files; patch is safe here.
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "pythia-410m"
RESULTS   = ROOT / "results" / "_leak"
OUT_DIR   = ROOT / "results" / "spectral_autopsy_sectional_3tier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAR_CACHE = OUT_DIR / "variance.pt"

RUNS = [
    ("v1_dpo_r16",  RESULTS / "checkpoints"       / "checkpoint-800",  16,  32),
    ("v2_dpo_r128", RESULTS / "v2" / "checkpoints" / "checkpoint-800", 128, 256),
    ("v3_clm_r128", RESULTS / "v3" / "checkpoints" / "checkpoint-800", 128, 256),
]

# 3-tier partition fractions
P_PS    = 0.333   # bottom (lowest variance) — dormant channels
P_SRN   = 0.662   # middle                  — active (SRN)
P_GAUGE = 0.005   # top  (highest variance) — gauge channels

D_MODEL  = 1024
N_LAYERS = 24

# n_bot / n_mid / n_top must sum to D_MODEL = 1024
N_BOT = max(1, round(P_PS    * D_MODEL))   # 341
N_TOP = max(1, round(P_GAUGE * D_MODEL))   #   5
N_MID = D_MODEL - N_BOT - N_TOP           # 678

assert N_BOT + N_MID + N_TOP == D_MODEL, f"{N_BOT}+{N_MID}+{N_TOP} != {D_MODEL}"

L_ISO_3TIER    = 1.0 - (P_PS**2 + P_SRN**2 + P_GAUGE**2)
L_ISO_4QUARTILE = 1.0 - 4 * (0.25**2)   # 0.750

VERDICT_TOL = 0.02  # ±0.02 around L_iso_3tier for isotropy zone


# ── Step 1: variance capture ────────────────────────────────────────────────

def load_hh_rlhf_prompts(tokenizer, n: int = 100, max_length: int = 256):
    from datasets import load_dataset
    log.info("Loading hh-rlhf prompts...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)
    except Exception as e:
        log.warning(f"hh-rlhf failed ({e}), using trl fallback")
        ds = load_dataset("trl-internal-testing/hh-rlhf-helpful-base", split="train")

    ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))

    def extract_prompt(ex):
        text = ex.get("chosen", ex.get("prompt", ""))
        sep = "\n\nAssistant:"
        if sep in text:
            text = text.rsplit(sep, 1)[0] + sep
        return text[:512]

    prompts = [extract_prompt(ex) for ex in ds]
    tokenized = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", max_length=max_length,
                        truncation=True, padding=False)
        tokenized.append(ids["input_ids"])   # (1, seq_len)
    return tokenized


def capture_variance(n_samples: int = 100) -> dict[int, torch.Tensor]:
    """Forward pass on base model. Returns {layer_idx: var_tensor[1024]}."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading base model from {MODEL_DIR}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = model.gpt_neox.layers
    n_layers = len(layers)
    d_model  = layers[0].attention.query_key_value.weight.shape[1]
    log.info(f"Architecture: {n_layers} layers, d_model={d_model}")

    # Welford accumulators (float32 on CPU)
    sum_acts  = [torch.zeros(d_model) for _ in range(n_layers)]
    sum2_acts = [torch.zeros(d_model) for _ in range(n_layers)]
    n_tokens  = [0] * n_layers

    def make_hook(i):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs_cpu = hs.detach().float().squeeze(0).cpu()   # (T, d)
            sum_acts[i]  += hs_cpu.sum(0)
            sum2_acts[i] += (hs_cpu ** 2).sum(0)
            n_tokens[i]  += hs_cpu.shape[0]
        return hook

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(layers)]

    prompts = load_hh_rlhf_prompts(tokenizer, n=n_samples)
    log.info(f"Running {len(prompts)} forward passes...")
    with torch.no_grad():
        for idx, ids in enumerate(prompts):
            if idx % 20 == 0:
                log.info(f"  prompt {idx+1}/{len(prompts)}")
            try:
                model(ids.to(device), use_cache=False)
            except Exception as e:
                log.warning(f"  prompt {idx} failed: {e}")

    for h in hooks:
        h.remove()

    log.info("Computing per-channel variance...")
    variances = {}
    for i in range(n_layers):
        N = n_tokens[i]
        if N == 0:
            log.warning(f"Layer {i}: no tokens, using zeros")
            variances[i] = torch.zeros(d_model)
            continue
        mean = sum_acts[i] / N
        var  = sum2_acts[i] / N - mean ** 2
        variances[i] = var.clamp(min=0.0)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return variances


# ── Step 2: build 3-tier blocks ─────────────────────────────────────────────

def build_3tier_blocks(var: torch.Tensor, d_rows: int) -> tuple[list, list]:
    """
    Returns (col_blocks, row_blocks) — each a list of 3 index tensors.
    col ordering: [PS, SRN, gauge] by ascending variance.
    row ordering: proportional split of d_rows same ordering.
    """
    order = torch.argsort(var)   # ascending
    col_blocks = [
        order[:N_BOT],
        order[N_BOT:N_BOT + N_MID],
        order[N_BOT + N_MID:],
    ]
    # Row blocks: split d_rows proportionally to col widths
    widths = [N_BOT, N_MID, N_TOP]   # col widths, sum=1024
    row_blocks = []
    running = 0
    for j, w in enumerate(widths):
        chunk = int(round(w / D_MODEL * d_rows))
        if j == len(widths) - 1:
            chunk = d_rows - running  # absorb rounding remainder
        end = running + chunk
        row_blocks.append(torch.arange(running, min(end, d_rows)))
        running += chunk
    # Any leftover (shouldn't happen after last block fix, but guard)
    if running < d_rows:
        row_blocks[-1] = torch.cat([row_blocks[-1],
                                    torch.arange(running, d_rows)])
    return col_blocks, row_blocks


# ── SVD metrics (from spectral_autopsy_sectional.py) ────────────────────────

def svd_metrics(W: torch.Tensor) -> tuple[float, float, int]:
    """(frobenius, stable_rank, k90) — W computed in float32."""
    if W.abs().sum().item() == 0.0:
        return 0.0, 0.0, 0
    s = torch.linalg.svdvals(W.float())
    s_sq  = s ** 2
    total = s_sq.sum().item()
    frob  = total ** 0.5
    spec  = s[0].item()
    srank = total / (spec ** 2) if spec > 0 else 0.0
    cumsum = torch.cumsum(s_sq, 0) / total
    idx = (cumsum >= 0.90).nonzero()
    k90 = int(idx[0].item()) + 1 if len(idx) > 0 else int(s.numel())
    return frob, srank, k90


# ── Step 3: adapter loading + sectional leak ────────────────────────────────

def load_adapter(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def extract_qkv_delta(tensors: dict, layer: int, alpha: int, r: int) -> torch.Tensor:
    a_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_A.weight")
    b_key = (f"base_model.model.gpt_neox.layers.{layer}"
             f".attention.query_key_value.lora_B.weight")
    A = tensors[a_key].float()
    B = tensors[b_key].float()
    return (alpha / r) * (B @ A)   # [3072, 1024]


@dataclass
class SectionalStats3:
    layer: int
    frob_on:    float
    frob_off:   float
    frob_total: float
    srank_on:   float
    srank_off:  float
    k90_on:     int
    k90_off:    int
    leak_fraction: float   # frob_off² / frob_total²


def analyze_layer_3tier(dw: torch.Tensor,
                        col_blocks: list, row_blocks: list) -> SectionalStats3:
    layer_idx = -1  # caller sets
    d_rows, d_cols = dw.shape

    # Build on-block mask
    mask = torch.zeros((d_rows, d_cols), dtype=torch.bool)
    for rb, cb in zip(row_blocks, col_blocks):
        mask[rb.unsqueeze(1), cb.unsqueeze(0)] = True

    dw_on  = dw *   mask
    dw_off = dw * (~mask)

    frob_on,  sr_on,  k90_on  = svd_metrics(dw_on)
    frob_off, sr_off, k90_off = svd_metrics(dw_off)
    frob_total = (frob_on**2 + frob_off**2) ** 0.5
    leak = (frob_off**2) / (frob_total**2) if frob_total > 0 else 0.0

    return SectionalStats3(
        layer=layer_idx,
        frob_on=frob_on, frob_off=frob_off, frob_total=frob_total,
        srank_on=sr_on, srank_off=sr_off, k90_on=k90_on, k90_off=k90_off,
        leak_fraction=leak,
    )


# ── Step 4: aggregate + verdict ──────────────────────────────────────────────

def avg(seq): return sum(seq) / len(seq) if seq else float('nan')


def determine_verdict(run_leaks: dict[str, float]) -> str:
    iso = L_ISO_3TIER
    tol = VERDICT_TOL
    verdicts = []
    for v in run_leaks.values():
        if abs(v - iso) < tol:
            verdicts.append('a')
        elif v < iso - tol:
            verdicts.append('b')
        else:
            verdicts.append('c')
    if len(set(verdicts)) == 1:
        return verdicts[0]
    return 'mixed'


VERDICT_TEXT = {
    'a': ("orthogonality DEAD: ΔW is isotropic on the 3-tier grid too; "
          "L-conservation is arithmetic isotropy on every partition."),
    'b': ("orthogonality RESURRECTED: ΔW avoids off-block on 3-tier grid; "
          "gradient is directionally orthogonal to leak operator on the operative partition."),
    'c': ("ANTI-orthogonal: ΔW prefers off-block on the operative 3-tier grid."),
    'mixed': "Mixed per-run verdict — see individual run results.",
}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Step 1 — variance (cached)
    if VAR_CACHE.exists():
        log.info(f"Loading cached variance from {VAR_CACHE}")
        variances = torch.load(str(VAR_CACHE), map_location="cpu")
    else:
        variances = capture_variance(n_samples=100)
        torch.save(variances, str(VAR_CACHE))
        log.info(f"Saved variance cache → {VAR_CACHE}")

    # Step 2 — build per-layer blocks (d_rows=3072 for QKV)
    D_ROWS_QKV = 3072
    per_layer_blocks = {}
    for li in range(N_LAYERS):
        var = variances[li]
        col_blocks, row_blocks = build_3tier_blocks(var, D_ROWS_QKV)
        per_layer_blocks[li] = (col_blocks, row_blocks)

    # Step 3 — iterate runs
    all_run_stats: dict[str, list[SectionalStats3]] = {}
    run_avg_leaks: dict[str, float] = {}

    for run_name, ckpt, r, alpha in RUNS:
        print(f"\n{'='*80}\n{run_name}  r={r}  alpha={alpha}\n{'='*80}")
        if not ckpt.exists():
            print(f"  MISSING checkpoint: {ckpt}")
            continue
        tensors = load_adapter(ckpt)
        per_layer: list[SectionalStats3] = []
        for li in range(N_LAYERS):
            dw = extract_qkv_delta(tensors, li, alpha, r)
            col_blocks, row_blocks = per_layer_blocks[li]
            s = analyze_layer_3tier(dw, col_blocks, row_blocks)
            s.layer = li
            per_layer.append(s)
            if li % 6 == 0:
                log.info(f"  layer {li:2d}: leak={s.leak_fraction:.4f}  "
                         f"frob_on={s.frob_on:.4f}  frob_off={s.frob_off:.4f}")

        all_run_stats[run_name] = per_layer

        avg_leak     = avg([x.leak_fraction for x in per_layer])
        avg_sr_on    = avg([x.srank_on      for x in per_layer])
        avg_sr_off   = avg([x.srank_off     for x in per_layer])
        avg_frob_on  = avg([x.frob_on       for x in per_layer])
        avg_frob_off = avg([x.frob_off      for x in per_layer])
        ratio        = avg_frob_off / avg_frob_on if avg_frob_on > 0 else float('nan')
        delta_iso    = avg_leak - L_ISO_3TIER

        run_avg_leaks[run_name] = avg_leak

        print(f"  leak_fraction (frob_off²/total²) : {avg_leak:.4f}  "
              f"[L_iso(3-tier)={L_ISO_3TIER:.4f}  Δ={delta_iso:+.4f}]")
        print(f"  srank_on = {avg_sr_on:6.2f}   srank_off = {avg_sr_off:6.2f}")
        print(f"  frob_on  = {avg_frob_on:.4f}   frob_off  = {avg_frob_off:.4f}   ratio off/on = {ratio:.4f}")

    # Step 4 — verdict
    verdict = determine_verdict(run_avg_leaks)

    print(f"\n{'='*80}")
    print("HEADLINE — β′: ΔW sectional leak on 3-tier partition")
    print(f"{'='*80}")
    print(f"Partition: PS={P_PS}  SRN={P_SRN}  gauge={P_GAUGE}")
    print(f"L_iso(3-tier)     = {L_ISO_3TIER:.4f}")
    print(f"L_iso(4-quartile) = {L_ISO_4QUARTILE:.4f}")
    print()
    print(f"{'run':<16}  {'leak':>8}  {'Δ_iso_3tier':>12}  {'srank_on':>9}  {'srank_off':>10}  {'ratio_off/on':>13}")
    print("-"*80)
    for run_name, per_layer in all_run_stats.items():
        lk  = avg([x.leak_fraction for x in per_layer])
        son = avg([x.srank_on      for x in per_layer])
        sof = avg([x.srank_off     for x in per_layer])
        fon = avg([x.frob_on       for x in per_layer])
        fof = avg([x.frob_off      for x in per_layer])
        r   = fof / fon if fon > 0 else float('nan')
        dlt = lk - L_ISO_3TIER
        print(f"  {run_name:<14}  {lk:>8.4f}  {dlt:>+12.4f}  {son:>9.2f}  {sof:>10.2f}  {r:>13.4f}")
    print()
    print(f"VERDICT ({verdict}): {VERDICT_TEXT[verdict]}")
    print()

    # Build results JSON
    runs_json = {}
    for run_name, per_layer in all_run_stats.items():
        avg_leak     = avg([x.leak_fraction for x in per_layer])
        avg_sr_on    = avg([x.srank_on      for x in per_layer])
        avg_sr_off   = avg([x.srank_off     for x in per_layer])
        avg_frob_on  = avg([x.frob_on       for x in per_layer])
        avg_frob_off = avg([x.frob_off      for x in per_layer])
        runs_json[run_name] = {
            "per_layer": [asdict(s) for s in per_layer],
            "avg": {
                "leak_fraction":   avg_leak,
                "srank_on":        avg_sr_on,
                "srank_off":       avg_sr_off,
                "frob_on":         avg_frob_on,
                "frob_off":        avg_frob_off,
                "ratio_off_on":    avg_frob_off / avg_frob_on if avg_frob_on > 0 else None,
                "delta_iso_3tier": avg_leak - L_ISO_3TIER,
            },
        }

    results = {
        "partition": {"PS": P_PS, "SRN": P_SRN, "gauge": P_GAUGE},
        "n_bot": N_BOT,
        "n_mid": N_MID,
        "n_top": N_TOP,
        "L_iso_3tier":    L_ISO_3TIER,
        "L_iso_4quartile": L_ISO_4QUARTILE,
        "runs": runs_json,
        "verdict": verdict,
        "headline": VERDICT_TEXT[verdict],
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    shutil.copy(__file__, OUT_DIR / "spectral_autopsy_sectional_3tier.py")
    log.info(f"Wrote: {out_json}")
    print(f"Artifacts:\n  script  : {__file__}\n  results : {out_json}")
    if VAR_CACHE.exists():
        print(f"  variance: {VAR_CACHE}")


if __name__ == "__main__":
    main()
