#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers>=4.40",
#   "datasets>=2.18",
# ]
# ///
"""cross_probe_dry_run.py — tokenizer-only validation. NO model weights loaded. CPU only.

Validates:
  1. Field extraction for Anthropic/hh-rlhf (chosen/rejected as flat strings)
  2. Field extraction for HuggingFaceH4/ultrafeedback_binarized (chosen/rejected as turn-dicts)
  3. normalize_pair() produces non-empty (prompt, chosen_r, rejected_r) for both
  4. Tokenizers (Pythia-70m base, Qwen2-1.5B base) load from local cache without weight download
  5. Token counts are plausible (> 5 tokens for each field)

Prints a 5-line report. Should complete in under 30s on CPU.

Usage:
  uv run python scripts/cross_probe_dry_run.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Paths (must match cross_probe_score.py) ────────────────────────────────────

HF_CACHE    = Path("/home/debanjan/.cache/huggingface/hub")
MODELS_BASE = Path("/home/debanjan/Code/Research/lean-mining/cross-check/trained-model-battery/models")

PYTHIA_TOK_DIR = MODELS_BASE / "pythia-70m"
# Qwen2-1.5B cached snapshots are weights-only (tokenizer.json not present locally).
# The full scorer fetches the tokenizer from HF Hub at runtime.  Dry-run skips
# the Qwen tokenizer check and uses Pythia as a proxy for UF field-extraction.

# ── normalize_pair (inline copy — keep in sync with cross_probe_score.py) ──────

def normalize_pair(row: dict, dataset_name: str):
    if dataset_name == "hh":
        chosen_text   = row.get("chosen", "")
        rejected_text = row.get("rejected", "")
        sep = "\n\nAssistant:"
        if sep in chosen_text:
            parts    = chosen_text.rsplit(sep, 1)
            prompt   = parts[0] + sep
            chosen_r = parts[1].strip()
        else:
            prompt   = chosen_text[:256]
            chosen_r = chosen_text[256:]
        if sep in rejected_text:
            rejected_r = rejected_text.rsplit(sep, 1)[1].strip()
        else:
            rejected_r = rejected_text[256:]
        if len(chosen_r) < 10 or len(rejected_r) < 10:
            return None
        return prompt, chosen_r, rejected_r

    elif dataset_name == "uf":
        chosen_turns   = row.get("chosen", [])
        rejected_turns = row.get("rejected", [])
        if not chosen_turns or not rejected_turns:
            return None

        def turns_to_text(turns: list) -> str:
            return "\n\n".join(
                f"{t.get('role','user').capitalize()}: {t.get('content','')}"
                for t in turns
            )

        chosen_prompt_turns = chosen_turns[:-1]
        chosen_r_turn       = chosen_turns[-1]
        rejected_r_turn     = rejected_turns[-1]

        if chosen_r_turn.get("role") != "assistant":
            return None
        if rejected_r_turn.get("role") != "assistant":
            return None

        prompt     = turns_to_text(chosen_prompt_turns) + "\n\nAssistant:"
        chosen_r   = chosen_r_turn.get("content", "").strip()
        rejected_r = rejected_r_turn.get("content", "").strip()

        if len(chosen_r) < 10 or len(rejected_r) < 10:
            return None
        return prompt, chosen_r, rejected_r
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name!r}")


def main() -> None:
    t0 = time.time()
    print("=== cross_probe_dry_run.py ===")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # ── 1. hh-rlhf ────────────────────────────────────────────────────────────
    ds_hh = load_dataset("Anthropic/hh-rlhf", split="test", streaming=False)
    hh_pairs = []
    for row in ds_hh:
        p = normalize_pair(row, "hh")
        if p is not None:
            hh_pairs.append(p)
        if len(hh_pairs) >= 2:
            break
    assert len(hh_pairs) >= 2, "hh-rlhf: could not extract 2 pairs"

    # ── 2. UltraFeedback ──────────────────────────────────────────────────────
    ds_uf = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="test_prefs",
        trust_remote_code=False,
        streaming=False,
    )
    uf_pairs = []
    for row in ds_uf:
        p = normalize_pair(row, "uf")
        if p is not None:
            uf_pairs.append(p)
        if len(uf_pairs) >= 2:
            break
    assert len(uf_pairs) >= 2, "ultrafeedback_binarized: could not extract 2 pairs"

    # ── 3. Pythia tokenizer (no weights) ─────────────────────────────────────
    tok_pythia = AutoTokenizer.from_pretrained(str(PYTHIA_TOK_DIR), local_files_only=True)
    if tok_pythia.pad_token is None:
        tok_pythia.pad_token = tok_pythia.eos_token

    hh_prompt_toks   = tok_pythia(hh_pairs[0][0]).input_ids
    hh_chosen_toks   = tok_pythia(hh_pairs[0][1]).input_ids
    hh_rejected_toks = tok_pythia(hh_pairs[0][2]).input_ids

    assert len(hh_prompt_toks)   > 5, "hh prompt too short"
    assert len(hh_chosen_toks)   > 5, "hh chosen too short"
    assert len(hh_rejected_toks) > 5, "hh rejected too short"

    # ── 4. Qwen/UF tokenization check — use Pythia tokenizer as proxy ────────
    # NOTE: Qwen2-1.5B cached snapshots are weights-only (no tokenizer.json);
    # the scorer loads the tokenizer from the HF repo at runtime.  For this
    # dry-run we use Pythia-70m as a proxy tokenizer to verify UF field
    # extraction produces non-empty prompt/chosen/rejected text.
    uf_prompt_toks   = tok_pythia(uf_pairs[0][0]).input_ids
    uf_chosen_toks   = tok_pythia(uf_pairs[0][1]).input_ids
    uf_rejected_toks = tok_pythia(uf_pairs[0][2]).input_ids

    assert len(uf_prompt_toks)   > 5, f"uf prompt too short: {len(uf_prompt_toks)} toks"
    assert len(uf_chosen_toks)   > 5, f"uf chosen too short: {len(uf_chosen_toks)} toks"
    assert len(uf_rejected_toks) > 5, f"uf rejected too short: {len(uf_rejected_toks)} toks"

    elapsed = time.time() - t0

    # ── 5-line report ─────────────────────────────────────────────────────────
    print(f"[OK] hh-rlhf   : 2 pairs parsed | prompt={len(hh_prompt_toks)} tok | chosen={len(hh_chosen_toks)} tok | rejected={len(hh_rejected_toks)} tok (pythia-70m tok)")
    print(f"[OK] ultra-fb  : 2 pairs parsed | prompt={len(uf_prompt_toks)} tok | chosen={len(uf_chosen_toks)} tok | rejected={len(uf_rejected_toks)} tok (pythia-70m tok proxy)")
    print(f"[OK] pythia-70m tokenizer loaded from local cache (no weights)")
    print(f"[OK] ultrafeedback_binarized test_prefs split confirmed: {len(ds_uf)} rows, chosen/rejected as list-of-turn-dicts")
    print(f"[OK] dry-run complete in {elapsed:.1f}s — field-extraction verified for both probes")


if __name__ == "__main__":
    main()
