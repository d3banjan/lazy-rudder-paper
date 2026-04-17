#!/usr/bin/env python3
"""Pull all paper-cited adapter checkpoints from HuggingFace into RESULTS_DIR.

Usage:
    python scripts/fetch_checkpoints.py            # idempotent
    python scripts/fetch_checkpoints.py --force    # ignore local cache

After this completes, every analysis script and `make analysis` target works
without needing GPU training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CHECKPOINTS_HF_REPO, RESULTS_DIR, download_checkpoints  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files match the remote hashes")
    args = ap.parse_args()

    print(f"Repo : https://huggingface.co/{CHECKPOINTS_HF_REPO}")
    print(f"Dest : {RESULTS_DIR}")
    print("Downloading (~1.9 GB, idempotent)…")
    out = download_checkpoints(force=args.force)
    print(f"Done. Files under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
