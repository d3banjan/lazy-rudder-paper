"""Path resolution shim for lazy-rudder-paper scripts.

Resolution order (first match wins):
  1. Environment variables:
       LAZY_RUDDER_MODELS_DIR   → MODELS_DIR
       LAZY_RUDDER_RESULTS_DIR  → RESULTS_DIR
       LAZY_RUDDER_BASE_DIR     → BASE_DIR
  2. paper/config.toml (copy from config.example.toml and edit):
       models_dir, results_dir, base_dir
  3. Fallback: ../cross-check/trained-model-battery/{models,results}
       relative to this file's parent's parent — the in-place dev layout.
       A warning is emitted so cloners notice the fallback is active.

Exports:
  MODELS_DIR  : Path — Pythia base-weight root (e.g. MODELS_DIR / "pythia-410m")
  RESULTS_DIR : Path — training outputs root   (e.g. RESULTS_DIR / "_leak/v2")
  BASE_DIR    : Path — battery root            (alias kept for legacy scripts)

Helper:
  download_model(model_name) → Path
    Downloads EleutherAI/{model_name} from HuggingFace into MODELS_DIR/{model_name}
    and returns the local path. Raises ImportError with install hint if
    huggingface_hub is not installed.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

# ── locate config.toml (lives next to scripts/, i.e. in paper/) ──────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent   # paper/scripts/
_PAPER_DIR   = _SCRIPTS_DIR.parent               # paper/
_CONFIG_TOML = _PAPER_DIR / "config.toml"

# ── parse TOML if present ─────────────────────────────────────────────────────
_toml_cfg: dict = {}
if _CONFIG_TOML.exists():
    import tomllib  # stdlib in Python ≥ 3.11
    with open(_CONFIG_TOML, "rb") as _fh:
        _toml_cfg = tomllib.load(_fh)


def _resolve(env_var: str, toml_key: str, fallback: Path, label: str) -> Path:
    """Return the first defined path in the resolution order."""
    # 1. env var
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)
    # 2. config.toml
    toml_val = _toml_cfg.get(toml_key)
    if toml_val:
        return Path(toml_val)
    # 3. fallback with warning
    warnings.warn(
        f"[_paths] {label}: neither ${env_var} nor config.toml '{toml_key}' set; "
        f"falling back to {fallback} (in-place dev layout). "
        f"Set ${env_var} or copy config.example.toml → config.toml to suppress.",
        stacklevel=3,
    )
    return fallback


# ── fallback paths (original in-place dev layout) ────────────────────────────
_BATTERY_FALLBACK = _PAPER_DIR.parent / "cross-check" / "trained-model-battery"

MODELS_DIR: Path = _resolve(
    "LAZY_RUDDER_MODELS_DIR",
    "models_dir",
    _BATTERY_FALLBACK / "models",
    "MODELS_DIR",
)

RESULTS_DIR: Path = _resolve(
    "LAZY_RUDDER_RESULTS_DIR",
    "results_dir",
    _BATTERY_FALLBACK / "results",
    "RESULTS_DIR",
)

# Paper-local analysis output dir. Generated JSONs live here and are committed
# to the paper repo so that `make paper` works without re-running analysis.
# Always resolves to paper/results/ regardless of where checkpoints live.
PAPER_RESULTS_DIR: Path = _PAPER_DIR / "results"

BASE_DIR: Path = _resolve(
    "LAZY_RUDDER_BASE_DIR",
    "base_dir",
    _BATTERY_FALLBACK,
    "BASE_DIR",
)


# ── optional download helper ──────────────────────────────────────────────────

def download_model(model_name: str) -> Path:
    """Download EleutherAI/{model_name} from HuggingFace into MODELS_DIR/{model_name}.

    Returns the local directory path. Requires huggingface_hub >= 0.23.
    If not installed, raises ImportError with an install hint.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for download_model(). "
            "Install it with: uv add huggingface_hub  (or pip install huggingface_hub)"
        ) from exc

    local_dir = MODELS_DIR / model_name
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=f"EleutherAI/{model_name}",
        local_dir=str(local_dir),
    )
    return local_dir


# ── checkpoint download helper ────────────────────────────────────────────────

CHECKPOINTS_HF_REPO: str = (
    os.environ.get("LAZY_RUDDER_CHECKPOINTS_HF_REPO")
    or _toml_cfg.get("checkpoints_hf_repo")
    or "d3banjan/lazy-rudder-checkpoints"
)


def download_checkpoints(force: bool = False) -> Path:
    """Download all paper-cited adapter checkpoints into RESULTS_DIR.

    Pulls from `d3banjan/lazy-rudder-checkpoints` (public, ~2.5 GB).
    Layout mirrors the on-disk dev tree, so analysis scripts work unchanged
    once this completes:

      RESULTS_DIR/
        _leak/v2/checkpoints/checkpoint-800/adapter_model.safetensors
        _leak/v3/checkpoints/checkpoint-800/adapter_model.safetensors
        ...

    By default, snapshot_download is no-op when files already match the
    remote hashes. Pass force=True to bypass the local cache check.

    Returns the RESULTS_DIR path.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for download_checkpoints(). "
            "Install it with: uv add huggingface_hub  (or pip install huggingface_hub)"
        ) from exc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=CHECKPOINTS_HF_REPO,
        local_dir=str(RESULTS_DIR),
        force_download=force,
    )
    return RESULTS_DIR
