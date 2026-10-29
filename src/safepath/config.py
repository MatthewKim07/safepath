"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve every entry under `paths` relative to the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: _resolve(v) for k, v in cfg.get("paths", {}).items()}
    return cfg


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
