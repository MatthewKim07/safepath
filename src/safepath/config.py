"""Configuration loading.

A config may start with `extends: <other.yaml>` (relative to its own directory): the other
file is loaded first and this one is deep-merged over it, so a second city's config only
lists what differs from Toronto's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
OTTAWA_CONFIG = REPO_ROOT / "configs" / "ottawa.yaml"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(_load_raw(path.parent / parent), cfg)
    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve every entry under `paths` relative to the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = _load_raw(path)
    cfg["paths"] = {k: _resolve(v) for k, v in cfg.get("paths", {}).items()}
    return cfg


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def ottawa_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Ottawa config: `cfg` itself if it already is one, else configs/ottawa.yaml."""
    if cfg is not None and cfg.get("city") == "ottawa":
        return cfg
    return load_config(OTTAWA_CONFIG)


def toronto_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The Toronto (training-city) config: `cfg` unless it is Ottawa's."""
    return load_config() if cfg.get("city") == "ottawa" else cfg
