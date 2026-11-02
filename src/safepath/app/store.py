"""App data from app_bundle/ only (numpy / pandas / scipy): what the deployed app runs on.

`SAFEPATH_APP_SOURCE=pipeline` (local use only) builds Toronto from the full pipeline
outputs instead, which needs the heavy dependencies; the deployed app never sets it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from safepath.router import Router, RouterData

BUNDLE_ENV = "SAFEPATH_BUNDLE"
SOURCE_ENV = "SAFEPATH_APP_SOURCE"
DEFAULT_BUNDLE = Path(__file__).resolve().parents[3] / "app_bundle"

# Bands within the top 5% of intersections by the chosen measure, darkest = riskiest.
RISK_BANDS = [(0.99, "top 1%", "#a3201f"), (0.975, "top 1-2.5%", "#e34948"),
              (0.95, "top 2.5-5%", "#eda100")]  # fmt: skip
# The per-vehicle ranking only considers intersections with at least this entering AADT:
# below it a handful of crashes over a small volume gives noisy, extreme rates.
LAYER_MIN_AADT = 5_000
LAYER_MEASURES = {
    "crashes": ("risk", "expected crashes/yr"),
    "rate": ("risk_per_mev", "crashes per million entering vehicles"),
}


@dataclass
class City:
    key: str
    meta: dict[str, Any]
    router: Router
    intersections: pd.DataFrame

    @property
    def relative(self) -> bool:
        return bool(self.meta.get("relative"))


def bundle_dir() -> Path:
    return Path(os.environ.get(BUNDLE_ENV, DEFAULT_BUNDLE))


def available_cities() -> list[str]:
    root = bundle_dir()
    order = ["toronto", "london"]
    found = [c for c in order if (root / c / "meta.json").exists()]
    return found or ["toronto"]


def load_city(key: str) -> City:
    if os.environ.get(SOURCE_ENV) == "pipeline":
        return _from_pipeline(key)
    folder = bundle_dir() / key
    meta = json.loads((folder / "meta.json").read_text())
    router = Router(RouterData.load(folder), k=float(meta["k_seconds_per_crash"]))
    inter = pd.read_parquet(folder / "intersections.parquet")
    return City(key, meta, router, inter)


def _from_pipeline(key: str) -> City:  # pragma: no cover - local development only
    from safepath import bundle

    root = Path(os.environ.get("SAFEPATH_PIPELINE_BUNDLE", "/tmp/safepath_pipeline_bundle"))
    bundle.build_city(key, root)
    os.environ[BUNDLE_ENV] = str(root)
    os.environ.pop(SOURCE_ENV)
    return load_city(key)


def risk_layer(city: City, measure: str = "crashes") -> pd.DataFrame:
    """Top 5% of intersections by expected crashes/yr ("crashes") or by crashes per million
    entering vehicles ("rate", busy-enough intersections only), with a quantile band."""
    column, _ = LAYER_MEASURES[measure]
    df = city.intersections
    if measure == "rate":
        df = df[df["entering_aadt"] >= LAYER_MIN_AADT]
    cuts = {q: df[column].quantile(q) for q, _, _ in RISK_BANDS}
    top = df[df[column] >= cuts[0.95]].copy()
    top["band"], top["color"] = RISK_BANDS[-1][1], RISK_BANDS[-1][2]
    for q, label, color in reversed(RISK_BANDS[:-1]):
        sel = top[column] >= cuts[q]
        top.loc[sel, "band"], top.loc[sel, "color"] = label, color
    top["size"] = top[column] / top[column].max()
    return top.sort_values(column).reset_index(drop=True)
