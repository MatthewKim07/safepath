"""Slim deployment bundle for the app: `python -m safepath bundle` -> app_bundle/<city>/.

The deployed app (Streamlit Community Cloud) reads only these files: no OSMnx / Overpass,
no LightGBM or torch, no pickled graph, no recalibration pickle. Per city:

- nodes.parquet, edges.parquet, geometry.parquet, geom_offsets.npy: the routing graph as
  flat arrays (`router.RouterData`), with per-vehicle risk and travel time on every edge
- intersections.parquet: id, lat/lon, OSM street names, expected crashes/yr, crashes per
  million entering vehicles, entering AADT, proxy flag, top-3 risk factors
- meta.json: label, map centre / zoom, presets and default trip, k, whether risk is
  relative only (London), file sizes
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from safepath.config import LONDON_CONFIG, REPO_ROOT, load_config
from safepath.explain import FACTOR_TERMS, INTERSECTION_FACTORS, spf_contributions, top_factors
from safepath.features.build import STATIC
from safepath.graph import INTERSECTIONS, NODE_MAP
from safepath.models.spf import SPF_MODEL, spf_covariates
from safepath.risk import INTERSECTION_RISK, ROUTING_GRAPH
from safepath.routing import PRESETS, ROUTING_K, router_data

log = logging.getLogger(__name__)

BUNDLE_DIR = REPO_ROOT / "app_bundle"
MAX_FILE_MB = 50
LONDON_PRESETS = {
    "Western University (UCC)": (43.0089, -81.2737),
    "Masonville Place": (43.0255, -81.2796),
    "White Oaks Mall": (42.9395, -81.2258),
    "Covent Garden Market (downtown)": (42.9832, -81.2497),
    "Fanshawe College": (43.0128, -81.2003),
    "Hyde Park area": (43.0080, -81.3340),  # Hyde Park Road retail strip
    "London International Airport area": (43.0303, -81.1569),
}
CITIES: dict[str, dict[str, Any]] = {
    "toronto": {
        "config": None,
        "label": "Toronto",
        "center": [43.70, -79.40],
        "zoom": 11,
        "presets": PRESETS,
        "default_trip": ["Union Station", "Yorkdale"],
        "relative": False,
    },
    "london": {
        "config": LONDON_CONFIG,
        "label": "London, ON",
        "center": [42.985, -81.245],
        "zoom": 12,
        "presets": LONDON_PRESETS,
        "default_trip": ["Western University (UCC)", "White Oaks Mall"],
        "relative": True,
    },
}


def london_factors(cfg: dict[str, Any], toronto: dict[str, Any]) -> pd.DataFrame:
    """SPF factor decomposition for London (Toronto's coefficients, London's covariates and
    average); no crash-history term, since London has none."""
    processed = Path(cfg["paths"]["processed_dir"])
    models = json.loads((Path(toronto["paths"]["models_dir"]) / SPF_MODEL).read_text())["models"]
    static = pd.read_parquet(processed / STATIC)
    expo = pd.read_parquet(processed / "intersection_exposure.parquet").set_index("intersection_id")
    rows = pd.DataFrame({"intersection_id": static.index, "year": 2025})
    df = spf_covariates(rows, static, expo)
    c = spf_contributions(cfg, df, models)
    return top_factors(c, [*FACTOR_TERMS, "intersection type"])


def intersection_table(cfg: dict[str, Any], city: str, toronto: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    risk = pd.read_parquet(processed / INTERSECTION_RISK)
    inter = gpd.read_parquet(processed / INTERSECTIONS)[["intersection_id", "street_names",
                                                        "geometry"]].to_crs(4326)  # fmt: skip
    inter["lat"], inter["lon"] = inter.geometry.y, inter.geometry.x
    df = risk[["intersection_id", "risk", "risk_per_mev", "entering_aadt",
               "is_proxy_fallback"]].merge(
        pd.DataFrame(inter.drop(columns="geometry")), on="intersection_id")  # fmt: skip
    if city == "london":
        factors = london_factors(cfg, toronto)
    else:
        factors = pd.read_parquet(processed / INTERSECTION_FACTORS)
    df = df.merge(factors[["intersection_id", "factor_1", "factor_2", "factor_3"]],
                  on="intersection_id", how="left")  # fmt: skip
    df[["factor_1", "factor_2", "factor_3"]] = df[["factor_1", "factor_2", "factor_3"]].fillna("")
    df["street_names"] = df["street_names"].replace("", "(unnamed)")
    for col in ("risk", "risk_per_mev", "entering_aadt", "lat", "lon"):
        df[col] = df[col].astype(np.float32 if col not in ("lat", "lon") else np.float64)
    return df


def build_city(city: str, out_root: Path = BUNDLE_DIR) -> dict[str, Any]:
    spec = CITIES[city]
    toronto = load_config()
    cfg = load_config(spec["config"]) if spec["config"] else toronto
    processed = Path(cfg["paths"]["processed_dir"])
    out = out_root / city
    with open(processed / ROUTING_GRAPH, "rb") as f:
        G = pickle.load(f)
    node_map = pd.read_parquet(processed / NODE_MAP)
    junction = dict(zip(node_map["osmid"], node_map["intersection_id"], strict=True))
    files = router_data(G, junction).save(out)
    inter = intersection_table(cfg, city, toronto)
    inter.to_parquet(out / "intersections.parquet", index=False, compression="zstd")
    files.append(out / "intersections.parquet")
    k = json.loads((processed / ROUTING_K).read_text())["k_seconds_per_crash"]
    meta = {k_: v for k_, v in spec.items() if k_ != "config"}
    meta["presets"] = {n: list(p) for n, p in spec["presets"].items()}
    meta["k_seconds_per_crash"] = k
    meta["files_mb"] = {p.name: round(p.stat().st_size / 1e6, 2) for p in files}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    big = {n: s for n, s in meta["files_mb"].items() if s > MAX_FILE_MB}
    if big:
        raise ValueError(f"bundle files over {MAX_FILE_MB} MB: {big}")
    log.info("bundle %s: %.1f MB in %d files", city, sum(meta["files_mb"].values()), len(files))
    return meta


def run_bundle(cfg: dict[str, Any]) -> dict[str, Any]:
    metas = {city: build_city(city) for city in CITIES}
    total = sum(sum(m["files_mb"].values()) for m in metas.values())
    log.info("app bundle: %.1f MB total", total)
    return metas
