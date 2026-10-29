"""Data access for the Streamlit app, kept free of Streamlit so it can be tested."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

from safepath.config import load_config
from safepath.graph import INTERSECTIONS
from safepath.risk import INTERSECTION_RISK
from safepath.routing import Route, Router, load_router


def build_router(cfg: dict[str, Any] | None = None) -> Router:
    router = load_router(cfg or load_config())
    if router.k == 1.0:
        raise RuntimeError("routing k is not calibrated; run `python -m safepath route` first")
    return router


def to_wgs84(crs: str) -> Transformer:
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def route_latlon(router: Router, route: Route, transformer: Transformer) -> list[list[float]]:
    """[lat, lon] points following each edge's geometry (straight line if it has none)."""
    xs: list[float] = []
    ys: list[float] = []
    for e in route.edges:
        u, v = router.nodes[router.u[e]], router.nodes[router.v[e]]
        geom = router.G.edges[u, v, router.key[e]].get("geometry")
        if geom is not None:
            x, y = geom.xy
        else:
            x = [router.G.nodes[u]["x"], router.G.nodes[v]["x"]]
            y = [router.G.nodes[u]["y"], router.G.nodes[v]["y"]]
        xs.extend(x)
        ys.extend(y)
    if not xs:
        return []
    lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))
    return [[float(a), float(b)] for a, b in zip(lat, lon, strict=True)]


# Bands within the top 5% of intersections by predicted risk, darkest = riskiest.
RISK_BANDS = [(0.99, "top 1%", "#a3201f"), (0.975, "top 1-2.5%", "#e34948"),
              (0.95, "top 2.5-5%", "#eda100")]  # fmt: skip


def risk_layer(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Top 5% riskiest intersections with lat/lon, OSM street names, and a quantile band."""
    cfg = cfg or load_config()
    processed = Path(cfg["paths"]["processed_dir"])
    risk = pd.read_parquet(processed / INTERSECTION_RISK)
    inter = pd.read_parquet(
        processed / INTERSECTIONS, columns=["intersection_id", "x", "y", "street_names"]
    )
    df = risk.merge(inter, on="intersection_id")
    cuts = {q: df["risk"].quantile(q) for q, _, _ in RISK_BANDS}
    top = df[df["risk"] >= cuts[0.95]].copy()
    top["band"], top["color"] = RISK_BANDS[-1][1], RISK_BANDS[-1][2]
    for q, label, color in reversed(RISK_BANDS[:-1]):
        sel = top["risk"] >= cuts[q]
        top.loc[sel, "band"], top.loc[sel, "color"] = label, color
    lon, lat = to_wgs84(cfg["graph"]["crs"]).transform(top["x"].to_numpy(), top["y"].to_numpy())
    top["lat"], top["lon"] = lat, lon
    top["street_names"] = top["street_names"].replace("", "(unnamed)")
    return top.sort_values("risk").reset_index(drop=True)
