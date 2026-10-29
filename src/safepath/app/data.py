"""Data access for the Streamlit app, kept free of Streamlit so it can be tested."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Transformer

from safepath.config import load_config
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
