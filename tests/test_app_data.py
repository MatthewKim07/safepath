import pytest
from shapely.geometry import LineString

from safepath.app.data import route_latlon, to_wgs84
from safepath.routing import Router, fastest_route


def test_route_latlon_follows_edge_geometry(toy_graph):
    G = toy_graph
    for *_, d in G.edges(keys=True, data=True):
        d["travel_time"] = d["length"] / 10
        d["risk"] = 0.0
    # give one edge a bent geometry; the drawn route must include its middle vertex
    G.edges[1, 2, 0]["geometry"] = LineString([(0, 0), (100, 30), (200, 0)])
    router = Router(G)
    route = fastest_route(router, 1, 6)
    pts = route_latlon(router, route, to_wgs84(G.graph["crs"]))
    assert len(pts) == 3 + 2  # bent edge (3 vertices) + straight 2-6 edge (2 nodes)
    lat, lon = pts[0]
    # toy coordinates are tiny UTM 17N values, i.e. near 0 N, 85.5 W
    assert lat == pytest.approx(0.0, abs=0.01) and lon == pytest.approx(-85.49, abs=0.05)


def test_risk_layer_keeps_top_five_percent_with_bands(tmp_path):
    import numpy as np
    import pandas as pd

    from safepath.app.data import risk_layer
    from safepath.config import load_config

    n = 400
    pd.DataFrame(
        {
            "intersection_id": range(n),
            "risk": np.arange(n, dtype=float),
            # per-vehicle rate runs the other way; the first 100 are too quiet to rank
            "risk_per_mev": np.arange(n, 0, -1, dtype=float),
            "entering_aadt": np.r_[np.full(100, 1_000.0), np.full(n - 100, 20_000.0)],
            "is_proxy_fallback": [False] * n,
        }
    ).to_parquet(tmp_path / "intersection_risk.parquet")
    pd.DataFrame(
        {
            "intersection_id": range(n),
            "x": 630_000.0,
            "y": 4_840_000.0,
            "street_names": ["A / B"] * (n - 1) + [""],
        }
    ).to_parquet(tmp_path / "intersections.parquet")
    cfg = load_config()
    cfg["paths"]["processed_dir"] = tmp_path
    top = risk_layer(cfg)
    assert len(top) == 20  # top 5% of 400
    assert top["band"].value_counts().to_dict() == {"top 2.5-5%": 10, "top 1-2.5%": 6, "top 1%": 4}
    assert top.iloc[-1]["street_names"] == "(unnamed)"  # highest risk, empty name
    assert 43 < top["lat"].iloc[0] < 44 and -80 < top["lon"].iloc[0] < -79

    by_rate = risk_layer(cfg, measure="rate")
    assert len(by_rate) == 15  # top 5% of the 300 busy-enough intersections
    assert by_rate["intersection_id"].min() >= 100  # quiet ones never enter the ranking
    assert by_rate["intersection_id"].max() <= 114  # highest rates among the busy ones
