import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from safepath.app.store import City, load_city, risk_layer
from safepath.router import Router as ArrayRouter
from safepath.router import RouterData, fastest_route
from safepath.routing import Router, router_data


def test_bundle_round_trip_routes_like_the_graph(toy_graph, tmp_path):
    G = toy_graph
    for *_, d in G.edges(keys=True, data=True):
        d["travel_time"] = d["length"] / 10
        d["risk"] = 1.0
    # a bent edge: the drawn route must include its middle vertex
    G.edges[1, 2, 0]["geometry"] = LineString([(0, 0), (100, 30), (200, 0)])
    graph_route = fastest_route(Router(G), 1, 6)
    router_data(G).save(tmp_path)
    loaded = ArrayRouter(RouterData.load(tmp_path))
    route = fastest_route(loaded, 1, 6)
    assert route.nodes == graph_route.nodes and route.risk == graph_route.risk
    pts = loaded.latlon(route)
    assert len(pts) == 3 + 2  # bent edge (3 vertices) + straight 2-6 edge (2 nodes)
    lat, lon = pts[0]
    # toy coordinates are tiny UTM 17N values, i.e. near 0 N, 85.5 W
    assert lat == pytest.approx(0.0, abs=0.01) and lon == pytest.approx(-85.49, abs=0.05)


def test_risk_layer_keeps_top_five_percent_with_bands():
    n = 400
    inter = pd.DataFrame(
        {
            "intersection_id": range(n),
            "risk": np.arange(n, dtype=float),
            # per-vehicle rate runs the other way; the first 100 are too quiet to rank
            "risk_per_mev": np.arange(n, 0, -1, dtype=float),
            "entering_aadt": np.r_[np.full(100, 1_000.0), np.full(n - 100, 20_000.0)],
        }
    )
    city = City("toy", {}, None, inter)  # type: ignore[arg-type]
    top = risk_layer(city)
    assert len(top) == 20  # top 5% of 400
    assert top["band"].value_counts().to_dict() == {"top 2.5-5%": 10, "top 1-2.5%": 6, "top 1%": 4}
    by_rate = risk_layer(city, measure="rate")
    assert len(by_rate) == 15  # top 5% of the 300 busy-enough intersections
    assert by_rate["intersection_id"].min() >= 100  # quiet ones never enter the ranking


@pytest.mark.parametrize("city", ["toronto", "london"])
def test_committed_bundle_loads(city):
    c = load_city(city)
    assert c.router.k > 0 and len(c.intersections) > 1000
    assert set(c.meta["default_trip"]) <= set(c.meta["presets"])
    assert c.relative == (city == "london")
