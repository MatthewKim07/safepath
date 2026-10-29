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
