import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from safepath.graph import consolidate, segment_frames
from safepath.risk import attach_risk
from safepath.routing import Router, fastest_route, route_stats, safest_route


def grid_graph(n: int = 8, seed: int = 0) -> nx.MultiDiGraph:
    """Directed grid with random travel times and risks, plus some parallel edges."""
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph(crs="EPSG:32617")
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, x=600_000 + 100.0 * i, y=4_840_000 + 100.0 * j)
    for i in range(n):
        for j in range(n):
            a = i * n + j
            for di, dj in ((1, 0), (0, 1)):
                if i + di < n and j + dj < n:
                    b = (i + di) * n + (j + dj)
                    for u, v in ((a, b), (b, a)):
                        for _ in range(1 + (rng.random() < 0.15)):  # some parallel edges
                            G.add_edge(
                                u,
                                v,
                                length=100.0,
                                travel_time=float(rng.uniform(5, 20)),
                                risk=float(rng.exponential(1.0)),
                                highway="secondary",
                            )
    return G


def _nx_cost(G, path, weight):
    return sum(
        min(d[weight] for d in G[u][v].values()) for u, v in zip(path[:-1], path[1:], strict=True)
    )


def test_lambda_zero_is_the_fastest_route():
    G = grid_graph()
    router = Router(G, k=10.0)
    for o, d in [(0, 63), (7, 56), (12, 50)]:
        expected = nx.shortest_path(G, o, d, weight="travel_time")
        route = fastest_route(router, o, d)
        assert route.nodes == expected
        assert route.minutes * 60 == pytest.approx(_nx_cost(G, expected, "travel_time"))
        assert safest_route(router, o, d, 0.0).nodes == route.nodes


def test_safest_route_matches_networkx_with_combined_cost():
    G = grid_graph(seed=3)
    router = Router(G, k=7.0)
    lam = 1.5
    for _, _, d in G.edges(data=True):
        d["combined"] = d["travel_time"] + lam * 7.0 * d["risk"]
    expected = nx.shortest_path(G, 0, 63, weight="combined")
    route = safest_route(router, 0, 63, lam)
    assert _nx_cost(G, route.nodes, "combined") == pytest.approx(_nx_cost(G, expected, "combined"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_route_risk_never_increases_with_lambda(seed):
    G = grid_graph(seed=seed)
    router = Router(G, k=5.0)
    for o, d in [(0, 63), (3, 60), (9, 54)]:
        risks = [safest_route(router, o, d, lam).risk for lam in (0, 0.1, 0.5, 1, 2, 5, 10)]
        assert all(b <= a + 1e-9 for a, b in zip(risks[:-1], risks[1:], strict=True)), risks


def test_route_stats():
    G = grid_graph()
    router = Router(G, k=5.0)
    fast = fastest_route(router, 0, 63)
    safe = safest_route(router, 0, 63, 5)
    s = route_stats(safe, fast)
    assert s["extra_minutes"] >= 0
    assert s["risk_reduction_pct"] == pytest.approx(100 * (1 - safe.risk / fast.risk))


def _scored_toy(toy_graph, class_rates):
    G = toy_graph
    for u, v, k in list(G.edges(keys=True)):
        G.edges[u, v, k]["travel_time"] = G.edges[u, v, k]["length"] / 10
        if {u, v} == {6, 3}:
            G.edges[u, v, k]["highway"] = "motorway"
    _, node_map = consolidate(G, tolerance=10)
    _, edge_map = segment_frames(G)
    per_mev = pd.Series(5.0, index=node_map["intersection_id"].unique())
    rates = class_rates(G, edge_map, {"secondary": 0.0, "motorway": 2.0})
    attach_risk(G, node_map, edge_map, per_mev, rates)
    return G


def test_route_through_jogged_intersection_is_charged_once(toy_graph, class_rates):
    G = _scored_toy(toy_graph, class_rates)
    route = fastest_route(Router(G), 1, 5)  # 1 -> 2 -> 6 -> 5: two OSM nodes, one intersection
    assert route.nodes == [1, 2, 6, 5]
    assert route.risk == pytest.approx(5.0)


def test_motorway_edges_carry_risk(toy_graph, class_rates):
    G = _scored_toy(toy_graph, class_rates)
    route = fastest_route(Router(G), 6, 3)  # the motorway edge only
    assert route.risk == pytest.approx(2.0 * 0.188)
    assert route.motorway_km == pytest.approx(0.188)


def test_routes_never_reverse_immediately():
    G = grid_graph(seed=5)
    router = Router(G, k=50.0)  # strong risk weight invites odd detours
    rng = np.random.default_rng(1)
    for _ in range(30):
        o, d = (int(x) for x in rng.choice(64, 2, replace=False))
        for lam in (0, 1, 5):
            nodes = safest_route(router, o, d, lam).nodes
            assert all(nodes[i] != nodes[i + 2] for i in range(len(nodes) - 2))


def divided_road(block_s: float) -> tuple[nx.MultiDiGraph, dict[int, int]]:
    """A divided Main Street (eastbound y=0, westbound y=20) with a median crossing at
    x=200 (nodes 2/12, one consolidated intersection) and a loop around the block through
    side streets 3 -> 7 -> 13. The origin is on the eastbound carriageway (node 1), the
    destination behind it on the westbound one (node 11).

        11 <- 12 <- 13 <-+
              |^         |  westbound (y = 20)
              v|         |
         1 -> 2  -> 3 ---+  eastbound (y = 0), loop 3 -> 7 -> 13 costs `block_s`
    """
    G = nx.MultiDiGraph(crs="EPSG:32617")
    xy = {1: (0, 0), 2: (200, 0), 3: (400, 0), 11: (0, 20), 12: (200, 20), 13: (400, 20),
          7: (500, 10)}  # fmt: skip
    for n, (x, y) in xy.items():
        G.add_node(n, x=float(x), y=float(y), street_count=3)

    def add(u, v, tt, name, risk=0.0):
        G.add_edge(u, v, length=100.0, travel_time=tt, risk=risk, name=name, highway="secondary")

    add(1, 2, 10, "Main Street")
    add(2, 3, 10, "Main Street")
    add(13, 12, 10, "Main Street")
    add(12, 11, 10, "Main Street")
    add(2, 12, 2, "Main Street")  # median crossing, internal to the junction {2, 12}
    add(3, 7, block_s / 2, "First Street")
    add(7, 13, block_s / 2, "Second Street")
    return G, {2: 0, 12: 0}


def test_median_u_turn_is_penalised():
    G, junction = divided_road(block_s=30)  # around the block: 30 s more than the U-turn
    route = fastest_route(Router(G, node_junction=junction), 1, 11)
    assert route.nodes == [1, 2, 3, 7, 13, 12, 11]
    assert route.uturns == 0
    # without the junction map the U-turn is invisible and taken
    plain = fastest_route(Router(G), 1, 11)
    assert plain.nodes == [1, 2, 12, 11]


def test_u_turn_still_allowed_when_it_saves_more_than_the_penalty():
    G, junction = divided_road(block_s=300)
    route = fastest_route(Router(G, node_junction=junction), 1, 11)
    assert route.nodes == [1, 2, 12, 11]
    assert route.uturns == 1


def test_snapping_avoids_dead_ends_and_prefers_arterial_access():
    G = nx.MultiDiGraph(crs="EPSG:32617")
    pts = {1: (0, 0), 2: (300, 0), 3: (600, 0), 4: (300, 400), 5: (300, 200)}
    for n, (x, y) in pts.items():
        G.add_node(n, x=600_000.0 + x, y=4_840_000.0 + y, street_count=3)
    G.nodes[4]["street_count"] = 1  # cul-de-sac end
    for u, v, hw in [(1, 2, "secondary"), (2, 3, "secondary"), (2, 5, "residential"),
                     (5, 4, "residential")]:  # fmt: skip
        for a, b in ((u, v), (v, u)):
            G.add_edge(a, b, length=100.0, travel_time=10.0, highway=hw, name=hw)
    router = Router(G)
    back = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)
    # 10 m from the residential node 5: an address on that street
    lon, lat = back.transform(600_310.0, 4_840_200.0)
    assert router.nearest_node(lat, lon) == 5
    # 150 m past the cul-de-sac end: dead end 4 is nearest but never chosen, and arterial
    # node 2 (550 m) is not within the slack of node 5 (350 m)
    lon, lat = back.transform(600_300.0, 4_840_550.0)
    assert router.nearest_node(lat, lon) == 5
    # 130 m from node 5 and 230 m from arterial node 2: parcel access goes to the arterial
    lon, lat = back.transform(600_430.0, 4_840_200.0)
    assert router.nearest_node(lat, lon) == 2
