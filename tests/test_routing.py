import networkx as nx
import numpy as np
import pandas as pd
import pytest

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
