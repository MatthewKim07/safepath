import pandas as pd
import pytest

from safepath.graph import consolidate, segment_frames
from safepath.risk import attach_risk


@pytest.fixture
def scored(toy_graph):
    G = toy_graph
    for u, v, k in list(G.edges(keys=True)):
        if {u, v} == {6, 3}:
            G.edges[u, v, k]["highway"] = "motorway"
    _, node_map = consolidate(G, tolerance=10)
    segments, edge_map = segment_frames(G)
    cid = node_map["intersection_id"].iloc[0]
    risk = pd.DataFrame({"intersection_id": [cid], "risk": [5.0]})
    rates = pd.DataFrame(
        {"crashes_per_km_year": [0.5, 2.0]}, index=pd.Index(["secondary", "motorway"])
    )
    attach_risk(G, node_map, edge_map, segments, risk, rates)
    return G


def _route_risk(G, path, attr="intersection_risk"):
    return sum(G.edges[u, v, 0][attr] for u, v in zip(path[:-1], path[1:], strict=True))


def test_intersection_charged_once_on_entry(scored):
    G = scored
    assert G.edges[1, 2, 0]["intersection_risk"] == 5.0  # enters the {2, 6} cluster
    assert G.edges[2, 6, 0]["intersection_risk"] == 0.0  # internal to the cluster
    assert G.edges[6, 3, 0]["intersection_risk"] == 0.0  # leaves to a dead end
    # straight through the jogged intersection: charged once, not once per OSM node
    assert _route_risk(G, [1, 2, 6, 3]) == 5.0
    assert _route_risk(G, [4, 2, 6, 5]) == 5.0


def test_segment_and_motorway_risk(scored):
    G = scored
    assert G.edges[1, 2, 0]["segment_risk"] == pytest.approx(0.5 * 0.2)  # 200 m secondary
    mw = G.edges[6, 3, 0]
    assert mw["segment_risk"] == pytest.approx(2.0 * 0.188) and mw["risk"] > 0
    assert G.edges[1, 2, 0]["risk"] == pytest.approx(5.0 + 0.1)
