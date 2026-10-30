import numpy as np
import pandas as pd
import pytest

from safepath.graph import consolidate, segment_frames
from safepath.risk import adjusted_labels, attach_risk, attribute_excess, edge_rates


@pytest.fixture
def scored(toy_graph, class_rates):
    G = toy_graph
    for u, v, k in list(G.edges(keys=True)):
        if {u, v} == {6, 3}:
            G.edges[u, v, k]["highway"] = "motorway"
    _, node_map = consolidate(G, tolerance=10)
    _, edge_map = segment_frames(G)
    cid = node_map["intersection_id"].iloc[0]
    rates = class_rates(G, edge_map, {"secondary": 0.5, "motorway": 2.0})
    attach_risk(G, node_map, edge_map, pd.Series({cid: 5.0}), rates)
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


def test_attribute_excess_moves_only_crashes_above_the_spf():
    labels = pd.DataFrame(
        {
            "intersection_id": [1, 1, 2, 2],
            "year": [2022, 2023, 2022, 2023],
            "crash_count": [30, 50, 2, 2],
        }
    )
    base = pd.Series({1: 10.0, 2: 5.0})
    levels = pd.Series({2022: 1.0, 2023: 1.0})
    cand = pd.DataFrame({"intersection_id": [1, 2], "expressway": "Gardiner", "dist_m": 10.0})
    out = attribute_excess(labels, base, levels, cand, [2022, 2023]).set_index("intersection_id")
    assert out.loc[1, "reassigned_per_year"] == pytest.approx((80 - 20) / 2)
    assert out.loc[1, "keep_share"] == pytest.approx(20 / 80)
    assert out.loc[2, "reassigned_per_year"] == 0  # below its SPF: keeps everything
    adj = adjusted_labels(labels, out.reset_index())
    assert adj.loc[adj["intersection_id"] == 1, "crash_count"].sum() == pytest.approx(20)
    assert adj.loc[adj["intersection_id"] == 2, "crash_count"].sum() == 4


def test_edge_rates_pool_by_class_and_use_mto_sections():
    edges = pd.DataFrame(
        {
            "u": [1, 2, 3, 4],
            "v": [2, 1, 4, 3],
            "key": 0,
            "segment_id": [10, 10, 20, 30],
            "highway": ["secondary", "secondary", "motorway", "motorway"],
            "aadt_dir": [1000.0, 3000.0, 50000.0, 40000.0],
            "length_m": 1000.0,
            "mto_lhrs": [np.nan, np.nan, np.nan, 7.0],
            "expressway": [None, None, "Gardiner", None],
        }
    )
    edges["mvkm"] = edges["aadt_dir"] * edges["length_m"] / 1000 * 365 / 1e6
    midblock = pd.Series({10: 2.92})
    mto = pd.DataFrame({"segment_id": [30], "cr": [0.9]})
    e, groups = edge_rates(edges, midblock, mto, pd.Series({"Gardiner": 18.25}))
    sec_mvkm = 4000 * 365 / 1e6
    assert e.loc[0, "rate_per_mvkm"] == pytest.approx(2.92 / sec_mvkm)
    assert e.loc[2, "rate_per_mvkm"] == pytest.approx(18.25 / (50000 * 365 / 1e6))
    assert e.loc[3, "rate_per_mvkm"] == 0.9
    assert groups.loc["400-series (MTO)", "rate_per_mvkm"] == pytest.approx(0.9)
