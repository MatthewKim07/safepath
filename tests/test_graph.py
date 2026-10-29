from safepath.graph import consolidate, intersection_frame, segment_frames


def test_consolidation_merges_jogged_intersection_and_drops_dead_ends(toy_graph):
    Gc, node_map = consolidate(toy_graph, tolerance=10)  # nodes within 20 m merge
    assert Gc.number_of_nodes() == 1
    assert set(node_map["osmid"]) == {2, 6}
    assert node_map["intersection_id"].nunique() == 1

    inter = intersection_frame(Gc, node_map, toy_graph)
    assert inter.loc[0, "n_osm_nodes"] == 2
    assert inter.loc[0, "street_names"] == "Cross Avenue / Main Street"
    assert abs(inter.loc[0, "x"] - 206) < 1e-6


def test_small_tolerance_keeps_nodes_separate(toy_graph):
    Gc, node_map = consolidate(toy_graph, tolerance=5)  # 12 m apart > 10 m
    assert Gc.number_of_nodes() == 2
    assert node_map["intersection_id"].nunique() == 2


def test_segments_collapse_reciprocal_edges(toy_graph):
    seg, edge_map = segment_frames(toy_graph)
    assert len(seg) == 5
    assert len(edge_map) == 10
    assert (edge_map.groupby("segment_id").size() == 2).all()
    # both directions of n1-n2 map to the same segment
    ids = edge_map.set_index(["u", "v"])["segment_id"]
    assert ids[(1, 2)] == ids[(2, 1)]
    assert set(seg["name"]) == {"Main Street", "Cross Avenue"}
