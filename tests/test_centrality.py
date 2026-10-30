import networkx as nx
import numpy as np
import pytest

from safepath.features.centrality import edge_betweenness, subtree_sizes


def test_matches_networkx_edge_betweenness():
    G = nx.gnm_random_graph(60, 220, seed=3, directed=True)
    rng = np.random.default_rng(0)
    for a, b in G.edges:
        G[a][b]["w"] = rng.uniform(1, 10)  # distinct weights: unique shortest paths
    u = np.array([a for a, _ in G.edges])
    v = np.array([b for _, b in G.edges])
    w = np.array([G[a][b]["w"] for a, b in G.edges])
    ours = edge_betweenness(u, v, w, 60, n_sources=60)
    ref = nx.edge_betweenness_centrality(G, weight="w", normalized=False)
    assert ours == pytest.approx([ref[e] for e in G.edges])


def test_parallel_edges_only_cheapest_carries_flow():
    u, v = np.array([0, 0, 1]), np.array([1, 1, 2])
    w = np.array([5.0, 1.0, 1.0])
    flow = edge_betweenness(u, v, w, 3, n_sources=3)
    assert flow[0] == 0
    assert flow[1] == 2  # 0->1 and 0->2
    assert flow[2] == 2  # 0->2 and 1->2


def test_subtree_sizes_on_a_chain_and_branch():
    # 0 -> 1 -> 2, 1 -> 3; node 4 unreachable
    pred = np.array([-9999, 0, 1, 1, -9999])
    assert subtree_sizes(pred, 0).tolist() == [4, 3, 1, 1, 0]


def test_limit_restricts_destinations():
    u, v, w = np.array([0, 1]), np.array([1, 2]), np.array([10.0, 10.0])
    flow = edge_betweenness(u, v, w, 3, n_sources=3, limit=15)
    assert flow[1] == 1  # only 1->2 is within 15 s; 0->2 takes 20 s
