"""Sampled edge betweenness on the directed drive graph, a proxy for traffic volume.

For each sampled source, Dijkstra on travel time gives a shortest-path tree; every node it
reaches is one destination, so the number of shortest paths from that source through a tree
edge (p -> w) is the size of w's subtree. Summed over sources and scaled by n / k, this is
the standard sampled estimate of edge betweenness (ties broken by the single predecessor
Dijkstra returns). A `limit` in seconds restricts destinations to nearby nodes, giving a
local version that reflects neighbourhood rather than cross-city trips.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

log = logging.getLogger(__name__)


def collapse_parallel(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, n: int
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Cheapest edge per (u, v) as a sparse weight matrix, plus sorted (u * n + v) keys and
    the edge id for each key, for vectorised lookups."""
    order = np.lexsort((w, v, u))
    first = np.ones(len(order), bool)
    first[1:] = (u[order][1:] != u[order][:-1]) | (v[order][1:] != v[order][:-1])
    best = order[first]  # sorted by (u, v), so keys are sorted too
    m = csr_matrix((w[best] + 1e-9, (u[best], v[best])), shape=(n, n))
    keys = u[best].astype(np.int64) * n + v[best]
    return m, keys, best


def subtree_sizes(pred: np.ndarray, source: int) -> np.ndarray:
    """Nodes in each node's subtree of a predecessor tree (itself included; 0 if
    unreached). Depths by pointer doubling, then sizes accumulated level by level."""
    n = len(pred)
    reached = pred >= 0
    reached[source] = True
    parent = np.where(pred >= 0, pred, np.arange(n))
    depth = (pred >= 0).astype(np.int64)
    anc = parent.copy()
    while True:  # depth[i] = edges from i to the root
        step = depth[anc]
        if not step.any():
            break
        depth = depth + step
        anc = anc[anc]
    size = reached.astype(np.float64)
    nodes = np.flatnonzero(pred >= 0)
    by_depth = nodes[np.argsort(-depth[nodes], kind="stable")]
    levels = np.split(by_depth, np.flatnonzero(np.diff(depth[by_depth])) + 1)
    for level in levels:  # deepest first, so children are final before their parents
        np.add.at(size, pred[level], size[level])
    return size


def edge_betweenness(
    u: np.ndarray,
    v: np.ndarray,
    weight: np.ndarray,
    n_nodes: int,
    n_sources: int,
    seed: int = 7,
    limit: float = np.inf,
) -> np.ndarray:
    """Estimated shortest paths through each directed edge (indexed like u/v). Parallel
    edges: only the cheapest carries flow."""
    m, keys, best = collapse_parallel(u, v, weight, n_nodes)
    rng = np.random.default_rng(seed)
    sources = rng.choice(n_nodes, size=min(n_sources, n_nodes), replace=False)
    flow = np.zeros(len(u))
    t0 = time.time()
    for s in sources:
        _, pred = dijkstra(m, indices=s, return_predecessors=True, limit=limit)
        pred = pred.astype(np.int64)
        size = subtree_sizes(pred, int(s))
        w = np.flatnonzero(pred >= 0)
        edge = best[np.searchsorted(keys, pred[w] * n_nodes + w)]
        np.add.at(flow, edge, size[w])
    log.info("betweenness: %d sources (limit %s) in %.0fs", len(sources), limit, time.time() - t0)
    return flow * n_nodes / len(sources)
