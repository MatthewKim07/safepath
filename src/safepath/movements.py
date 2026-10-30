"""Turn-aware routing graph: states are directed edges, transitions are movements.

A node-based shortest path cannot see a U-turn made through a divided-road intersection:
the route arrives on one carriageway, crosses the median on a short internal edge, and
leaves on the other carriageway, visiting no node twice. Routing over movements fixes this.

Each consolidated intersection (or a lone graph node) is a junction. A movement is a pair
(edge entering the junction, edge leaving it) plus the cheapest path between them over the
junction's internal edges (by travel time). A movement that drives straight back along the
edge it came in on, or turns more than U_TURN_DEG back onto the same named street (the other
carriageway of a divided road), is a U-turn and costs U_TURN_PENALTY_S extra. Loop ramps
also turn sharply but join a different road, so they are not U-turns. Internal edges are
never states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import csgraph_from_dense, floyd_warshall

U_TURN_DEG = 150.0
U_TURN_PENALTY_S = 60.0


def bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(dy, dx)) % 360


def turn_angle(heading_in: np.ndarray, heading_out: np.ndarray) -> np.ndarray:
    """Absolute change of heading in degrees, 0-180."""
    return np.abs((heading_out - heading_in + 180) % 360 - 180)


@dataclass
class Movements:
    """Movement table and the internal paths needed to expand a route back into edges."""

    external: np.ndarray  # edge ids that are states
    state_of: np.ndarray  # edge id -> state index (-1 for internal edges)
    junction: np.ndarray  # node index -> junction id
    move_from: np.ndarray  # state index
    move_to: np.ndarray  # state index
    int_tt: np.ndarray
    int_risk: np.ndarray
    uturn: np.ndarray
    paths: dict[tuple[int, int], tuple[float, float, list[int]]] = field(default_factory=dict)

    def internal(self, a: int, b: int) -> tuple[float, float, list[int]] | None:
        """(travel time, risk, edge ids) from node a to node b inside one junction."""
        if a == b:
            return 0.0, 0.0, []
        return self.paths.get((a, b))


def internal_paths(
    u: np.ndarray,
    v: np.ndarray,
    tt: np.ndarray,
    risk: np.ndarray,
    junction: np.ndarray,
    internal: np.ndarray,
) -> dict[tuple[int, int], tuple[float, float, list[int]]]:
    """All-pairs shortest internal paths (by travel time) within each multi-node junction."""
    paths: dict[tuple[int, int], tuple[float, float, list[int]]] = {}
    ids = np.flatnonzero(internal)
    if len(ids) == 0:
        return paths
    j = junction[u[ids]]
    ids = ids[np.lexsort((tt[ids], j))]  # by junction, cheapest parallel edge first
    bounds = np.flatnonzero(np.diff(junction[u[ids]])) + 1
    for grp in np.split(ids, bounds):
        a, b = u[grp], v[grp]
        nodes = np.unique(np.r_[a, b])
        ai, bi = np.searchsorted(nodes, a), np.searchsorted(nodes, b)
        n = len(nodes)
        dense = np.full((n, n), np.inf)
        edge_at = np.full((n, n), -1)
        # assign in reverse so the cheapest of parallel edges is written last and wins
        dense[ai[::-1], bi[::-1]] = tt[grp[::-1]] + 1e-9
        edge_at[ai[::-1], bi[::-1]] = grp[::-1]
        dist, pred = floyd_warshall(
            csgraph_from_dense(dense, null_value=np.inf), directed=True, return_predecessors=True
        )
        for x in range(n):
            for y in range(n):
                if x == y or not np.isfinite(dist[x, y]):
                    continue
                chain, cur = [], y
                while cur != x:
                    p = pred[x, cur]
                    chain.append(int(edge_at[p, cur]))
                    cur = p
                chain.reverse()
                paths[(int(nodes[x]), int(nodes[y]))] = (
                    float(tt[chain].sum()),
                    float(risk[chain].sum()),
                    chain,
                )
    return paths


def build_movements(
    u: np.ndarray,
    v: np.ndarray,
    tt: np.ndarray,
    risk: np.ndarray,
    junction: np.ndarray,
    heading_start: np.ndarray,
    heading_end: np.ndarray,
    name: np.ndarray,
) -> Movements:
    """`u`, `v` are node indices; `junction` maps node index -> junction id; headings are
    each edge's direction at its start and at its end; `name` is the street name ("" if
    unnamed)."""
    internal = junction[u] == junction[v]
    paths = internal_paths(u, v, tt, risk, junction, internal)
    external = np.flatnonzero(~internal)
    state_of = np.full(len(u), -1)
    state_of[external] = np.arange(len(external))
    ins = pd.DataFrame({"e": external, "j": junction[v[external]], "a": v[external]})
    outs = pd.DataFrame({"f": external, "j": junction[u[external]], "b": u[external]})
    pairs = ins.merge(outs, on="j")
    same = pairs["a"].to_numpy() == pairs["b"].to_numpy()
    int_tt = np.zeros(len(pairs))
    int_risk = np.zeros(len(pairs))
    ok = same.copy()
    a_arr, b_arr = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    for i in np.flatnonzero(~same):
        p = paths.get((int(a_arr[i]), int(b_arr[i])))
        if p is not None:
            int_tt[i], int_risk[i], ok[i] = p[0], p[1], True
    pairs, int_tt, int_risk = pairs[ok], int_tt[ok], int_risk[ok]
    e, f = pairs["e"].to_numpy(), pairs["f"].to_numpy()
    reverse = (u[f] == v[e]) & (v[f] == u[e])
    same_street = (name[e] == name[f]) & (name[e] != "")
    uturn = reverse | (same_street & (turn_angle(heading_end[e], heading_start[f]) > U_TURN_DEG))
    return Movements(
        external=external,
        state_of=state_of,
        junction=junction,
        move_from=state_of[e],
        move_to=state_of[f],
        int_tt=int_tt,
        int_risk=int_risk,
        uturn=uturn,
        paths=paths,
    )
