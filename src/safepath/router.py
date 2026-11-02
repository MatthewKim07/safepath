"""Turn-aware risk routing over compact arrays (numpy / scipy / pandas only).

This is the part of routing the deployed app needs. The pipeline builds `RouterData` from
the networkx routing graph (`routing.router_data`); the app loads it from app_bundle/
(`RouterData.load`), so it runs without networkx, pyproj, OSMnx or a pickled graph.

States are directed edges and transitions are movements through junctions (movements.py),
so U-turns through divided-road intersections cost a penalty. Origins and destinations snap
like a navigation app (`Router.nearest_node`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from safepath.movements import U_TURN_PENALTY_S, build_movements

MOTORWAY = ("motorway", "motorway_link")
RESIDENTIAL = ("residential", "living_street")
# A click within this distance of a street is an address on that street; farther away it is
# inside a large parcel (mall, campus, park) and snaps to the nearest arterial access.
ADDRESS_M = 100
ARTERIAL_SLACK_M = 150
ACCESS_CLASSES = (
    "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "trunk", "trunk_link",
)  # fmt: skip
EARTH_M = 6_371_008.8


class NoRoute(Exception):
    """No path between the two nodes."""


@dataclass
class Route:
    nodes: list[int]
    edges: np.ndarray  # indices into Router edge arrays
    minutes: float
    risk: float
    km: float
    motorway_km: float
    residential_km: float
    uturns: int = 0


@dataclass
class RouterData:
    """Everything routing needs, as flat arrays. Node arrays have length N, edge arrays E;
    `u` / `v` are node indices. Edge geometry (for drawing) is lat/lon vertex lists
    concatenated in `geom_lat` / `geom_lon`, edge i spanning geom_offsets[i]:[i + 1]."""

    nodes: np.ndarray  # OSM node id
    lat: np.ndarray
    lon: np.ndarray
    street_count: np.ndarray
    in_scc: np.ndarray  # in the largest strongly connected component
    junction: np.ndarray  # consolidated intersection (or a unique id for lone nodes)
    u: np.ndarray
    v: np.ndarray
    key: np.ndarray
    travel_time: np.ndarray  # seconds
    risk: np.ndarray  # expected crashes per million vehicles
    length: np.ndarray  # metres
    highway: np.ndarray  # first OSM highway tag
    name: np.ndarray  # first OSM name, "" if none
    head_start: np.ndarray  # degrees, projected coordinates
    head_end: np.ndarray
    geom_offsets: np.ndarray
    geom_lat: np.ndarray
    geom_lon: np.ndarray

    NODE = ("nodes", "lat", "lon", "street_count", "in_scc", "junction")
    EDGE = ("u", "v", "key", "travel_time", "risk", "length", "highway", "name", "head_start",
            "head_end")  # fmt: skip

    def save(self, folder: Path) -> list[Path]:
        folder.mkdir(parents=True, exist_ok=True)
        nodes = pd.DataFrame({c: getattr(self, c) for c in self.NODE})
        edges = pd.DataFrame({c: getattr(self, c) for c in self.EDGE})
        edges["highway"] = edges["highway"].astype("category")
        geom = pd.DataFrame({"lat": self.geom_lat, "lon": self.geom_lon})
        paths = [folder / "nodes.parquet", folder / "edges.parquet", folder / "geometry.parquet"]
        nodes.to_parquet(paths[0], index=False, compression="zstd")
        edges.to_parquet(paths[1], index=False, compression="zstd")
        geom.to_parquet(paths[2], index=False, compression="zstd")
        np.save(folder / "geom_offsets.npy", self.geom_offsets)
        return [*paths, folder / "geom_offsets.npy"]

    @classmethod
    def load(cls, folder: Path) -> RouterData:
        nodes = pd.read_parquet(folder / "nodes.parquet")
        edges = pd.read_parquet(folder / "edges.parquet")
        geom = pd.read_parquet(folder / "geometry.parquet")
        kw = {c: nodes[c].to_numpy() for c in cls.NODE}
        kw |= {c: edges[c].astype(str if c in ("highway", "name") else edges[c].dtype)
               .to_numpy() for c in cls.EDGE}  # fmt: skip
        kw |= {"geom_lat": geom["lat"].to_numpy(), "geom_lon": geom["lon"].to_numpy(),
               "geom_offsets": np.load(folder / "geom_offsets.npy")}  # fmt: skip
        return cls(**kw)

    def __post_init__(self) -> None:
        names = {f.name for f in fields(self)}
        missing = names - set(vars(self))
        if missing:
            raise ValueError(f"RouterData missing {missing}")


def local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Metres on a local equirectangular plane: within a city the error is well under 1%,
    which is all snapping thresholds need."""
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_M
    y = np.radians(lat - lat0) * EARTH_M
    return np.column_stack([x, y])


class Router:
    """Shortest paths for any lambda: cost = travel_time + lambda * k * risk."""

    def __init__(self, data: RouterData, k: float = 1.0):
        self.data = data
        self.k = k
        self.nodes = np.asarray(data.nodes)
        self.index = {int(n): i for i, n in enumerate(self.nodes)}
        self.u, self.v, self.key = np.asarray(data.u), np.asarray(data.v), np.asarray(data.key)
        self.travel_time = np.asarray(data.travel_time, float)
        self.risk = np.asarray(data.risk, float)
        self.length = np.asarray(data.length, float)
        hw = np.asarray(data.highway).astype(str)
        self.is_motorway = np.isin(hw, MOTORWAY)
        self.is_residential = np.isin(hw, RESIDENTIAL)
        self.junction = np.asarray(data.junction, dtype=np.int64)
        self.moves = build_movements(
            self.u, self.v, self.travel_time, self.risk, self.junction,
            np.asarray(data.head_start, float), np.asarray(data.head_end, float),
            np.asarray(data.name).astype(str),
        )  # fmt: skip
        self.lat0, self.lon0 = float(np.mean(data.lat)), float(np.mean(data.lon))
        self.xy = local_xy(np.asarray(data.lat), np.asarray(data.lon), self.lat0, self.lon0)
        ok = (np.asarray(data.street_count) >= 2) & np.asarray(data.in_scc, bool)
        access = np.zeros(len(self.nodes), bool)
        is_access = np.isin(hw, ACCESS_CLASSES)
        access[self.u[is_access]] = True
        access[self.v[is_access]] = True
        self.snap_nodes = np.flatnonzero(ok)
        self.snap_tree = cKDTree(self.xy[self.snap_nodes])
        self.access_nodes = np.flatnonzero(ok & access)
        self.access_tree = cKDTree(self.xy[self.access_nodes])
        self._cache: dict[float, csr_matrix] = {}

    def cost(self, lam: float) -> np.ndarray:
        return self.travel_time + lam * self.k * self.risk

    def _matrix(self, lam: float) -> csr_matrix:
        """Movement-to-movement costs (state = directed edge), with two spare rows/columns
        for a query's virtual origin and destination."""
        if lam not in self._cache:
            m = self.moves
            c = self.cost(lam)
            to_edge = m.external[m.move_to]
            w = (
                m.int_tt
                + lam * self.k * m.int_risk
                + c[to_edge]
                + U_TURN_PENALTY_S * m.uturn
                + 1e-9
            )
            n = len(m.external) + 2
            self._cache = {lam: csr_matrix((w, (m.move_from, m.move_to)), shape=(n, n))}
        return self._cache[lam]

    def nearest_node(self, lat: float, lon: float) -> int:
        """Snap a point like a navigation app: to the nearest street if it is within
        ADDRESS_M, otherwise to the nearest arterial or collector access if it is no more
        than ARTERIAL_SLACK_M farther than the nearest street. Never to a dead end."""
        xy = local_xy(np.array([lat]), np.array([lon]), self.lat0, self.lon0)[0]
        d_any, i_any = self.snap_tree.query(xy)
        if d_any > ADDRESS_M and len(self.access_nodes):
            d_acc, i_acc = self.access_tree.query(xy)
            if d_acc <= d_any + ARTERIAL_SLACK_M:
                return int(self.nodes[self.access_nodes[i_acc]])
        return int(self.nodes[self.snap_nodes[i_any]])

    def route(self, origin: int, destination: int, lam: float) -> Route:
        m = self.moves
        o, d = self.index[origin], self.index[destination]
        if m.junction[o] == m.junction[d]:
            inside = m.internal(o, d)
            if inside is None:
                raise NoRoute(f"no route from {origin} to {destination}")
            return self._route_from_edges(o, np.array(inside[2], dtype=int), 0)
        c = self.cost(lam)
        S, T = len(m.external), len(m.external) + 1
        ext_u, ext_v = self.u[m.external], self.v[m.external]
        rows, cols, w = [], [], []
        for s in np.flatnonzero(m.junction[ext_u] == m.junction[o]):
            inside = m.internal(o, int(ext_u[s]))
            if inside is not None:
                e = m.external[s]
                rows.append(S), cols.append(s)
                w.append(inside[0] + lam * self.k * inside[1] + c[e] + 1e-9)
        for s in np.flatnonzero(m.junction[ext_v] == m.junction[d]):
            inside = m.internal(int(ext_v[s]), d)
            if inside is not None:
                rows.append(s), cols.append(T)
                w.append(inside[0] + lam * self.k * inside[1] + 1e-9)
        n = len(m.external) + 2
        mat = self._matrix(lam) + csr_matrix((w, (rows, cols)), shape=(n, n))
        _, pred = dijkstra(mat, indices=S, return_predecessors=True)
        if pred[T] < 0:
            raise NoRoute(f"no route from {origin} to {destination}")
        states = []
        cur = pred[T]
        while cur != S:
            states.append(int(cur))
            cur = pred[cur]
        states.reverse()
        edges: list[int] = []
        prev_node = o
        uturns = 0
        for i, s in enumerate(states):
            e = int(m.external[s])
            edges += m.internal(prev_node, int(self.u[e]))[2]
            if i > 0 and self._is_uturn(states[i - 1], s):
                uturns += 1
            edges.append(e)
            prev_node = int(self.v[e])
        edges += m.internal(prev_node, d)[2]
        return self._route_from_edges(o, np.array(edges, dtype=int), uturns)

    def _is_uturn(self, a: int, b: int) -> bool:
        m = self.moves
        hit = np.flatnonzero((m.move_from == a) & (m.move_to == b))
        return bool(len(hit) and m.uturn[hit[0]])

    def _route_from_edges(self, o: int, edges: np.ndarray, uturns: int) -> Route:
        nodes = [int(self.nodes[o])] + [int(self.nodes[self.v[e]]) for e in edges]
        return Route(
            nodes=nodes,
            edges=edges,
            minutes=float(self.travel_time[edges].sum() / 60),
            risk=float(self.risk[edges].sum()),
            km=float(self.length[edges].sum() / 1000),
            motorway_km=float(self.length[edges][self.is_motorway[edges]].sum() / 1000),
            residential_km=float(self.length[edges][self.is_residential[edges]].sum() / 1000),
            uturns=uturns,
        )

    def latlon(self, route: Route) -> list[list[float]]:
        """[lat, lon] points following each edge's geometry."""
        off, la, lo = self.data.geom_offsets, self.data.geom_lat, self.data.geom_lon
        pts: list[list[float]] = []
        for e in route.edges:
            a, b = off[e], off[e + 1]
            pts.extend([float(x), float(y)] for x, y in zip(la[a:b], lo[a:b], strict=True))
        return pts


def fastest_route(router: Router, origin: int, destination: int) -> Route:
    return router.route(origin, destination, 0.0)


def safest_route(router: Router, origin: int, destination: int, lam: float) -> Route:
    return router.route(origin, destination, lam)


def route_stats(route: Route, fastest: Route) -> dict[str, float]:
    """Minutes, expected crashes per million trips, and the trade-off against the fastest
    route."""
    return {
        "minutes": route.minutes,
        "expected_crashes": route.risk,
        "km": route.km,
        "extra_minutes": route.minutes - fastest.minutes,
        "risk_reduction_pct": 100 * (1 - route.risk / fastest.risk) if fastest.risk > 0 else 0.0,
    }


def load_k(path: Path) -> float:
    return float(json.loads(path.read_text())["k_seconds_per_crash"])
