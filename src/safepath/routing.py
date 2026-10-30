"""Risk-aware routing: cost = travel_time + lambda * k * risk.

`risk` on an edge is the expected number of crashes for one vehicle driving it, per million
vehicles (intersection entered plus the segment itself, risk.py), so a route's risk is
expected crashes per million trips. `k` converts it to seconds so lambda is interpretable:
at lambda = 1 the median route's risk cost equals its travel time (calibrated over random
origin/destination pairs, see `calibrate_k`). lambda = 0 is the fastest route.

Routing is turn-aware (movements.py): Dijkstra runs over directed edges as states, so a
U-turn through a divided-road intersection costs a penalty instead of being invisible.
Origins and destinations snap like a navigation app (`Router.nearest_node`).
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.graph import NODE_MAP, SEGMENTS
from safepath.locations import MOTORWAY_CLASSES, markdown_table
from safepath.movements import U_TURN_PENALTY_S, bearing, build_movements
from safepath.risk import EDGE_RISK, ROUTING_GRAPH

log = logging.getLogger(__name__)

ROUTING_K = "routing_k.json"
PRESETS = {
    "Union Station": (43.6453, -79.3806),
    "Yorkdale": (43.7255, -79.4522),
    "Scarborough Town Centre": (43.7757, -79.2578),
    # Pearson airport itself is in Mississauga, outside the Toronto graph; this is the
    # Toronto side of the airport at Dixon Road and Carlingview Drive.
    "Pearson area": (43.6867, -79.5941),
    "High Park": (43.6465, -79.4637),
    "UofT St. George": (43.6629, -79.3957),
}
EXAMPLE_PAIRS = [
    ("Union Station", "Yorkdale"),
    ("High Park", "Scarborough Town Centre"),
    ("UofT St. George", "Pearson area"),
]
SWEEP_LAMBDAS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10]


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


# A click within this distance of a street is an address on that street; farther away it is
# inside a large parcel (mall, campus, park) and snaps to the nearest arterial access.
ADDRESS_M = 100
ARTERIAL_SLACK_M = 150
ACCESS_CLASSES = {
    "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "trunk", "trunk_link",
}  # fmt: skip


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) else value


class Router:
    """Turn-aware shortest paths over the directed routing graph for any lambda.

    States are directed edges and transitions are movements through junctions
    (`movements.py`), so U-turns through divided-road intersections can be penalised.
    `node_junction` maps OSM node -> consolidated intersection id; nodes without one are
    junctions on their own.
    """

    def __init__(self, G: nx.MultiDiGraph, k: float = 1.0, node_junction: dict | None = None):
        self.G = G
        self.k = k
        self.nodes = np.array(list(G.nodes))
        self.index = {n: i for i, n in enumerate(self.nodes)}
        rows = [(u, v, key, d) for u, v, key, d in G.edges(keys=True, data=True)]
        self.u = np.array([self.index[r[0]] for r in rows])
        self.v = np.array([self.index[r[1]] for r in rows])
        self.key = np.array([r[2] for r in rows])
        self.travel_time = np.array([float(r[3]["travel_time"]) for r in rows])
        self.risk = np.array([float(r[3].get("risk", 0.0)) for r in rows])
        self.length = np.array([float(r[3]["length"]) for r in rows])
        hw = [_first(r[3].get("highway")) for r in rows]
        self.is_motorway = np.isin(hw, list(MOTORWAY_CLASSES))
        self.is_residential = np.isin(hw, ["residential", "living_street"])
        xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in self.nodes])
        self.xy = xy
        self.to_xy = Transformer.from_crs("EPSG:4326", G.graph["crs"], always_xy=True)
        node_junction = node_junction or {}
        n_cl = max(node_junction.values(), default=-1) + 1
        self.junction = np.array(
            [node_junction.get(n, n_cl + i) for i, n in enumerate(self.nodes)], dtype=np.int64
        )
        start, end = self._headings(rows)
        names = np.array([str(_first(r[3].get("name")) or "") for r in rows])
        self.moves = build_movements(
            self.u, self.v, self.travel_time, self.risk, self.junction, start, end, names
        )
        self._snap_setup(rows, hw)
        self._cache: dict[float, csr_matrix] = {}

    def _headings(self, rows: list) -> tuple[np.ndarray, np.ndarray]:
        """Heading of each edge as it leaves its start node and as it reaches its end node,
        from its geometry when it has one."""
        start = np.empty(len(rows))
        end = np.empty(len(rows))
        for i, (a, b, _, d) in enumerate(rows):
            geom = d.get("geometry")
            if geom is not None and len(geom.coords) >= 2:
                c = np.asarray(geom.coords)
                p0, p1, q0, q1 = c[0], c[1], c[-2], c[-1]
            else:
                p0 = q0 = self.xy[self.index[a]]
                p1 = q1 = self.xy[self.index[b]]
            start[i] = bearing(p1[0] - p0[0], p1[1] - p0[1])
            end[i] = bearing(q1[0] - q0[0], q1[1] - q0[1])
        return start, end

    def _snap_setup(self, rows: list, hw: list) -> None:
        """Snap candidates: no dead ends, in the largest strongly connected component; plus
        the subset that touches an arterial or collector road."""
        scc = max(nx.strongly_connected_components(self.G), key=len)
        degree = np.array([self.G.nodes[n].get("street_count", 2) for n in self.nodes])
        ok = (degree >= 2) & np.isin(self.nodes, list(scc))
        access = np.zeros(len(self.nodes), bool)
        is_access = np.isin(hw, list(ACCESS_CLASSES))
        access[self.u[is_access]] = True
        access[self.v[is_access]] = True
        self.snap_nodes = np.flatnonzero(ok)
        self.snap_tree = cKDTree(self.xy[self.snap_nodes])
        self.access_nodes = np.flatnonzero(ok & access)
        self.access_tree = cKDTree(self.xy[self.access_nodes])

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
        x, y = self.to_xy.transform(lon, lat)
        d_any, i_any = self.snap_tree.query([x, y])
        if d_any > ADDRESS_M and len(self.access_nodes):
            d_acc, i_acc = self.access_tree.query([x, y])
            if d_acc <= d_any + ARTERIAL_SLACK_M:
                return int(self.nodes[self.access_nodes[i_acc]])
        return int(self.nodes[self.snap_nodes[i_any]])

    def route(self, origin: int, destination: int, lam: float) -> Route:
        m = self.moves
        o, d = self.index[origin], self.index[destination]
        if m.junction[o] == m.junction[d]:
            inside = m.internal(o, d)
            if inside is None:
                raise nx.NetworkXNoPath(f"no route from {origin} to {destination}")
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
            raise nx.NetworkXNoPath(f"no route from {origin} to {destination}")
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


def random_pairs(
    router: Router, n: int, seed: int = 7, min_km: float = 3, max_km: float = 25
) -> list[tuple[int, int]]:
    """Random origin/destination nodes in the largest strongly connected component, between
    `min_km` and `max_km` apart in a straight line."""
    scc = max(nx.strongly_connected_components(router.G), key=len)
    nodes = np.array(sorted(scc))
    xy = np.array([(router.G.nodes[n]["x"], router.G.nodes[n]["y"]) for n in nodes])
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n:
        a, b = rng.integers(0, len(nodes), 2)
        dist = np.hypot(*(xy[a] - xy[b])) / 1000
        if min_km <= dist <= max_km:
            pairs.append((int(nodes[a]), int(nodes[b])))
    return pairs


def calibrate_k(router: Router, pairs: list[tuple[int, int]], iterations: int = 6) -> dict:
    """Fixed point of k = median(travel time) / median(risk) over the lambda = 1 routes.

    Starts from the fastest routes; each round re-routes at lambda = 1 with the current k and
    re-estimates k from those routes, so that at lambda = 1 the typical route's risk cost
    equals its travel time.
    """
    fast = [router.route(o, d, 0) for o, d in pairs]
    k = float(np.median([r.minutes * 60 for r in fast]) / np.median([r.risk for r in fast]))
    history = [{"iteration": 0, "k": k, "on": "fastest routes"}]
    for i in range(1, iterations + 1):
        router.k = k
        routes = [router.route(o, d, 1.0) for o, d in pairs]
        new_k = float(
            np.median([r.minutes * 60 for r in routes]) / np.median([r.risk for r in routes])
        )
        history.append({"iteration": i, "k": new_k, "on": "lambda=1 routes"})
        converged = abs(new_k - k) / k < 0.01
        k = new_k
        if converged:
            break
    router.k = k
    routes = [router.route(o, d, 1.0) for o, d in pairs]
    check = float(
        np.median([k * r.risk for r in routes]) / np.median([r.minutes * 60 for r in routes])
    )
    return {"k_seconds_per_crash": k, "history": history, "check_ratio_at_lambda1": check}


def tradeoff_table(
    router: Router, pairs: list[tuple[int, int]], lambdas: list[float]
) -> pd.DataFrame:
    rows = []
    for o, d in pairs:
        fast = router.route(o, d, 0)
        for lam in lambdas:
            r = router.route(o, d, lam)
            rows.append(
                {
                    "lam": lam,
                    **route_stats(r, fast),
                    "fast_minutes": fast.minutes,
                    "fast_km": fast.km,
                    "motorway_share": r.motorway_km / r.km if r.km else 0,
                    "fast_motorway_share": fast.motorway_km / fast.km if fast.km else 0,
                    "residential_share": r.residential_km / r.km if r.km else 0,
                    "uturns": r.uturns,
                    "fast_residential_share": fast.residential_km / fast.km if fast.km else 0,
                }
            )
    return pd.DataFrame(rows)


def summarise(trade: pd.DataFrame) -> pd.DataFrame:
    g = trade.groupby("lam")
    return pd.DataFrame(
        {
            "median extra min": g["extra_minutes"].median(),
            "p90 extra min": g["extra_minutes"].quantile(0.9),
            "median % risk reduction": g["risk_reduction_pct"].median(),
            "p90 % risk reduction": g["risk_reduction_pct"].quantile(0.9),
            "median detour (km ratio)": (trade["km"] / trade["fast_km"])
            .groupby(trade["lam"])
            .median(),
            "p90 detour (km ratio)": (trade["km"] / trade["fast_km"])
            .groupby(trade["lam"])
            .quantile(0.9),
            "motorway share": g["motorway_share"].mean(),
            "residential share": g["residential_share"].mean(),
            "routes with a U-turn": g["uturns"].apply(lambda x: (x > 0).mean()),
        }
    ).reset_index()


def plot_pareto(
    router: Router, name: str, origin: str, destination: str, path: Path
) -> pd.DataFrame:
    o = router.nearest_node(*PRESETS[origin])
    d = router.nearest_node(*PRESETS[destination])
    fast = router.route(o, d, 0)
    pts = pd.DataFrame(
        [{"lam": lam, **route_stats(router.route(o, d, lam), fast)} for lam in SWEEP_LAMBDAS]
    )
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    ax.plot(
        pts["minutes"],
        pts["expected_crashes"],
        color=BAR,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    # one label per distinct route, naming the range of lambdas that produce it
    key = pts["minutes"].round(2).astype(str) + "_" + pts["expected_crashes"].round(2).astype(str)
    for _, g in pts.groupby(key, sort=False):
        lo, hi = g["lam"].min(), g["lam"].max()
        r = g.iloc[0]
        ax.annotate(
            f"λ={lo:g}" if lo == hi else f"λ={lo:g}–{hi:g}",
            (r.minutes, r.expected_crashes),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_MUTED,
        )
    ax.set_title(
        f"{origin} → {destination}: time vs risk as λ rises", loc="left", fontsize=11, color=TEXT
    )
    ax.set_xlabel("route time (minutes, free-flow)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("expected crashes per million trips", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pts.assign(pair=name)


def load_router(cfg: dict[str, Any], G: nx.MultiDiGraph | None = None) -> Router:
    processed = Path(cfg["paths"]["processed_dir"])
    if G is None:
        with open(processed / ROUTING_GRAPH, "rb") as f:
            G = pickle.load(f)
    node_map = pd.read_parquet(processed / NODE_MAP)
    junction = dict(zip(node_map["osmid"], node_map["intersection_id"], strict=True))
    router = Router(G, node_junction=junction)
    k_path = processed / ROUTING_K
    if k_path.exists():
        router.k = json.loads(k_path.read_text())["k_seconds_per_crash"]
    return router


# --------------------------------------------------------------------------- sensitivity

CP4_BASELINE = "cp4_baseline.json"
CITY_XWAY = ("Gardiner", "DVP", "Allen Road")


def rate_groups(cfg: dict[str, Any], G: nx.MultiDiGraph) -> pd.Series:
    """Rate group of every edge of G (risk.py's edge_risk.parquet), indexed like G.edges."""
    processed = Path(cfg["paths"]["processed_dir"])
    er = pd.read_parquet(processed / EDGE_RISK, columns=["u", "v", "key", "rate_group"])
    er = er.set_index(["u", "v", "key"])["rate_group"]
    return er.reindex(pd.MultiIndex.from_tuples(list(G.edges(keys=True))))


def scaled_graph(G: nx.MultiDiGraph, groups: pd.Series, scale: dict[str, float]) -> nx.MultiDiGraph:
    """Copy of G with segment risk multiplied by `scale[rate_group]` (others unchanged)."""
    H = G.copy()
    factor = groups.map(scale).fillna(1.0).to_numpy()
    for (u, v, k), f in zip(H.edges(keys=True), factor, strict=True):
        if f != 1.0:
            d = H.edges[u, v, k]
            d["segment_risk"] = d["segment_risk"] * f
            d["risk"] = d["intersection_risk"] + d["segment_risk"]
    return H


def sensitivity_variants() -> dict[str, dict[str, float]]:
    freeway = ["400-series (MTO)", *[f"city expressway: {x}" for x in CITY_XWAY],
               "other expressway"]  # fmt: skip
    city = [f"city expressway: {x}" for x in CITY_XWAY]
    return {
        "baseline": {},
        "freeway rates x0.5": {g: 0.5 for g in freeway},
        "freeway rates x2": {g: 2.0 for g in freeway},
        "Gardiner/DVP/Allen volume +30%": {g: 1 / 1.3 for g in city},
        "Gardiner/DVP/Allen volume -30%": {g: 1 / 0.7 for g in city},
    }


def run_sensitivity(
    cfg: dict[str, Any], base: Router, pairs: list[tuple[int, int]], lambdas: list[float]
) -> pd.DataFrame:
    """The 200-trip evaluation under scaled freeway risk, with k held at its baseline value
    so that lambda keeps its meaning and the scaling shows up fully in behaviour."""
    groups = rate_groups(cfg, base.G)
    rows = []
    o = base.nearest_node(*PRESETS["Union Station"])
    d = base.nearest_node(*PRESETS["Yorkdale"])
    for name, scale in sensitivity_variants().items():
        router = base if not scale else load_router(cfg, scaled_graph(base.G, groups, scale))
        router.k = base.k
        trade = tradeoff_table(router, pairs, [0, *lambdas])
        case = router.route(o, d, 2.0)
        rows.append(
            summarise(trade).assign(
                variant=name, union_yorkdale_km=case.km, union_yorkdale_min=case.minutes
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_before_after(
    router: Router, cp4: dict[str, Any], lam: float, segments: gpd.GeoDataFrame, path: Path
) -> dict[str, Any]:
    """CP4's count-based safest route vs the CP6 per-vehicle route, Union Station ->
    Yorkdale at the same lambda, both scored with CP6 per-vehicle risk."""
    o = router.nearest_node(*PRESETS["Union Station"])
    d = router.nearest_node(*PRESETS["Yorkdale"])
    fast = router.route(o, d, 0)
    new = router.route(o, d, lam)
    old_nodes = cp4["nodes"]
    old_edges = []
    for a, b in zip(old_nodes[:-1], old_nodes[1:], strict=True):
        keys = list(router.G[a][b])
        k_best = min(keys, key=lambda k: router.G.edges[a, b, k]["travel_time"])
        old_edges.append(np.flatnonzero(
            (router.u == router.index[a]) & (router.v == router.index[b]) & (router.key == k_best)
        )[0])  # fmt: skip
    old = router._route_from_edges(router.index[old_nodes[0]], np.array(old_edges), 0)

    def line(route: Route) -> np.ndarray:
        return router.xy[[router.index[n] for n in route.nodes]]

    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    pts = np.vstack([line(fast), line(new), line(old)])
    pad = 1500
    box = (
        pts[:, 0].min() - pad,
        pts[:, 0].max() + pad,
        pts[:, 1].min() - pad,
        pts[:, 1].max() + pad,
    )
    bg = segments.cx[box[0] : box[1], box[2] : box[3]]
    bg[bg["highway"].isin(["primary", "secondary", "motorway", "tertiary"])].plot(
        ax=ax, color=GRID, linewidth=0.6
    )
    styles = [
        (old, "#e34948", 5, f"CP4 safest, λ={lam:g}: count-based risk"),
        (new, "#008300", 3.5, f"CP6 safest, λ={lam:g}: per-vehicle risk"),
        (fast, TEXT_MUTED, 2, "fastest"),
    ]
    for route, color, width, label in styles:
        xy = line(route)
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=width, alpha=0.85,
                label=f"{label} — {route.km:.1f} km, {route.minutes:.1f} min, "
                f"{route.risk:.0f} per M trips")  # fmt: skip
    for name in ("Union Station", "Yorkdale"):
        n = o if name == "Union Station" else d
        x, y = router.xy[router.index[n]]
        ax.scatter([x], [y], s=60, color=TEXT, zorder=5)
        ax.annotate(name, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=9, color=TEXT)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT)
    ax.set_title("Union Station → Yorkdale: CP4 vs CP6 safest route (risk scored per vehicle)",
                 loc="left", fontsize=10, color=TEXT)  # fmt: skip
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return {
        name: {"km": r.km, "minutes": r.minutes, "risk_per_million_trips": r.risk,
               "motorway_km": r.motorway_km, "uturns": r.uturns}
        for name, r in (("fastest", fast), ("cp6", new), ("cp4", old))
    }  # fmt: skip


def run_routing(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    reports, figures = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["figures_dir"])
    router = load_router(cfg)
    pairs = random_pairs(router, 200)
    calib = calibrate_k(router, pairs)
    (processed / ROUTING_K).write_text(json.dumps(calib, indent=2))
    lambdas = [0.5, 1, 2]
    t0 = time.time()
    trade = tradeoff_table(router, pairs, [0, *lambdas])
    per_route_s = (time.time() - t0) / (len(pairs) * 5)  # one fastest + four lambdas each
    summary = summarise(trade)

    sweeps = []
    for origin, destination in EXAMPLE_PAIRS:
        slug = f"{origin}_{destination}".lower().replace(" ", "_").replace(".", "")
        sweeps.append(
            plot_pareto(router, slug, origin, destination, figures / f"pareto_{slug}.png")
        )
    sweep = pd.concat(sweeps)
    cp4 = json.loads((Path(cfg["paths"]["reports_dir"]) / CP4_BASELINE).read_text())
    segments = gpd.read_parquet(processed / SEGMENTS)
    case = plot_before_after(
        router, cp4["union_yorkdale_lambda2"], 2.0, segments,
        figures / "before_after_union_yorkdale.png",
    )  # fmt: skip
    sens = run_sensitivity(cfg, router, pairs, lambdas)
    write_routing_report(
        calib, summary, sweep, per_route_s, cp4, case, sens, reports / "routing.md"
    )
    log.info("k = %.1f s per crash per million trips; wrote routing report",
             calib["k_seconds_per_crash"])  # fmt: skip
    return {"calibration": calib, "summary": summary, "case": case, "sensitivity": sens}


def cp4_vs_cp6(cp4: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    old = pd.DataFrame(cp4["summary_200_trips"]).set_index("lam")
    new = summary.set_index("lam")
    cols = ["median extra min", "p90 extra min", "median % risk reduction",
            "median detour (km ratio)", "p90 detour (km ratio)", "motorway share",
            "residential share"]  # fmt: skip
    rows = []
    for lam in new.index:
        for label, src in (("CP4 (count-based)", old), ("CP6 (per-vehicle)", new)):
            if lam in src.index:
                rows.append({"λ": f"{lam:g}", "version": label, **src.loc[lam, cols].to_dict()})
    return pd.DataFrame(rows)


def sensitivity_table(sens: pd.DataFrame) -> pd.DataFrame:
    cols = ["variant", "lam", "median extra min", "median % risk reduction",
            "median detour (km ratio)", "motorway share", "residential share"]  # fmt: skip
    return sens.loc[sens["lam"] > 0, cols].rename(columns={"lam": "λ"})


def case_by_variant(sens: pd.DataFrame) -> pd.DataFrame:
    first = sens.drop_duplicates("variant")
    return first[["variant", "union_yorkdale_km", "union_yorkdale_min"]].rename(
        columns={"union_yorkdale_km": "Union→Yorkdale λ=2 km",
                 "union_yorkdale_min": "minutes"}
    )  # fmt: skip


def sensitivity_verdict(sens: pd.DataFrame) -> list[str]:
    at1 = sens[sens["lam"] == 1].set_index("variant")
    base = at1.loc["baseline"]
    lines = []
    for v, r in at1.drop(index="baseline").iterrows():
        lines.append(
            f"- {v}: motorway share {base['motorway share']:.0%} -> {r['motorway share']:.0%}, "
            f"median extra minutes {base['median extra min']:.2f} -> {r['median extra min']:.2f}, "
            f"median risk reduction {base['median % risk reduction']:.0f}% -> "
            f"{r['median % risk reduction']:.0f}% (λ = 1)."
        )
    spread = at1["motorway share"].max() - at1["motorway share"].min()
    fastest = sens[sens["lam"] == 0].set_index("variant")["motorway share"]
    above = (at1["motorway share"] > fastest.reindex(at1.index)).all()
    lines.append(
        f"- Across all variants at λ = 1 the motorway share spans {spread:.0%} "
        f"({at1['motorway share'].min():.0%}-{at1['motorway share'].max():.0%}), against "
        f"{fastest.iloc[0]:.0%} on the fastest routes."
    )
    lines.append(
        "- **Conclusion:** "
        + (
            "the direction of the behaviour does not depend on the MTO/TPS comparability "
            "assumption: halving or doubling every freeway rate still leaves safer routes "
            "leaning on freeways (motorway share above the fastest routes' in every "
            "variant), and moves extra time and risk reduction by a fraction of their "
            "baseline values. The size of the freeway lean does depend on it."
            if above
            else "the freeway lean disappears in at least one variant, so the routing "
            "behaviour depends on the MTO/TPS comparability assumption."
        )
    )
    return lines


def write_routing_report(
    calib: dict,
    summary: pd.DataFrame,
    sweep: pd.DataFrame,
    per_route_s: float,
    cp4: dict[str, Any],
    case: dict[str, Any],
    sens: pd.DataFrame,
    path: Path,
) -> None:
    k = calib["k_seconds_per_crash"]
    hist = pd.DataFrame(calib["history"])
    s = summary.set_index("lam")
    old = pd.DataFrame(cp4["summary_200_trips"]).set_index("lam")
    c4, c6, cf = case["cp4"], case["cp6"], case["fastest"]
    lines = [
        "# Routing trade-off (CP6: per-vehicle risk)",
        "",
        "Edge cost = travel_time (s, free-flow) + λ · k · risk, where risk is the expected "
        "number of crashes for one vehicle driving the edge, per million vehicles "
        "(reports/risk.md). A route's risk is therefore expected crashes per million trips. "
        "Routing is turn-aware: a U-turn (driving back along the edge just used, or turning "
        f"more than 150° back onto the same street through an intersection) costs "
        f"{U_TURN_PENALTY_S:.0f} s.",
        "",
        "## Scaling constant k",
        "",
        f"k = **{k:.1f} seconds per (crash per million trips)**, the fixed point of "
        "k = median travel time / median route risk on λ = 1 routes over 200 random "
        "origin/destination pairs (3-25 km apart, seed 7). Check: median risk cost / median "
        f"travel time on the final λ = 1 routes = {calib['check_ratio_at_lambda1']:.3f}.",
        "",
        markdown_table(hist, floats="{:.2f}"),
        "",
        "## Trade-off over 200 random pairs",
        "",
        markdown_table(summary, floats="{:.2f}"),
        "",
        f"Mean routing time: {1000 * per_route_s:.0f} ms per route.",
        "",
        "## CP4 vs CP6",
        "",
        "CP4 routed on expected crashes per year at the locations a route passes (no "
        "exposure). The '% risk reduction' columns are each version's own risk measure, so "
        "they are not directly comparable; times, detours and road-class shares are.",
        "",
        markdown_table(cp4_vs_cp6(cp4, summary), floats="{:.2f}"),
        "",
        f"- Residential share at λ = 1: {old.loc[1, 'residential share']:.0%} in CP4, "
        f"{s.loc[1, 'residential share']:.0%} now (fastest routes: "
        f"{old.loc[0, 'residential share']:.0%} / {s.loc[0, 'residential share']:.0%}).",
        f"- Motorway share at λ = 1: {old.loc[1, 'motorway share']:.0%} in CP4, "
        f"{s.loc[1, 'motorway share']:.0%} now (fastest routes: "
        f"{old.loc[0, 'motorway share']:.0%} / {s.loc[0, 'motorway share']:.0%}).",
        "",
        "## Case study: Union Station → Yorkdale at λ = 2",
        "",
        "![Before and after](figures/before_after_union_yorkdale.png)",
        "",
        "| route | km | minutes | expected crashes per million trips | motorway km |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {label} | {r['km']:.1f} | {r['minutes']:.1f} | {r['risk_per_million_trips']:.1f} "
            f"| {r['motorway_km']:.1f} |"
            for label, r in (
                ("fastest", cf),
                ("CP4 safest (count-based)", c4),
                ("CP6 safest (per-vehicle)", c6),
            )
        ],  # fmt: skip
        "",
        "All three are scored with CP6 per-vehicle risk. The CP4 route ends at the node CP4 "
        "snapped Yorkdale to (a turning circle on Highland Hill, behind the mall).",
        "",
        "**The Yorkdale hook.** CP4 snapped the Yorkdale preset (the middle of the mall) to the "
        "nearest graph node: a turning circle at the end of Highland Hill, a residential "
        "cul-de-sac behind the mall. Every route therefore left the 401 at Dufferin and wound "
        "south through Jane Osler, Groton, Bentworth and Ranee before turning back north "
        "into the cul-de-sac, which is the hook. It was a snapping artifact, not a routing "
        "one: the route had no repeated nodes. Fix: snap like a navigation app (never to a "
        "dead end; a point more than 100 m from any street snaps to the nearest arterial or "
        "collector access if it is within 150 m of the nearest street), so Yorkdale now snaps "
        "to Yorkdale Road. Separately, node-based routing could not see U-turns through "
        "divided-road intersections, which per-vehicle routing uses as a cheap way to dodge a "
        "risky intersection; routing now runs over turn movements with a U-turn penalty "
        "(tests: no immediate reversals; a median U-turn loses to a 30 s detour).",
        "",
        "## Sensitivity: is the behaviour robust to the MTO/TPS comparability assumption?",
        "",
        "Freeway segment rates (400-series from MTO, city expressways from TPS) scaled by 0.5 "
        "and 2; Gardiner/DVP/Allen volumes ±30% (their per-vehicle rate scales by 1/1.3 and "
        "1/0.7). k stays at its baseline value.",
        "",
        markdown_table(sensitivity_table(sens), floats="{:.2f}"),
        "",
        markdown_table(case_by_variant(sens), floats="{:.1f}"),
        "",
        *sensitivity_verdict(sens),
        "",
        "## Things to watch",
        "",
        "- Free-flow travel times flatter long freeway detours; congestion would cut them.",
        "- Per-vehicle routing minimises the driver's own risk. It can still send traffic "
        "through local streets when they are safer per vehicle, which shifts risk onto "
        "pedestrians and residents there (a neighbourhood externality the cost does not "
        "include).",
        "- EB overpredicts the quietest intersections (lowest decile, reports/"
        "model_comparison.md), so per-vehicle risk on quiet local streets is if anything too "
        "high, which works against cut-throughs.",
        "",
        "## Pareto sweeps",
        "",
    ]
    for pair, g in sweep.groupby("pair", sort=False):
        lines += [
            f"### {pair.replace('_', ' ')}",
            "",
            f"![Pareto {pair}](figures/pareto_{pair}.png)",
            "",
            markdown_table(g.drop(columns="pair"), floats="{:.2f}"),
            "",
        ]
    path.write_text("\n".join(lines))
