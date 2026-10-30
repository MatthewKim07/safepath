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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.graph import NODE_MAP
from safepath.locations import MOTORWAY_CLASSES, markdown_table
from safepath.movements import U_TURN_PENALTY_S, bearing, build_movements
from safepath.risk import ROUTING_GRAPH

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
    ax.set_ylabel("expected crashes/yr at route locations", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pts.assign(pair=name)


def load_router(cfg: dict[str, Any]) -> Router:
    processed = Path(cfg["paths"]["processed_dir"])
    with open(processed / ROUTING_GRAPH, "rb") as f:
        G = pickle.load(f)
    node_map = pd.read_parquet(processed / NODE_MAP)
    junction = dict(zip(node_map["osmid"], node_map["intersection_id"], strict=True))
    router = Router(G, node_junction=junction)
    k_path = processed / ROUTING_K
    if k_path.exists():
        router.k = json.loads(k_path.read_text())["k_seconds_per_crash"]
    return router


def run_routing(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    reports, figures = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["figures_dir"])
    router = load_router(cfg)
    pairs = random_pairs(router, 200)
    calib = calibrate_k(router, pairs)
    (processed / ROUTING_K).write_text(json.dumps(calib, indent=2))
    t0 = time.time()
    trade = tradeoff_table(router, pairs, [0, 0.5, 1, 2])
    per_route_s = (time.time() - t0) / (len(pairs) * 5)  # one fastest + four lambdas each
    summary = summarise(trade)

    sweeps = []
    for origin, destination in EXAMPLE_PAIRS:
        slug = f"{origin}_{destination}".lower().replace(" ", "_").replace(".", "")
        sweeps.append(
            plot_pareto(router, slug, origin, destination, figures / f"pareto_{slug}.png")
        )
    sweep = pd.concat(sweeps)
    write_routing_report(calib, summary, sweep, per_route_s, reports / "routing.md")
    log.info("k = %.1f s per crash/yr; wrote routing report", calib["k_seconds_per_crash"])
    return {"calibration": calib, "summary": summary}


def write_routing_report(
    calib: dict, summary: pd.DataFrame, sweep: pd.DataFrame, per_route_s: float, path: Path
) -> None:
    k = calib["k_seconds_per_crash"]
    hist = pd.DataFrame(calib["history"])
    s0 = summary.set_index("lam").loc[0]
    s1 = summary.set_index("lam").loc[1]
    lines = [
        "# Routing trade-off",
        "",
        "Edge cost = travel_time (s, free-flow OSMnx speeds) + λ · k · risk, where risk is "
        "expected crashes per year at the edge's location (intersection entered + segment).",
        "",
        "## Scaling constant k",
        "",
        f"k = **{k:.1f} seconds per (crash/yr)**. Chosen as a fixed point over 200 random "
        "origin/destination pairs (3-25 km apart, seed 7): starting from the fastest routes, "
        "re-route at λ = 1 and set k = median travel time / median route risk until it moves "
        f"less than 1%. Check: on the final λ = 1 routes, median risk cost / median travel "
        f"time = {calib['check_ratio_at_lambda1']:.3f}. So λ = 1 means 'a typical route's "
        "risk is weighted as much as its travel time'.",
        "",
        markdown_table(hist, floats="{:.1f}"),
        "",
        "## Trade-off over 200 random pairs",
        "",
        markdown_table(summary, floats="{:.2f}"),
        "",
        "Detour = route km / fastest-route km. Shares are mean fractions of route km on "
        "motorways (incl. ramps) and on residential streets; λ = 0 values are in the "
        "Pareto tables below via the fastest route.",
        "",
        f"Mean routing time: {1000 * per_route_s:.0f} ms per route (scipy Dijkstra).",
        "",
        "## Things to watch",
        "",
        f"- Residential share of route km rises from {s0['residential share']:.0%} at λ = 0 "
        f"to {s1['residential share']:.0%} at λ = 1: safer routes cut through local streets, "
        "whose small intersections carry little predicted risk, to avoid large arterial "
        "intersections.",
        f"- Motorway share rises from {s0['motorway share']:.0%} to {s1['motorway share']:.0%}. "
        "Motorway risk is a per-km base rate that is low next to arterial intersections, and "
        "most 400-series highways in Toronto are patrolled by the OPP, so TPS data undercounts "
        "their crashes. Some safest routes take long motorway detours as a result (worst "
        "example below: Union Station to Yorkdale).",
        "- Risk is expected crashes per year at the locations on a route, with no traffic "
        "volume term, so a '% risk reduction' means the route avoids crash-prone places, not "
        "that a driver's own crash probability falls by that much. Travel times are "
        "free-flow, which flatters motorways further.",
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
