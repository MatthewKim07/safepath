"""Build the Toronto drive graph, consolidated intersections, and the undirected segment view.

Outputs in data/processed/:
- toronto_drive.graphml            directed graph with speeds and travel times (WGS84)
- toronto_drive_proj.graphml       the same graph projected to UTM 17N, used for routing
- toronto_intersections.graphml    consolidated-intersection graph at the chosen tolerance
- intersections.parquet            one row per consolidated intersection (GeoParquet)
- intersections_tol{T}.parquet     the same for every candidate tolerance, for comparison
- node_to_intersection.parquet     original OSM node -> intersection_id
- segments.parquet                 undirected, deduplicated road segments (GeoParquet)
- edge_to_segment.parquet          directed edge (u, v, key) -> segment_id
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

log = logging.getLogger(__name__)

RAW_GRAPH = "toronto_drive.graphml"
PROJ_GRAPH = "toronto_drive_proj.graphml"
CONS_GRAPH = "toronto_intersections.graphml"
INTERSECTIONS = "intersections.parquet"
NODE_MAP = "node_to_intersection.parquet"
SEGMENTS = "segments.parquet"
EDGE_MAP = "edge_to_segment.parquet"
GRAPH_STATS = "graph_stats.json"


def configure_osmnx(cfg: dict[str, Any]) -> None:
    ox.settings.cache_folder = str(Path(cfg["paths"]["raw_dir"]) / "osmnx_cache")
    ox.settings.use_cache = True
    ox.settings.log_console = False


def _tag(value: Any) -> str | None:
    """First value of a possibly list-valued OSM tag, as a string; None if missing."""
    value = _as_list(value)
    return str(value[0]) if value else None


def _as_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return value if isinstance(value, list) else [value]


def download_graph(place: str, network_type: str) -> nx.MultiDiGraph:
    G = ox.graph_from_place(place, network_type=network_type, simplify=True)
    G = ox.add_edge_speeds(G)
    return ox.add_edge_travel_times(G)


def load_or_build_projected(cfg: dict[str, Any]) -> nx.MultiDiGraph:
    out = Path(cfg["paths"]["processed_dir"])
    raw_path, proj_path = out / RAW_GRAPH, out / PROJ_GRAPH
    if proj_path.exists():
        return ox.load_graphml(proj_path)
    if raw_path.exists():
        G = ox.load_graphml(raw_path)
    else:
        g = cfg["graph"]
        t0 = time.time()
        G = download_graph(g["place"], g["network_type"])
        log.info("downloaded graph in %.0fs", time.time() - t0)
        ox.save_graphml(G, raw_path)
    Gp = ox.project_graph(G, to_crs=cfg["graph"]["crs"])
    ox.save_graphml(Gp, proj_path)
    return Gp


def consolidate(Gp: nx.MultiDiGraph, tolerance: float) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """Merge intersection nodes within 2 * tolerance (OSMnx buffers each node by `tolerance`).

    Dead-end nodes are dropped, so the result contains street intersections only. Returns the
    consolidated graph and a frame mapping every original intersection node to its cluster.
    """
    Gc = ox.consolidate_intersections(
        Gp, tolerance=tolerance, rebuild_graph=True, dead_ends=False, reconnect_edges=True
    )
    rows = [
        (orig, cid)
        for cid, data in Gc.nodes(data=True)
        for orig in _as_list(data["osmid_original"])
    ]
    node_map = pd.DataFrame(rows, columns=["osmid", "intersection_id"])
    return Gc, node_map


def intersection_frame(
    Gc: nx.MultiDiGraph, node_map: pd.DataFrame, Gp: nx.MultiDiGraph
) -> gpd.GeoDataFrame:
    """One row per consolidated intersection with its member count, street names, and road
    classes taken from the original edges incident to its member nodes."""
    names: dict[int, set[str]] = {}
    classes: dict[int, set[str]] = {}
    member_of = dict(zip(node_map["osmid"], node_map["intersection_id"], strict=True))
    for u, v, data in Gp.edges(data=True):
        for node in (u, v):
            cid = member_of.get(node)
            if cid is None:
                continue
            names.setdefault(cid, set()).update(_as_list(data.get("name")))
            classes.setdefault(cid, set()).update(_as_list(data.get("highway")))

    nodes = pd.DataFrame(
        [(cid, d["x"], d["y"], d.get("street_count", 0)) for cid, d in Gc.nodes(data=True)],
        columns=["intersection_id", "x", "y", "street_count"],
    )
    nodes["n_osm_nodes"] = nodes["intersection_id"].map(node_map.groupby("intersection_id").size())
    nodes["street_names"] = nodes["intersection_id"].map(
        lambda c: " / ".join(sorted(names.get(c, set())))
    )
    nodes["highway_classes"] = nodes["intersection_id"].map(
        lambda c: ",".join(sorted(classes.get(c, set())))
    )
    return gpd.GeoDataFrame(
        nodes, geometry=gpd.points_from_xy(nodes["x"], nodes["y"]), crs=Gp.graph["crs"]
    )


def segment_frames(Gp: nx.MultiDiGraph) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Collapse each two-way street's reciprocal directed edges into one undirected segment.

    Two directed edges are the same segment when they join the same node pair with the same
    length (to 0.1 m); one-way carriageways of a divided road stay separate segments.
    """
    edges = ox.graph_to_gdfs(Gp, nodes=False).reset_index()
    a = np.minimum(edges["u"], edges["v"])
    b = np.maximum(edges["u"], edges["v"])
    canon = pd.DataFrame({"a": a, "b": b, "len": edges["length"].round(1)})
    edges["segment_id"] = canon.groupby(["a", "b", "len"], sort=True).ngroup()

    edge_map = edges[["u", "v", "key", "segment_id"]].copy()
    first = edges.sort_values(["segment_id", "u"]).drop_duplicates("segment_id")
    seg = first[["segment_id", "u", "v", "key", "length", "geometry"]].copy()
    for col in ("highway", "name", "oneway", "maxspeed", "lanes", "speed_kph"):
        values = first[col] if col in first else pd.Series(None, index=first.index)
        seg[col] = values.map(_tag)
    seg["n_directed_edges"] = seg["segment_id"].map(edge_map.groupby("segment_id").size())
    seg = gpd.GeoDataFrame(seg, geometry="geometry", crs=Gp.graph["crs"])
    return seg.reset_index(drop=True), edge_map


def build_graph(cfg: dict[str, Any]) -> dict[str, Any]:
    configure_osmnx(cfg)
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    gcfg = cfg["graph"]

    Gp = load_or_build_projected(cfg)
    stats: dict[str, Any] = {
        "crs": gcfg["crs"],
        "directed_nodes": Gp.number_of_nodes(),
        "directed_edges": Gp.number_of_edges(),
        "nodes_by_street_count": {
            str(k): v
            for k, v in sorted(
                pd.Series(dict(Gp.nodes(data="street_count"))).value_counts().items()
            )
        },
        "consolidation": {},
    }
    log.info(
        "projected graph: %d nodes, %d edges", stats["directed_nodes"], stats["directed_edges"]
    )

    chosen = gcfg["consolidation_tolerance_m"]
    for tol in sorted(set(gcfg["tolerance_candidates_m"]) | {chosen}):
        t0 = time.time()
        Gc, node_map = consolidate(Gp, tol)
        inter = intersection_frame(Gc, node_map, Gp)
        inter.to_parquet(out / f"intersections_tol{tol}.parquet")
        sizes = inter["n_osm_nodes"]
        stats["consolidation"][str(tol)] = {
            "intersections": Gc.number_of_nodes(),
            "edges": Gc.number_of_edges(),
            "original_intersection_nodes": len(node_map),
            "merged_clusters": int((sizes > 1).sum()),
            "max_nodes_in_cluster": int(sizes.max()),
            "clusters_with_5plus_nodes": int((sizes >= 5).sum()),
        }
        log.info(
            "tolerance %sm: %s (%.0fs)", tol, stats["consolidation"][str(tol)], time.time() - t0
        )
        if tol == chosen:
            ox.save_graphml(Gc, out / CONS_GRAPH)
            inter.to_parquet(out / INTERSECTIONS)
            node_map.to_parquet(out / NODE_MAP, index=False)

    seg, edge_map = segment_frames(Gp)
    seg.to_parquet(out / SEGMENTS)
    edge_map.to_parquet(out / EDGE_MAP, index=False)
    stats["segments"] = len(seg)
    stats["chosen_tolerance_m"] = chosen
    (out / GRAPH_STATS).write_text(json.dumps(stats, indent=2))
    log.info("segments: %d undirected from %d directed edges", len(seg), len(edge_map))
    return stats


def run_graph(cfg: dict[str, Any]) -> None:
    build_graph(cfg)
