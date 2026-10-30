"""Features for the traffic-volume model: OSM attributes and sampled betweenness.

Intersections reuse the 27 CP3 static features and add betweenness summed over entering
edges; segments get road attributes, endpoint context, nearby activity, and betweenness
summed over their directed edges. No coordinates or neighbourhood IDs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from safepath.features.build import STATIC
from safepath.features.centrality import edge_betweenness
from safepath.features.roads import class_rank
from safepath.features.static import TRUTHY, count_within, first_number
from safepath.graph import EDGE_MAP, NODE_MAP
from safepath.locations import MOTORWAY_CLASSES
from safepath.risk import ROUTING_GRAPH
from safepath.routing import PRESETS, Router

EDGE_CENTRALITY = "edge_centrality.parquet"
# global: random sources, every destination; local: destinations within 5 minutes
CENTRALITY = {"bt_global": {"n_sources": 2000}, "bt_local": {"n_sources": 5000, "limit": 300}}


def edge_centrality(cfg: dict[str, Any]) -> pd.DataFrame:
    """Global and local sampled betweenness per directed edge (cached)."""
    processed = Path(cfg["paths"]["processed_dir"])
    path = processed / EDGE_CENTRALITY
    if path.exists():
        return pd.read_parquet(path)
    with open(processed / ROUTING_GRAPH, "rb") as f:
        router = Router(pickle.load(f))
    out = pd.DataFrame(
        {"u": router.nodes[router.u], "v": router.nodes[router.v], "key": router.key}
    )
    for name, kw in CENTRALITY.items():
        out[name] = edge_betweenness(
            router.u, router.v, router.travel_time, len(router.nodes), **kw
        )
    out.to_parquet(path, index=False)
    return out


def downtown_xy(crs: str) -> tuple[float, float]:
    lat, lon = PRESETS["Union Station"]
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)


def intersection_features(
    cfg: dict[str, Any], cent: pd.DataFrame, intersections: gpd.GeoDataFrame
) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    static = pd.read_parquet(processed / STATIC)
    cluster = pd.read_parquet(processed / NODE_MAP).set_index("osmid")["intersection_id"]
    e = cent.assign(cu=cent["u"].map(cluster), cv=cent["v"].map(cluster))
    entering = e[e["cv"].notna() & (e["cu"] != e["cv"])]
    g = entering.groupby(entering["cv"].astype(int))
    out = static.copy()
    out["bt_global_in"] = g["bt_global"].sum().reindex(out.index).fillna(0)
    out["bt_local_in"] = g["bt_local"].sum().reindex(out.index).fillna(0)
    out["bt_global_in_max"] = g["bt_global"].max().reindex(out.index).fillna(0)
    xy = intersections.set_index("intersection_id").loc[out.index, ["x", "y"]].to_numpy()
    dx, dy = downtown_xy(intersections.crs)
    out["dist_downtown_m"] = np.hypot(xy[:, 0] - dx, xy[:, 1] - dy)
    out["n_intersections_500m"] = cKDTree(xy).query_ball_point(xy, r=500, return_length=True)
    out["x"], out["y"] = xy[:, 0], xy[:, 1]
    return out


def divided_carriageways(seg: gpd.GeoDataFrame, max_m: float = 40) -> np.ndarray:
    """One-way segments with a same-named one-way segment running the opposite way within
    `max_m`: one carriageway of a divided road, carrying about half its two-way volume."""
    one = seg[seg["is_oneway"] & seg["name"].notna()].copy()
    start = np.column_stack(
        [one.geometry.apply(lambda g: g.coords[0][0]), one.geometry.apply(lambda g: g.coords[0][1])]
    )
    end = np.column_stack(
        [
            one.geometry.apply(lambda g: g.coords[-1][0]),
            one.geometry.apply(lambda g: g.coords[-1][1]),
        ]
    )
    d = end - start
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    pos = {sid: i for i, sid in enumerate(one["segment_id"])}
    left, right = one.sindex.query(one.geometry.buffer(max_m), predicate="intersects")
    keep = left != right
    left, right = left[keep], right[keep]
    names = one["name"].to_numpy()
    same = names[left] == names[right]
    opposite = (d[left] * d[right]).sum(axis=1) < -0.7
    hit = np.zeros(len(one), bool)
    hit[left[same & opposite]] = True
    out = np.zeros(len(seg), bool)
    sid_pos = seg["segment_id"].map(pos)
    found = sid_pos.notna().to_numpy()
    out[found] = hit[sid_pos[found].astype(int).to_numpy()]
    return out


def segment_features(
    cfg: dict[str, Any],
    cent: pd.DataFrame,
    segments: gpd.GeoDataFrame,
    inter_feats: pd.DataFrame,
) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    edge_map = pd.read_parquet(processed / EDGE_MAP)
    cluster = pd.read_parquet(processed / NODE_MAP).set_index("osmid")["intersection_id"]
    c = cent.merge(edge_map, on=["u", "v", "key"])
    g = c.groupby("segment_id")
    seg = segments.copy()
    seg["bt_global"] = seg["segment_id"].map(g["bt_global"].sum()).fillna(0)
    seg["bt_local"] = seg["segment_id"].map(g["bt_local"].sum()).fillna(0)
    seg["class_rank"] = class_rank(seg["highway"])
    seg["lanes_n"] = seg["lanes"].map(first_number)
    seg["maxspeed_kph"] = seg["maxspeed"].map(first_number)
    for col in ("lanes_n", "maxspeed_kph"):
        seg[col.replace("_n", "").replace("_kph", "") + "_missing"] = seg[col].isna()
        seg[col] = seg[col].fillna(seg.groupby("highway")[col].transform("median"))
    seg["is_oneway"] = seg["oneway"].astype(str).isin(TRUTHY)
    seg["is_divided"] = divided_carriageways(seg)
    seg["is_bridge_tunnel"] = seg["bridge"].isin(TRUTHY) | seg["tunnel"].isin(TRUTHY)
    ends = pd.DataFrame({"a": seg["u"].map(cluster), "b": seg["v"].map(cluster)})
    rank = inter_feats["max_class_rank"]
    sig = inter_feats["has_signal"].astype(float)
    seg["end_max_class_rank"] = np.fmax(ends["a"].map(rank), ends["b"].map(rank)).fillna(0)
    seg["end_has_signal"] = np.fmax(ends["a"].map(sig), ends["b"].map(sig)).fillna(0)
    mid = seg.geometry.interpolate(0.5, normalized=True)
    xy = np.column_stack([mid.x, mid.y])
    layers = {n: gpd.read_parquet(processed / f"osm_{n}.parquet") for n in ("pois", "bus_stops")}
    seg["n_pois_100m"] = count_within(xy, layers["pois"].to_crs(seg.crs), 100)
    seg["n_bus_stops_50m"] = count_within(xy, layers["bus_stops"].to_crs(seg.crs), 50)
    tram = gpd.read_parquet(processed / "osm_tram.parquet").to_crs(seg.crs)
    seg["dist_tram_m"] = gpd.GeoSeries(mid, crs=seg.crs).distance(tram.union_all()).to_numpy()
    motorway = seg[seg["highway"].isin(MOTORWAY_CLASSES)].union_all()
    seg["dist_motorway_m"] = mid.distance(motorway).to_numpy()
    dx, dy = downtown_xy(seg.crs)
    seg["dist_downtown_m"] = np.hypot(xy[:, 0] - dx, xy[:, 1] - dy)
    ixy = inter_feats[["x", "y"]].to_numpy()
    seg["n_intersections_500m"] = cKDTree(ixy).query_ball_point(xy, r=500, return_length=True)
    seg["x"], seg["y"] = xy[:, 0], xy[:, 1]
    return pd.DataFrame(seg.drop(columns="geometry"))
