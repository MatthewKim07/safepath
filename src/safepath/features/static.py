"""Static (time-invariant) OSM features per consolidated intersection.

Computed once from the current OSM snapshot. No coordinates or neighbourhood IDs: they let a
model memorise locations instead of learning transferable risk factors.
"""

from __future__ import annotations

import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from safepath.features.roads import approaches, class_combo
from safepath.locations import MOTORWAY_CLASSES, nearest

LEG_MERGE_DEG = 30  # approach bearings closer than this are one leg (e.g. divided carriageways)
BEARING_SAMPLE_M = 20  # bearing measured from the intersection to this far along the approach
TRUTHY = {"yes", "viaduct", "movable", "building_passage", "True", "true"}


def first_number(value: Any) -> float:
    """First number in an OSM tag value ('50', '40;50', "['3', '2']", '30 mph' -> km/h)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    m = re.search(r"\d+(\.\d+)?", text)
    if not m:
        return np.nan
    num = float(m.group())
    return num * 1.609344 if "mph" in text else num


def outward_bearings(appr: gpd.GeoDataFrame) -> np.ndarray:
    """Bearing (degrees, 0-360) of each approach, pointing away from the intersection."""
    length = appr.geometry.length.to_numpy()
    step = np.minimum(BEARING_SAMPLE_M, length / 2)
    at_u = appr["at_u"].to_numpy()
    start = shapely.line_interpolate_point(appr.geometry.values, np.where(at_u, 0, length))
    end = shapely.line_interpolate_point(appr.geometry.values, np.where(at_u, step, length - step))
    dx = shapely.get_x(end) - shapely.get_x(start)
    dy = shapely.get_y(end) - shapely.get_y(start)
    return np.degrees(np.arctan2(dy, dx)) % 360


def legs_and_min_angle(bearings: np.ndarray) -> tuple[int, float]:
    """Group circular bearings into legs (gaps < LEG_MERGE_DEG merge) and return the leg
    count and the smallest angle between adjacent legs (180 when there is one leg)."""
    b = np.sort(bearings)
    if len(b) == 0:
        return 0, 180.0
    gaps = np.diff(np.append(b, b[0] + 360))
    breaks = gaps >= LEG_MERGE_DEG
    n_legs = int(breaks.sum())
    if n_legs <= 1:
        return max(n_legs, 1), 180.0
    # leg direction = circular mean of its bearings; groups start after each break
    starts = np.roll(breaks, 1)
    group = np.cumsum(starts) % n_legs
    rad = np.radians(b)
    centres = np.array(
        [
            np.degrees(np.arctan2(np.sin(rad[group == k]).mean(), np.cos(rad[group == k]).mean()))
            % 360
            for k in range(n_legs)
        ]
    )
    c = np.sort(centres)
    angles = np.diff(np.append(c, c[0] + 360))
    return n_legs, float(min(angles.min(), 180.0))


def approach_features(segments: gpd.GeoDataFrame, node_map: pd.DataFrame) -> pd.DataFrame:
    appr = approaches(segments, node_map)
    appr = gpd.GeoDataFrame(appr, geometry="geometry", crs=segments.crs)
    appr["bearing"] = outward_bearings(appr)
    appr["maxspeed_kph"] = appr["maxspeed"].map(first_number)
    appr["lanes_n"] = appr["lanes"].map(first_number)
    # Impute missing tags with the median for the approach's road class.
    for col in ("maxspeed_kph", "lanes_n"):
        appr[f"{col}_tagged"] = appr[col].notna()
        by_class = appr.groupby("highway")[col].transform("median")
        appr[col] = appr[col].fillna(by_class).fillna(appr[col].median())
    appr["is_oneway"] = appr["oneway"].astype(str).isin(TRUTHY)
    appr["is_bridge_tunnel"] = appr["bridge"].isin(TRUTHY) | appr["tunnel"].isin(TRUTHY)

    g = appr.groupby("intersection_id")
    out = class_combo(appr)
    legs = g["bearing"].apply(lambda s: legs_and_min_angle(s.to_numpy()))
    out["n_legs"] = legs.map(lambda t: t[0])
    out["min_leg_angle_deg"] = legs.map(lambda t: t[1])
    out["n_approaches"] = g.size()
    out["max_maxspeed_kph"] = g["maxspeed_kph"].max()
    out["min_maxspeed_kph"] = g["maxspeed_kph"].min()
    out["maxspeed_missing"] = ~g["maxspeed_kph_tagged"].any()
    out["max_lanes"] = g["lanes_n"].max()
    out["lanes_missing"] = ~g["lanes_n_tagged"].any()
    out["n_oneway_approaches"] = g["is_oneway"].sum()
    out["any_bridge_tunnel"] = g["is_bridge_tunnel"].any()
    out["has_motorway_approach"] = g["highway"].agg(lambda s: bool(set(s) & MOTORWAY_CLASSES))
    return out


def count_within(centres: np.ndarray, points: gpd.GeoDataFrame, radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(len(centres), dtype=int)
    tree = cKDTree(np.column_stack([points.geometry.x, points.geometry.y]))
    return tree.query_ball_point(centres, r=radius, return_length=True)


def nearest_dist(intersections: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> np.ndarray:
    pts = intersections[["intersection_id", "geometry"]].rename(
        columns={"intersection_id": "point_id"}
    )
    tg = targets[["geometry"]].reset_index(drop=True).rename_axis("tid").reset_index()
    return nearest(pts, tg, "tid")["dist"].reindex(pts["point_id"]).to_numpy()


def context_features(
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    xy = np.column_stack([intersections.geometry.x, intersections.geometry.y])
    control = layers["traffic_control"]
    out = pd.DataFrame(index=intersections["intersection_id"].to_numpy())
    out["n_signals_30m"] = count_within(xy, control[control["highway"] == "traffic_signals"], 30)
    out["n_stop_signs_30m"] = count_within(xy, control[control["highway"] == "stop"], 30)
    out["n_crossings_30m"] = count_within(xy, control[control["highway"] == "crossing"], 30)
    out["has_signal"] = out["n_signals_30m"] > 0
    out["n_bus_stops_50m"] = count_within(xy, layers["bus_stops"], 50)
    out["n_pois_100m"] = count_within(xy, layers["pois"], 100)
    out["dist_tram_m"] = nearest_dist(intersections, layers["tram"])
    out["streetcar_20m"] = out["dist_tram_m"] <= 20
    out["dist_motorway_m"] = nearest_dist(
        intersections, segments[segments["highway"].isin(MOTORWAY_CLASSES)]
    )
    out["dist_retail_m"] = nearest_dist(intersections, layers["retail"])
    # spacing: distance to the nearest other intersection
    dist, _ = cKDTree(xy).query(xy, k=2)
    out["dist_nearest_intersection_m"] = dist[:, 1]
    return out.rename_axis("intersection_id")


def static_features(
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    node_map: pd.DataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    base = intersections.set_index("intersection_id")[["n_osm_nodes", "street_count"]]
    out = base.join(approach_features(segments, node_map)).join(
        context_features(intersections, segments, layers)
    )
    # Intersections whose every approach is internal have no approach rows (none in Toronto
    # at tolerance 10 m, but keep the fill so the pipeline never emits NaNs).
    fill = {
        "max_class_rank": 0,
        "min_class_rank": 0,
        "class_combo": "local x local",
        "n_legs": 0,
        "min_leg_angle_deg": 180.0,
        "n_approaches": 0,
        "n_oneway_approaches": 0,
        "maxspeed_missing": True,
        "lanes_missing": True,
        "any_bridge_tunnel": False,
        "has_motorway_approach": False,
    }
    out = out.fillna(fill)
    for col in ("max_maxspeed_kph", "min_maxspeed_kph", "max_lanes"):
        out[col] = out[col].fillna(out[col].median())
    return out
