"""Assign every located crash to a consolidated intersection, a mid-block segment, or nothing.

Rules, applied per distinct crash point and in this order:
1. inside an OSM off-street parking polygon      -> excluded, reason "in_parking"
2. within `intersection_m` of an intersection     -> that intersection
3. within `edge_m` of a drivable segment          -> that (undirected) segment
4. otherwise                                      -> excluded, reason "beyond_snap_thresholds"

Parking is checked first because a crash inside a lot is a lot crash even when the lot
touches an intersection. `snap_distance_m` is the distance to the assigned unit, or to the
nearest drivable segment for excluded crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from safepath.data.clean import CLEAN_FILENAME
from safepath.graph import INTERSECTIONS, SEGMENTS
from safepath.locations import crash_points, in_parking, nearest
from safepath.parking import load_or_fetch_parking

log = logging.getLogger(__name__)

ASSIGNMENTS = "crash_assignments.parquet"
INTERSECTION, SEGMENT, EXCLUDED = "intersection", "segment", "excluded"
IN_PARKING, BEYOND = "in_parking", "beyond_snap_thresholds"


def assign_points(
    points: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
    intersection_m: float,
    edge_m: float,
) -> pd.DataFrame:
    """Return one row per point: unit_type, unit_id, snap_distance_m, exclusion_reason."""
    ni = nearest(points, intersections, "intersection_id").reindex(points["point_id"])
    ns = nearest(points, segments, "segment_id").reindex(points["point_id"])
    d_int, d_seg = ni["dist"].to_numpy(), ns["dist"].to_numpy()

    parked = in_parking(points, parking) if len(parking) else np.zeros(len(points), bool)
    to_int = ~parked & (d_int <= intersection_m)
    to_seg = ~parked & ~to_int & (d_seg <= edge_m)

    out = pd.DataFrame({"point_id": points["point_id"].to_numpy()})
    out["unit_type"] = np.select([to_int, to_seg], [INTERSECTION, SEGMENT], EXCLUDED)
    out["unit_id"] = pd.array(
        np.where(to_int, ni["intersection_id"], np.where(to_seg, ns["segment_id"], -1)),
        dtype="Int64",
    )
    out.loc[out["unit_type"] == EXCLUDED, "unit_id"] = pd.NA
    out["snap_distance_m"] = np.where(to_int, d_int, d_seg)
    out["exclusion_reason"] = pd.Series(
        np.select([parked, ~to_int & ~to_seg], [IN_PARKING, BEYOND], ""), dtype="string"
    ).replace("", pd.NA)
    return out


def assign_crashes(clean: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    points = crash_points(clean, cfg["graph"]["crs"])
    snap = cfg["snap"]
    by_point = assign_points(
        points,
        gpd.read_parquet(processed / INTERSECTIONS),
        gpd.read_parquet(processed / SEGMENTS),
        load_or_fetch_parking(cfg),
        snap["intersection_m"],
        snap["edge_m"],
    )
    keys = points[["point_id", "lon", "lat"]].merge(by_point, on="point_id")
    crashes = clean[["event_id", "year", "lon", "lat"]].merge(
        keys, on=["lon", "lat"], how="left", validate="many_to_one"
    )
    if crashes["point_id"].isna().any():
        raise RuntimeError("some crashes did not match a crash point")
    return crashes.drop(columns=["lon", "lat"])


def run_snap(cfg: dict[str, Any]) -> Path:
    processed = Path(cfg["paths"]["processed_dir"])
    clean = pd.read_parquet(processed / CLEAN_FILENAME)
    assignments = assign_crashes(clean, cfg)
    out = processed / ASSIGNMENTS
    assignments.to_parquet(out, index=False)
    counts = assignments["unit_type"].value_counts().to_dict()
    reasons = assignments["exclusion_reason"].value_counts().to_dict()
    log.info(
        "assigned %d crashes: %s; exclusions %s; wrote %s", len(assignments), counts, reasons, out
    )
    return out
