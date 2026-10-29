"""Where do TPS crash points sit relative to the road network?

Crash locations are quantised to a small set of points, so the analysis works on distinct
points weighted by their crash counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from safepath.data.clean import CLEAN_FILENAME
from safepath.graph import INTERSECTIONS, SEGMENTS
from safepath.parking import load_or_fetch_parking

log = logging.getLogger(__name__)

POINTS = "crash_points.parquet"


def crash_points(clean: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Distinct crash coordinates with crash counts, projected to `crs`."""
    pts = (
        clean.groupby(["lon", "lat"])
        .agg(
            n_crashes=("event_id", "size"),
            n_severe=("injury_or_fatal", "sum"),
            n_fail_to_remain=("is_fail_to_remain", "sum"),
        )
        .reset_index()
    )
    pts.insert(0, "point_id", range(len(pts)))
    gdf = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326"
    )
    return gdf.to_crs(crs)


def nearest(points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame:
    """Nearest target id and distance for every point (ties broken by lowest id)."""
    joined = gpd.sjoin_nearest(
        points[["point_id", "geometry"]],
        targets[[id_col, "geometry"]],
        how="left",
        distance_col="dist",
    )
    joined = joined.sort_values(["point_id", id_col]).drop_duplicates("point_id")
    return joined.set_index("point_id")[[id_col, "dist"]]


def in_parking(points: gpd.GeoDataFrame, parking: gpd.GeoDataFrame) -> pd.Series:
    hits = gpd.sjoin(
        points[["point_id", "geometry"]], parking[["geometry"]], how="inner", predicate="within"
    )
    return points["point_id"].isin(hits["point_id"]).to_numpy()


def annotate_points(
    points: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add nearest intersection / segment (id and distance) and a parking flag to each point."""
    out = points.copy()
    ni = nearest(out, intersections, "intersection_id").reindex(out["point_id"])
    ns = nearest(out, segments, "segment_id").reindex(out["point_id"])
    out["intersection_id"] = ni["intersection_id"].to_numpy()
    out["dist_intersection"] = ni["dist"].to_numpy()
    out["segment_id"] = ns["segment_id"].to_numpy()
    out["dist_edge"] = ns["dist"].to_numpy()
    out["in_parking"] = in_parking(out, parking)
    return out


def load_inputs(cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, ...]:
    processed = Path(cfg["paths"]["processed_dir"])
    clean = pd.read_parquet(processed / CLEAN_FILENAME)
    points = crash_points(clean, cfg["graph"]["crs"])
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    parking = load_or_fetch_parking(cfg)
    return points, intersections, segments, parking


def top_points_table(
    pts: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    n: int = 50,
) -> pd.DataFrame:
    top = pts.nlargest(n, "n_crashes").copy()
    top["nearest_intersection"] = top["intersection_id"].map(
        intersections.set_index("intersection_id")["street_names"]
    )
    seg = segments.set_index("segment_id")
    top["nearest_road"] = top["segment_id"].map(seg["name"]).fillna("(unnamed)")
    top["road_class"] = top["segment_id"].map(seg["highway"])
    top["pct_severe"] = 100 * top["n_severe"] / top["n_crashes"]
    top["pct_ftr"] = 100 * top["n_fail_to_remain"] / top["n_crashes"]
    return top


def write_top_points_report(top: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# Top {len(top)} busiest TPS crash points",
        "",
        "Distances are to the nearest consolidated intersection and nearest drivable segment "
        "(UTM 17N metres). `parking` = inside an OSM off-street parking polygon.",
        "",
        "| # | crashes | % severe | % FTR | m to int. | m to road | parking | nearest road "
        "(class) | nearest intersection |",
        "|---:|---:|---:|---:|---:|---:|:---:|---|---|",
    ]
    for rank, r in enumerate(top.itertuples(), start=1):
        lines.append(
            f"| {rank} | {r.n_crashes:,} | {r.pct_severe:.1f} | {r.pct_ftr:.1f} | "
            f"{r.dist_intersection:.0f} | {r.dist_edge:.0f} | {'yes' if r.in_parking else ''} | "
            f"{r.nearest_road} ({r.road_class}) | {r.nearest_intersection or '(unnamed)'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_locations(cfg: dict[str, Any]) -> None:
    points, intersections, segments, parking = load_inputs(cfg)
    pts = annotate_points(points, intersections, segments, parking)
    pts.to_parquet(Path(cfg["paths"]["processed_dir"]) / POINTS)
    top = top_points_table(pts, intersections, segments)
    write_top_points_report(top, Path(cfg["paths"]["reports_dir"]) / "top_crash_points.md")
    log.info("annotated %d crash points; wrote top-50 table", len(pts))
