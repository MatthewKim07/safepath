"""Where do TPS crash points sit relative to the road network?

Crash locations are quantised to a small set of points, so the analysis works on distinct
points weighted by their crash counts. This is the evidence for choosing the unit of
analysis (intersection vs mid-block segment), the consolidation tolerance, and the snapping
thresholds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safepath.data.clean import CLEAN_FILENAME
from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.graph import GRAPH_STATS, INTERSECTIONS, SEGMENTS
from safepath.parking import load_or_fetch_parking

log = logging.getLogger(__name__)

POINTS = "crash_points.parquet"
MOTORWAY_CLASSES = {"motorway", "motorway_link"}
ORANGE, AQUA, RED = "#eb6834", "#1baf7a", "#e34948"


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


def in_parking(points: gpd.GeoDataFrame, parking: gpd.GeoDataFrame) -> np.ndarray:
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
    """Add nearest intersection / segment / motorway / parking lot distances to each point."""
    out = points.copy()
    ni = nearest(out, intersections, "intersection_id").reindex(out["point_id"])
    ns = nearest(out, segments, "segment_id").reindex(out["point_id"])
    out["intersection_id"] = ni["intersection_id"].to_numpy()
    out["dist_intersection"] = ni["dist"].to_numpy()
    out["segment_id"] = ns["segment_id"].to_numpy()
    out["dist_edge"] = ns["dist"].to_numpy()
    seg = segments.set_index("segment_id")
    out["road_class"] = out["segment_id"].map(seg["highway"])
    out["road_oneway"] = out["segment_id"].map(seg["oneway"]).eq("True")
    out["intersection_n_nodes"] = out["intersection_id"].map(
        intersections.set_index("intersection_id")["n_osm_nodes"]
    )
    motorways = segments[segments["highway"].isin(MOTORWAY_CLASSES)]
    out["dist_motorway"] = (
        nearest(out, motorways, "segment_id").reindex(out["point_id"])["dist"].to_numpy()
    )
    lots = parking.reset_index(drop=True).rename_axis("lot").reset_index()
    out["dist_parking"] = nearest(out, lots, "lot").reindex(out["point_id"])["dist"].to_numpy()
    out["in_parking"] = in_parking(out, parking)
    return out


def location_summary(pts: pd.DataFrame) -> pd.DataFrame:
    """Crash-weighted and point shares for the categories that decide the unit of analysis."""
    d, e = pts["dist_intersection"], pts["dist_edge"]
    cats = {f"within {k} m of an intersection": d <= k for k in (5, 10, 20, 30, 50)}
    cats |= {
        "10-30 m from an intersection": (d > 10) & (d <= 30),
        "mid-block: <= 20 m from an edge, > 30 m from any intersection": (e <= 20) & (d > 30),
        "off-network: > 50 m from any drivable edge": e > 50,
        "inside an OSM parking polygon": pts["in_parking"],
        "within 25 m of a motorway / ramp": pts["dist_motorway"] <= 25,
        "within 25 m of a motorway and > 30 m from any intersection": (
            (pts["dist_motorway"] <= 25) & (d > 30)
        ),
    }
    w, total = pts["n_crashes"], pts["n_crashes"].sum()
    rows = [
        {
            "category": name,
            "crashes": int(w[mask].sum()),
            "pct_crashes": 100 * w[mask].sum() / total,
            "points": int(mask.sum()),
            "pct_points": 100 * mask.mean(),
        }
        for name, mask in cats.items()
    ]
    return pd.DataFrame(rows)


def tolerance_comparison(
    points: gpd.GeoDataFrame, processed: Path, tolerances: list[float], graph_stats: dict
) -> pd.DataFrame:
    """For each consolidation tolerance: how close crash points sit to cluster centroids, and
    how many clusters absorb more than one distinct TPS point (a sign of over-merging)."""
    w, total = points["n_crashes"].to_numpy(), points["n_crashes"].sum()
    rows = []
    for tol in tolerances:
        inter = gpd.read_parquet(processed / f"intersections_tol{tol}.parquet")
        n = nearest(points, inter, "intersection_id").reindex(points["point_id"])
        dist = n["dist"].to_numpy()
        near = dist <= 20
        per_cluster = pd.Series(n["intersection_id"].to_numpy()[near]).value_counts()
        multi = set(per_cluster[per_cluster > 1].index)
        in_multi = near & np.isin(n["intersection_id"].to_numpy(), list(multi))
        rows.append(
            {
                "tolerance_m": tol,
                "merge_distance_m": 2 * tol,
                "intersections": graph_stats["consolidation"][str(tol)]["intersections"],
                "merged_clusters": graph_stats["consolidation"][str(tol)]["merged_clusters"],
                "pct_crashes_within_5m": 100 * w[dist <= 5].sum() / total,
                "pct_crashes_within_10m": 100 * w[dist <= 10].sum() / total,
                "pct_crashes_within_20m": 100 * w[near].sum() / total,
                "median_dist_m": float(np.median(np.repeat(dist, w))),
                "clusters_with_2plus_tps_points": len(multi),
                "pct_crashes_in_those_clusters": 100 * w[in_multi].sum() / total,
            }
        )
    return pd.DataFrame(rows)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_distance_histograms(pts: pd.DataFrame, thresholds: dict, path: Path) -> None:
    panels = [
        (
            "dist_intersection",
            "Distance to nearest consolidated intersection (m)",
            100,
            2,
            thresholds["intersection_m"],
        ),
        ("dist_edge", "Distance to nearest drivable segment (m)", 60, 1, thresholds["edge_m"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE, layout="constrained")
    total = pts["n_crashes"].sum()
    for ax, (col, label, cap, width, thr) in zip(axes, panels, strict=True):
        _style(ax)
        bins = np.arange(0, cap + width, width)
        ax.hist(
            pts[col].clip(upper=cap - 1e-9),
            bins=bins,
            weights=pts["n_crashes"],
            color=BAR,
            edgecolor=SURFACE,
            linewidth=0.5,
        )
        ax.set_yscale("log")
        ax.axvline(thr, color=TEXT_MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"snap threshold {thr} m",
            (thr, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_MUTED,
        )
        over = pts.loc[pts[col] > cap, "n_crashes"].sum()
        ax.set_title(
            f"{label}\nlast bin includes > {cap} m ({100 * over / total:.1f}% of crashes)",
            loc="left",
            fontsize=10,
            color=TEXT,
        )
        ax.set_ylabel("crashes (log scale)", fontsize=9, color=TEXT_MUTED)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_offnetwork_map(
    pts: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
    thresholds: dict,
    path: Path,
) -> None:
    d, e = pts["dist_intersection"], pts["dist_edge"]
    layers = [
        (
            "mid-block (beyond intersection threshold, near an edge)",
            (d > thresholds["intersection_m"]) & (e <= thresholds["edge_m"]),
            BAR,
        ),
        (
            "off-network (beyond both thresholds)",
            (d > thresholds["intersection_m"]) & (e > thresholds["edge_m"]),
            ORANGE,
        ),
        ("inside parking polygon", pts["in_parking"], RED),
    ]
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    parking.plot(ax=ax, color="#dcdad3", linewidth=0)
    segments.plot(ax=ax, color="#c9c8c1", linewidth=0.25)
    motorways = segments[segments["highway"].isin(MOTORWAY_CLASSES)]
    motorways.plot(ax=ax, color="#8e8d86", linewidth=0.8)
    for label, mask, color in layers:
        sub = pts[mask]
        sizes = 4 + 2 * np.sqrt(sub["n_crashes"])
        ax.scatter(
            sub.geometry.x,
            sub.geometry.y,
            s=sizes,
            color=color,
            alpha=0.75,
            edgecolors=SURFACE,
            linewidths=0.4,
            label=f"{label}: {len(sub):,} points, {sub['n_crashes'].sum():,} crashes",
        )
    ax.set_title(
        "Crash points not at an intersection (marker area ~ crash count; dark grey "
        "lines = motorways, light fill = parking)",
        loc="left",
        fontsize=10,
        color=TEXT,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


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
    top["pct_severe"] = 100 * top["n_severe"] / top["n_crashes"]
    top["pct_ftr"] = 100 * top["n_fail_to_remain"] / top["n_crashes"]
    return top


def write_top_points_report(top: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# Top {len(top)} busiest TPS crash points",
        "",
        "Distances in metres (UTM 17N) to the nearest consolidated intersection, drivable "
        "segment, motorway segment, and OSM off-street parking polygon. "
        "`in lot` = inside a parking polygon.",
        "",
        "| # | crashes | % severe | % FTR | m to int. | m to road | m to motorway | "
        "m to lot | in lot | nearest road (class) | nearest intersection |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|",
    ]
    for rank, r in enumerate(top.itertuples(), start=1):
        lines.append(
            f"| {rank} | {r.n_crashes:,} | {r.pct_severe:.1f} | {r.pct_ftr:.1f} | "
            f"{r.dist_intersection:.0f} | {r.dist_edge:.0f} | {r.dist_motorway:.0f} | "
            f"{r.dist_parking:.0f} | {'yes' if r.in_parking else ''} | "
            f"{r.nearest_road} ({r.road_class}) | {r.nearest_intersection or '(unnamed)'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame, floats: str = "{:.1f}") -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float | np.floating):
            return floats.format(v)
        if isinstance(v, int | np.integer):
            return f"{v:,}"
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = (
        "|"
        + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns)
        + "|"
    )
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_location_report(
    summary: pd.DataFrame, tol: pd.DataFrame, pts: pd.DataFrame, cfg: dict, path: Path
) -> None:
    chosen = cfg["graph"]["consolidation_tolerance_m"]
    snap = cfg["snap"]
    total = pts["n_crashes"].sum()
    lots_near = pts[(pts["dist_parking"] <= 30) & ~pts["in_parking"]]
    lots_near_lowsev = lots_near[
        (lots_near["n_fail_to_remain"] / lots_near["n_crashes"] >= 0.35)
        & (lots_near["n_crashes"] >= 100)
    ]
    d, e = pts["dist_intersection"], pts["dist_edge"]
    w = pts["n_crashes"]
    at_int = w[d <= snap["intersection_m"]].sum() / total
    mid = (d > snap["intersection_m"]) & (e <= snap["edge_m"])
    mid_motorway = w[mid & (pts["dist_motorway"] <= 25)].sum() / w[mid].sum()
    mid_top = w[mid].sort_values(ascending=False)
    band = (d > 10) & (d <= snap["intersection_m"])
    band_merged = w[band & (pts["intersection_n_nodes"] > 1)].sum() / w[band].sum()
    band_oneway = w[band & pts["road_oneway"]].sum() / w[band].sum()
    edge_q99 = float(np.quantile(np.repeat(e.to_numpy(), w.to_numpy()), 0.99))
    lines = [
        "# Crash location analysis",
        "",
        f"{len(pts):,} distinct TPS crash points carrying {total:,} located crashes. Shares "
        "are crash-weighted unless marked as points. Intersections are OSMnx-consolidated "
        f"at tolerance {chosen} m (nodes within {2 * chosen} m merge).",
        "",
        "## Where crash points sit",
        "",
        markdown_table(
            summary.rename(columns={"pct_crashes": "% crashes", "pct_points": "% points"})
        ),
        "",
        "## Reading",
        "",
        f"- {at_int:.1%} of located crashes sit within {snap['intersection_m']} m of a "
        "consolidated intersection (median 2 m). The intersection is the natural unit.",
        f"- Mid-block points (> {snap['intersection_m']} m from an intersection, <= "
        f"{snap['edge_m']} m from an edge) carry {w[mid].sum() / total:.1%} of crashes on "
        f"{int(mid.sum()):,} points; {mid_motorway:.0%} of those crashes are within 25 m of "
        f"a motorway or ramp, and the 20 largest mid-block points hold "
        f"{mid_top.head(20).sum() / w[mid].sum():.0%} of them. They look like expressway "
        "crashes and a few address-geocoded points, not a systematic mid-block signal.",
        "- Recommendation for CP3: intersection-only labels. Keep mid-block assignments in "
        "`crash_assignments.parquet` so a segment model can be added later, but a segment "
        "unit would be trained on ~3% of crashes, a large part of which are motorway events "
        "whose TPS location is approximate.",
        "",
        "## Consolidation tolerance",
        "",
        "OSMnx buffers every intersection node by the tolerance and merges overlaps, so the "
        "effective merge distance is twice the tolerance.",
        "",
        markdown_table(tol),
        "",
        f"Chosen: {chosen} m. A larger tolerance moves cluster centroids away from where TPS "
        "places crashes (crashes within 10 m fall as tolerance grows) and folds more distinct "
        "TPS points into one cluster; 20 m visibly merges separate intersections (e.g. "
        "Dixon Road / Wincott Drive / Northcrest Road / Chetta Place become one node). TPS "
        "itself places hundreds of point pairs 10-20 m apart at divided-road intersections, "
        "so merging within 20 m matches its own notion of one intersection.",
        "",
        "## Snapping thresholds",
        "",
        f"- Intersection: {snap['intersection_m']} m. The distance histogram flattens after "
        f"~20 m. Of crashes 10-{snap['intersection_m']} m out, {band_merged:.0%} are at "
        f"multi-node (merged) intersections and {band_oneway:.0%} sit beside a one-way "
        "carriageway: divided-road intersections whose TPS point is on one side. Beyond 30 m "
        "the counts are flat, i.e. no longer intersection-shaped.",
        f"- Mid-block edge: {snap['edge_m']} m. 99% of crashes are within {edge_q99:.1f} m of a "
        "drivable edge; 20 m leaves room for wide roads while excluding points with no "
        "drivable road nearby (Toronto Islands, interiors of large sites).",
        "",
        "## Parking",
        "",
        f"- Points inside an OSM off-street parking polygon: {int(pts['in_parking'].sum())}.",
        f"- Points outside but within 30 m of a lot: {len(lots_near):,} "
        f"({100 * lots_near['n_crashes'].sum() / total:.1f}% of crashes). Of those, points with "
        f">= 100 crashes and >= 35% fail-to-remain: {len(lots_near_lowsev):,} "
        f"({100 * lots_near_lowsev['n_crashes'].sum() / total:.1f}% of crashes).",
        "",
        "## Figures",
        "",
        "![Distance histograms](figures/location_distance_hist.png)",
        "",
        "![Off-intersection crash points](figures/location_offnetwork_map.png)",
        "",
    ]
    path.write_text("\n".join(lines))


def load_inputs(cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, ...]:
    processed = Path(cfg["paths"]["processed_dir"])
    clean = pd.read_parquet(processed / CLEAN_FILENAME)
    points = crash_points(clean, cfg["graph"]["crs"])
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    parking = load_or_fetch_parking(cfg)
    return points, intersections, segments, parking


def run_locations(cfg: dict[str, Any]) -> None:
    processed = Path(cfg["paths"]["processed_dir"])
    reports, figures = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["figures_dir"])
    points, intersections, segments, parking = load_inputs(cfg)
    pts = annotate_points(points, intersections, segments, parking)
    pts.to_parquet(processed / POINTS)

    summary = location_summary(pts)
    graph_stats = json.loads((processed / GRAPH_STATS).read_text())
    tol = tolerance_comparison(
        points, processed, cfg["graph"]["tolerance_candidates_m"], graph_stats
    )
    write_top_points_report(
        top_points_table(pts, intersections, segments), reports / "top_crash_points.md"
    )
    plot_distance_histograms(pts, cfg["snap"], figures / "location_distance_hist.png")
    plot_offnetwork_map(
        pts, segments, parking, cfg["snap"], figures / "location_offnetwork_map.png"
    )
    write_location_report(summary, tol, pts, cfg, reports / "location_analysis.md")
    log.info("annotated %d crash points; wrote location analysis", len(pts))
