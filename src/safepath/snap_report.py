"""Report on crash-to-unit assignments: coverage, exclusions, and the busiest intersections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from safepath.data.clean import CLEAN_FILENAME
from safepath.data.report import SEQ_BLUE, SURFACE, TEXT, TEXT_MUTED
from safepath.graph import INTERSECTIONS, SEGMENTS
from safepath.locations import MOTORWAY_CLASSES, markdown_table, nearest
from safepath.snapping import ASSIGNMENTS

log = logging.getLogger(__name__)

ARTERIALS = ["Dundas", "Eglinton", "Finch", "Steeles", "Sheppard", "Lawrence"]
MINOR_CLASSES = {"residential", "unclassified", "living_street", "tertiary"}


def intersection_stats(
    assignments: pd.DataFrame, intersections: gpd.GeoDataFrame, segments: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Crash totals per intersection (zeros included) with names and simple anomaly flags."""
    at = assignments[assignments["unit_type"] == "intersection"]
    g = at.groupby("unit_id").agg(
        crashes=("event_id", "size"),
        severe=("injury_or_fatal", "sum"),
        fail_to_remain=("is_fail_to_remain", "sum"),
    )
    out = intersections.set_index("intersection_id")
    out = out.join(g).fillna({"crashes": 0, "severe": 0, "fail_to_remain": 0})
    out["crashes"] = out["crashes"].astype(int)
    motorways = segments[segments["highway"].isin(MOTORWAY_CLASSES)]
    pts = out.reset_index().rename(columns={"intersection_id": "point_id"})
    out["dist_motorway"] = nearest(pts, motorways, "segment_id")["dist"].reindex(out.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pct_severe"] = 100 * out["severe"] / out["crashes"]
        out["pct_ftr"] = 100 * out["fail_to_remain"] / out["crashes"]
    classes = out["highway_classes"].str.split(",")
    out["minor_roads_only"] = classes.map(lambda c: set(c) <= MINOR_CLASSES)
    busy = out["crashes"] >= 500
    # Retail-lot proxy: very low severity with many hit-and-runs, typical of parking-lot
    # crashes geocoded to the nearest public intersection.
    out["flag_retail_proxy"] = busy & (out["pct_severe"] < 6) & (out["pct_ftr"] > 20)
    # Expressway proxy: busy minor-road intersection within 300 m of a motorway.
    out["flag_expressway_proxy"] = busy & out["minor_roads_only"] & (out["dist_motorway"] <= 300)
    return out.sort_values("crashes", ascending=False)


def plot_intersection_map(stats: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    hit = stats[stats["crashes"] > 0].sort_values("crashes")
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    segments.plot(ax=ax, color="#d9d8d1", linewidth=0.2)
    sc = ax.scatter(
        hit.geometry.x,
        hit.geometry.y,
        c=hit["crashes"],
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=hit["crashes"].max()),
        s=1 + 0.02 * hit["crashes"],
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("crashes per intersection, all years (log scale)", color=TEXT_MUTED, fontsize=9)
    cb.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cb.outline.set_visible(False)
    span = f"{len(hit):,} of {len(stats):,} consolidated intersections have at least one crash"
    ax.set_title(f"Crashes per intersection, Toronto ({span})", loc="left", fontsize=10, color=TEXT)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def arterial_check(stats: pd.DataFrame) -> pd.DataFrame:
    names = stats["street_names"].fillna("")
    rows = []
    for street in ARTERIALS:
        ranks = np.flatnonzero(names.str.contains(street).to_numpy()) + 1
        rows.append(
            {
                "street": street,
                "best rank": int(ranks[0]) if len(ranks) else None,
                "in top 100": int((ranks <= 100).sum()),
                "in top 500": int((ranks <= 500).sum()),
                "intersections on street": len(ranks),
            }
        )
    return pd.DataFrame(rows)


def build_report(assignments: pd.DataFrame, stats: gpd.GeoDataFrame) -> str:
    total = len(assignments)
    by_type = assignments["unit_type"].value_counts()
    reasons = assignments["exclusion_reason"].value_counts()
    year = pd.crosstab(assignments["year"], assignments["unit_type"])
    year["% excluded"] = 100 * year.get("excluded", 0) / year.sum(axis=1)
    year = year.reset_index()
    year["year"] = year["year"].astype(str)

    counts = stats["crashes"]
    ranked = counts.sort_values(ascending=False)
    shares = []
    for q in (0.01, 0.05, 0.10):
        k = int(round(len(ranked) * q))
        shares.append(
            {
                "top intersections": f"{q:.0%} ({k:,})",
                "% of intersection crashes": 100 * ranked.head(k).sum() / ranked.sum(),
                "% of all located crashes": 100 * ranked.head(k).sum() / total,
            }
        )
    quant = counts.quantile([0.25, 0.5, 0.75, 0.9, 0.99]).round(0).astype(int)

    top = stats.head(20).reset_index()
    flags = np.select(
        [top["flag_retail_proxy"], top["flag_expressway_proxy"]], ["retail?", "expressway?"], ""
    )
    top_tbl = pd.DataFrame(
        {
            "#": np.arange(1, len(top) + 1),
            "crashes": top["crashes"],
            "% severe": top["pct_severe"],
            "% FTR": top["pct_ftr"],
            "OSM nodes": top["n_osm_nodes"],
            "m to motorway": top["dist_motorway"].round(0).astype(int),
            "flag": flags,
            "intersection": top["street_names"].replace("", "(unnamed)"),
            "road classes": top["highway_classes"],
        }
    )
    retail = stats[stats["flag_retail_proxy"]]
    near_mw = stats["dist_motorway"] <= 100
    near_mw_share = stats.loc[near_mw, "crashes"].sum() / counts.sum()
    express = stats[stats["flag_expressway_proxy"]]

    def flagged(df: pd.DataFrame) -> str:
        return "; ".join(
            f"{n} ({c:,})" for n, c in zip(df["street_names"], df["crashes"], strict=True)
        )

    lines = [
        "# Snapping report",
        "",
        "Every located crash is assigned to a consolidated intersection, a mid-block segment, "
        "or excluded (rules in `src/safepath/snapping.py`, thresholds in "
        "`configs/default.yaml`).",
        "",
        "## Assignments",
        "",
        "| unit type | crashes | share |",
        "|---|---:|---:|",
        *[f"| {k} | {v:,} | {100 * v / total:.2f}% |" for k, v in by_type.items()],
        f"| **total** | **{total:,}** | |",
        "",
        "Exclusions by reason:",
        "",
        "| reason | crashes |",
        "|---|---:|",
        *[f"| {k} | {v:,} |" for k, v in reasons.items()],
        "",
        "## By year",
        "",
        markdown_table(year),
        "",
        "## Crashes per intersection",
        "",
        f"{len(stats):,} consolidated intersections; {int((counts > 0).sum()):,} have at "
        f"least one crash in {assignments['year'].min()}-{assignments['year'].max()}. "
        "Quantiles of crashes per intersection (all years, zeros included): "
        + ", ".join(f"p{int(q * 100)} = {v:,}" for q, v in quant.items())
        + f", max = {int(counts.max()):,}.",
        "",
        markdown_table(pd.DataFrame(shares)),
        "",
        "## Top 20 intersections",
        "",
        "Names are the OSM street names on edges touching the intersection's nodes. "
        "Flags: `retail?` = >= 500 crashes, < 6% severe, > 20% fail-to-remain (pattern of "
        "parking-lot crashes geocoded to the nearest public intersection); `expressway?` = "
        ">= 500 crashes on minor roads only, within 300 m of a motorway (pattern of "
        "expressway crashes geocoded to a nearby surface intersection).",
        "",
        markdown_table(top_tbl),
        "",
        "## Anomalous intersections",
        "",
        f"- Retail-lot proxies ({len(retail)}, {int(retail['crashes'].sum()):,} crashes): "
        f"{flagged(retail)}.",
        f"- Expressway proxies ({len(express)}, {int(express['crashes'].sum()):,} crashes): "
        f"{flagged(express)}.",
        f"- More broadly, {int(near_mw.sum()):,} intersections within 100 m of a motorway or "
        f"ramp hold {near_mw_share:.1%} of intersection crashes (e.g. Lake Shore at Spadina, "
        "Jameson, Bay and Jarvis under the Gardiner). Some of these are genuine ramp-terminal "
        "crashes; some are expressway crashes placed at the nearest surface point.",
        "- The flags are heuristics: a busy downtown intersection can also show low severity "
        "and many hit-and-runs (e.g. Bremner / York), and the two lists overlap.",
        "- Both patterns put risk on an intersection that is not where the crashes happened. "
        "They need an explicit decision before CP3 labels are built.",
        "",
        "## Arterial sanity check",
        "",
        "Rank of the first intersection whose street names contain each arterial.",
        "",
        markdown_table(arterial_check(stats)),
        "",
        "Steeles Avenue is the city's northern boundary; crashes on its north side are "
        "York Region's, and TPS covers only the Toronto side, so it ranks lower than its "
        "traffic suggests.",
        "",
        "![Crashes per intersection](figures/intersection_crash_map.png)",
        "",
    ]
    return "\n".join(lines)


def run_snap_report(cfg: dict[str, Any]) -> Path:
    processed = Path(cfg["paths"]["processed_dir"])
    reports, figures = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["figures_dir"])
    clean = pd.read_parquet(
        processed / CLEAN_FILENAME, columns=["event_id", "injury_or_fatal", "is_fail_to_remain"]
    )
    assignments = pd.read_parquet(processed / ASSIGNMENTS).merge(clean, on="event_id")
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    stats = intersection_stats(assignments, intersections, segments)
    plot_intersection_map(stats, segments, figures / "intersection_crash_map.png")
    out = reports / "snapping_report.md"
    out.write_text(build_report(assignments, stats))
    log.info("wrote %s", out)
    return out
