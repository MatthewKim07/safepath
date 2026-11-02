"""Ottawa graph, location-precision check, snapping, labels and features.

Every computation reuses the Toronto modules with the Ottawa config (configs/ottawa.yaml):
graph.build_graph, locations.annotate_points / location_summary, snapping.assign_crashes,
features.build (proxies, labels, static features, history, model table). Only the report is
Ottawa's own: reports/ottawa/build.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from safepath.data.ottawa import CLEAN_FILENAME, FREEWAY_CRASHES
from safepath.features.build import (
    build_model_table,
    build_static,
    compute_proxies,
)
from safepath.features.labels import build_labels, label_summary
from safepath.features.proxies import EXPRESSWAY, RETAIL
from safepath.fetch import study_area
from safepath.graph import EDGE_MAP, GRAPH_STATS, INTERSECTIONS, PROJ_GRAPH, SEGMENTS, build_graph
from safepath.locations import (
    MOTORWAY_CLASSES,
    POINTS,
    annotate_points,
    crash_points,
    location_summary,
    markdown_table,
    nearest,
    tolerance_comparison,
)
from safepath.parking import load_or_fetch_parking
from safepath.snapping import ASSIGNMENTS, assign_crashes

log = logging.getLogger(__name__)

BUILD_STATS = "build_stats.json"
FREEWAY_HIGHWAYS = ("416", "417")


def bands(d: pd.Series) -> pd.Series:
    return pd.cut(d, [-1, 5, 10, 30, 50, np.inf], labels=["<=5 m", "5-10", "10-30", "30-50",
                                                          "> 50 m"])  # fmt: skip


def precision_tables(clean: pd.DataFrame, pts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """How the City's own location fields line up with distance to the nearest consolidated
    intersection: per crash, by the location-string kind and (2013-2016) by the City's
    intersection / non-intersection field."""
    d = pts.set_index(["lon", "lat"])["dist_intersection"]
    c = clean.assign(dist=d.reindex(pd.MultiIndex.from_frame(clean[["lon", "lat"]])).to_numpy())
    c["band"] = bands(c["dist"])
    by_kind = pd.crosstab(c["location_kind"], c["band"], normalize="index") * 100
    by_kind["crashes"] = c.groupby("location_kind").size()
    by_kind["median m"] = c.groupby("location_kind")["dist"].median()
    al = c[c["accident_location"].notna()]
    by_field = pd.crosstab(al["accident_location"], al["band"], normalize="index") * 100
    by_field["crashes"] = al.groupby("accident_location").size()
    by_field["median m"] = al.groupby("accident_location")["dist"].median()
    # how concentrated are crash coordinates? (TPS: 704k crashes on 18,651 points)
    per_point = clean.groupby(["lon", "lat"]).size()
    geo = clean.groupby("geo_id").agg(
        points=("lon", lambda s: s.nunique()), crashes=("lon", "size")
    )
    conc = pd.DataFrame(
        [
            {"measure": "crashes", "value": len(clean)},
            {"measure": "distinct crash coordinates", "value": len(per_point)},
            {"measure": "crashes per distinct point (mean)", "value": per_point.mean()},
            {"measure": "distinct City location ids (Geo_ID)", "value": len(geo)},
            {
                "measure": "% of Geo_IDs with a single coordinate",
                "value": 100 * (geo["points"] == 1).mean(),
            },  # fmt: skip
            {
                "measure": "% crashes on points with >= 10 crashes",
                "value": 100 * per_point[per_point >= 10].sum() / len(clean),
            },  # fmt: skip
        ]
    )
    return {"by_kind": by_kind.reset_index(), "by_field": by_field.reset_index(), "conc": conc}


def freeway_check(cfg: dict[str, Any], segments: gpd.GeoDataFrame) -> dict[str, Any]:
    """Cross-check the text rule with OSM: removed crashes vs distance to a 416 / 417
    motorway edge, and kept crashes that sit on one but were not flagged."""
    fw = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / FREEWAY_CRASHES)
    clean = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / CLEAN_FILENAME)
    processed = Path(cfg["paths"]["processed_dir"])
    G = ox.load_graphml(processed / PROJ_GRAPH)
    refs = pd.DataFrame(
        [(u, v, k, str(d.get("ref", ""))) for u, v, k, d in G.edges(keys=True, data=True)],
        columns=["u", "v", "key", "ref"],
    ).merge(pd.read_parquet(processed / EDGE_MAP), on=["u", "v", "key"])
    pattern = r"\b(?:" + "|".join(FREEWAY_HIGHWAYS) + r")\b"
    hwy_ids = set(refs.loc[refs["ref"].str.contains(pattern), "segment_id"])
    mw = segments[segments["highway"].isin(MOTORWAY_CLASSES)]
    mw = mw[mw["segment_id"].isin(hwy_ids)]
    surface = segments[~segments["highway"].isin(MOTORWAY_CLASSES)]
    m_cut = cfg["ottawa"]["freeway"]["motorway_m"]

    def dists(df: pd.DataFrame) -> pd.DataFrame:
        pts = crash_points(df.assign(injury_or_fatal=False, is_fail_to_remain=False),
                           cfg["graph"]["crs"])  # fmt: skip
        d_mw = nearest(pts, mw[["segment_id", "geometry"]], "segment_id")["dist"]
        d_s = nearest(pts, surface[["segment_id", "geometry"]], "segment_id")["dist"]
        return pts.assign(d_mw=d_mw.reindex(pts["point_id"]).to_numpy(),
                          d_surface=d_s.reindex(pts["point_id"]).to_numpy())  # fmt: skip

    area = gpd.GeoSeries([study_area(cfg)], crs="EPSG:4326").to_crs(cfg["graph"]["crs"]).iloc[0]
    f = dists(fw)
    f_in = f[f.within(area)]
    k = dists(clean)
    k_in = k[k.within(area)]
    missed = k_in[(k_in["d_mw"] <= m_cut) & (k_in["d_surface"] > m_cut)]
    return {
        "removed_in_study_area": int(f_in["n_crashes"].sum()),
        "removed_within_motorway_m": int(f_in.loc[f_in["d_mw"] <= m_cut, "n_crashes"].sum()),
        "removed_median_m_to_4xx": float(np.median(np.repeat(f_in["d_mw"], f_in["n_crashes"]))),
        "kept_on_4xx_only": int(missed["n_crashes"].sum()),
        "kept_in_study_area": int(k_in["n_crashes"].sum()),
        "motorway_m": m_cut,
    }


def retail_check(flags: pd.DataFrame, rules: dict[str, Any], n_years: int) -> pd.DataFrame:
    """Ottawa publishes no fail-to-remain flag, so the retail rule cannot fire. Busy,
    low-severity intersections near malls are listed to see whether the lot pattern exists."""
    r = rules["retail"]
    busy = flags["train_crashes"] >= r["min_train_crashes"] * n_years / 6  # same rate per year
    near = flags["dist_retail"] <= r["max_dist_retail_m"]
    low = flags["train_severe_share"] <= r["max_severe_share"]
    return flags[busy & near & low].sort_values("train_crashes", ascending=False)


def run_ottawa_build(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    stats: dict[str, Any] = {"graph": build_graph(cfg)}
    parking = load_or_fetch_parking(cfg)
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    area = gpd.GeoSeries([study_area(cfg)], crs="EPSG:4326").to_crs(cfg["graph"]["crs"])
    stats["study_area_km2"] = float(area.area.iloc[0] / 1e6)

    # location precision, on crashes inside the study area
    clean_all = pd.read_parquet(processed / CLEAN_FILENAME)
    pts_all = crash_points(clean_all, cfg["graph"]["crs"])
    inside = pts_all.within(area.iloc[0])
    pts = annotate_points(pts_all[inside].reset_index(drop=True), intersections, segments,
                          parking)  # fmt: skip
    pts.to_parquet(processed / POINTS)
    keys = pts[["lon", "lat"]].assign(_in=True)
    clean = clean_all.merge(keys, on=["lon", "lat"]).drop(columns="_in")
    stats["crashes_outside_study_area"] = len(clean_all) - len(clean)
    stats["crashes_in_study_area"] = len(clean)
    summary = location_summary(pts)
    tol = tolerance_comparison(
        pts_all[inside].reset_index(drop=True), processed, cfg["graph"]["tolerance_candidates_m"],
        json.loads((processed / GRAPH_STATS).read_text()),
    )  # fmt: skip
    prec = precision_tables(clean, pts)
    stats["freeway_check"] = freeway_check(cfg, segments)

    # snapping (same thresholds and rule order as Toronto); crashes outside the study area
    # are out of scope and are not assigned
    assignments = assign_crashes(clean, cfg)
    assignments.to_parquet(processed / ASSIGNMENTS, index=False)
    kind = clean.set_index("event_id")["location_kind"]
    a = assignments.assign(location_kind=assignments["event_id"].map(kind))
    snap_tab = pd.crosstab(a["location_kind"], a["unit_type"].astype(str), margins=True)

    # labels, proxies, static + history features
    flags = compute_proxies(cfg)
    excluded = flags[flags["exclusion_reason"].notna()]
    labels = build_labels(cfg, excluded)
    static = build_static(cfg)
    table = build_model_table(cfg, labels, static)
    n_train = len(cfg["splits"]["train"])
    retail = retail_check(flags, cfg["proxies"], n_train)
    # what the candidates are: a roundabout's crashes are low-severity sideswipes too
    ctl = a.merge(clean[["event_id", "traffic_control"]], on="event_id")
    ctl = ctl[ctl["unit_type"] == "intersection"]
    rb = ctl.groupby(ctl["unit_id"].astype(int))["traffic_control"].agg(
        lambda s: 100 * s.astype(str).str.startswith("11").mean()
    )
    retail["% roundabout"] = retail["intersection_id"].map(rb)
    express = flags[flags[EXPRESSWAY]].copy()
    if len(express):
        at_share = a[a["unit_type"] == "intersection"].assign(
            at=lambda x: x["location_kind"] == "at"
        )
        share = at_share.groupby(at_share["unit_id"].astype(int))["at"].mean()
        express["share_at_records"] = express["intersection_id"].map(share)
    stats.update(
        intersections=int(len(intersections)),
        segments=int(len(segments)),
        assignments={k: int(v) for k, v in a["unit_type"].value_counts().items()},
        exclusions={k: int(v) for k, v in a["exclusion_reason"].value_counts().items()},
        proxies={RETAIL: int(flags[RETAIL].sum()), EXPRESSWAY: int(flags[EXPRESSWAY].sum())},
        model_table_rows=len(table),
    )
    stats["location_summary"] = summary.to_dict(orient="records")
    stats["precision"] = {k: v.to_dict(orient="records") for k, v in prec.items()}
    stats["snap_by_kind"] = snap_tab.reset_index().to_dict(orient="records")
    stats["retail_candidates"] = int(len(retail))
    stats["expressway_flags"] = express.to_dict(orient="records") if len(express) else []
    (processed / BUILD_STATS).write_text(json.dumps(stats, indent=2, default=str))
    write_build_report(cfg, stats, summary, tol, prec, snap_tab, labels, retail, express,
                       reports / "build.md")  # fmt: skip
    log.info("ottawa build: %d intersections, %d model rows", len(intersections), len(table))
    return stats


def write_build_report(
    cfg: dict[str, Any],
    stats: dict[str, Any],
    summary: pd.DataFrame,
    tol: pd.DataFrame,
    prec: dict[str, pd.DataFrame],
    snap_tab: pd.DataFrame,
    labels: pd.DataFrame,
    retail: pd.DataFrame,
    express: pd.DataFrame,
    path: Path,
) -> None:
    g = stats["graph"]
    chosen = str(cfg["graph"]["consolidation_tolerance_m"])
    fc = stats["freeway_check"]
    flag_cols = [
        "intersection_id", "street_names", "train_crashes", "train_severe_share",
        "dist_retail", "dist_motorway", "class_combo", "ratio_to_combo_median",
    ]  # fmt: skip
    lines = [
        "# Ottawa graph, location precision, snapping and labels",
        "",
        "## Study area",
        "",
        f"{cfg['graph']['boundary']['name']} ({cfg['graph']['boundary']['url']}), "
        f"{stats['study_area_km2']:.0f} km² (Toronto: 630 km²). The amalgamated city is "
        "2,790 km², mostly rural townships and villages with no counterpart in Toronto. "
        f"{stats['crashes_outside_study_area']:,} non-freeway crashes fall outside the "
        f"boundary and are out of scope; {stats['crashes_in_study_area']:,} are inside.",
        "",
        f"OSM drive graph: {g['directed_nodes']:,} nodes, {g['directed_edges']:,} directed "
        f"edges, {g['segments']:,} undirected segments. Consolidation at {chosen} m (as "
        f"Toronto): {g['consolidation'][chosen]['intersections']:,} intersections.",
        "",
        markdown_table(tol),
        "",
        "## Location precision",
        "",
        markdown_table(prec["conc"]),
        "",
        "Where crash points sit (distinct points in the study area, crash-weighted):",
        "",
        markdown_table(
            summary.rename(columns={"pct_crashes": "% crashes", "pct_points": "% points"})
        ),
        "",
        "Distance to the nearest consolidated intersection by the City's location string "
        "('A @ B' = at, 'A btwn B & C' = btwn), % of crashes per row:",
        "",
        markdown_table(prec["by_kind"]),
        "",
        "By the City's own intersection field (2013-2016 layers only; the 2017-2024 layer "
        "dropped it):",
        "",
        markdown_table(prec["by_field"]),
        "",
        "## Freeway cross-check (OSM)",
        "",
        f"Of the {fc['removed_in_study_area']:,} removed freeway crashes inside the study "
        f"area, {fc['removed_within_motorway_m']:,} sit within {fc['motorway_m']} m of an OSM "
        f"Hwy 416 / 417 motorway or ramp edge (median {fc['removed_median_m_to_4xx']:.0f} m). "
        f"Kept crashes within {fc['motorway_m']} m of a 416 / 417 edge and farther than that "
        f"from any surface road: {fc['kept_on_4xx_only']:,} of "
        f"{fc['kept_in_study_area']:,}.",
        "",
        "## Snapping",
        "",
        f"Same rule order and thresholds as Toronto: parking polygon -> excluded; <= "
        f"{cfg['snap']['intersection_m']} m from an intersection -> intersection; <= "
        f"{cfg['snap']['edge_m']} m from an edge -> segment; else excluded. Crash counts by "
        "the City's location kind:",
        "",
        markdown_table(snap_tab.reset_index(), floats="{:.0f}"),
        "",
        "## Proxy rules",
        "",
        f"- Retail: cannot fire (Ottawa publishes no fail-to-remain flag). Busy (>= "
        f"{cfg['proxies']['retail']['min_train_crashes']} train crashes per 6 years, scaled "
        f"to {len(cfg['splits']['train'])}), severe share <= "
        f"{cfg['proxies']['retail']['max_severe_share']:.0%} and within "
        f"{cfg['proxies']['retail']['max_dist_retail_m']} m of a mall / large retail area: "
        f"{len(retail)} intersections.",
        "",
        markdown_table(retail[[*flag_cols, "% roundabout"]].head(15)) if len(retail) else "(none)",
        "",
        "A candidate whose crashes are mostly coded '11 - Roundabout' is a real roundabout "
        "(many low-severity sideswipes at the junction itself, recorded as 'A @ B' at many "
        "distinct points), not parking-lot crashes moved to a public intersection, so it is "
        "kept. No retail exclusions were needed.",
        "",
        f"- Expressway: {len(express)} intersections meet Toronto's rule.",
        "",
        markdown_table(express[flag_cols + ["share_at_records"]]) if len(express) else "",
        "",
        "## Labels",
        "",
        markdown_table(label_summary(labels)),
        "",
    ]
    path.write_text("\n".join(lines))
