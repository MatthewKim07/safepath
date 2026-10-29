"""Orchestrates labels, proxy exclusions, and features for `python -m safepath features`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from safepath.features.labels import build_labels, label_summary, load_assignments
from safepath.features.proxies import EXPRESSWAY, RETAIL, distances, flag_proxies, train_stats
from safepath.features.roads import approaches, class_combo
from safepath.graph import INTERSECTIONS, NODE_MAP, SEGMENTS
from safepath.locations import MOTORWAY_CLASSES, markdown_table
from safepath.osm_context import load_or_fetch_layer

log = logging.getLogger(__name__)

PROXIES = "intersection_proxies.parquet"


def compute_proxies(cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    node_map = pd.read_parquet(processed / NODE_MAP)
    stats = train_stats(load_assignments(cfg), cfg["splits"]["train"])
    dist = distances(
        intersections,
        segments[segments["highway"].isin(MOTORWAY_CLASSES)],
        load_or_fetch_layer(cfg, "retail"),
    )
    combo = class_combo(approaches(segments, node_map))[["class_combo"]]
    flags = flag_proxies(stats, dist, combo, cfg["proxies"])
    names = intersections.set_index("intersection_id")["street_names"]
    flags["street_names"] = flags["intersection_id"].map(names).replace("", "(unnamed)")
    flags.to_parquet(processed / PROXIES, index=False)
    return flags


def _flag_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": df["intersection_id"].astype(str),
            "intersection": df["street_names"],
            "train crashes": df["train_crashes"],
            "% severe": 100 * df["train_severe_share"],
            "% FTR": 100 * df["train_ftr_share"],
            "m to retail": df["dist_retail"].round(0).astype(int),
            "m to motorway": df["dist_motorway"].round(0).astype(int),
            "class combo": df["class_combo"],
            "x combo median": df["ratio_to_combo_median"],
        }
    )


def write_labels_report(
    flags: pd.DataFrame, labels: pd.DataFrame, cfg: dict[str, Any], path: Path
) -> None:
    r, e = cfg["proxies"]["retail"], cfg["proxies"]["expressway"]
    excluded = flags[flags["exclusion_reason"].notna()].sort_values(
        "train_crashes", ascending=False
    )
    retail = excluded[excluded["exclusion_reason"] == RETAIL]
    express = excluded[excluded["exclusion_reason"] == EXPRESSWAY]
    both = flags[flags[RETAIL] & flags[EXPRESSWAY]]
    busy = flags["train_crashes"] >= r["min_train_crashes"]
    near_retail = flags[
        busy
        & flags["exclusion_reason"].isna()
        & (flags["train_severe_share"] <= 0.10)
        & (flags["train_ftr_share"] >= 0.20)
    ].sort_values("train_crashes", ascending=False)
    near_express = flags[
        (flags["dist_motorway"] <= e["max_dist_motorway_m"])
        & (flags["train_crashes"] >= e["min_train_crashes"])
        & (flags["ratio_to_combo_median"] >= 10)
        & ~flags[EXPRESSWAY]
        & flags["exclusion_reason"].isna()
    ].sort_values("train_crashes", ascending=False)
    total_train = labels.loc[labels["split"] == "train", "crash_count"].sum()
    train_excl = labels.loc[(labels["split"] == "train") & labels["is_excluded"], "crash_count"]
    n_rows = len(labels)
    lines = [
        "# Labels and proxy exclusions",
        "",
        f"Intersection-year table: {labels['intersection_id'].nunique():,} consolidated "
        f"intersections x {labels['year'].nunique()} years "
        f"({labels['year'].min()}-{labels['year'].max()}) = {n_rows:,} rows, "
        f"{100 * (labels['crash_count'] == 0).mean():.1f}% of them zero-crash.",
        "",
        "## Proxy rules (train label years only: "
        + ", ".join(map(str, cfg["splits"]["train"]))
        + ")",
        "",
        f"- **Retail proxy**: >= {r['min_train_crashes']} train crashes AND severe share <= "
        f"{r['max_severe_share']:.0%} AND fail-to-remain share >= {r['min_ftr_share']:.0%} "
        f"AND within {r['max_dist_retail_m']} m of an OSM shop=mall or a landuse=retail polygon "
        ">= 2 ha. (Busy intersections citywide have a median severe share of ~14% and FTR "
        "share of ~13%.)",
        f"- **Expressway proxy**: within {e['max_dist_motorway_m']} m of a motorway / "
        f"motorway_link AND >= {e['min_train_crashes']} train crashes AND >= "
        f"{e['min_ratio_to_combo_median']}x the median train crashes of intersections with "
        "the same approach-class combo (median over intersections with >= 1 train crash).",
        "- When both fire, the reason is `retail_proxy` (the more specific signature). "
        f"Both fired for {len(both)}: "
        + ("; ".join(both["street_names"]) if len(both) else "none")
        + ".",
        "",
        "Tuning notes:",
        "",
        "- Retail: the four known lot proxies are caught (Highland Hill by Yorkdale, Borough / "
        "Progress at STC, Parkway Forest / Sheppard at Fairview, Sherway Gardens). Bremner / "
        "York is not: it has low severity and 21% FTR but no mall or large retail within "
        "150 m. An FTR cutoff of 20% instead of 25% would also flag Yonge / Eglinton, a "
        "genuinely busy intersection.",
        "- Expressway: the suggested ~100 m radius misses the two clearest cases, Airley "
        "Crescent (173 m from the DVP, 353x its class median) and Ferrand Drive (122 m, "
        "257x). At 200 m a 5x or 10x cutoff starts flagging real downtown intersections "
        "(Bremner / York, Fleet / Strachan, Billy Bishop Way), so the ratio bar is 20x.",
        "",
        f"Excluded: {len(excluded)} intersections ({len(retail)} retail, {len(express)} "
        f"expressway) carrying {int(train_excl.sum()):,} of {int(total_train):,} train-year "
        f"intersection crashes ({100 * train_excl.sum() / total_train:.1f}%). Full list in "
        "`reports/excluded_intersections.csv`.",
        "",
        f"### Retail proxies ({len(retail)})",
        "",
        markdown_table(_flag_table(retail)),
        "",
        f"### Expressway proxies ({len(express)})",
        "",
        markdown_table(_flag_table(express)),
        "",
        "### Near misses (not excluded, listed for transparency)",
        "",
        "Busy, severe <= 10%, FTR >= 20%, but not flagged as retail:",
        "",
        markdown_table(_flag_table(near_retail)),
        "",
        "Within the motorway radius at 10-20x their class median:",
        "",
        markdown_table(_flag_table(near_express)),
        "",
        "## Label stats per split (clean intersections)",
        "",
        markdown_table(label_summary(labels)),
        "",
    ]
    path.write_text("\n".join(lines))


def run_labels(cfg: dict[str, Any]) -> pd.DataFrame:
    flags = compute_proxies(cfg)
    excluded = flags[flags["exclusion_reason"].notna()]
    labels = build_labels(cfg, excluded)
    reports = Path(cfg["paths"]["reports_dir"])
    cols = [
        "intersection_id",
        "street_names",
        "exclusion_reason",
        "train_crashes",
        "train_severe_share",
        "train_ftr_share",
        "dist_retail",
        "dist_motorway",
        "class_combo",
        "ratio_to_combo_median",
    ]
    excluded.sort_values("train_crashes", ascending=False)[cols].round(3).to_csv(
        reports / "excluded_intersections.csv", index=False
    )
    write_labels_report(flags, labels, cfg, reports / "labels.md")
    log.info("labels: %d rows; excluded %d intersections", len(labels), len(excluded))
    return labels


def run_features(cfg: dict[str, Any]) -> None:
    run_labels(cfg)
