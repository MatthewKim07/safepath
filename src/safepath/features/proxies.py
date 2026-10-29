"""Flag intersections whose TPS crash counts mostly belong somewhere else.

Two documented rules, computed from TRAIN label years only (never val/test):

retail proxy
    Parking-lot crashes geocoded to the nearest public intersection. Busy, very low severe
    share, high fail-to-remain share, and close to a mall or large retail area.

expressway proxy
    Expressway crashes geocoded to a nearby surface intersection. Close to a motorway or ramp
    and far busier than other intersections with the same approach-road class combo.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from safepath.locations import nearest

RETAIL, EXPRESSWAY = "retail_proxy", "expressway_proxy"


def train_stats(assignments: pd.DataFrame, train_years: list[int]) -> pd.DataFrame:
    """Per-intersection crash totals and severe / fail-to-remain shares over train years."""
    at = assignments[
        (assignments["unit_type"] == "intersection") & assignments["year"].isin(train_years)
    ]
    g = at.groupby(at["unit_id"].astype(int).rename("intersection_id")).agg(
        train_crashes=("event_id", "size"),
        train_severe_share=("injury_or_fatal", "mean"),
        train_ftr_share=("is_fail_to_remain", "mean"),
    )
    g["train_crashes_per_year"] = g["train_crashes"] / len(train_years)
    return g


def distances(
    intersections: gpd.GeoDataFrame, motorways: gpd.GeoDataFrame, retail: gpd.GeoDataFrame
) -> pd.DataFrame:
    pts = intersections[["intersection_id", "geometry"]].rename(
        columns={"intersection_id": "point_id"}
    )
    mw = motorways.reset_index(drop=True).rename_axis("mw").reset_index()
    rt = retail.reset_index(drop=True).rename_axis("rt").reset_index()
    return pd.DataFrame(
        {
            "dist_motorway": nearest(pts, mw, "mw")["dist"],
            "dist_retail": nearest(pts, rt, "rt")["dist"],
        }
    ).rename_axis("intersection_id")


def flag_proxies(
    stats: pd.DataFrame, dist: pd.DataFrame, combo: pd.DataFrame, rules: dict[str, Any]
) -> pd.DataFrame:
    """Apply both rules. Returns every intersection with its inputs, flags, and a reason
    (retail takes precedence when both fire: its severity / hit-and-run signature is the more
    specific of the two)."""
    df = combo.join(dist, how="outer").join(stats, how="left")
    df["train_crashes"] = df["train_crashes"].fillna(0).astype(int)

    r = rules["retail"]
    df[RETAIL] = (
        (df["train_crashes"] >= r["min_train_crashes"])
        & (df["train_severe_share"] <= r["max_severe_share"])
        & (df["train_ftr_share"] >= r["min_ftr_share"])
        & (df["dist_retail"] <= r["max_dist_retail_m"])
    )

    e = rules["expressway"]
    # Expected count = median train crashes among intersections of the same class combo that
    # had at least one train crash (zeros would make the median uninformative).
    has = df["train_crashes"] > 0
    df["combo_median_train_crashes"] = df["class_combo"].map(
        df[has].groupby("class_combo")["train_crashes"].median()
    )
    df["ratio_to_combo_median"] = df["train_crashes"] / df["combo_median_train_crashes"]
    df[EXPRESSWAY] = (
        (df["dist_motorway"] <= e["max_dist_motorway_m"])
        & (df["train_crashes"] >= e["min_train_crashes"])
        & (df["ratio_to_combo_median"] >= e["min_ratio_to_combo_median"])
    )
    df["exclusion_reason"] = np.select(
        [df[RETAIL], df[EXPRESSWAY]], [RETAIL, EXPRESSWAY], default=None
    )
    return df.rename_axis("intersection_id").reset_index()
