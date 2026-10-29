"""Intersection-year labels, splits, and proxy exclusions.

Every consolidated intersection gets one row per year 2014-2025, zero-crash years included.
Split by label year: train 2016-2019 + 2022-2023, val 2024, test 2025; 2014-2015 and
2020-2021 are history only (used for lagged features, never as labels); 2026 is partial and
dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from safepath.data.clean import CLEAN_FILENAME
from safepath.graph import INTERSECTIONS
from safepath.snapping import ASSIGNMENTS

log = logging.getLogger(__name__)

LABELS = "intersection_year_labels.parquet"
FIRST_YEAR, LAST_YEAR = 2014, 2025


def split_of(year: int, splits: dict[str, list[int]]) -> str:
    for name in ("train", "val", "test"):
        if year in splits[name]:
            return name
    return "history"


def crash_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Crash and severe counts per (intersection_id, year) from intersection assignments."""
    at = assignments[assignments["unit_type"] == "intersection"]
    return (
        at.groupby([at["unit_id"].astype(int).rename("intersection_id"), "year"])
        .agg(crash_count=("event_id", "size"), severe_count=("injury_or_fatal", "sum"))
        .reset_index()
    )


def label_table(
    counts: pd.DataFrame, intersection_ids: pd.Series, splits: dict[str, list[int]]
) -> pd.DataFrame:
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    grid = pd.MultiIndex.from_product(
        [sorted(intersection_ids), years], names=["intersection_id", "year"]
    ).to_frame(index=False)
    out = grid.merge(counts, on=["intersection_id", "year"], how="left")
    out[["crash_count", "severe_count"]] = (
        out[["crash_count", "severe_count"]].fillna(0).astype("int32")
    )
    out["any_crash"] = out["crash_count"] > 0
    out["split"] = out["year"].map(lambda y: split_of(y, splits))
    return out


def load_assignments(cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    clean = pd.read_parquet(
        processed / CLEAN_FILENAME, columns=["event_id", "injury_or_fatal", "is_fail_to_remain"]
    )
    return pd.read_parquet(processed / ASSIGNMENTS).merge(clean, on="event_id")


def build_labels(cfg: dict[str, Any], exclusions: pd.DataFrame) -> pd.DataFrame:
    """Label table with is_excluded / exclusion_reason joined from the proxy flags."""
    processed = Path(cfg["paths"]["processed_dir"])
    intersections = pd.read_parquet(processed / INTERSECTIONS, columns=["intersection_id"])
    labels = label_table(
        crash_counts(load_assignments(cfg)), intersections["intersection_id"], cfg["splits"]
    )
    labels = labels.merge(
        exclusions[["intersection_id", "exclusion_reason"]], on="intersection_id", how="left"
    )
    labels["is_excluded"] = labels["exclusion_reason"].notna()
    labels.to_parquet(processed / LABELS, index=False)
    return labels


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-split stats on clean (non-excluded) intersections, plus all-intersection counts."""
    rows = []
    for split, g in labels.groupby("split", sort=False):
        clean = g[~g["is_excluded"]]
        rows.append(
            {
                "split": split,
                "years": ", ".join(str(y) for y in sorted(g["year"].unique())),
                "rows (clean)": len(clean),
                "% zero-crash": 100 * (clean["crash_count"] == 0).mean(),
                "mean crashes": clean["crash_count"].mean(),
                "max crashes": int(clean["crash_count"].max()),
                "mean severe": clean["severe_count"].mean(),
                "% any severe": 100 * (clean["severe_count"] > 0).mean(),
                "crashes (clean)": int(clean["crash_count"].sum()),
                "crashes (excluded ints)": int(g.loc[g["is_excluded"], "crash_count"].sum()),
            }
        )
    order = {"history": 0, "train": 1, "val": 2, "test": 3}
    return pd.DataFrame(rows).sort_values("split", key=lambda s: s.map(order))
