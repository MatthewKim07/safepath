"""Clean raw TPS collision records into one row per located collision event."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from safepath.data.download import RAW_FILENAME

log = logging.getLogger(__name__)

CLEAN_FILENAME = "collisions_clean.parquet"
STATS_FILENAME = "collisions_clean_stats.json"
TIMEZONE = "America/Toronto"

# "N/R" (not recorded) and missing values become <NA>, not False.
_YES_NO = {"YES": True, "NO": False}

_FLAG_COLUMNS = {
    "INJURY_COLLISIONS": "is_injury",
    "FTR_COLLISIONS": "is_fail_to_remain",
    "PD_COLLISIONS": "is_property_damage",
    "AUTOMOBILE": "involves_automobile",
    "MOTORCYCLE": "involves_motorcycle",
    "PASSENGER": "involves_passenger",
    "BICYCLE": "involves_bicycle",
    "PEDESTRIAN": "involves_pedestrian",
}


def yes_no(series: pd.Series) -> pd.Series:
    return series.str.strip().str.upper().map(_YES_NO).astype("boolean")


def parse_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Type and rename raw columns and derive date, severity, and road-user flags.

    OCC_DATE is epoch milliseconds at local midnight, so it is converted to Toronto time
    before taking the calendar date. The hour comes from OCC_HOUR.
    """
    occ = pd.to_datetime(pd.to_numeric(raw["OCC_DATE"]), unit="ms", utc=True).dt.tz_convert(
        TIMEZONE
    )
    df = pd.DataFrame(
        {
            "objectid": pd.to_numeric(raw["OBJECTID"]).astype("int64"),
            "event_id": raw["EVENT_UNIQUE_ID"].astype(str).str.strip().str.upper(),
            "occ_date": occ.dt.tz_localize(None).dt.normalize(),
            "year": occ.dt.year.astype("int16"),
            "month": occ.dt.month.astype("int8"),
            "dow": occ.dt.day_of_week.astype("int8"),
            "hour": pd.to_numeric(raw["OCC_HOUR"], errors="coerce").astype("Int8"),
            "division": raw["DIVISION"].astype(str),
            "hood_158": raw["HOOD_158"].astype(str),
            "neighbourhood": raw["NEIGHBOURHOOD_158"].astype(str),
            "lon": pd.to_numeric(raw["LONG_WGS84"], errors="coerce"),
            "lat": pd.to_numeric(raw["LAT_WGS84"], errors="coerce"),
            "fatalities": pd.to_numeric(raw["FATALITIES"], errors="coerce")
            .fillna(0)
            .astype("int16"),
        }
    )
    for src, dst in _FLAG_COLUMNS.items():
        df[dst] = yes_no(raw[src].astype("string"))

    df["is_fatal"] = df["fatalities"] > 0
    # Fatal collisions are usually not also marked INJURY_COLLISIONS=YES, so both count.
    df["injury_or_fatal"] = (df["is_injury"].fillna(False) | df["is_fatal"]).astype(bool)
    # Kleene logic: True if any VRU is involved, <NA> if none is but some flag is unknown.
    df["involves_vru"] = (
        df["involves_pedestrian"] | df["involves_bicycle"] | df["involves_motorcycle"]
    )
    df["occ_year_mismatch"] = pd.to_numeric(raw["OCC_YEAR"], errors="coerce").ne(df["year"])
    return df


def clean_collisions(
    raw: pd.DataFrame, bbox: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return located, deduplicated collision events and a dict of what was dropped and why."""
    df = parse_records(raw)
    stats: dict[str, Any] = {"n_raw": len(df)}

    dup_mask = df.duplicated("event_id", keep=False)
    stats["n_duplicate_rows"] = int(dup_mask.sum())
    stats["n_event_ids_with_duplicates"] = int(df.loc[dup_mask, "event_id"].nunique())
    df = df.sort_values("objectid").drop_duplicates("event_id", keep="first")
    stats["n_dropped_duplicate"] = stats["n_raw"] - len(df)

    missing = df["lon"].isna() | df["lat"].isna() | df["lon"].eq(0) | df["lat"].eq(0)
    in_bbox = df["lon"].between(bbox["min_lon"], bbox["max_lon"]) & df["lat"].between(
        bbox["min_lat"], bbox["max_lat"]
    )
    stats["n_dropped_missing_coords"] = int(missing.sum())
    stats["n_dropped_outside_bbox"] = int((~missing & ~in_bbox).sum())
    stats["missing_coords_rate_by_severity"] = {
        "injury_or_fatal": float(missing[df["injury_or_fatal"]].mean()),
        "fatal": float(missing[df["is_fatal"]].mean()),
        "other": float(missing[~df["injury_or_fatal"]].mean()),
    }
    df = df[~missing & in_bbox]

    stats["n_clean"] = len(df)
    stats["n_occ_year_mismatch"] = int(df["occ_year_mismatch"].sum())
    stats["n_hour_missing"] = int(df["hour"].isna().sum())
    df = df.drop(columns="occ_year_mismatch").reset_index(drop=True)
    return df, stats


def run_clean(cfg: dict[str, Any]) -> Path:
    paths = cfg["paths"]
    raw = pd.read_parquet(Path(paths["raw_dir"]) / RAW_FILENAME)
    df, stats = clean_collisions(raw, cfg["clean"]["bbox"])
    out_dir = Path(paths["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / CLEAN_FILENAME
    df.to_parquet(out, index=False)
    (out_dir / STATS_FILENAME).write_text(json.dumps(stats, indent=2))
    log.info(
        "kept %d of %d rows (duplicates %d, missing coords %d, outside bbox %d); wrote %s",
        stats["n_clean"],
        stats["n_raw"],
        stats["n_dropped_duplicate"],
        stats["n_dropped_missing_coords"],
        stats["n_dropped_outside_bbox"],
        out,
    )
    return out
