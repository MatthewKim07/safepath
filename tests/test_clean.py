from pathlib import Path

import pandas as pd
import pytest

from safepath.config import load_config
from safepath.data.clean import clean_collisions, parse_records

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "collisions_sample.csv"
BBOX = load_config()["clean"]["bbox"]


@pytest.fixture
def raw() -> pd.DataFrame:
    # Read everything as strings, like an untyped export, to exercise type coercion.
    return pd.read_csv(SAMPLE, dtype=str, keep_default_na=False, na_values=[""])


def row(df: pd.DataFrame, objectid: int) -> pd.Series:
    return df.set_index("objectid").loc[objectid]


def test_dates_use_toronto_local_calendar(raw):
    df = parse_records(raw)
    assert (df["year"] == pd.to_numeric(raw["OCC_YEAR"]).to_numpy()).all()
    # 1388552400000 ms would be 2014-01-01 05:00 UTC; ensure no row slips a day.
    months = pd.to_datetime(raw["OCC_MONTH"], format="%B").dt.month.to_numpy()
    assert (df["month"].to_numpy() == months).all()
    assert df["hour"].between(0, 23).all()


def test_fatal_counts_as_severe_even_without_injury_flag(raw):
    df = parse_records(raw)
    fatal_no_injury = row(df, 411947)  # FATALITIES=1, INJURY_COLLISIONS=NO
    assert fatal_no_injury["is_fatal"] and not fatal_no_injury["is_injury"]
    assert fatal_no_injury["injury_or_fatal"]
    injury = row(df, 119776)
    assert injury["injury_or_fatal"] and not injury["is_fatal"]
    pd_only = row(df, 40482)
    assert not pd_only["injury_or_fatal"]


def test_not_recorded_flags_become_missing_not_false(raw):
    df = parse_records(raw)
    nr = row(df, 309573)  # AUTOMOBILE..PEDESTRIAN = N/R
    assert pd.isna(nr["involves_pedestrian"]) and pd.isna(nr["involves_vru"])
    null_flags = row(df, 164842)  # all flag columns empty, FATALITIES=1
    assert pd.isna(null_flags["is_injury"]) and null_flags["injury_or_fatal"]


def test_vru_flag(raw):
    df = parse_records(raw)
    assert row(df, 119776)["involves_vru"]  # bicycle
    assert row(df, 411947)["involves_vru"]  # pedestrian
    assert not row(df, 40482)["involves_vru"]


def test_event_ids_are_normalised(raw):
    df = parse_records(raw)
    assert row(df, 436330)["event_id"] == "GO-20192465950"


def test_clean_drops_missing_coords_and_reports_counts(raw):
    df, stats = clean_collisions(raw, BBOX)
    n_zero = int((pd.to_numeric(raw["LONG_WGS84"]) == 0).sum())
    assert n_zero > 0
    assert stats["n_dropped_missing_coords"] == n_zero
    assert stats["n_clean"] == len(raw) - n_zero == len(df)
    assert df["lon"].between(BBOX["min_lon"], BBOX["max_lon"]).all()
    assert df["lat"].between(BBOX["min_lat"], BBOX["max_lat"]).all()


def test_clean_drops_points_outside_bbox(raw):
    raw = raw.copy()
    located = raw.index[pd.to_numeric(raw["LONG_WGS84"]) != 0][0]
    raw.loc[located, ["LONG_WGS84", "LAT_WGS84"]] = ["-75.69", "45.42"]  # Ottawa
    df, stats = clean_collisions(raw, BBOX)
    assert stats["n_dropped_outside_bbox"] == 1
    assert int(raw.loc[located, "OBJECTID"]) not in set(df["objectid"])


def test_clean_dedupes_on_event_id_keeping_lowest_objectid(raw):
    raw = raw.copy()
    first = raw[pd.to_numeric(raw["LONG_WGS84"]) != 0].iloc[0]
    dup = first.copy()
    dup["OBJECTID"] = "9999999"
    dup["EVENT_UNIQUE_ID"] = " " + first["EVENT_UNIQUE_ID"].lower()
    raw = pd.concat([raw, dup.to_frame().T], ignore_index=True)
    df, stats = clean_collisions(raw, BBOX)
    assert stats["n_dropped_duplicate"] == 1
    assert stats["n_event_ids_with_duplicates"] == 1
    assert df["event_id"].is_unique
    assert 9999999 not in set(df["objectid"])
