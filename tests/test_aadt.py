import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from safepath.data import aadt
from safepath.data.volumes import window_bins

CRS = "EPSG:32617"


def test_street_key_normalises_abbreviations():
    assert aadt.street_key("Steeles Ave E") == aadt.street_key("Steeles Avenue East")
    assert aadt.street_key("Don Mills Rd") == "don mills"
    assert aadt.street_key("Pape Ave (id: 13458569)") == "pape"
    assert aadt.counted_street("Danforth Ave: Moberly Ave - Woodbine Ave") == "Danforth Ave"
    assert aadt.counted_street("Scarlett Rd to Edenbridge Dr") == "Scarlett Rd"


def _to_lonlat(x, y):
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=CRS).to_crs("EPSG:4326")
    return pts.x.to_numpy(), pts.y.to_numpy()


def test_match_tmc_prefers_shared_street_names_over_nearest():
    inter = pd.DataFrame(
        {
            "intersection_id": [1, 2, 3],
            "x": [630000.0, 630012.0, 631000.0],
            "y": [4840000.0, 4840000.0, 4840000.0],
            "street_names": ["Side Street / Lane", "Main Street / Oak Avenue", "Far Road"],
        }
    )
    inter = gpd.GeoDataFrame(inter, geometry=gpd.points_from_xy(inter.x, inter.y), crs=CRS)
    lon, lat = _to_lonlat([630002.0, 631100.0], [4840000.0, 4840000.0])
    tmc = pd.DataFrame(
        {"location_name": ["Main St / Oak Ave", "Nowhere Rd"], "lon": lon, "lat": lat}
    )
    out = aadt.match_tmc(tmc, inter, max_m=40, k=3)
    assert out.loc[0, "intersection_id"] == 2  # 10 m away but shares both names
    assert out.loc[0, "names_shared"] == 2
    assert np.isnan(out.loc[1, "intersection_id"])  # 100 m from anything
    assert np.isnan(out.loc[1, "match_m"])


def test_match_svc_requires_name_unless_very_close():
    seg = gpd.GeoDataFrame(
        {
            "segment_id": [10, 11],
            "name": ["Cross Street", "Main Street"],
            "highway": ["residential", "secondary"],
            "oneway": [False, False],
        },
        geometry=[
            LineString([(630000, 4839900), (630000, 4840100)]),
            LineString([(629900, 4840025), (630100, 4840025)]),
        ],
        crs=CRS,
    )
    lon, lat = _to_lonlat([630003.0, 630003.0], [4840000.0, 4840000.0])
    svc = pd.DataFrame(
        {"location_name": ["Main St: A - B", "Unknown Rd: A - B"], "lon": lon, "lat": lat}
    )
    out = aadt.match_svc(svc, seg, max_m=40, unnamed_max_m=15)
    assert out.loc[0, "segment_id"] == 11  # 25 m away, but it is the named street
    assert bool(out.loc[0, "name_match"])
    assert out.loc[1, "segment_id"] == 10  # no name match: nearest, 3 m away
    assert not bool(out.loc[1, "name_match"])


def test_window_ratio_recovers_flat_profile():
    # a flat profile puts 8/24 of the day in an 8-hour window
    mat = pd.DataFrame(
        np.full((4, 96), 10.0),
        index=pd.MultiIndex.from_tuples(
            [(c, pd.Timestamp("2024-05-14")) for c in (1, 2, 3, 4)], names=["count_id", "date"]
        ),
        columns=range(0, 1440, 15),
    )
    svc = pd.DataFrame({"count_id": [1, 2, 3, 4], "highway": ["primary"] * 2 + ["residential"] * 2})
    days = aadt.day_table(mat, svc)
    R = aadt.window_ratios(days).set_index(["group", "window"])["R"]
    assert R[("major", "8R")] == pytest.approx(3.0)
    assert R[("minor", "14")] == pytest.approx(24 / 14)
    assert days["h0"].iloc[0] == 40


def test_month_factors_recover_seasonality():
    rng = np.random.default_rng(0)
    true = np.array([0.8, 0.85, 0.95, 1.0, 1.05, 1.08, 1.0, 1.02, 1.04, 1.06, 1.03, 0.95])
    rows = []
    for loc in range(300):
        base = rng.uniform(1000, 30000)
        for month in rng.choice(np.arange(1, 13), size=3, replace=False):
            rows.append(
                {
                    "centreline_id": loc,
                    "date_start": pd.Timestamp(2018, int(month), 10),
                    "avg_weekday_daily": base * true[month - 1] * rng.lognormal(0, 0.02),
                }
            )
    svc = pd.DataFrame(rows)
    M, _ = aadt.month_factors(svc, (2010, 2026))
    days = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    expected = true / np.average(true, weights=days)
    assert np.allclose(M["M"], expected, atol=0.02)


def test_dow_factors_relative_to_weekday_mean():
    dates = pd.date_range("2024-05-13", periods=7)  # Monday..Sunday
    days = pd.DataFrame(
        {
            "count_id": 1,
            "dow": dates.day_of_week,
            "daily": [100, 100, 100, 100, 100, 80, 60],
        }
    )
    D = aadt.dow_factors(days)
    assert D.loc[5, "D"] == pytest.approx(0.8)
    assert D["D"].mean() == pytest.approx(640 / 700)


def test_combine_siblings_sums_directions_and_dedupes_unknown():
    base = {"centreline_id": 1, "date_start": pd.Timestamp("2024-05-14")}
    svc = pd.DataFrame(
        [
            {**base, "count_id": 1, "directions": "EB", "n_directions": 1, "avg_daily": 500,
             "avg_weekday_daily": 500, "avg_weekend_daily": np.nan},
            {**base, "count_id": 2, "directions": "WB", "n_directions": 1, "avg_daily": 400,
             "avg_weekday_daily": 400, "avg_weekend_daily": np.nan},
            {"centreline_id": 2, "date_start": pd.Timestamp("2010-05-14"), "count_id": 3,
             "directions": "", "n_directions": 0, "avg_daily": 900, "avg_weekday_daily": 900,
             "avg_weekend_daily": np.nan},
            {"centreline_id": 2, "date_start": pd.Timestamp("2010-05-14"), "count_id": 4,
             "directions": "", "n_directions": 0, "avg_daily": 950, "avg_weekday_daily": 950,
             "avg_weekend_daily": np.nan},
        ]
    )  # fmt: skip
    out = aadt.combine_svc_siblings(svc).set_index("centreline_id")
    assert out.loc[1, "avg_weekday_daily"] == 900
    assert out.loc[1, "directions"] == "EB,WB"
    assert out.loc[2, "avg_weekday_daily"] == 950
    assert out.loc[2, "combined"] == "kept largest"


def test_tmc_aadt_formula():
    tmc = pd.DataFrame(
        {
            "intersection_id": [7],
            "date": [pd.Timestamp("2024-03-12")],  # a Tuesday in March
            "duration": ["14"],
            "entering": [10_000],
            "raw_bins": [52.0],  # one hour of the 14 missing
        }
    )
    R = pd.DataFrame(
        {
            "era": ["2020on"],
            "group": ["major"],
            "daytype": ["weekday"],
            "window": ["14"],
            "R": [1.2],
        }
    )
    D = pd.DataFrame({"D": [1.0, 0.95, 1.0, 1.0, 1.0, 0.9, 0.8]})
    M = pd.DataFrame({"month": range(1, 13), "M": [1.0] * 2 + [1.05] + [1.0] * 9})
    out = aadt.tmc_aadt(tmc, pd.Series({7: "major"}), R, D, W=0.95, M=M)
    expected = 10_000 * (56 / 52) * 1.2 / 0.95 * 0.95 / 1.05
    assert out.loc[0, "aadt"] == pytest.approx(expected)
    assert len(window_bins("14")) == 56


def test_latest_by_unit_keeps_recent_and_noncovid():
    counts = pd.DataFrame(
        {
            "intersection_id": [1, 1, 1, 2],
            "year": [2012, 2019, 2021, 2024],
            "aadt": [10_000, 12_000, 8_000, 5_000],
        }
    )
    out = aadt.latest_by_unit(counts, "intersection_id").set_index("intersection_id")
    assert out.loc[1, "latest_year"] == 2021
    assert out.loc[1, "latest_aadt"] == 8_000
    assert out.loc[1, "latest_noncovid_year"] == 2019
    assert out.loc[1, "latest_noncovid_aadt"] == 12_000
    assert out.loc[1, "n_counts"] == 3


def test_ramp_counts_never_match_expressway_mainline():
    seg = gpd.GeoDataFrame(
        {
            "segment_id": [20, 21],
            "name": ["Gardiner Expressway", None],
            "highway": ["motorway", "motorway_link"],
            "oneway": [True, True],
        },
        geometry=[
            LineString([(630000, 4840000), (630300, 4840000)]),
            LineString([(630000, 4840010), (630300, 4840030)]),
        ],
        crs=CRS,
    )
    lon, lat = _to_lonlat([630100.0, 630100.0, 630100.0], [4840002.0] * 3)
    svc = pd.DataFrame(
        {
            "location_name": [
                "Gardiner W / Sherbourne Ramp",  # a ramp: mainline is 2 m away, link 15 m
                "Gardiner Expy E (id: 20044024)",  # named mainline count
                "Unnamed Rd: A - B",  # no name match: never the mainline
            ],
            "lon": lon,
            "lat": lat,
        }
    )
    out = aadt.match_svc(svc, seg, max_m=40, unnamed_max_m=15)
    assert out["segment_id"].tolist()[:2] == [21, 20]
    assert out.loc[2, "segment_id"] == 21
