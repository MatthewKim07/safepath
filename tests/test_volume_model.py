import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from safepath.features.volume_features import divided_carriageways
from safepath.models import volume


def test_blend_weights_recent_counts_more():
    pred = pd.Series([np.log(10_000.0)] * 3)
    obs = pd.Series([np.log(20_000.0), np.log(20_000.0), np.nan])
    year = pd.Series([2025, 1995, np.nan])
    var = {"s0_sq": 0.01, "tau_sq": 0.005}
    out = volume.blend(pred, obs, year, v_model=0.2, var=var)
    w_new, w_old = out["weight_count"].iloc[0], out["weight_count"].iloc[1]
    assert w_new == pytest.approx(0.2 / 0.21)
    assert w_old == pytest.approx(0.2 / (0.2 + 0.01 + 0.005 * 30))
    assert w_new > w_old
    assert out["log_aadt"].iloc[2] == pytest.approx(np.log(10_000))
    expected = w_new * np.log(20_000) + (1 - w_new) * np.log(10_000)
    assert out["log_aadt"].iloc[0] == pytest.approx(expected)


def test_repeat_count_variance_recovers_noise_and_drift():
    rng = np.random.default_rng(1)
    s0, tau = 0.1, 0.05
    rows = []
    for unit in range(4000):
        level = rng.normal(9, 1)
        years = np.sort(rng.choice(np.arange(2000, 2026), size=3, replace=False))
        prev = None
        for y in years:
            level = level if prev is None else level + rng.normal(0, tau * np.sqrt(y - prev))
            rows.append(
                {
                    "unit": unit,
                    "year": y,
                    "date": pd.Timestamp(int(y), 6, 1),
                    "aadt": np.exp(level + rng.normal(0, s0)),
                }
            )
            prev = y
    var = volume.repeat_count_variance(pd.DataFrame(rows), "unit")
    assert var["s0_sq"] == pytest.approx(s0**2, rel=0.15)
    assert var["tau_sq"] == pytest.approx(tau**2, rel=0.15)


def test_unit_targets_skip_covid_and_old_counts():
    counts = pd.DataFrame(
        {
            "intersection_id": [1, 1, 1, 2, 3],
            "year": [2015, 2019, 2021, 2005, 2023],
            "aadt": [100.0, 200.0, 50.0, 300.0, 400.0],
        }
    )
    out = volume.unit_targets(counts, "intersection_id")
    assert out.loc[1, "count_year"] == 2019
    assert out.loc[1, "log_aadt"] == pytest.approx(np.log(200))
    assert 2 not in out.index  # only a pre-2010 count


def test_divided_carriageways():
    seg = gpd.GeoDataFrame(
        {
            "segment_id": [1, 2, 3, 4],
            "name": ["Main Street", "Main Street", "Side Street", "Main Street"],
            "is_oneway": [True, True, True, False],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 20), (0, 20)]),  # opposite direction, 20 m away
            LineString([(0, 40), (100, 40)]),  # one-way street, no partner
            LineString([(0, 60), (100, 60)]),  # two-way
        ],
        crs="EPSG:32617",
    )
    assert divided_carriageways(seg).tolist() == [True, True, False, False]


def test_directional_split():
    edges = pd.DataFrame(
        {
            "is_oneway": [False, True, True, True, True],
            "is_divided": [False, False, True, True, True],
            "mto_lhrs": [np.nan, np.nan, np.nan, 7, 7],
            "mto_hwy": [None, None, None, "401", "401"],
            "name": ["A", "B", "C", "Highway 401 Express", "Highway 401 Collector"],
            "lanes_n": [2, 2, 2, 4, 2],
        }
    )
    share = volume.directional_split(edges)
    assert share.tolist() == pytest.approx([0.5, 1.0, 0.5, 0.5 * 4 / 6, 0.5 * 2 / 6])
