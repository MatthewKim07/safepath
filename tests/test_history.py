import numpy as np
import pandas as pd
import pytest

from safepath.features.history import YEARS_SINCE_CAP, history_features

YEARS = list(range(2014, 2026))
LABEL_YEARS = list(range(2016, 2026))


def _labels(seed: int = 0, n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = pd.MultiIndex.from_product([range(n), YEARS], names=["intersection_id", "year"])
    df = grid.to_frame(index=False)
    df["crash_count"] = rng.poisson(2.0, len(df)) * rng.integers(0, 2, len(df))
    df["severe_count"] = rng.binomial(df["crash_count"], 0.15)
    return df


@pytest.mark.parametrize("year", LABEL_YEARS)
def test_no_label_year_or_future_information_leaks(year):
    labels = _labels()
    before = history_features(labels, [year])

    rng = np.random.default_rng(year)
    future = labels["year"] >= year
    perturbed = labels.copy()
    perturbed.loc[future, "crash_count"] = rng.integers(0, 500, future.sum())
    perturbed.loc[future, "severe_count"] = rng.integers(0, 50, future.sum())
    after = history_features(perturbed, [year])

    pd.testing.assert_frame_equal(before, after)


def test_past_changes_do_change_features():
    labels = _labels()
    before = history_features(labels, [2020])
    changed = labels.copy()
    changed.loc[changed["year"] == 2019, "crash_count"] += 7
    after = history_features(changed, [2020])
    assert not before["prev1_crashes"].equals(after["prev1_crashes"])


def test_hand_computed_values():
    rows = []
    # intersection 0: crashes 0,4,0,2 in 2014-2017; intersection 1: 1 crash per year
    for year, (a, b) in zip(range(2014, 2018), [(0, 1), (4, 1), (0, 1), (2, 1)], strict=True):
        rows += [(0, year, a, a // 2), (1, year, b, 0)]
    labels = pd.DataFrame(rows, columns=["intersection_id", "year", "crash_count", "severe_count"])
    f = history_features(labels, [2016, 2017]).set_index(["intersection_id", "year"])

    r = f.loc[(0, 2017)]  # window 2014-2016
    assert r["prev1_crashes"] == 0
    assert r["prev3_crashes_mean"] == pytest.approx(4 / 3)
    assert r["prev1_crash_share"] == 0
    assert r["prev3_crash_share_mean"] == pytest.approx((0 / 1 + 4 / 5 + 0 / 1) / 3)
    assert r["prev3_severe_mean"] == pytest.approx(2 / 3)
    assert r["years_since_last_crash"] == 2  # last crash in 2015

    r = f.loc[(0, 2016)]  # only 2014-2015 exist
    assert r["n_history_years"] == 2
    assert r["prev3_crashes_mean"] == pytest.approx(2.0)
    assert f.loc[(1, 2016), "years_since_last_crash"] == 1


def test_years_since_cap_and_covid_flag():
    labels = _labels()
    labels.loc[labels["intersection_id"] == 0, "crash_count"] = 0
    f = history_features(labels, LABEL_YEARS).set_index(["intersection_id", "year"])
    assert (f.loc[0, "years_since_last_crash"] == YEARS_SINCE_CAP).all()
    covid = f.xs(0, level="intersection_id")["covid_in_window"]
    assert covid.to_dict() == {
        2016: 0,
        2017: 0,
        2018: 0,
        2019: 0,
        2020: 0,
        2021: 1,
        2022: 1,
        2023: 1,
        2024: 1,
        2025: 0,
    }
