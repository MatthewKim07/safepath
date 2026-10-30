import numpy as np
import pandas as pd
import pytest

from safepath.models import spf


def _frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    major = np.exp(rng.uniform(np.log(2000), np.log(60000), n))
    minor = np.exp(rng.uniform(np.log(200), np.log(20000), n))
    df = pd.DataFrame(
        {
            "intersection_id": np.arange(n) % (n // 2),
            "year": np.where(np.arange(n) < n // 2, 2016, 2017),
            "log_major": np.log(major),
            "log_minor": np.log(minor),
            "log_entering_single": 0.0,
            "single_road": 0.0,
            "has_signal": rng.integers(0, 2, n).astype(float),
            "streetcar_20m": 0.0,
            "min_leg_angle_10deg": rng.uniform(6, 9, n),
            "legs": rng.choice(["3-", "4"], n),
            "class_combo": rng.choice(["arterial x local", "local x local"], n),
            "is_excluded": False,
            "group": "unsignalised",
        }
    )
    mu = np.exp(-6 + 0.5 * df["log_major"] + 0.3 * df["log_minor"] + 0.4 * df["has_signal"])
    k = 0.8
    df["crash_count"] = rng.negative_binomial(1 / k, 1 / (1 + k * mu))
    return df


def test_nb2_recovers_elasticities_and_k():
    df = _frame()
    models = spf.fit_models(df, grouped=False)
    m = models[spf.POOLED]
    assert m["params"]["log_major"] == pytest.approx(0.5, abs=0.08)
    assert m["params"]["log_minor"] == pytest.approx(0.3, abs=0.08)
    assert m["k"] == pytest.approx(0.8, abs=0.2)
    assert "year_2016" in m["reference"]


def test_predict_base_ignores_year_effects():
    df = _frame()
    models = spf.fit_models(df, grouped=False)
    pair = pd.concat([df.iloc[[0]], df.iloc[[0]].assign(year=2017)], ignore_index=True)
    base = spf.predict_base(models, pair)
    assert base[0] == pytest.approx(base[1])


def test_forecast_uses_previous_year_level_only():
    df = pd.DataFrame(
        {
            "intersection_id": [1, 2, 1, 2, 1, 2],
            "year": [2020, 2020, 2021, 2021, 2022, 2022],
            "crash_count": [1, 1, 4, 4, 100, 100],
            "is_excluded": False,
            "group": "x",
        }
    )
    base = pd.Series(1.0, index=df.index)
    out = spf.with_levels(df, base)
    assert out.loc[2, "spf_known"] == pytest.approx(4.0)
    assert out.loc[2, "spf_forecast"] == pytest.approx(1.0)  # 2021 forecast uses 2020 level
    assert out.loc[4, "spf_forecast"] == pytest.approx(4.0)  # not the huge 2022 total
    assert np.isnan(out.loc[0, "spf_forecast"])
