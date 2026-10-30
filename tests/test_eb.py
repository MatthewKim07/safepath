import numpy as np
import pandas as pd
import pytest

from safepath.models import eb


def _labels(n=50, years=range(2014, 2026), seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        {"intersection_id": i, "year": y, "crash_count": int(rng.poisson(1 + i % 7)),
         "is_excluded": False}
        for i in range(n) for y in years
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_eb_uses_only_years_before_target():
    labels = _labels()
    base = pd.Series(np.linspace(0.5, 5, 50), index=pd.RangeIndex(50, name="intersection_id"))
    k = pd.Series(0.8, index=base.index)
    before = eb.eb_all(base, k, labels, range(2022, 2023), window=5)
    future = labels["year"] >= 2022
    perturbed = labels.copy()
    perturbed.loc[future, "crash_count"] = perturbed.loc[future, "crash_count"] * 10 + 7
    after = eb.eb_all(base, k, perturbed, range(2022, 2023), window=5)
    assert np.allclose(before["eb"], after["eb"])
    # and the history does matter: perturbing 2021 changes it
    past = perturbed.copy()
    past.loc[past["year"] == 2021, "crash_count"] += 5
    changed = eb.eb_all(base, k, past, range(2022, 2023), window=5)
    assert not np.allclose(before["eb"], changed["eb"])


def test_eb_formula_and_limits():
    base = pd.Series([2.0, 2.0], index=pd.Index([1, 2], name="intersection_id"))
    labels = pd.DataFrame(
        {
            "intersection_id": [1, 2, 1, 2],
            "year": [2020, 2020, 2021, 2021],
            "crash_count": [0, 8, 0, 8],
        }
    )
    levels = pd.Series({2020: 1.0, 2021: 1.0})
    k = pd.Series(0.5, index=base.index)
    out = eb.eb_for_year(base, k, labels, levels, 2022, window=2).set_index("intersection_id")
    w = 1 / (1 + 0.5 * 4.0)
    assert out.loc[1, "eb"] == pytest.approx(2.0 * w)
    assert out.loc[2, "eb"] == pytest.approx(2.0 * (w + (1 - w) * 16 / 4))
    # no overdispersion -> trust the SPF entirely
    out0 = eb.eb_for_year(base, k * 0, labels, levels, 2022, window=2)
    assert np.allclose(out0["eb"], 2.0)
