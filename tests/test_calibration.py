import numpy as np
import pandas as pd
import pytest

from safepath.models import calibration


def _frame(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    aadt = np.exp(rng.uniform(np.log(500), np.log(50000), n))
    truth = 0.5 * (aadt / 5000) ** 0.6
    # a model that overpredicts quiet intersections 3x, tapering to 1x at high volume
    bias = 1 + 2 * np.clip((np.log(5000) - np.log(aadt)) / np.log(10), 0, 1)
    return pd.DataFrame(
        {
            "year": rng.choice([2016, 2017, 2018], n),
            "entering_aadt": aadt,
            "eb": truth * bias,
            "obs_history": rng.poisson(truth * 5),
            "crash_count": rng.poisson(truth),
        }
    )


def test_volume_recalibration_removes_volume_bias():
    df = calibration.add_terms(_frame())
    recal = calibration.Recalibration(calibration.FORMULAS["EB x volume correction"], df)
    df["fixed"] = recal.apply(df)
    before = calibration.decile_table(df, "eb", "entering_aadt", n_boot=1)
    after = calibration.decile_table(df, "fixed", "entering_aadt", n_boot=1)
    spread = lambda t: t["predicted / observed"].max() / t["predicted / observed"].min()  # noqa: E731
    assert spread(before) > 2.5
    assert spread(after) < 1.15  # flat across volume: the per-vehicle bias is gone
    # the correction changes shape, not the overall level on the data it was fitted on
    assert df["fixed"].sum() == pytest.approx(df["eb"].sum(), rel=1e-6)


def test_decile_table_ratio_and_per_vehicle_rates():
    df = pd.DataFrame(
        {
            "entering_aadt": np.repeat(np.arange(1, 11) * 1000.0, 10),
            "crash_count": 2.0,
            "pred": 3.0,
        }
    )
    t = calibration.decile_table(df, "pred", "entering_aadt", n_boot=5)
    assert (t["predicted / observed"] == 1.5).all()
    mev = 10 * 1000 * 365 / 1e6
    assert t.loc[0, "observed per MEV"] == pytest.approx(20 / mev)


def test_choice_uses_validation_per_vehicle_error_then_level():
    metrics = pd.DataFrame(
        [
            {"split": "val", "model": "a", "per-vehicle CE (AADT deciles)": 0.0381,
             "total predicted / observed": 0.90, "Poisson deviance": 1.0},
            {"split": "val", "model": "b", "per-vehicle CE (AADT deciles)": 0.0379,
             "total predicted / observed": 0.99, "Poisson deviance": 1.2},
            {"split": "val", "model": "c", "per-vehicle CE (AADT deciles)": 0.050,
             "total predicted / observed": 1.00, "Poisson deviance": 0.5},
            {"split": "test", "model": "c", "per-vehicle CE (AADT deciles)": 0.001,
             "total predicted / observed": 1.00, "Poisson deviance": 0.1},
        ]
    )  # fmt: skip
    # a and b tie at 3 decimals; b has the better level; test rows are never consulted
    assert calibration.choose(metrics) == "b"
