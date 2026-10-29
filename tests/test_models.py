import numpy as np
import pandas as pd

from safepath.models import lgbm
from safepath.models.baselines import fit_baselines, predict_baselines
from safepath.models.train import spatial_folds


def _table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "intersection_id": np.repeat(np.arange(n // 3), 3),
            "class_combo": rng.choice(["local x local", "arterial x local"], n),
            "prev3_crashes_mean": rng.random(n) * 3,
            "crash_count": rng.poisson(2, n),
        }
    )
    return df


def test_baselines():
    df = _table()
    fitted = fit_baselines(df)
    preds = predict_baselines(df, fitted)
    assert np.allclose(preds["baseline_prev3_mean"], df["prev3_crashes_mean"])
    by_int = df.groupby("intersection_id")["crash_count"].mean()
    assert np.allclose(preds["baseline_persistence"], df["intersection_id"].map(by_int))
    unseen = df.head(1).assign(intersection_id=999, class_combo="highway x highway")
    p = predict_baselines(unseen, fitted).iloc[0]
    assert p["baseline_persistence"] == p["baseline_class_combo"] == df["crash_count"].mean()


def test_prepare_uses_fixed_categories_and_ints_for_bools():
    df = pd.DataFrame({"class_combo": ["b x b"], "has_signal": [True]})
    X = lgbm.prepare(df, ["class_combo", "has_signal"], {"class_combo": ["a x a", "b x b"]})
    assert list(X["class_combo"].cat.categories) == ["a x a", "b x b"]
    assert X["has_signal"].dtype.kind == "i"


def test_main_features_exclude_coordinates_and_ids():
    for name in ("main", "static_only"):
        assert not {"x", "y", "intersection_id", "year", "hood_158"} & set(lgbm.FEATURE_SETS[name])
    assert {"x", "y"} <= set(lgbm.FEATURE_SETS["with_coords"])


def test_spatial_folds_keep_blocks_together():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.uniform(0, 20_000, 500), "y": rng.uniform(0, 20_000, 500)})
    folds = spatial_folds(df, block_m=2000, n_folds=5)
    block = (
        (df["x"] // 2000).astype(int).astype(str) + "_" + (df["y"] // 2000).astype(int).astype(str)
    )
    assert (pd.DataFrame({"b": block, "f": folds}).groupby("b")["f"].nunique() == 1).all()
    assert set(folds) == set(range(5))
