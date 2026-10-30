"""LightGBM Poisson models on intersection-year features."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STATIC_FEATURES = [
    "n_osm_nodes", "street_count", "max_class_rank", "min_class_rank", "class_combo", "n_legs",
    "min_leg_angle_deg", "n_approaches", "max_maxspeed_kph", "min_maxspeed_kph",
    "maxspeed_missing", "max_lanes", "lanes_missing", "n_oneway_approaches",
    "any_bridge_tunnel", "has_motorway_approach", "n_signals_30m", "n_stop_signs_30m",
    "n_crossings_30m", "has_signal", "n_bus_stops_50m", "n_pois_100m", "dist_tram_m",
    "streetcar_20m", "dist_motorway_m", "dist_retail_m", "dist_nearest_intersection_m",
]  # fmt: skip
HISTORY_FEATURES = [
    "prev1_crashes", "prev1_severe", "prev3_crashes_mean", "prev3_severe_mean",
    "prev1_crash_share", "prev1_severe_share", "prev3_crash_share_mean",
    "prev3_severe_share_mean", "years_since_last_crash", "covid_in_window",
]  # fmt: skip
AADT_FEATURES = ["log_aadt_entering", "log_aadt_major", "log_aadt_minor"]  # CP5 volumes
COORD_FEATURES = ["x", "y"]  # ablation only: lets the model memorise locations
CATEGORICAL = ["class_combo"]

FEATURE_SETS = {
    "main": STATIC_FEATURES + HISTORY_FEATURES,
    "static_only": STATIC_FEATURES,
    "with_coords": STATIC_FEATURES + HISTORY_FEATURES + COORD_FEATURES,
    "aadt": STATIC_FEATURES + HISTORY_FEATURES + AADT_FEATURES,
}

BASE_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 7,
    "deterministic": True,
    "num_threads": 4,
}
GRID = {"num_leaves": [15, 31, 63], "min_data_in_leaf": [50, 200]}


def prepare(df: pd.DataFrame, features: list[str], categories: dict[str, list]) -> pd.DataFrame:
    X = df[features].copy()
    for col in X.columns:
        if col in CATEGORICAL:
            X[col] = pd.Categorical(X[col], categories=categories[col])
        elif X[col].dtype == bool:
            X[col] = X[col].astype(int)
    return X


def fit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target: str,
    categories: dict[str, list],
    grid: dict[str, list] | None = None,
    max_rounds: int = 3000,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Small grid search; each config early-stops on val Poisson deviance. Returns the best
    booster (trained on train only) and a summary of the search."""
    grid = grid or GRID
    X_tr, X_va = prepare(train, features, categories), prepare(val, features, categories)
    keys = list(grid)
    results, best = [], None
    for values in itertools.product(*(grid[k] for k in keys)):
        params = {**BASE_PARAMS, **dict(zip(keys, values, strict=True))}
        dtrain = lgb.Dataset(
            X_tr,
            train[target],
            categorical_feature=CATEGORICAL if "class_combo" in features else "auto",
            free_raw_data=False,
        )
        dval = lgb.Dataset(X_va, val[target], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        score = booster.best_score["valid_0"]["poisson"]
        results.append(
            {
                **dict(zip(keys, values, strict=True)),
                "best_iteration": booster.best_iteration,
                "val_poisson": score,
            }
        )
        if best is None or score < best[1]:
            best = (booster, score, params)
    booster, score, params = best
    log.info(
        "%s on %s: best %s, val poisson %.4f, %d rounds",
        target,
        len(features),
        {k: params[k] for k in keys},
        score,
        booster.best_iteration,
    )
    return booster, {
        "search": results,
        "best_params": {k: params[k] for k in keys},
        "best_iteration": booster.best_iteration,
        "best_val_poisson": score,
    }


def predict(
    booster: lgb.Booster, df: pd.DataFrame, features: list[str], categories: dict[str, list]
) -> np.ndarray:
    return booster.predict(prepare(df, features, categories), num_iteration=booster.best_iteration)
