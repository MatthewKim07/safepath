"""Fit baselines and LightGBM models on clean train rows, early-stopping on val (2024)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from safepath.features.build import MODEL_TABLE
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.graph import INTERSECTIONS
from safepath.models import lgbm
from safepath.models.baselines import fit_baselines

log = logging.getLogger(__name__)

# name -> (feature set, target)
MODELS = {
    "lgbm_main": ("main", "crash_count"),
    "lgbm_static_only": ("static_only", "crash_count"),
    "lgbm_with_coords": ("with_coords", "crash_count"),
    "lgbm_severe": ("main", "severe_count"),
    "lgbm_aadt": ("aadt", "crash_count"),
}


def load_table(cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    table = pd.read_parquet(processed / MODEL_TABLE)
    xy = pd.read_parquet(processed / INTERSECTIONS, columns=["intersection_id", "x", "y"])
    table = table.merge(xy, on="intersection_id", how="left")
    expo = pd.read_parquet(processed / INTERSECTION_EXPOSURE)
    expo = pd.DataFrame(
        {
            "intersection_id": expo["intersection_id"],
            "log_aadt_entering": np.log(expo["entering_aadt"]),
            "log_aadt_major": np.log(expo["major_aadt"]),
            "log_aadt_minor": np.log(expo["minor_aadt"]),  # NaN with a single road
        }
    )
    return table.merge(expo, on="intersection_id", how="left")


def categories(table: pd.DataFrame) -> dict[str, list]:
    return {"class_combo": sorted(table["class_combo"].unique())}


def models_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["models_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


SPATIAL_OOF = "static_only_spatial_oof.parquet"
STATIC_DEPLOY = "lgbm_static_deploy.txt"
BLOCK_M = 2000
N_FOLDS = 5


def spatial_folds(
    table: pd.DataFrame, block_m: float = BLOCK_M, n_folds: int = N_FOLDS, seed: int = 7
) -> pd.Series:
    """Fold id per row from square spatial blocks, so a fold's intersections are never seen
    in training. Blocks are shuffled into folds with a fixed seed."""
    block = (
        (table["x"] // block_m).astype(int).astype(str)
        + "_"
        + (table["y"] // block_m).astype(int).astype(str)
    )
    blocks = np.sort(block.unique())
    order = np.random.default_rng(seed).permutation(len(blocks))
    fold_of = dict(zip(blocks[order], np.arange(len(blocks)) % n_folds, strict=True))
    return block.map(fold_of)


def spatial_holdout_static(
    clean: pd.DataFrame, params: dict[str, Any], cats: dict[str, list]
) -> tuple[pd.DataFrame, list[int]]:
    """Out-of-fold static-only predictions for val/test rows: each fold's model never saw
    those intersections, so this measures transfer to intersections with no crash record.
    Also returns each fold's early-stopping round count."""
    folds = spatial_folds(clean)
    features = lgbm.FEATURE_SETS["static_only"]
    parts, iterations = [], []
    for k in range(N_FOLDS):
        inside = folds == k
        # early-stop on a different set of unseen blocks, so the held-out fold stays untouched
        stop = folds == (k + 1) % N_FOLDS
        tr = clean[~inside & ~stop & (clean["split"] == "train")]
        va = clean[stop & (clean["split"] == "val")]
        booster, info = lgbm.fit(
            tr, va, features, "crash_count", cats, grid={key: [v] for key, v in params.items()}
        )
        iterations.append(info["best_iteration"])
        held = clean[inside & clean["split"].isin(["val", "test"])]
        parts.append(
            pd.DataFrame(
                {
                    "intersection_id": held["intersection_id"],
                    "year": held["year"],
                    "fold": k,
                    "pred": lgbm.predict(booster, held, features, cats),
                }
            )
        )
    return pd.concat(parts, ignore_index=True), iterations


def fit_static_deploy(
    clean: pd.DataFrame, params: dict[str, Any], rounds: int, cats: dict[str, list]
) -> lgb.Booster:
    """Static-only model on every clean label year (2016-2019, 2022-2025), for intersections
    whose own counts cannot be trusted. Its round count comes from the spatial holdout, i.e.
    the setting where the model has not seen the intersection, which is the deployment case."""
    rows = clean[clean["split"] != "history"]
    features = lgbm.FEATURE_SETS["static_only"]
    data = lgb.Dataset(
        lgbm.prepare(rows, features, cats),
        rows["crash_count"],
        categorical_feature=lgbm.CATEGORICAL,
    )
    return lgb.train({**lgbm.BASE_PARAMS, **params}, data, num_boost_round=rounds)


def run_train(cfg: dict[str, Any]) -> dict[str, Any]:
    table = load_table(cfg)
    clean = table[~table["is_excluded"]]
    train, val = clean[clean["split"] == "train"], clean[clean["split"] == "val"]
    cats = categories(table)
    out = models_dir(cfg)

    summary: dict[str, Any] = {"train_rows": len(train), "val_rows": len(val), "models": {}}
    for target in ("crash_count", "severe_count"):
        fitted = fit_baselines(train, target)
        pd.to_pickle(fitted, out / f"baselines_{target}.pkl")

    for name, (feature_set, target) in MODELS.items():
        features = lgbm.FEATURE_SETS[feature_set]
        booster, info = lgbm.fit(train, val, features, target, cats)
        booster.save_model(out / f"{name}.txt", num_iteration=booster.best_iteration)
        summary["models"][name] = {
            "feature_set": feature_set,
            "target": target,
            "n_features": len(features),
            **info,
        }
    static_params = summary["models"]["lgbm_static_only"]["best_params"]
    oof, iterations = spatial_holdout_static(clean, static_params, cats)
    oof.to_parquet(out / SPATIAL_OOF, index=False)
    rounds = int(np.median(iterations))
    fit_static_deploy(clean, static_params, rounds, cats).save_model(out / STATIC_DEPLOY)
    summary["spatial_holdout"] = {
        "block_m": BLOCK_M,
        "n_folds": N_FOLDS,
        "rows": len(oof),
        "best_iterations": iterations,
    }
    summary["static_deploy"] = {
        "rounds": rounds,
        "params": static_params,
        "label_years": "all clean train/val/test years",
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("trained %d models on %d rows", len(MODELS), len(train))
    return summary
