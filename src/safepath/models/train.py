"""Fit baselines and LightGBM models on clean train rows, early-stopping on val (2024)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safepath.features.build import MODEL_TABLE
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
}


def load_table(cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    table = pd.read_parquet(processed / MODEL_TABLE)
    xy = pd.read_parquet(processed / INTERSECTIONS, columns=["intersection_id", "x", "y"])
    return table.merge(xy, on="intersection_id", how="left")


def categories(table: pd.DataFrame) -> dict[str, list]:
    return {"class_combo": sorted(table["class_combo"].unique())}


def models_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["models_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


SPATIAL_OOF = "static_only_spatial_oof.parquet"
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
) -> pd.DataFrame:
    """Out-of-fold static-only predictions for val/test rows: each fold's model never saw
    those intersections, so this measures transfer to intersections with no crash record."""
    folds = spatial_folds(clean)
    features = lgbm.FEATURE_SETS["static_only"]
    parts = []
    for k in range(N_FOLDS):
        inside = folds == k
        # early-stop on a different set of unseen blocks, so the held-out fold stays untouched
        stop = folds == (k + 1) % N_FOLDS
        tr = clean[~inside & ~stop & (clean["split"] == "train")]
        va = clean[stop & (clean["split"] == "val")]
        booster, _ = lgbm.fit(
            tr, va, features, "crash_count", cats, grid={key: [v] for key, v in params.items()}
        )
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
    return pd.concat(parts, ignore_index=True)


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
    oof = spatial_holdout_static(clean, summary["models"]["lgbm_static_only"]["best_params"], cats)
    oof.to_parquet(out / SPATIAL_OOF, index=False)
    summary["spatial_holdout"] = {"block_m": BLOCK_M, "n_folds": N_FOLDS, "rows": len(oof)}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("trained %d models on %d rows", len(MODELS), len(train))
    return summary
