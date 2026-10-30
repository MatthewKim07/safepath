"""Traffic-volume (AADT) model for every intersection and road segment.

Targets: log AADT at counted intersections (entering volume, TMC) and counted segments
(two-way volume, SVC), from data/aadt.py. Features are OSM road attributes plus sampled
betweenness on the travel-time graph (global and within 5 minutes), which is the main
volume signal for uncounted roads. Evaluated by 5-fold spatial-block CV (2 km blocks, as in
CP3) against a class-median baseline.

Final AADT per counted unit is a precision-weighted blend in log space of its most recent
count and the out-of-fold model prediction:

    w = v_model / (v_model + v_count(age)),   v_count(age) = s0^2 + tau^2 * age

where v_model is the CV error variance, s0^2 the count-to-count noise and tau^2 the yearly
drift, both estimated from repeat counts at the same location. A count from this year gets
most of the weight; a 20-year-old count much less. Uncounted units use the model; the
400-series use MTO AADT.

Outputs in data/processed/: aadt_intersections.parquet, aadt_edges.parquet (one row per
directed edge with two-way and directional AADT). Orchestration and the report live in
models/volume_report.py; features in features/volume_features.py.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from safepath.data.aadt import (
    COVID_YEARS,
)
from safepath.models.lgbm import STATIC_FEATURES
from safepath.models.train import spatial_folds

log = logging.getLogger(__name__)

AADT_INTERSECTIONS = "aadt_intersections.parquet"
AADT_EDGES = "aadt_edges.parquet"
VOLUME_CV = "volume_cv.json"

PREDICT_YEAR = 2025  # predictions describe current volumes
MIN_TARGET_YEAR = 2010  # older counts describe a different network

INTERSECTION_FEATURES = [
    *STATIC_FEATURES,
    "bt_global_in", "bt_local_in", "bt_global_in_max", "dist_downtown_m",
    "n_intersections_500m", "count_year",
]  # fmt: skip
SEGMENT_FEATURES = [
    "highway", "class_rank", "lanes_n", "lanes_missing", "maxspeed_kph", "maxspeed_missing",
    "is_oneway", "is_divided", "length", "is_bridge_tunnel", "bt_global", "bt_local",
    "end_max_class_rank", "end_has_signal", "n_pois_100m", "n_bus_stops_50m", "dist_tram_m",
    "dist_motorway_m", "dist_downtown_m", "n_intersections_500m", "count_year",
]  # fmt: skip
CATEGORICAL = ["class_combo", "highway"]
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 7,
    "deterministic": True,
    "num_threads": 4,
}
EDGE_COLUMNS = [
    "u", "v", "key", "segment_id", "highway", "aadt_segment", "dir_share", "aadt_dir", "source",
    "mto_hwy", "mto_lhrs", "aadt_dir_model",
]  # fmt: skip
SEGMENT_CLASSES = [
    "motorway", "motorway_link", "primary", "secondary", "tertiary", "unclassified", "residential",
]  # fmt: skip


# --------------------------------------------------------------------------- targets


def unit_targets(counts: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Per unit: mean log AADT of its most recent non-2020/21 count year since 2010."""
    c = counts.dropna(subset=[unit, "aadt"])
    c = c[(c["aadt"] > 0) & (c["year"] >= MIN_TARGET_YEAR) & ~c["year"].isin(COVID_YEARS)]
    latest = c.groupby(unit)["year"].transform("max")
    last = c[c["year"] == latest]
    out = last.groupby(unit).agg(count_year=("year", "first"), log_aadt=("aadt", "mean"))
    out["log_aadt"] = np.log(out["log_aadt"])
    out.index = out.index.astype(int)
    return out


def repeat_count_variance(counts: pd.DataFrame, unit: str) -> dict[str, float]:
    """Fit E[(log a2 - log a1)^2] = 2 s0^2 + tau^2 * gap over consecutive counts at the same
    unit (non-negative least squares)."""
    c = counts.dropna(subset=[unit, "aadt"])
    c = c[(c["aadt"] > 0) & ~c["year"].isin(COVID_YEARS)].copy()
    c["t"] = c["year"] + c.get("date", c.get("date_start")).dt.day_of_year / 365.25
    c = c.sort_values([unit, "t"])
    c["log"] = np.log(c["aadt"])
    d = c.groupby(unit)[["log", "t"]].diff().dropna()
    gap, sq = d["t"].to_numpy(), d["log"].to_numpy() ** 2
    keep = gap <= 30
    A = np.column_stack([np.full(keep.sum(), 2.0), gap[keep]])
    (s0_sq, tau_sq), _ = nnls(A, sq[keep])
    return {"s0_sq": float(s0_sq), "tau_sq": float(tau_sq), "pairs": int(keep.sum())}


# --------------------------------------------------------------------------- model


def prepare(df: pd.DataFrame, features: list[str], cats: dict[str, list]) -> pd.DataFrame:
    X = df[features].copy()
    for col in X.columns:
        if col in CATEGORICAL:
            X[col] = pd.Categorical(X[col], categories=cats[col])
        elif X[col].dtype == bool:
            X[col] = X[col].astype(int)
    return X


def fit(
    tr: pd.DataFrame,
    features: list[str],
    cats: dict[str, list],
    rounds: int | None = None,
    va: pd.DataFrame | None = None,
) -> lgb.Booster:
    dtr = lgb.Dataset(prepare(tr, features, cats), tr["log_aadt"], free_raw_data=False)
    if va is None:
        return lgb.train(PARAMS, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(prepare(va, features, cats), va["log_aadt"], reference=dtr)
    return lgb.train(
        PARAMS,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def cross_validate(
    data: pd.DataFrame, features: list[str], cats: dict[str, list], class_col: str
) -> tuple[pd.DataFrame, list[int]]:
    """Out-of-fold LightGBM and class-median predictions over spatial blocks. Each fold
    early-stops on the next fold, so the held-out fold is never used for fitting."""
    folds = spatial_folds(data)
    n = folds.max() + 1
    oof = pd.DataFrame(index=data.index, columns=["fold", "lgbm", "baseline"], dtype=float)
    iters = []
    for k in range(n):
        held, stop = folds == k, folds == (k + 1) % n
        tr, va = data[~held & ~stop], data[stop]
        booster = fit(tr, features, cats, va=va)
        iters.append(booster.best_iteration)
        test = data[held]
        oof.loc[held, "lgbm"] = booster.predict(
            prepare(test, features, cats), num_iteration=booster.best_iteration
        )
        fit_all = data[~held]
        med = fit_all.groupby(class_col)["log_aadt"].median()
        oof.loc[held, "baseline"] = (
            test[class_col].map(med).fillna(fit_all["log_aadt"].median()).to_numpy()
        )
        oof.loc[held, "fold"] = k
    return oof, iters


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    ape = np.abs(np.exp(err) - 1)
    return {
        "n": len(y),
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "median_abs_pct_err": float(100 * np.median(ape)),
        "r2_log": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "bias_log": float(np.mean(err)),
    }


def metrics_table(data: pd.DataFrame, oof: pd.DataFrame, class_col: str) -> pd.DataFrame:
    rows = []
    groups = [("all", data.index), *data.groupby(class_col).groups.items()]
    for name, idx in groups:
        if len(idx) < 20:
            continue
        y = data.loc[idx, "log_aadt"].to_numpy()
        for model in ("baseline", "lgbm"):
            m = metrics(y, oof.loc[idx, model].astype(float).to_numpy())
            rows.append({class_col: name, "model": model, **m})
    return pd.DataFrame(rows)


def blend(
    pred_log: pd.Series,
    obs_log: pd.Series,
    obs_year: pd.Series,
    v_model: float,
    var: dict[str, float],
) -> pd.DataFrame:
    """Precision-weighted log-space blend of the latest count and the model."""
    age = (PREDICT_YEAR - obs_year).clip(lower=0)
    v_count = var["s0_sq"] + var["tau_sq"] * age
    w = (v_model / (v_model + v_count)).where(obs_log.notna(), 0.0)
    final = np.where(obs_log.notna(), w * obs_log + (1 - w) * pred_log, pred_log)
    return pd.DataFrame({"weight_count": w, "log_aadt": final})


# --------------------------------------------------------------------------- outputs


def directional_split(edges: pd.DataFrame) -> pd.Series:
    """Share of the two-way AADT on one directed edge: 1/2 on a two-way street or one
    carriageway of a divided road, all of it on a one-way street. Where a motorway has
    collector and express carriageways (401, 427), each direction's half is split between
    them by their median lane counts in that MTO section."""
    share = np.where(edges["is_oneway"] & ~edges["is_divided"], 1.0, 0.5)
    share = pd.Series(share, index=edges.index)
    mto = edges["mto_lhrs"].notna()
    if mto.any():
        e = edges[mto].copy()
        e["lane_type"] = np.select(
            [
                e["name"].str.contains("Express", na=False),
                e["name"].str.contains("Collector", na=False),
            ],
            ["express", "collector"],
            "single",
        )
        lanes = e.groupby(["mto_hwy", "mto_lhrs", "lane_type"])["lanes_n"].median()

        def frac(r: pd.Series) -> float:
            if r["lane_type"] == "single":
                return 1.0
            key = (r["mto_hwy"], r["mto_lhrs"])
            both = [lanes.get((*key, t)) for t in ("express", "collector")]
            if any(x is None or np.isnan(x) for x in both):
                return 1.0
            return lanes[(*key, r["lane_type"])] / sum(both)

        share[mto] = 0.5 * e.apply(frac, axis=1)
    return share


def lane_scaled_expressways(
    edges: pd.DataFrame, svc_all: pd.DataFrame, sx: pd.DataFrame
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """City expressways (motorway class, not 400-series: Gardiner, DVP, Allen, ...) have no
    usable counts, and the model, whose only freeway targets are 400-series, gives them
    400-series volumes. Instead: directional AADT = the 400-series' median AADT per through
    lane (MTO AADT x directional share / OSM lanes) x the carriageway's lanes."""
    out = edges.copy()
    mto = out[out["source"] == "mto"]
    per_lane = float((mto["aadt_dir"] / mto["lanes_n"]).median())
    city = (out["highway"] == "motorway") & out["mto_lhrs"].isna()
    model_dir = out.loc[city, "aadt_dir"].copy()
    out["aadt_dir_model"] = np.where(city, out["aadt_dir"], np.nan)  # kept for sensitivity
    out.loc[city, "aadt_dir"] = per_lane * out.loc[city, "lanes_n"]
    out.loc[city, "aadt"] = out.loc[city, "aadt_dir"] / out.loc[city, "dir_share"]
    out.loc[city, "source"] = "lane-scaled"
    e = out[city].assign(model_dir=model_dir, length=out.loc[city, "segment_id"].map(sx["length"]))
    hist = svc_all[svc_all["expressway_mainline"]].assign(
        name=lambda d: d["segment_id"].map(sx["name"])
    )
    rows = []
    for name, g in e.groupby("name"):
        h = hist[hist["name"] == name]
        rows.append(
            {
                "expressway": name,
                "carriageway km": g["length"].sum() / 1000,
                "lanes (median)": g["lanes_n"].median(),
                "model two-way AADT": float((g["model_dir"] / g["dir_share"]).median()),
                "lane-scaled two-way AADT": float(g["aadt"].median()),
                "City tube counts (n, years, median daily)": (
                    f"{len(h)}, {h['year'].min()}-{h['year'].max()}, {h['aadt'].median():,.0f}"
                    if len(h)
                    else ""
                ),
            }
        )
    info = {"segment_ids": e["segment_id"].unique(), "table": pd.DataFrame(rows)}
    return out, per_lane, info


def distribution(values: pd.Series, classes: pd.Series, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"aadt": values, label: classes}).dropna()
    g = df.groupby(label)["aadt"]
    out = pd.DataFrame(
        {"n": g.size(), "p10": g.quantile(0.1), "median": g.median(), "p90": g.quantile(0.9)}
    ).reset_index()
    return out.sort_values("median", ascending=False)


def run_part(
    data: pd.DataFrame, features: list[str], cats: dict[str, list], class_col: str
) -> dict[str, Any]:
    oof, iters = cross_validate(data, features, cats, class_col)
    booster = fit(data, features, cats, rounds=int(np.median(iters)))
    return {
        "data": data,
        "oof": oof,
        "iters": iters,
        "booster": booster,
        "metrics": metrics_table(data, oof, class_col),
        "v_model": float(np.mean((oof["lgbm"].astype(float) - data["log_aadt"]) ** 2)),
    }


def latest_obs(counts: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Log of the mean AADT in the most recent non-2020/21 count year, and that year."""
    c = counts.dropna(subset=[unit, "aadt"])
    c = c[(c["aadt"] > 0) & ~c["year"].isin(COVID_YEARS)]
    latest = c.groupby(unit)["year"].transform("max")
    last = c[c["year"] == latest]
    g = last.groupby(last[unit].astype(int))
    return pd.DataFrame({"log_obs": np.log(g["aadt"].mean()), "obs_year": g["year"].first()})


def mto_check(data: pd.DataFrame, oof: pd.DataFrame, mto: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold predictions on the MTO-target segments, by highway."""
    rows = []
    m = data[data["target"] == "mto"]
    df = pd.DataFrame(
        {"hwy": mto.loc[m.index, "hwy"], "obs": m["log_aadt"], "pred": oof.loc[m.index, "lgbm"]}
    ).astype({"pred": float})
    for hwy, g in [*df.groupby("hwy"), ("all", df)]:
        ratio = np.exp(g["pred"] - g["obs"])
        rows.append(
            {
                "highway": hwy,
                "OSM segments": len(g),
                "MTO AADT (median)": float(np.exp(g["obs"]).median()),
                "out-of-fold model (median)": float(np.exp(g["pred"]).median()),
                "median model / MTO": float(ratio.median()),
                "median abs % err": float(100 * np.median(np.abs(ratio - 1))),
            }
        )
    return pd.DataFrame(rows)
