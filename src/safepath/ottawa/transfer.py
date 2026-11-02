"""Crash-model transfer to Ottawa (the CP9 result): reports/ottawa.md.

Evaluated on Ottawa validation (2022) and test (2024), clean intersections (proxies and
freeway crashes out), with Toronto's metrics and bootstrap. Groups:

(A) Toronto-trained, no Ottawa crash data at all: road-only LightGBM (CP3 static model),
    road-only GraphSAGE and MLP, the Toronto SPF on Ottawa AADT.
(B) Toronto-trained, Ottawa's own crash history as features (still no Ottawa training
    labels): LightGBM + AADT, LightGBM + AADT + neighbours, EB (Toronto SPF + Ottawa
    history), GraphSAGE with history.
(C) Ottawa-native baselines: persistence (Ottawa-train mean per intersection), prev-3 mean,
    Ottawa class-combo mean (Ottawa-train years).
(D) Upper bound: LightGBM + AADT trained on Ottawa-train years (2016-2019), early-stopped
    on Ottawa validation 2022 (so its validation numbers flatter it).

Ranking metrics are primary: reporting levels differ between cities. Calibration is shown
raw and after one level-adjustment scalar per model, fit on the prior year only (sum of
observed / sum of predicted, clean intersections): 2021 for validation 2022, and 2022 for
test 2024 (2023 is unpublished).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from safepath import tracking
from safepath.evaluate import N_BOOT, bootstrap, hotspot_recall, point_metrics
from safepath.features.build import STATIC
from safepath.features.labels import LABELS
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.models import lgbm
from safepath.models.baselines import fit_baselines, predict_baselines
from safepath.models.calibration import decile_table
from safepath.models.eb import EB_PREDICTIONS, eb_all
from safepath.models.spf import SPF_MODEL, SPF_PREDICTIONS, predict_base, spf_frame
from safepath.models.train import load_table
from safepath.ottawa.volumes import COUNTFREE_EXPOSURE

log = logging.getLogger(__name__)

TRANSFER_ROWS = "transfer_predictions.parquet"
TRANSFER_JSON = "transfer_results.json"
GNN_FILE = "gnn_predictions.parquet"
EB_WINDOW_FALLBACK = 5

GROUPS = {
    "A": "Toronto-trained, no Ottawa crash labels",
    "B": "Toronto-trained, Ottawa crash history as features (no Ottawa training labels)",
    "C": "Ottawa-native baselines",
    "D": "Ottawa-trained upper bound",
}
MODELS: dict[str, tuple[str, str]] = {  # column -> (group, label)
    "road_lgbm": ("A", "LightGBM, road features only"),
    "road_sage": ("A", "GraphSAGE, road features only"),
    "road_mlp": ("A", "MLP, road features only"),
    "spf": ("A", "SPF with Ottawa AADT"),
    "spf_countfree": ("A", "SPF with Toronto-model AADT (count-free)"),
    "lgbm_aadt": ("B", "LightGBM + AADT"),
    "lgbm_aadt_nbr": ("B", "LightGBM + AADT + neighbours"),
    "eb": ("B", "EB (Toronto SPF + Ottawa history)"),
    "sage_history": ("B", "GraphSAGE with history"),
    "persistence": ("C", "persistence (Ottawa-train mean)"),
    "prev3_mean": ("C", "prev-3-year mean"),
    "class_combo_mean": ("C", "Ottawa class-combo mean"),
    "upper_lgbm_aadt": ("D", "LightGBM + AADT trained on Ottawa"),
}
# What Ottawa data each model's inputs touch. "Counts" = City traffic counts, through the
# Ottawa AADT (blended counts + Ottawa-trained volume model) or, for road-only GraphSAGE,
# through edge AADT whose betweenness variant and level correction were chosen on counts.
DATA_USE = {
    "road_lgbm": "OSM only",
    "road_sage": "OSM + counts (edge AADT)",
    "road_mlp": "OSM only",
    "spf": "OSM + counts",
    "spf_countfree": "OSM only",
    "lgbm_aadt": "OSM + counts + crash history",
    "lgbm_aadt_nbr": "OSM + counts + crash history",
    "eb": "OSM + counts + crash history",
    "sage_history": "OSM + counts + crash history",
    "persistence": "crash labels 2016-19",
    "prev3_mean": "crash history",
    "class_combo_mean": "OSM + crash labels 2016-19",
    "upper_lgbm_aadt": "OSM + counts + crash labels 2016-19",
}
COUNT_FREE_A = ["road_lgbm", "road_mlp", "spf_countfree"]
N_SIM = 200
SHIFT_FEATURES = [
    "entering_aadt", "has_signal", "n_legs", "max_class_rank", "max_maxspeed_kph", "max_lanes",
    "n_pois_100m", "n_bus_stops_50m", "dist_nearest_intersection_m", "n_osm_nodes",
    "streetcar_20m", "dist_motorway_m",
]  # fmt: skip


def label(col: str) -> str:
    g, name = MODELS[col]
    return f"({g}) {name}"


# --------------------------------------------------------------------------- predictions


def toronto_categories(toronto: dict[str, Any]) -> dict[str, list]:
    static = pd.read_parquet(Path(toronto["paths"]["processed_dir"]) / STATIC)
    return {"class_combo": sorted(static["class_combo"].unique())}


def countfree_volumes(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """SPF covariates rebuilt from the count-free exposure (Toronto volume models only)."""
    expo = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / COUNTFREE_EXPOSURE)
    expo = expo.set_index("intersection_id")
    df = df.drop(columns=["entering_aadt", "major_aadt", "minor_aadt", "n_roads"],
                 errors="ignore").join(expo, on="intersection_id")  # fmt: skip
    has_minor = df["minor_aadt"].notna()
    df["log_major"] = np.where(has_minor, np.log(df["major_aadt"]), 0.0)
    df["log_minor"] = np.where(has_minor, np.log(df["minor_aadt"].fillna(1)), 0.0)
    df["log_entering_single"] = np.where(has_minor, 0.0, np.log(df["entering_aadt"]))
    df["single_road"] = (~has_minor).astype(float)
    return df


def spf_predictions(
    cfg: dict[str, Any], toronto: dict[str, Any], countfree: bool = False
) -> pd.DataFrame:
    """Toronto's SPF coefficients on Ottawa covariates. Raw level: the Toronto citywide level
    its own 2025 forecast used (2024's); EB uses Ottawa's yearly levels instead."""
    spf = json.loads((Path(toronto["paths"]["models_dir"]) / SPF_MODEL).read_text())
    df = spf_frame(cfg)
    if countfree:
        df = countfree_volumes(cfg, df)
    base = predict_base(spf["models"], df)
    t_pred = pd.read_parquet(Path(toronto["paths"]["processed_dir"]) / SPF_PREDICTIONS)
    t_level = float((t_pred["spf_known"] / t_pred["spf_base"])[t_pred["year"] == 2024].iloc[0])
    groups = df["group"]
    k = groups.map({g: m["k"] for g, m in spf["models"].items()}) if "pooled" not in spf[
        "models"] else pd.Series(spf["models"]["pooled"]["k"], index=df.index)  # fmt: skip
    return pd.DataFrame({"intersection_id": df["intersection_id"], "year": df["year"],
                         "spf_base": base, "k": k, "spf": base * t_level,
                         "is_excluded": df["is_excluded"],
                         "crash_count": df["crash_count"]})  # fmt: skip


def eb_predictions(cfg: dict[str, Any], spf: pd.DataFrame, toronto: dict[str, Any],
                   years: list[int]) -> pd.DataFrame:  # fmt: skip
    """EB with Toronto's SPF and window, Ottawa's history and yearly levels."""
    processed = Path(cfg["paths"]["processed_dir"])
    labels = pd.read_parquet(processed / LABELS)
    t_eb = pd.read_parquet(Path(toronto["paths"]["processed_dir"]) / EB_PREDICTIONS,
                           columns=["window"])  # fmt: skip
    window = int(t_eb["window"].iloc[0]) if len(t_eb) else EB_WINDOW_FALLBACK
    one = spf.drop_duplicates("intersection_id").set_index("intersection_id")
    out = eb_all(one["spf_base"], one["k"], labels, years, window)
    out["window"] = window
    out.to_parquet(processed / EB_PREDICTIONS, index=False)
    return out[["intersection_id", "year", "eb"]]


def gnn_predictions(cfg: dict[str, Any], toronto: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["paths"]["processed_dir"]) / GNN_FILE
    if not path.exists():
        run_gnn_transfer(toronto)
    g = pd.read_parquet(path)
    g["model"] = g["model"].replace({"sage_road": "road_sage", "mlp_road": "road_mlp"})
    return g.pivot_table(index=["intersection_id", "year"], columns="model",
                         values="pred").reset_index()  # fmt: skip


def run_gnn_transfer(toronto: dict[str, Any]) -> None:
    """GNN predictions in a separate process (torch and LightGBM must not share one)."""
    cmd = [sys.executable, "-m", "safepath.gnn.transfer"]
    if toronto.get("_config_path"):
        cmd += ["--config", str(toronto["_config_path"])]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GNN transfer exited with code {result.returncode}")


def with_entering(cfg: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    expo = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / INTERSECTION_EXPOSURE)
    return table.merge(expo[["intersection_id", "entering_aadt"]], on="intersection_id")


def build_rows(cfg: dict[str, Any], toronto: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Every Ottawa intersection-year in the model table with every model's prediction."""
    table = with_entering(cfg, load_table(cfg))
    t_models = Path(toronto["paths"]["models_dir"])
    cats = toronto_categories(toronto)
    info: dict[str, Any] = {
        "unseen_class_combos": sorted(set(table["class_combo"]) - set(cats["class_combo"])),
    }
    for col, name, fs in (("road_lgbm", "lgbm_static_only", "static_only"),
                          ("lgbm_aadt", "lgbm_aadt", "aadt"),
                          ("lgbm_aadt_nbr", "lgbm_aadt_nbr", "aadt_nbr")):  # fmt: skip
        booster = lgb.Booster(model_file=str(t_models / f"{name}.txt"))
        table[col] = lgbm.predict(booster, table, lgbm.FEATURE_SETS[fs], cats)

    spf = spf_predictions(cfg, toronto)
    years = sorted(table["year"].unique())
    eb = eb_predictions(cfg, spf, toronto, years)
    table = table.merge(spf[["intersection_id", "year", "spf"]], on=["intersection_id", "year"])
    cf = spf_predictions(cfg, toronto, countfree=True)[["intersection_id", "year", "spf"]]
    table = table.merge(cf.rename(columns={"spf": "spf_countfree"}),
                        on=["intersection_id", "year"])  # fmt: skip
    table = table.merge(eb, on=["intersection_id", "year"], how="left")
    table = table.merge(gnn_predictions(cfg, toronto), on=["intersection_id", "year"],
                        how="left")  # fmt: skip

    clean = table[~table["is_excluded"]]
    train = clean[clean["split"] == "train"]
    val = clean[clean["split"] == "val"]
    base = predict_baselines(table, fit_baselines(train))
    table["persistence"] = base["baseline_persistence"].to_numpy()
    table["prev3_mean"] = base["baseline_prev3_mean"].to_numpy()
    table["class_combo_mean"] = base["baseline_class_combo"].to_numpy()

    o_cats = {"class_combo": sorted(table["class_combo"].unique())}
    booster, fit = lgbm.fit(train, val, lgbm.FEATURE_SETS["aadt"], "crash_count", o_cats)
    table["upper_lgbm_aadt"] = lgbm.predict(booster, table, lgbm.FEATURE_SETS["aadt"], o_cats)
    info["upper_bound"] = {k: fit[k] for k in ("best_params", "best_iteration",
                                               "best_val_poisson")}  # fmt: skip
    missing = [c for c in MODELS if table.loc[~table["is_excluded"], c].isna().any()]
    if missing:
        raise ValueError(f"missing predictions for {missing}")
    return table, info


# --------------------------------------------------------------------------- evaluation


def prior_year(years: list[int], year: int) -> int:
    return max(y for y in years if y < year)


def level_scalars(clean: pd.DataFrame, year: int) -> dict[str, float]:
    """One scalar per model from the prior year: sum observed / sum predicted."""
    years = sorted(clean["year"].unique())
    p = clean[clean["year"] == prior_year(years, year)]
    return {m: float(p["crash_count"].sum() / p[m].sum()) for m in MODELS}


def per_vehicle_ce(d: pd.DataFrame, col: str) -> float:
    t = decile_table(d, col, "entering_aadt", n_boot=1)
    return float(np.abs(np.log(t["predicted / observed"])).mean())


def score_split(d: pd.DataFrame, scal: dict[str, float], seed: int) -> pd.DataFrame:
    y = d["crash_count"].to_numpy(float)
    preds = {m: d[m].to_numpy(float) for m in MODELS}
    boot = bootstrap(y, preds, "class_combo_mean", N_BOOT, seed)
    rows = []
    for m in MODELS:
        pm = point_metrics(y, preds[m])
        adj = point_metrics(y, preds[m] * scal[m])
        lo, hi = boot[m]["spearman_ci"]
        rlo, rhi = boot[m]["recall_top5_ci"]
        rows.append({
            "group": MODELS[m][0], "model": MODELS[m][1], "Ottawa data": DATA_USE[m],
            "Spearman [95% CI]": f"{pm['spearman']:.3f} [{lo:.3f}, {hi:.3f}]",
            "PR-AUC": pm["pr_auc"], "recall@1%": pm["recall_top1"],
            "recall@5% [95% CI]": f"{pm['recall_top5']:.3f} [{rlo:.3f}, {rhi:.3f}]",
            "recall@10%": pm["recall_top10"],
            "dev. raw": pm["poisson_deviance"], "dev. level-adj.": adj["poisson_deviance"],
            "pred / obs raw": pm["mean_pred"] / pm["mean_obs"], "level scalar": scal[m],
            "CE raw": per_vehicle_ce(d, m),
            "CE level-adj.": per_vehicle_ce(d.assign(**{m: d[m] * scal[m]}), m),
            "_col": m, "_spearman": pm["spearman"],
        })  # fmt: skip
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame, pairs: list[tuple[str, str]], seed: int) -> pd.DataFrame:
    """a - b with paired bootstrap 95% CIs (same intersection resamples for both models) for
    Spearman, recall@5% and PR-AUC: co-primary in a sparse city, where Spearman is dominated by
    ties among the ~83% zero-crash intersection-years."""
    y = d["crash_count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))
    metrics = {
        "spearman": lambda yy, pp: spearmanr(pp, yy).statistic,
        "recall5": lambda yy, pp: hotspot_recall(yy, pp, 0.05),
        "pr_auc": lambda yy, pp: average_precision_score(yy > 0, pp),
    }
    out = []
    for a, b in pairs:
        pa, pb = d[a].to_numpy(float), d[b].to_numpy(float)
        row: dict[str, Any] = {"comparison": f"{label(a)} − {label(b)}", "_a": a, "_b": b}
        for name, f in metrics.items():
            point = f(y, pa) - f(y, pb)
            diffs = np.array([f(y[i], pa[i]) - f(y[i], pb[i]) for i in idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row[{"spearman": "Spearman", "recall5": "recall@5%", "pr_auc": "PR-AUC"}[name]] = (
                f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]")  # fmt: skip
            row[f"_{name}"] = (float(point), (float(lo), float(hi)))
        row["_d"], row["_ci"] = row["_spearman"]
        out.append(row)
    return pd.DataFrame(out)


def best_of(val_table: pd.DataFrame, group: str, among: list[str] | None = None) -> str:
    """Best model of a group on VALIDATION Spearman (never chosen on test)."""
    g = val_table[val_table["group"] == group]
    if among is not None:
        g = g[g["_col"].isin(among)]
    return str(g.loc[g["_spearman"].idxmax(), "_col"])


def subset(d: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    s = d[mask]
    y = s["crash_count"].to_numpy(float)
    rows = []
    for m in MODELS:
        pm = point_metrics(y, s[m].to_numpy(float))
        rows.append({"model": label(m), "rows": len(s), "crashes": int(y.sum()),
                     "Spearman": pm["spearman"], "recall@5%": pm["recall_top5"]})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ceiling


def poisson_ceiling(lam: np.ndarray, seed: int, n_sim: int = N_SIM) -> tuple[float, float, float]:
    """Spearman a perfect model could reach if `lam` were the true expected counts and the
    observed year a Poisson draw from them: the noise ceiling at this sparsity."""
    rng = np.random.default_rng(seed)
    sims = [spearmanr(lam, rng.poisson(lam)).statistic for _ in range(n_sim)]
    return float(np.mean(sims)), *np.percentile(sims, [2.5, 97.5]).tolist()


def ceiling_table(
    ottawa: pd.DataFrame, t_rows: pd.DataFrame, toronto: dict[str, Any],
    test_table: pd.DataFrame, seed: int,
) -> pd.DataFrame:  # fmt: skip
    """How much of the Toronto -> Ottawa drop is sparsity, not transfer.

    1. Noise ceiling in each city: expected counts = the best-calibrated in-city model
       (Ottawa: LightGBM + AADT trained on Ottawa, level-adjusted to the observed total;
       Toronto: LightGBM + AADT on its own test year), observed = Poisson draws.
    2. Toronto at Ottawa's sparsity: thin Toronto's observed test crashes binomially to
       Ottawa's crashes per intersection-year and re-score Toronto's own LightGBM + AADT.
    """
    booster = lgb.Booster(model_file=str(Path(toronto["paths"]["models_dir"]) / "lgbm_aadt.txt"))
    t_pred = lgbm.predict(booster, t_rows, lgbm.FEATURE_SETS["aadt"], toronto_categories(toronto))
    t_pred = t_pred * t_rows["crash_count"].sum() / t_pred.sum()
    t_y = t_rows["crash_count"].to_numpy()
    o_pred = ottawa["upper_lgbm_aadt"].to_numpy(float)
    o_pred = o_pred * ottawa["crash_count"].sum() / o_pred.sum()
    o_y = ottawa["crash_count"].to_numpy()
    keep = t_y.mean() and ottawa["crash_count"].mean() / t_y.mean()
    rng = np.random.default_rng(seed)
    thinned = [spearmanr(t_pred, rng.binomial(t_y, keep)).statistic for _ in range(50)]
    rows = []
    for city, lam, y in (("Toronto 2025", t_pred, t_y), ("Ottawa 2024", o_pred, o_y)):
        mean, lo, hi = poisson_ceiling(lam, seed)
        rows.append({"setting": f"{city}: noise ceiling", "crashes per intersection-year":
                     y.mean(), "Spearman": mean, "95% range": f"[{lo:.3f}, {hi:.3f}]"})  # fmt: skip
    rows.append({"setting": "Toronto 2025: LightGBM + AADT, observed", "crashes per "
                 "intersection-year": t_y.mean(), "Spearman": spearmanr(t_pred, t_y).statistic,
                 "95% range": ""})  # fmt: skip
    rows.append({"setting": f"Toronto 2025: LightGBM + AADT, crashes thinned x{keep:.2f}",
                 "crashes per intersection-year": t_y.mean() * keep,
                 "Spearman": float(np.mean(thinned)),
                 "95% range": f"[{np.percentile(thinned, 2.5):.3f}, "
                              f"{np.percentile(thinned, 97.5):.3f}]"})  # fmt: skip
    sp = test_table.set_index("_col")["_spearman"]
    o_ceiling = rows[1]["Spearman"]
    for col in ("upper_lgbm_aadt", "eb", "lgbm_aadt", "prev3_mean"):
        rows.append({"setting": f"Ottawa 2024: {label(col)}, observed",
                     "crashes per intersection-year": o_y.mean(), "Spearman": sp[col],
                     "95% range": f"{100 * sp[col] / o_ceiling:.0f}% of the ceiling"})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- shift


def shift_table(toronto_rows: pd.DataFrame, ottawa_rows: pd.DataFrame) -> pd.DataFrame:
    """Key feature distributions, Toronto vs Ottawa (one row per clean intersection, test
    year), with the standardised mean difference (Ottawa - Toronto) / pooled sd."""
    rows = []
    for f in SHIFT_FEATURES:
        a = toronto_rows[f].astype(float)
        b = ottawa_rows[f].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"feature": f, "Toronto median": a.median(), "Ottawa median": b.median(),
                     "Toronto mean": a.mean(), "Ottawa mean": b.mean(),
                     "SMD": (b.mean() - a.mean()) / sd if sd > 0 else 0.0})  # fmt: skip
    return pd.DataFrame(rows)


def mix_table(toronto_rows: pd.DataFrame, ottawa_rows: pd.DataFrame) -> pd.DataFrame:
    t = toronto_rows["class_combo"].value_counts(normalize=True) * 100
    o = ottawa_rows["class_combo"].value_counts(normalize=True) * 100
    out = pd.DataFrame({"Toronto %": t, "Ottawa %": o}).fillna(0)
    out["difference"] = out["Ottawa %"] - out["Toronto %"]
    return out.sort_values("Toronto %", ascending=False).head(12).rename_axis(
        "class combo").reset_index()  # fmt: skip


def rate_table(toronto_rows: pd.DataFrame, ottawa_rows: pd.DataFrame) -> pd.DataFrame:
    """Crash levels: per intersection-year and per million entering vehicles, by signal."""
    rows = []
    for city, d in (("Toronto", toronto_rows), ("Ottawa", ottawa_rows)):
        for name, g in [("all", d), ("signalised", d[d["has_signal"]]),
                        ("unsignalised", d[~d["has_signal"]])]:  # fmt: skip
            mev = (g["entering_aadt"] * 365 / 1e6).sum()
            rows.append({"city": city, "intersections": name, "rows": len(g),
                         "crashes per intersection-year": g["crash_count"].mean(),
                         "% zero-crash": 100 * (g["crash_count"] == 0).mean(),
                         "crashes per MEV": g["crash_count"].sum() / mev})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step


def run_ottawa_transfer(cfg: dict[str, Any], toronto: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    table, info = build_rows(cfg, toronto)
    table.to_parquet(processed / TRANSFER_ROWS, index=False)
    clean = table[~table["is_excluded"]]
    r: dict[str, Any] = {"info": info}
    for i, split in enumerate(("val", "test")):
        year = int(cfg["splits"][split][0])
        d = clean[clean["year"] == year].reset_index(drop=True)
        scal = level_scalars(clean, year)
        r[split] = {"year": year, "prior": prior_year(sorted(clean["year"].unique()), year),
                    "rows": len(d), "crashes": int(d["crash_count"].sum()),
                    "table": score_split(d, scal, 51 + i)}  # fmt: skip
    best_a, best_b = best_of(r["val"]["table"], "A"), best_of(r["val"]["table"], "B")
    best_cf = best_of(r["val"]["table"], "A", COUNT_FREE_A)
    pairs = [(best_a, "class_combo_mean"), (best_cf, "class_combo_mean"),
             ("road_sage", "road_lgbm"), ("road_sage", "road_mlp"), ("road_lgbm", "road_mlp"),
             ("eb", "persistence"), (best_b, "persistence")]  # fmt: skip
    pairs.insert(2, ("spf_countfree", "class_combo_mean"))
    pairs += [(m, "upper_lgbm_aadt") for m in MODELS if MODELS[m][0] == "B"]
    pairs += [(best_b, "prev3_mean"), ("upper_lgbm_aadt", "prev3_mean")]
    pairs = list(dict.fromkeys(pairs))  # the best models may coincide with fixed pairs
    for i, split in enumerate(("val", "test")):
        d = clean[clean["year"] == r[split]["year"]].reset_index(drop=True)
        r[split]["paired"] = paired(d, pairs, 61 + i)
    test = clean[clean["year"] == r["test"]["year"]]
    r["zero"] = subset(test, test["prev3_crashes_mean"] == 0)
    r["best"] = {"A": best_a, "A_countfree": best_cf, "B": best_b}

    # shift: Toronto clean test-2025 rows vs Ottawa clean test-2024 rows
    t_table = with_entering(toronto, load_table(toronto))
    t_rows = t_table[~t_table["is_excluded"] & (t_table["split"] == "test")]
    r["shift"] = shift_table(t_rows, test)
    r["mix"] = mix_table(t_rows, test)
    r["rates"] = rate_table(t_rows, test)
    r["ceiling"] = ceiling_table(test, t_rows, toronto, r["test"]["table"], 71)
    r["deciles"] = {m: decile_table(test.reset_index(drop=True), m, "entering_aadt", n_boot=1)
                    for m in (best_b, "eb", "upper_lgbm_aadt")}  # fmt: skip

    save = {k: (v.drop(columns=[c for c in v if c.startswith("_")]).to_dict(orient="records")
                if isinstance(v, pd.DataFrame) else v) for k, v in r.items()
            if k in ("shift", "mix", "rates", "zero", "ceiling")}  # fmt: skip
    for split in ("val", "test"):
        save[split] = {k: (v.drop(columns=[c for c in v if c.startswith("_")]).to_dict(
            orient="records") if isinstance(v, pd.DataFrame) else v)
            for k, v in r[split].items()}  # fmt: skip
    save["best"], save["info"] = r["best"], info
    (processed / TRANSFER_JSON).write_text(json.dumps(save, indent=2, default=float))
    from safepath.ottawa.report import write_ottawa_report

    write_ottawa_report(cfg, toronto, r)
    with tracking.run(cfg, "ottawa-transfer", "ottawa", {
        "val_year": r["val"]["year"], "test_year": r["test"]["year"], "n_boot": N_BOOT,
        "best_A": best_a, "best_B": best_b, "upper_bound": info["upper_bound"]}) as wb:  # fmt: skip
        for split in ("val", "test"):
            t = r[split]["table"]
            wb.table(f"{split}_metrics", t.drop(columns=[c for c in t if c.startswith("_")]))
            p = r[split]["paired"]
            wb.table(f"{split}_paired", p.drop(columns=[c for c in p if c.startswith("_")]))
            wb.summary({f"{split}/spearman/{c}": s for c, s in zip(t["_col"], t["_spearman"],
                                                                     strict=True)})  # fmt: skip
        wb.table("shift", r["shift"])
        wb.table("rates", r["rates"])
    log.info("ottawa transfer: best A %s, best B %s", best_a, best_b)
    return r
