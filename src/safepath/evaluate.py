"""Evaluate baselines and LightGBM models on val (2024) and test (2025).

Headline rows use clean (non-excluded) intersections only; one extra row scores the main
model on all intersections for transparency. Ranking metrics come first because the routing
use case needs a good ordering of intersections, not calibrated counts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_poisson_deviance, roc_auc_score

from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.locations import markdown_table
from safepath.models import lgbm
from safepath.models.baselines import predict_baselines
from safepath.models.eb import EB_PREDICTIONS
from safepath.models.train import MODELS, SPATIAL_OOF, categories, load_table

log = logging.getLogger(__name__)

HOTSPOT_FRACS = (0.01, 0.05, 0.10)
DEVIANCE_FLOOR = 0.05  # crashes/year; baselines predict exact zeros, which deviance cannot score
N_BOOT = 1000
REFERENCE = "baseline_persistence"
ORANGE, AQUA = "#eb6834", "#1baf7a"

CRASH_MODELS = [
    "baseline_prev3_mean", "baseline_persistence", "baseline_class_combo", "lgbm_main",
    "lgbm_static_only", "lgbm_static_only_spatial", "lgbm_with_coords",
]  # fmt: skip
SEVERE_MODELS = [
    "severe_baseline_prev3_mean", "severe_baseline_persistence",
    "severe_baseline_class_combo", "lgbm_severe",
]  # fmt: skip


def hotspot_recall(y: np.ndarray, pred: np.ndarray, frac: float) -> float:
    """Share of all crashes that fall in the top `frac` of intersections ranked by `pred`.

    Ties at the cut-off get fractional credit (expected recall under random tie-breaking), so
    baselines with many equal predictions are not helped or hurt by sort order.
    """
    k = int(round(frac * len(y)))
    if k == 0 or y.sum() == 0:
        return float("nan")
    order = np.argsort(-pred, kind="stable")
    t = pred[order[k - 1]]
    above = pred > t
    tied = pred == t
    need = k - above.sum()
    captured = y[above].sum() + y[tied].sum() * need / tied.sum()
    return float(captured / y.sum())


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    any_ = y > 0
    out = {
        "spearman": float(spearmanr(pred, y).statistic),
        "pr_auc": float(average_precision_score(any_, pred)),
        "roc_auc": float(roc_auc_score(any_, pred)),
    }
    for f in HOTSPOT_FRACS:
        out[f"recall_top{int(f * 100)}"] = hotspot_recall(y, pred, f)
    out["poisson_deviance"] = float(mean_poisson_deviance(y, np.maximum(pred, DEVIANCE_FLOOR)))
    out["mean_pred"] = float(pred.mean())
    out["mean_obs"] = float(y.mean())
    return out


def bootstrap(
    y: np.ndarray, preds: dict[str, np.ndarray], reference: str, n_boot: int, seed: int
) -> dict[str, dict[str, list[float]]]:
    """Percentile 95% CIs over resampled intersections for Spearman and top-5% recall, plus
    paired CIs for each model's difference from `reference` (same resamples for all)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {m: {"spearman": [], "recall_top5": []} for m in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for m, p in preds.items():
            pb = p[idx]
            stats[m]["spearman"].append(spearmanr(pb, yb).statistic)
            stats[m]["recall_top5"].append(hotspot_recall(yb, pb, 0.05))
    out: dict[str, dict[str, list[float]]] = {}
    for m, s in stats.items():
        out[m] = {}
        for metric, values in s.items():
            v = np.asarray(values)
            out[m][f"{metric}_ci"] = np.percentile(v, [2.5, 97.5]).tolist()
            if reference in stats and m != reference:
                d = v - np.asarray(stats[reference][metric])
                out[m][f"{metric}_diff_vs_ref"] = float(np.mean(d))
                out[m][f"{metric}_diff_vs_ref_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
    return out


def predictions(cfg: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    """Every model's prediction for every val/test row (excluded intersections included)."""
    models_dir = Path(cfg["paths"]["models_dir"])
    rows = table[table["split"].isin(["val", "test"])].copy()
    cats = categories(table)
    preds = predict_baselines(rows, pd.read_pickle(models_dir / "baselines_crash_count.pkl"))
    severe = predict_baselines(rows, pd.read_pickle(models_dir / "baselines_severe_count.pkl"))
    severe["baseline_prev3_mean"] = rows["prev3_severe_mean"].to_numpy()
    preds = preds.join(severe.add_prefix("severe_"))
    for name, (feature_set, _) in MODELS.items():
        booster = lgb.Booster(model_file=str(models_dir / f"{name}.txt"))
        preds[name] = lgbm.predict(booster, rows, lgbm.FEATURE_SETS[feature_set], cats)
    oof = pd.read_parquet(models_dir / SPATIAL_OOF)
    rows = rows.join(preds)
    eb = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / EB_PREDICTIONS)
    rows = rows.merge(eb[["intersection_id", "year", "spf", "eb"]], on=["intersection_id", "year"])
    rows = rows.merge(
        oof[["intersection_id", "year", "pred"]].rename(
            columns={"pred": "lgbm_static_only_spatial"}
        ),
        on=["intersection_id", "year"],
        how="left",
    )
    return rows


def evaluate_split(rows: pd.DataFrame, seed: int) -> dict[str, Any]:
    clean = rows[~rows["is_excluded"]]
    result: dict[str, Any] = {"n_intersections": len(clean), "crash": {}, "severe": {}}
    y = clean["crash_count"].to_numpy(float)
    preds = {m: clean[m].to_numpy(float) for m in CRASH_MODELS}
    if np.isnan(preds["lgbm_static_only_spatial"]).any():
        raise ValueError("spatial out-of-fold predictions are missing for some rows")
    boot = bootstrap(y, preds, REFERENCE, N_BOOT, seed)
    for m, p in preds.items():
        result["crash"][m] = {**point_metrics(y, p), **boot[m]}
    ys = clean["severe_count"].to_numpy(float)
    spreds = {m: clean[m].to_numpy(float) for m in SEVERE_MODELS}
    sboot = bootstrap(ys, spreds, "severe_baseline_persistence", N_BOOT, seed)
    for m, p in spreds.items():
        result["severe"][m] = {**point_metrics(ys, p), **sboot[m]}
    result["crash"]["lgbm_main (all intersections)"] = point_metrics(
        rows["crash_count"].to_numpy(float), rows["lgbm_main"].to_numpy(float)
    )
    return result


def _fmt_ci(m: dict[str, Any], key: str) -> str:
    lo, hi = m[f"{key}_ci"]
    return f"{m[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def metrics_table(block: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, m in block.items():
        has_ci = "spearman_ci" in m
        rows.append(
            {
                "model": model,
                "Spearman [95% CI]": _fmt_ci(m, "spearman") if has_ci else f"{m['spearman']:.3f}",
                "PR-AUC": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                "recall@1%": m["recall_top1"],
                "recall@5% [95% CI]": _fmt_ci(m, "recall_top5")
                if has_ci
                else f"{m['recall_top5']:.3f}",
                "recall@10%": m["recall_top10"],
                "Poisson dev.": m["poisson_deviance"],
                "mean pred / obs": f"{m['mean_pred']:.2f} / {m['mean_obs']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def diff_table(block: dict[str, dict[str, Any]], reference: str) -> pd.DataFrame:
    rows = []
    for model, m in block.items():
        if "spearman_diff_vs_ref" not in m:
            continue
        s_lo, s_hi = m["spearman_diff_vs_ref_ci"]
        r_lo, r_hi = m["recall_top5_diff_vs_ref_ci"]
        rows.append(
            {
                "model": model,
                f"Spearman - {reference}": f"{m['spearman_diff_vs_ref']:+.3f} "
                f"[{s_lo:+.3f}, {s_hi:+.3f}]",
                f"recall@5% - {reference}": f"{m['recall_top5_diff_vs_ref']:+.3f} "
                f"[{r_lo:+.3f}, {r_hi:+.3f}]",
            }
        )
    return pd.DataFrame(rows)


def plot_importance(booster: lgb.Booster, path: Path, top: int = 25) -> pd.Series:
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(
        ascending=False
    )
    share = gain / gain.sum()
    shown = share.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    ax.barh(shown.index, shown.to_numpy(), color=BAR, height=0.7)
    ax.set_title(
        "LightGBM (main) feature importance: share of total gain",
        loc="left",
        fontsize=11,
        color=TEXT,
    )
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return share


def plot_calibration(rows: pd.DataFrame, path: Path) -> None:
    clean = rows[~rows["is_excluded"]]
    series = [
        ("lgbm_main", BAR),
        ("baseline_persistence", ORANGE),
        ("lgbm_static_only_spatial", AQUA),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE, layout="constrained")
    for ax, split in zip(axes, ["val", "test"], strict=True):
        d = clean[clean["split"] == split]
        ax.set_facecolor(SURFACE)
        top = 0
        for model, color in series:
            dec = pd.qcut(d[model].rank(method="first"), 10, labels=False)
            g = d.groupby(dec).agg(pred=(model, "mean"), obs=("crash_count", "mean"))
            ax.plot(
                g["pred"], g["obs"], marker="o", markersize=5, linewidth=2, color=color, label=model
            )
            top = max(top, g["pred"].max(), g["obs"].max())
        ax.plot(
            [0.01, top * 1.2],
            [0.01, top * 1.2],
            color=TEXT_MUTED,
            linewidth=1,
            linestyle="--",
            label="perfect calibration",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        year = int(d["year"].iloc[0])
        ax.set_title(
            f"{split} {year}: mean observed vs predicted crashes, by prediction decile",
            loc="left",
            fontsize=10,
            color=TEXT,
        )
        ax.set_xlabel("mean predicted crashes", fontsize=9, color=TEXT_MUTED)
        ax.set_ylabel("mean observed crashes", fontsize=9, color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, labelcolor=TEXT)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run_evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    table = load_table(cfg)
    rows = predictions(cfg, table)
    results = {
        split: evaluate_split(rows[rows["split"] == split], seed=11 + i)
        for i, split in enumerate(["val", "test"])
    }
    reports, figures = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["figures_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    booster = lgb.Booster(model_file=str(models_dir / "lgbm_main.txt"))
    share = plot_importance(booster, figures / "feature_importance.png")
    plot_calibration(rows, figures / "calibration.png")
    results["feature_importance_gain_share"] = share.round(5).to_dict()
    results["training"] = json.loads((models_dir / "training_summary.json").read_text())
    (reports / "metrics.json").write_text(json.dumps(results, indent=2))
    write_metrics_md(results, rows, reports / "metrics.md")
    run_comparison(cfg, rows)
    log.info("wrote metrics for val and test")
    return results


def _verdict(m: dict[str, Any], metric: str, label: str) -> str:
    diff, (lo, hi) = m[f"{metric}_diff_vs_ref"], m[f"{metric}_diff_vs_ref_ci"]
    if lo > 0:
        word = "better than"
    elif hi < 0:
        word = "worse than"
    else:
        word = "not distinguishable from"
    return f"{label} {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]: {word}"


def write_metrics_md(results: dict[str, Any], rows: pd.DataFrame, path: Path) -> None:
    training = results["training"]
    n_excluded = rows.loc[rows["is_excluded"], "intersection_id"].nunique()
    lines = [
        "# Model metrics",
        "",
        f"Unit: consolidated intersection x year. Train label years 2016-2019 + 2022-2023 "
        f"({training['train_rows']:,} clean rows), validation 2024, test 2025. Headline rows "
        f"exclude {n_excluded} proxy intersections (`reports/excluded_intersections.csv`). "
        f"Val is used for early stopping and the small hyper-parameter grid, so val numbers "
        f"are slightly optimistic; test 2025 is untouched. CIs: {N_BOOT} bootstrap resamples "
        "of intersections; differences are paired (same resamples).",
        "",
        "Models: `baseline_prev3_mean` = mean of the 3 prior years; `baseline_persistence` = "
        "the intersection's mean over all train label years; `baseline_class_combo` = mean "
        "per approach-class combo (no history); `lgbm_main` = static + history; "
        "`lgbm_static_only` = static features only, same intersections as train; "
        "`lgbm_static_only_spatial` = static only, scored on 2 km spatial blocks the model "
        "never saw (5-fold); `lgbm_with_coords` = main + projected x/y (ablation).",
        "",
        f"Poisson deviance clips predictions at {DEVIANCE_FLOOR} crashes/year so baselines "
        "that predict exact zeros can be scored; it is a secondary metric here.",
        "",
    ]
    for split in ("val", "test"):
        r = results[split]
        year = 2024 if split == "val" else 2025
        lines += [
            f"## {split.capitalize()} {year} ({r['n_intersections']:,} clean intersections)",
            "",
            "### Any-crash count",
            "",
            markdown_table(metrics_table(r["crash"]), floats="{:.3f}"),
            "",
            f"Paired differences vs `{REFERENCE}`:",
            "",
            markdown_table(diff_table(r["crash"], REFERENCE)),
            "",
            "### Severe (injury or fatal) count",
            "",
            markdown_table(metrics_table(r["severe"]), floats="{:.3f}"),
            "",
            markdown_table(diff_table(r["severe"], "severe_baseline_persistence")),
            "",
        ]
    t = results["test"]["crash"]
    main, ref = t["lgbm_main"], t[REFERENCE]
    lines += [
        "## Reading (test 2025)",
        "",
        "- `lgbm_main` vs persistence: "
        + _verdict(main, "spearman", "Spearman")
        + "; "
        + _verdict(main, "recall_top5", "recall@5%")
        + f". Absolute recall@5%: {main['recall_top5']:.3f} vs {ref['recall_top5']:.3f}.",
        "- Severe model vs severe persistence: "
        + _verdict(results["test"]["severe"]["lgbm_severe"], "spearman", "Spearman")
        + ".",
        f"- Static only, seen intersections: Spearman {t['lgbm_static_only']['spearman']:.3f}; "
        f"unseen spatial blocks: {t['lgbm_static_only_spatial']['spearman']:.3f}. The gap is "
        "memorisation: continuous static features (distances, counts) act as an intersection "
        "fingerprint under a temporal split, so only the spatial number measures transfer to "
        "intersections with no crash record.",
        f"- Adding coordinates: Spearman {t['lgbm_with_coords']['spearman']:.3f} vs "
        f"{main['spearman']:.3f} without.",
        "",
        "## Top 15 features (gain share, `lgbm_main`)",
        "",
        "| feature | gain share |",
        "|---|---:|",
        *[
            f"| {k} | {v:.1%} |"
            for k, v in list(results["feature_importance_gain_share"].items())[:15]
        ],
        "",
        "![Feature importance](figures/feature_importance.png)",
        "",
        "![Calibration](figures/calibration.png)",
        "",
    ]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- CP6 comparison

COMPARE_MODELS = {
    "baseline_persistence": "persistence (train-period mean)",
    "baseline_prev3_mean": "prev-3 mean",
    "lgbm_main": "LightGBM (CP3)",
    "lgbm_aadt": "LightGBM + log AADT",
    "spf": "SPF alone",
    "eb": "Empirical Bayes",
}
HOTSPOT_MIN_AADT = 5_000  # entering vehicles per day


def calibration_slope(y: np.ndarray, pred: np.ndarray) -> float:
    """Slope of a Poisson regression of y on log(pred): 1 = calibrated, < 1 = predictions
    too spread out, > 1 = too compressed."""
    import statsmodels.api as sm

    X = sm.add_constant(np.log(np.maximum(pred, DEVIANCE_FLOOR)))
    return float(sm.GLM(y, X, family=sm.families.Poisson()).fit().params[1])


def compare_split(rows: pd.DataFrame, seed: int) -> dict[str, Any]:
    clean = rows[~rows["is_excluded"]]
    y = clean["crash_count"].to_numpy(float)
    preds = {m: clean[m].to_numpy(float) for m in COMPARE_MODELS}
    vs_persistence = bootstrap(y, preds, "baseline_persistence", N_BOOT, seed)
    vs_aadt = bootstrap(y, preds, "lgbm_aadt", N_BOOT, seed)  # same resamples (same seed)
    out: dict[str, Any] = {"n": len(clean), "models": {}}
    for m, p in preds.items():
        out["models"][m] = {
            **point_metrics(y, p),
            **vs_persistence[m],
            "calibration_slope": calibration_slope(y, p),
        }
        if m != "lgbm_aadt":
            out["models"][m]["vs_lgbm_aadt"] = {
                k: v for k, v in vs_aadt[m].items() if "diff_vs_ref" in k
            }
    return out


def comparison_table(block: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for m, label in COMPARE_MODELS.items():
        r = block["models"][m]
        rows.append(
            {
                "model": label,
                "Spearman [95% CI]": _fmt_ci(r, "spearman"),
                "PR-AUC": r["pr_auc"],
                "recall@1%": r["recall_top1"],
                "recall@5% [95% CI]": _fmt_ci(r, "recall_top5"),
                "Poisson dev.": r["poisson_deviance"],
                "mean pred / obs": r["mean_pred"] / r["mean_obs"],
                "calibration slope": r["calibration_slope"],
            }
        )
    return pd.DataFrame(rows)


def paired_table(block: dict[str, Any]) -> pd.DataFrame:
    def fmt(d: dict[str, Any], metric: str) -> str:
        lo, hi = d[f"{metric}_diff_vs_ref_ci"]
        return f"{d[f'{metric}_diff_vs_ref']:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    eb = block["models"]["eb"]
    aadt = block["models"]["lgbm_aadt"]
    rows = [
        {"comparison": "EB - persistence", "Spearman": fmt(eb, "spearman"),
         "recall@5%": fmt(eb, "recall_top5")},
        {"comparison": "EB - LightGBM + AADT", "Spearman": fmt(eb["vs_lgbm_aadt"], "spearman"),
         "recall@5%": fmt(eb["vs_lgbm_aadt"], "recall_top5")},
        {"comparison": "LightGBM + AADT - persistence", "Spearman": fmt(aadt, "spearman"),
         "recall@5%": fmt(aadt, "recall_top5")},
        {"comparison": "LightGBM (CP3) - LightGBM + AADT",
         "Spearman": fmt(block["models"]["lgbm_main"]["vs_lgbm_aadt"], "spearman"),
         "recall@5%": fmt(block["models"]["lgbm_main"]["vs_lgbm_aadt"], "recall_top5")},
    ]  # fmt: skip
    return pd.DataFrame(rows)


def rate_hotspots(cfg: dict[str, Any]) -> pd.DataFrame:
    """Top 20 clean intersections by 2026 EB expected crashes per million entering
    vehicles, among those with at least HOTSPOT_MIN_AADT entering vehicles a day."""
    from safepath.features.labels import LABELS
    from safepath.graph import INTERSECTIONS
    from safepath.models.volume import AADT_INTERSECTIONS

    processed = Path(cfg["paths"]["processed_dir"])
    eb = pd.read_parquet(processed / EB_PREDICTIONS)
    eb = eb[eb["year"] == 2026].set_index("intersection_id")
    aadt = pd.read_parquet(processed / AADT_INTERSECTIONS).set_index("intersection_id")
    labels = pd.read_parquet(processed / LABELS)
    excluded = labels.groupby("intersection_id")["is_excluded"].any()
    recent = (
        labels[labels["year"].between(2021, 2025)].groupby("intersection_id")["crash_count"].mean()
    )
    names = pd.read_parquet(processed / INTERSECTIONS).set_index("intersection_id")["street_names"]
    df = pd.DataFrame(
        {
            "intersection": names.reindex(eb.index).replace("", "(unnamed)"),
            "class combo": aadt["class_combo"].reindex(eb.index),
            "EB crashes/yr (2026)": eb["eb"],
            "observed/yr 2021-25": recent.reindex(eb.index),
            "entering AADT": aadt["aadt"].reindex(eb.index),
            "AADT source": aadt["source"].reindex(eb.index),
        }
    )
    df["EB per MEV"] = df["EB crashes/yr (2026)"] / (df["entering AADT"] * 365 / 1e6)
    df["rank by EB count"] = df["EB crashes/yr (2026)"].rank(ascending=False).astype(int)
    df = df[~excluded.reindex(df.index).fillna(False) & (df["entering AADT"] >= HOTSPOT_MIN_AADT)]
    return df.sort_values("EB per MEV", ascending=False).head(20)


def plot_comparison_calibration(rows: pd.DataFrame, path: Path) -> None:
    clean = rows[~rows["is_excluded"] & (rows["split"] == "test")]
    series = [("baseline_persistence", ORANGE), ("lgbm_aadt", BAR), ("spf", TEXT_MUTED),
              ("eb", AQUA)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    top = 0.0
    for model, color in series:
        dec = pd.qcut(clean[model].rank(method="first"), 10, labels=False)
        g = clean.groupby(dec).agg(pred=(model, "mean"), obs=("crash_count", "mean"))
        ax.plot(g["pred"], g["obs"], marker="o", markersize=5, linewidth=2, color=color,
                label=COMPARE_MODELS[model])  # fmt: skip
        top = max(top, g["pred"].max(), g["obs"].max())
    ax.plot([0.01, top * 1.2], [0.01, top * 1.2], color=GRID, linewidth=1, linestyle="--",
            label="perfect calibration")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Test 2025: mean observed vs predicted crashes by decile", loc="left",
                 fontsize=10, color=TEXT)  # fmt: skip
    ax.set_xlabel("mean predicted crashes", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("mean observed crashes", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run_comparison(cfg: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    results = {
        split: compare_split(rows[rows["split"] == split], seed=21 + i)
        for i, split in enumerate(["val", "test"])
    }
    reports, figures = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["figures_dir"])
    plot_comparison_calibration(rows, figures / "calibration_cp6.png")
    hot = rate_hotspots(cfg)
    (reports / "model_comparison.json").write_text(json.dumps(results, indent=2))
    write_comparison_md(results, hot, reports / "model_comparison.md")
    return results


def comparison_verdict(t: dict[str, Any]) -> list[str]:
    m = t["models"]
    best_rank = max(COMPARE_MODELS, key=lambda k: m[k]["spearman"])
    best_recall = max(COMPARE_MODELS, key=lambda k: m[k]["recall_top5"])
    best_dev = min(COMPARE_MODELS, key=lambda k: m[k]["poisson_deviance"])
    best_slope = min(COMPARE_MODELS, key=lambda k: abs(m[k]["calibration_slope"] - 1))
    best_level = min(COMPARE_MODELS, key=lambda k: abs(m[k]["mean_pred"] / m[k]["mean_obs"] - 1))
    eb = m["eb"]
    # the bootstrap stored CP3 LightGBM minus LightGBM + AADT; flip it to read as a gain
    d = m["lgbm_main"]["vs_lgbm_aadt"]
    flipped = {
        "spearman_diff_vs_ref": -d["spearman_diff_vs_ref"],
        "spearman_diff_vs_ref_ci": [-x for x in d["spearman_diff_vs_ref_ci"][::-1]],
    }
    return [
        f"- Ranking (Spearman): best is {COMPARE_MODELS[best_rank]} "
        f"({m[best_rank]['spearman']:.3f}); top-5% recall: best is "
        f"{COMPARE_MODELS[best_recall]} ({m[best_recall]['recall_top5']:.3f}).",
        f"- Calibration: lowest Poisson deviance is {COMPARE_MODELS[best_dev]} "
        f"({m[best_dev]['poisson_deviance']:.3f}); calibration slope closest to 1 is "
        f"{COMPARE_MODELS[best_slope]} ({m[best_slope]['calibration_slope']:.3f}); total "
        f"predicted / observed closest to 1 is {COMPARE_MODELS[best_level]} "
        f"({m[best_level]['mean_pred'] / m[best_level]['mean_obs']:.3f}, LightGBM + AADT "
        f"{m['lgbm_aadt']['mean_pred'] / m['lgbm_aadt']['mean_obs']:.3f}).",
        "- EB vs persistence: "
        + _verdict(eb, "spearman", "Spearman")
        + "; "
        + _verdict(eb, "recall_top5", "recall@5%")
        + ".",
        "- EB vs LightGBM + AADT: "
        + _verdict(eb["vs_lgbm_aadt"], "spearman", "Spearman")
        + "; "
        + _verdict(eb["vs_lgbm_aadt"], "recall_top5", "recall@5%")
        + ".",
        "- LightGBM + AADT vs CP3 LightGBM: " + _verdict(flipped, "spearman", "Spearman") + ".",
    ]


def write_comparison_md(results: dict[str, Any], hot: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Model comparison (CP6)",
        "",
        "Clean intersections, same metrics and bootstrap as reports/metrics.md "
        f"({N_BOOT} resamples of intersections; paired differences use the same resamples). "
        "`SPF alone` is the grouped NB2 SPF with the previous year's citywide level; "
        "`Empirical Bayes` blends it with each intersection's previous 5 years "
        "(reports/spf.md, reports/eb.md). `LightGBM + log AADT` is the CP3 model plus "
        "log entering, major-road and minor-road AADT. Calibration slope: Poisson "
        "regression of observed on log(predicted); 1 is ideal, below 1 means the "
        "predictions are too spread out.",
        "",
    ]
    for split, year in (("val", 2024), ("test", 2025)):
        r = results[split]
        lines += [
            f"## {split.capitalize()} {year} ({r['n']:,} clean intersections)",
            "",
            markdown_table(comparison_table(r), floats="{:.3f}"),
            "",
            "Paired differences:",
            "",
            markdown_table(paired_table(r)),
            "",
        ]
    lines += [
        "![Calibration by decile, test 2025](figures/calibration_cp6.png)",
        "",
        "## Reading (test 2025)",
        "",
        *comparison_verdict(results["test"]),
        "",
        "## Rate hotspots: EB expected crashes per million entering vehicles (2026)",
        "",
        f"Clean intersections with at least {HOTSPOT_MIN_AADT:,} entering vehicles a day. "
        "`rank by EB count` is the intersection's rank among all intersections by EB "
        "expected crashes per year.",
        "",
        markdown_table(hot, floats="{:,.2f}"),
    ]
    path.write_text("\n".join(lines) + "\n")
