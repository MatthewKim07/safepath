"""GNN comparison for CP8: reports/gnn.md (torch-free; reads the GNN step's predictions).

Same rows, metrics and bootstrap as reports/metrics.md: clean intersections, validation 2024
and test 2025. Models: persistence, LightGBM + AADT, LightGBM + AADT + neighbour aggregates,
recalibrated EB (the routing model), MLP (no graph), GraphSAGE, GATv2, and an ensemble of the
best graph model with LightGBM + AADT (weighted mean of log predictions, weight chosen on
validation deviance).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safepath import tracking
from safepath.evaluate import N_BOOT, bootstrap, point_metrics, predictions
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.locations import markdown_table
from safepath.models.calibration import RECALIBRATION, decile_table
from safepath.models.eb import EB_PREDICTIONS
from safepath.models.train import SPATIAL_OOF, load_table

log = logging.getLogger(__name__)

GNN_PREDICTIONS = "gnn_predictions.parquet"  # written by safepath.gnn.train
GNN_SPATIAL_OOF = "gnn_spatial_oof.parquet"
GNN_SUMMARY = "gnn_summary.json"
LABELS = {
    "baseline_persistence": "persistence",
    "lgbm_aadt": "LightGBM + AADT",
    "lgbm_aadt_nbr": "LightGBM + AADT + neighbours",
    "eb_recal": "EB, recalibrated (routing)",
    "gnn_mlp": "MLP (no graph)",
    "gnn_sage": "GraphSAGE",
    "gnn_gatv2": "GATv2",
    "ensemble": "ensemble (best GNN + LightGBM + AADT)",
}
ROAD_LABELS = {
    "road_lgbm": "LightGBM, road features (CP3)",
    "road_mlp": "MLP, road features",
    "road_sage": "GraphSAGE, road features",
}
GRAPH_MODELS = ("gnn_sage", "gnn_gatv2")
CE = "per-vehicle CE"
NAMES = {**LABELS, **ROAD_LABELS}
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def per_vehicle_ce(d: pd.DataFrame, col: str) -> float:
    t = decile_table(d, col, "entering_aadt", n_boot=1)
    return float(np.abs(np.log(t["predicted / observed"])).mean())


def comparison_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    """Every model's prediction for clean val/test rows."""
    processed = Path(cfg["paths"]["processed_dir"])
    table = load_table(cfg)
    rows = predictions(cfg, table)
    rows = rows[~rows["is_excluded"]].copy()
    eb = pd.read_parquet(processed / EB_PREDICTIONS, columns=["intersection_id", "year",
                                                                "obs_history"])  # fmt: skip
    expo = pd.read_parquet(processed / INTERSECTION_EXPOSURE)[["intersection_id", "entering_aadt"]]
    rows = rows.merge(eb, on=["intersection_id", "year"]).merge(expo, on="intersection_id")
    with open(Path(cfg["paths"]["models_dir"]) / RECALIBRATION, "rb") as f:
        rows["eb_recal"] = pickle.load(f).apply(rows)
    gnn = pd.read_parquet(processed / GNN_PREDICTIONS)
    wide = gnn.pivot_table(index=["intersection_id", "year"], columns="model", values="pred")
    wide.columns = [f"gnn_{c}" for c in wide.columns]
    return rows.merge(wide.reset_index(), on=["intersection_id", "year"])


def choose_ensemble(rows: pd.DataFrame) -> tuple[str, float, pd.DataFrame]:
    """Best graph model on validation deviance, and the log-space weight on it (vs LightGBM +
    AADT) that minimises validation deviance."""
    val = rows[rows["split"] == "val"]
    devs = {m: point_metrics(val["crash_count"].to_numpy(float), val[m].to_numpy(float))
            ["poisson_deviance"] for m in GRAPH_MODELS}  # fmt: skip
    best = min(devs, key=devs.get)
    curve = []
    for w in WEIGHTS:
        p = np.exp(w * np.log(val[best]) + (1 - w) * np.log(val["lgbm_aadt"]))
        curve.append({"weight on GNN": w, "val deviance": point_metrics(
            val["crash_count"].to_numpy(float), p.to_numpy())["poisson_deviance"]})  # fmt: skip
    curve = pd.DataFrame(curve)
    w = float(curve.loc[curve["val deviance"].idxmin(), "weight on GNN"])
    return best, w, curve


def score(d: pd.DataFrame, models: list[str], reference: str, seed: int) -> pd.DataFrame:
    y = d["crash_count"].to_numpy(float)
    preds = {m: d[m].to_numpy(float) for m in models}
    boot = bootstrap(y, preds, reference, N_BOOT, seed)
    rows = []
    for m in models:
        pm = point_metrics(y, preds[m])
        lo, hi = boot[m]["spearman_ci"]
        rlo, rhi = boot[m]["recall_top5_ci"]
        rows.append({
            "model": {**LABELS, **ROAD_LABELS}.get(m, m),
            "Spearman [95% CI]": f"{pm['spearman']:.3f} [{lo:.3f}, {hi:.3f}]",
            "PR-AUC": pm["pr_auc"],
            "recall@1%": pm["recall_top1"],
            "recall@5% [95% CI]": f"{pm['recall_top5']:.3f} [{rlo:.3f}, {rhi:.3f}]",
            "recall@10%": pm["recall_top10"],
            "Poisson dev.": pm["poisson_deviance"],
            "pred / obs": pm["mean_pred"] / pm["mean_obs"],
            CE: per_vehicle_ce(d, m),
            "_spearman": pm["spearman"],
        })  # fmt: skip
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame, pairs: list[tuple[str, str]], seed: int) -> pd.DataFrame:
    """a - b with paired bootstrap CIs (same resamples), for Spearman and recall@5%."""
    y = d["crash_count"].to_numpy(float)
    out = []
    for a, b in pairs:
        boot = bootstrap(y, {a: d[a].to_numpy(float), b: d[b].to_numpy(float)}, b, N_BOOT, seed)
        m = boot[a]
        s_lo, s_hi = m["spearman_diff_vs_ref_ci"]
        r_lo, r_hi = m["recall_top5_diff_vs_ref_ci"]
        out.append({
            "comparison": f"{NAMES[a]} − {NAMES[b]}",
            "Spearman": f"{m['spearman_diff_vs_ref']:+.3f} [{s_lo:+.3f}, {s_hi:+.3f}]",
            "recall@5%": f"{m['recall_top5_diff_vs_ref']:+.3f} [{r_lo:+.3f}, {r_hi:+.3f}]",
            "_spearman_ci": (s_lo, s_hi),
        })  # fmt: skip
    return pd.DataFrame(out)


def subset_table(d: pd.DataFrame, models: list[str], mask: pd.Series) -> pd.DataFrame:
    s = d[mask]
    y = s["crash_count"].to_numpy(float)
    rows = []
    for m in models:
        pm = point_metrics(y, s[m].to_numpy(float))
        rows.append({"model": LABELS.get(m, m), "rows": len(s), "crashes": int(y.sum()),
                     "Spearman": pm["spearman"], "recall@5%": pm["recall_top5"],
                     "Poisson dev.": pm["poisson_deviance"],
                     "pred / obs": pm["mean_pred"] / pm["mean_obs"]})  # fmt: skip
    return pd.DataFrame(rows)


def spatial_table(
    cfg: dict[str, Any], rows: pd.DataFrame, split: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Road-only models on held-out 2 km blocks: CP3's LightGBM vs the GNN step's MLP and
    GraphSAGE (same folds)."""
    processed, models_dir = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["models_dir"])
    lgb = pd.read_parquet(models_dir / SPATIAL_OOF).rename(columns={"pred": "road_lgbm"})
    gnn = pd.read_parquet(processed / GNN_SPATIAL_OOF)
    gnn = gnn.pivot_table(index=["intersection_id", "year"], columns="model", values="pred")
    gnn.columns = [f"road_{c}" for c in gnn.columns]
    d = rows[rows["split"] == split][["intersection_id", "year", "crash_count", "entering_aadt"]]
    d = d.merge(lgb[["intersection_id", "year", "road_lgbm"]], on=["intersection_id", "year"])
    d = d.merge(gnn.reset_index(), on=["intersection_id", "year"])
    t = score(d, list(ROAD_LABELS), "road_lgbm", seed)
    p = paired(d, [("road_sage", "road_mlp"), ("road_sage", "road_lgbm")], seed)
    return t, p


def seed_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for kind, s in summary["models"].items():
        seeds = pd.DataFrame(s["seeds"])
        rows.append({
            "model": LABELS[f"gnn_{kind}"],
            "val deviance, mean ± sd": f"{seeds['val_deviance'].mean():.4f} ± {seeds['val_deviance'].std():.4f}",  # noqa: E501
            "test Spearman, mean ± sd": f"{seeds['test_spearman'].mean():.4f} ± {seeds['test_spearman'].std():.4f}",  # noqa: E501
            "test deviance, mean ± sd": f"{seeds['test_deviance'].mean():.4f} ± {seeds['test_deviance'].std():.4f}",  # noqa: E501
            "5-seed average: test Spearman": s["ensemble_of_seeds"]["test_spearman"],
            "best epochs": ", ".join(str(int(e)) for e in seeds["best_epoch"]),
        })  # fmt: skip
    return pd.DataFrame(rows)


def run_gnn_evaluation(cfg: dict[str, Any]) -> dict[str, Any] | None:
    processed, models_dir = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["models_dir"])
    if not (processed / GNN_PREDICTIONS).exists():
        log.info("no GNN predictions (run `python -m safepath gnn`); skipping reports/gnn.md")
        return None
    summary = json.loads((models_dir / GNN_SUMMARY).read_text())
    rows = comparison_rows(cfg)
    best, w, curve = choose_ensemble(rows)
    rows["ensemble"] = np.exp(w * np.log(rows[best]) + (1 - w) * np.log(rows["lgbm_aadt"]))
    models = list(LABELS)
    r: dict[str, Any] = {"best_gnn": best, "weight": w, "curve": curve, "summary": summary}
    for i, split in enumerate(("val", "test")):
        d = rows[rows["split"] == split].reset_index(drop=True)
        r[f"{split}_table"] = score(d, models, "baseline_persistence", 31 + i)
        r[f"{split}_paired"] = paired(d, [(best, "gnn_mlp"), (best, "lgbm_aadt"),
                                          (best, "lgbm_aadt_nbr"), ("ensemble", "lgbm_aadt")],
                                      31 + i)  # fmt: skip
        zero = d["prev3_crashes_mean"] == 0
        low = d["entering_aadt"] <= d["entering_aadt"].quantile(0.3)
        r[f"{split}_zero"] = subset_table(d, models, zero)
        r[f"{split}_low"] = subset_table(d, models, low)
        r[f"{split}_spatial"], r[f"{split}_spatial_paired"] = spatial_table(cfg, rows, split,
                                                                             41 + i)  # fmt: skip
    write_report(cfg, r, Path(cfg["paths"]["reports_dir"]) / "gnn.md")
    with tracking.run(cfg, "gnn-evaluation", "evaluate", {"best_gnn": best,
                      "ensemble_weight": w, "n_boot": N_BOOT}) as wb:  # fmt: skip
        for key in ("val_table", "test_table", "test_paired", "test_zero", "test_low",
                    "test_spatial", "test_spatial_paired"):  # fmt: skip
            wb.table(key, r[key].drop(columns=[c for c in r[key] if c.startswith("_")]))
    return r


def conclusions(r: dict[str, Any]) -> list[str]:
    t = r["test_table"].set_index("model")
    p = r["test_paired"].set_index("comparison")
    best = LABELS[r["best_gnn"]]

    def verdict(row: str) -> str:
        lo, hi = p.loc[row, "_spearman_ci"]
        word = "better" if lo > 0 else "worse" if hi < 0 else "no different"
        return f"{p.loc[row, 'Spearman']} ({word})"

    sp = r["test_spatial_paired"].set_index("comparison")
    st = r["test_spatial"].set_index("model")
    v = r["val_table"].set_index("model")
    eb_ce = v.loc[LABELS["eb_recal"], CE]
    beats_eb = [m for m in v.index if m != LABELS["eb_recal"] and v.loc[m, CE] < eb_ce]
    return [
        f"- **Does message passing help with crash history in the model?** {best} minus the "
        f"MLP on the same features, test Spearman: {verdict(f'{best} − MLP (no graph)')}.",
        f"- **Against the trees:** {best} minus LightGBM + AADT: "
        f"{verdict(f'{best} − LightGBM + AADT')}; minus LightGBM with hand-built neighbour "
        f"features: {verdict(f'{best} − LightGBM + AADT + neighbours')}. Ensemble minus "
        f"LightGBM + AADT: {verdict('ensemble (best GNN + LightGBM + AADT) − LightGBM + AADT')} "
        f"(weight on the GNN {r['weight']:.2f}, chosen on validation).",
        f"- **Road features only, unseen blocks:** GraphSAGE "
        f"{st.loc['GraphSAGE, road features', '_spearman']:.3f}, MLP "
        f"{st.loc['MLP, road features', '_spearman']:.3f}, LightGBM "
        f"{st.loc['LightGBM, road features (CP3)', '_spearman']:.3f} test Spearman. GraphSAGE "
        f"minus MLP: {sp.iloc[0]['Spearman']}; minus LightGBM: {sp.iloc[1]['Spearman']}.",
        "- **Per-vehicle calibration** (routing's criterion, mean |log pred/obs| over AADT "
        f"deciles): validation, recalibrated EB {eb_ce:.4f}; "
        + (
            "lower: " + ", ".join(f"{m} {v.loc[m, CE]:.4f}" for m in beats_eb) + ". "
            if beats_eb
            else "no other model is lower. "
        )
        + "Test: recalibrated EB "
        f"{t.loc[LABELS['eb_recal'], CE]:.4f}, "
        + ", ".join(f"{m} {t.loc[m, CE]:.4f}" for m in beats_eb)
        + ". Routing stays on recalibrated EB unless you decide otherwise (CP8 decision).",
        f"- Test Spearman: {best} {t.loc[best, '_spearman']:.3f}, LightGBM + AADT "
        f"{t.loc['LightGBM + AADT', '_spearman']:.3f}, persistence "
        f"{t.loc['persistence', '_spearman']:.3f}.",
    ]


def write_report(cfg: dict[str, Any], r: dict[str, Any], path: Path) -> None:
    s = r["summary"]
    g = s["graph"]

    def table(df: pd.DataFrame, floats: str = "{:.3f}") -> str:
        return markdown_table(df.drop(columns=[c for c in df if c.startswith("_")]), floats)

    configs = pd.DataFrame([{"model": LABELS[f"gnn_{k}"], **c} for k, c in s["configs"].items()])
    runs = [x for m in s["models"].values() for x in m["seeds"]]
    n_runs, latest = len(runs), max(x["best_epoch"] for x in runs)
    capped = sum(len(x["history"]) >= s.get("max_epochs", 150) for x in runs)
    lines = [
        "# Graph neural networks on the intersection graph (CP8)",
        "",
        "## Graph",
        "",
        f"- Nodes: {g['nodes']:,} consolidated intersections; undirected edges: "
        f"{g['undirected_edges']:,} (road segments joining two different intersections, "
        f"one per pair); {g['node_features']} node features (the LightGBM + AADT set, class "
        f"combo one-hot, standardised on clean train years) and {g['edge_features']} edge "
        "features (class rank, length, lanes, speed limit, log AADT, motorway flag).",
        f"- One snapshot per label year ({', '.join(str(y) for y in g['years'])}); same "
        "topology, year-specific history features and labels. Proxy intersections are never "
        "in the loss or the headline numbers.",
        "",
        "## Training",
        "",
        "Poisson NLL on log expected crashes; full batch, one step per train-year snapshot per "
        "epoch; early stopping on validation-2024 deviance (patience "
        f"{s.get('patience', 15)}; at most {s.get('max_epochs', 150)} epochs for the final "
        f"seeds and spatial folds, {s.get('grid_max_epochs', 150)} in the grid search). "
        f"Grid on validation only: hidden {{64, 128}} x layers {{2, 3}} x lr {{1e-3, 3e-3}} x "
        f"dropout {{0.1, 0.3}} per model. Device: {s['device']}. Final configs, 5 seeds each; "
        "a model's prediction is the mean of its 5 seeds' expected counts.",
        "",
        table(configs, "{:g}"),
        "",
        f"Final runs stopped by the epoch cap: {capped} of {n_runs} (latest best epoch "
        f"{latest}). CP8 used a 150-epoch cap, which cut off one GraphSAGE seed (best epoch "
        "140 of 150); with the cap at 400 (CP9) every run stops early, the 5-seed test "
        "Spearman moves by at most 0.0001 and no conclusion changes. GATv2 seeds moved a "
        "little without touching the old cap: MPS training is not bit-for-bit reproducible. "
        "The grid search was not re-run; its configs come from the 150-epoch grid.",
        "",
        "Seed variance:",
        "",
        table(seed_table(s), "{:.4f}"),
        "",
        "![Training curves](figures/gnn_training_curves.png)",
        "",
        "Timing, final GraphSAGE config, one seed, full training to early stop:",
        "",
        table(pd.DataFrame(s["timing"]), "{:.3f}"),
        "",
    ]
    for split, year in (("val", 2024), ("test", 2025)):
        lines += [
            f"## {split.capitalize()} {year}: all intersections",
            "",
            table(r[f"{split}_table"]),
            "",
            "Paired differences (same bootstrap resamples):",
            "",
            table(r[f"{split}_paired"]),
            "",
        ]
    lines += [
        "## Test 2025 subsets",
        "",
        "Zero-history intersections (no crash in the prior 3 years):",
        "",
        table(r["test_zero"]),
        "",
        "Low-volume intersections (bottom 3 entering-AADT deciles):",
        "",
        table(r["test_low"]),
        "",
        "## Road features only, unseen 2 km blocks (test 2025)",
        "",
        "CP3's 5 spatial folds; each fold's labels never enter the loss or early stopping "
        "(fold k + 1 early-stops fold k). Message passing still sees held-out nodes' road "
        "features: transductive on features, inductive on labels. No history, no AADT.",
        "",
        table(r["test_spatial"]),
        "",
        table(r["test_spatial_paired"]),
        "",
        "## Ensemble weight (validation)",
        "",
        f"Best graph model on validation: {LABELS[r['best_gnn']]}; log-space weight on it "
        f"{r['weight']:.2f}.",
        "",
        "## Conclusions",
        "",
        *conclusions(r),
    ]
    path.write_text("\n".join(lines) + "\n")
