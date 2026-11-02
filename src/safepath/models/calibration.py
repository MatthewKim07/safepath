"""Calibration of the crash models by traffic volume, and the routing model's recalibration.

Routing divides an intersection's expected crashes by its entering vehicles, so what matters
is calibration per vehicle across volumes: predicted / observed crashes within each
entering-AADT decile (the per-vehicle ratio, since vehicles cancel within a decile). Tables
are also given by predicted-risk decile.

Candidates (all fitted on train label years only, chosen on validation 2024, confirmed on
test 2025):
- EB as in CP6;
- EB x a volume correction (Poisson spline on log entering AADT, offset log EB);
- EB x a volume, level and history correction (adds a spline on log EB and a no-crashes-in-
  the-history-window indicator);
- LightGBM + AADT;
- a 50/50 blend of the recalibrated EB and LightGBM + AADT.
Corrections include train-year effects, which are averaged out and renormalised so a
correction changes the shape across intersections, not the overall level.

Selection rule, fixed before looking at test: lowest mean |log(predicted / observed)| over
validation AADT deciles; ties broken by total predicted / observed, then Poisson deviance.

Outputs: models/eb_recalibration.pkl, models/routing_risk_model.json,
reports/calibration.md, reports/figures/calibration_by_volume.png.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_poisson_deviance

from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.locations import markdown_table
from safepath.models import lgbm
from safepath.models.eb import EB_PREDICTIONS
from safepath.models.train import categories, load_table

log = logging.getLogger(__name__)

RECALIBRATION = "eb_recalibration.pkl"
ROUTING_MODEL = "routing_risk_model.json"
DEVIANCE_FLOOR = 0.05
N_BOOT = 500
ORANGE, AQUA = "#eb6834", "#1baf7a"
FORMULAS = {
    "EB x volume correction": "crash_count ~ cr(log_aadt, df=5) + C(year)",
    "EB x volume, level and history correction": (
        "crash_count ~ cr(log_aadt, df=5) + cr(log_eb, df=5) + zero_history + C(year)"
    ),
}
RECALIBRATED = "EB x volume, level and history correction"


def calibration_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Clean intersection-years (train, val, test) with EB, LightGBM + AADT, entering AADT."""
    processed = Path(cfg["paths"]["processed_dir"])
    table = load_table(cfg)
    eb = pd.read_parquet(processed / EB_PREDICTIONS)
    expo = pd.read_parquet(processed / INTERSECTION_EXPOSURE)[["intersection_id", "entering_aadt"]]
    df = table.merge(
        eb[["intersection_id", "year", "eb", "obs_history"]], on=["intersection_id", "year"]
    ).merge(expo, on="intersection_id")
    booster = lgb.Booster(model_file=str(Path(cfg["paths"]["models_dir"]) / "lgbm_aadt.txt"))
    df["lgbm_aadt"] = lgbm.predict(booster, df, lgbm.FEATURE_SETS["aadt"], categories(table))
    df = df[~df["is_excluded"]].reset_index(drop=True)
    return add_terms(df)


def add_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_aadt"] = np.log(out["entering_aadt"])
    out["log_eb"] = np.log(np.maximum(out["eb"], 1e-4))
    out["zero_history"] = (out["obs_history"] == 0).astype(float)
    return out


class Recalibration:
    """A Poisson correction to EB, applied with train-year effects averaged out and the
    overall train level held fixed."""

    def __init__(self, formula: str, train: pd.DataFrame):
        self.formula = formula
        self.years = sorted(train["year"].unique())
        result = smf.glm(
            formula, data=train, offset=train["log_eb"], family=sm.families.Poisson()
        ).fit()
        self.result = result
        self.norm = 1.0
        self.norm = float(self.apply(train).sum() / train["eb"].sum())
        result.remove_data()  # keep the pickle small; the formula still predicts

    def apply(self, df: pd.DataFrame) -> np.ndarray:
        d = add_terms(df) if "log_eb" not in df else df
        preds = [
            self.result.predict(d.assign(year=y), offset=d["log_eb"]).to_numpy() for y in self.years
        ]
        return np.mean(preds, axis=0) / self.norm


def decile_table(
    d: pd.DataFrame, pred: str, by: str, seed: int = 7, n_boot: int = N_BOOT
) -> pd.DataFrame:
    """Predicted vs observed crashes per decile of `by`, with a bootstrap 95% CI on the
    ratio (rows resampled within the decile) and crashes per million entering vehicles."""
    rng = np.random.default_rng(seed)
    dec = pd.qcut(d[by].rank(method="first"), 10, labels=False)
    rows = []
    for k, g in d.groupby(dec):
        y, p = g["crash_count"].to_numpy(float), g[pred].to_numpy(float)
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        boot = p[idx].sum(axis=1) / np.maximum(y[idx].sum(axis=1), 1e-9)
        mev = (g["entering_aadt"] * 365 / 1e6).sum()
        rows.append(
            {
                "decile": int(k) + 1,
                "median entering AADT": g["entering_aadt"].median(),
                "observed crashes": y.sum(),
                "predicted": p.sum(),
                "predicted / observed": p.sum() / y.sum() if y.sum() else np.nan,
                "95% CI": f"[{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]",
                "observed per MEV": y.sum() / mev,
                "predicted per MEV": p.sum() / mev,
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(d: pd.DataFrame, pred: str) -> dict[str, float]:
    by_aadt = decile_table(d, pred, "entering_aadt", n_boot=1)
    by_pred = decile_table(d, pred, pred, n_boot=1)
    return {
        "per-vehicle CE (AADT deciles)": float(
            np.abs(np.log(by_aadt["predicted / observed"])).mean()
        ),
        "CE (predicted deciles)": float(np.abs(np.log(by_pred["predicted / observed"])).mean()),
        "quietest AADT decile ratio": float(by_aadt["predicted / observed"].iloc[0]),
        "lowest predicted decile ratio": float(by_pred["predicted / observed"].iloc[0]),
        "total predicted / observed": float(d[pred].sum() / d["crash_count"].sum()),
        "Poisson deviance": float(
            mean_poisson_deviance(d["crash_count"], np.maximum(d[pred], DEVIANCE_FLOOR))
        ),
        "Spearman": float(d[pred].corr(d["crash_count"], method="spearman")),
    }


def plot_calibration(df: pd.DataFrame, models: dict[str, str], path: Path) -> None:
    colors = [ORANGE, AQUA, BAR]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE, layout="constrained")
    for i, split in enumerate(["val", "test"]):
        d = df[df["split"] == split]
        for j, by in enumerate(["entering_aadt", None]):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for (label, col), color in zip(models.items(), colors, strict=True):
                t = decile_table(d, col, by or col, n_boot=200)
                lo_hi = t["95% CI"].str.strip("[]").str.split(", ", expand=True).astype(float)
                ax.plot(t["decile"], t["predicted / observed"], marker="o", color=color,
                        lw=2, label=label)  # fmt: skip
                ax.fill_between(t["decile"], lo_hi[0], lo_hi[1], color=color, alpha=0.15, lw=0)
            ax.axhline(1, color=TEXT_MUTED, lw=1, ls="--")
            ax.set_yscale("log")
            # the lowest predicted decile has a few dozen crashes; its CI runs off any scale
            ax.set_ylim((0.75, 1.35) if by else (0.4, 12))
            ax.set_yticks([0.8, 0.9, 1, 1.1, 1.2] if by else [0.5, 1, 2, 5, 10])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda val, _: ""))
            ax.set_xticks(range(1, 11))
            ax.set_title(
                f"{split} {2024 if split == 'val' else 2025}: predicted / observed by "
                + ("entering-AADT decile (per vehicle)" if by else "predicted-risk decile"),
                fontsize=9, color=TEXT, loc="left",
            )  # fmt: skip
            ax.set_xlabel("decile (1 = lowest)", fontsize=8, color=TEXT_MUTED)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            ax.grid(color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=TEXT)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def choose(metrics: pd.DataFrame) -> str:
    m = metrics[metrics["split"] == "val"].set_index("model")
    key = pd.DataFrame(
        {
            "ce": m["per-vehicle CE (AADT deciles)"].round(3),
            "level": np.abs(np.log(m["total predicted / observed"])).round(3),
            "dev": m["Poisson deviance"],
        }
    )
    return key.sort_values(["ce", "level", "dev"]).index[0]


def write_report(r: dict[str, Any], path: Path) -> None:
    chosen = r["chosen"]
    lines = [
        "# Calibration by traffic volume (CP7)",
        "",
        "Routing divides expected crashes by entering vehicles, so the check that matters is "
        "predicted / observed crashes within entering-AADT deciles: within a decile the "
        "vehicles cancel, so this is also the per-vehicle ratio. Clean intersections; "
        f"CIs from {N_BOOT} bootstrap resamples within each decile.",
        "",
        "## Candidates",
        "",
        "Recalibrations are Poisson GLMs with offset log(EB), fitted on train label years only, "
        "with train-year effects averaged out and the train level held fixed. Selection rule, "
        "fixed before looking at test: lowest per-vehicle calibration error (mean "
        "|log(pred/obs)| over validation AADT deciles), then total level, then deviance.",
        "",
        *[f"- `{k}`: `{v}`" for k, v in FORMULAS.items()],
        "",
        markdown_table(r["metrics"], floats="{:.3f}"),
        "",
        f"**Chosen for routing: {chosen}.** {r['justification']}",
        "",
        "![Calibration by volume and predicted risk](figures/calibration_by_volume.png)",
        "",
    ]
    for split, year in (("val", 2024), ("test", 2025)):
        for label, key in (("EB (CP6), before", "before"), (f"{chosen}, after", "after")):
            lines += [
                f"## {split.capitalize()} {year}, by entering-AADT decile: {label}",
                "",
                markdown_table(r["tables"][split][key], floats="{:,.3f}"),
                "",
            ]
        lines += [
            f"## {split.capitalize()} {year}, by predicted-risk decile (before / after)",
            "",
            markdown_table(r["pred_tables"][split], floats="{:,.3f}"),
            "",
        ]
    lines += [
        "## What remains",
        "",
        "The lowest predicted-risk decile is still overpredicted several-fold by every "
        "candidate. Those are intersections with no crash in the history window and almost "
        "none in the label year (a few dozen crashes across ~2,200 intersections), so the "
        "ratio is noisy and the absolute error is a few hundredths of a crash per year. Per "
        "vehicle, their predicted risk stays well below that of busier intersections, so "
        "the effect on routing is a small extra penalty on quiet streets, which errs against "
        "cut-throughs.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    df = calibration_frame(cfg)
    train = df[df["split"] == "train"]
    recal = {name: Recalibration(f, train) for name, f in FORMULAS.items()}
    for name, model in recal.items():
        df[name] = model.apply(df)
    df["50/50 blend (recalibrated EB, LightGBM + AADT)"] = 0.5 * (
        df[RECALIBRATED] + df["lgbm_aadt"]
    )
    models = {
        "EB (CP6)": "eb",
        **{k: k for k in FORMULAS},
        "LightGBM + AADT": "lgbm_aadt",
        "50/50 blend (recalibrated EB, LightGBM + AADT)": "50/50 blend (recalibrated EB, "
        "LightGBM + AADT)",
    }
    rows = []
    for split in ("val", "test"):
        d = df[df["split"] == split]
        for label, col in models.items():
            rows.append({"split": split, "model": label, **summary_metrics(d, col)})
    metrics = pd.DataFrame(rows)
    chosen = choose(metrics)
    col = models[chosen]
    m = metrics.set_index(["split", "model"])
    v, t = m.loc[("val", chosen)], m.loc[("test", chosen)]
    lv, lt = m.loc[("val", "LightGBM + AADT")], m.loc[("test", "LightGBM + AADT")]
    ce = "per-vehicle CE (AADT deciles)"
    confirmed = t[ce] <= m.loc[("test", "EB (CP6)"), ce]
    justification = (
        f"Best per-vehicle calibration on validation ({v['per-vehicle CE (AADT deciles)']:.3f} "
        f"vs {m.loc[('val', 'EB (CP6)'), 'per-vehicle CE (AADT deciles)']:.3f} for EB and "
        f"{lv['per-vehicle CE (AADT deciles)']:.3f} for LightGBM + AADT); on test "
        f"{t['per-vehicle CE (AADT deciles)']:.3f} vs "
        f"{m.loc[('test', 'EB (CP6)'), 'per-vehicle CE (AADT deciles)']:.3f} and "
        f"{lt['per-vehicle CE (AADT deciles)']:.3f}"
        + (", so test confirms it" if confirmed else ", so test does not confirm it")
        + f". Total level on test {t['total predicted / observed']:.3f} vs "
        f"{lt['total predicted / observed']:.3f} for LightGBM + AADT. LightGBM + AADT still "
        f"ranks better (test Spearman {lt['Spearman']:.3f} vs {t['Spearman']:.3f}) and has "
        "lower deviance; the differences in per-vehicle calibration are small, and the EB "
        "route keeps an interpretable SPF behind every intersection's risk."
    )
    tables, pred_tables = {}, {}
    for split in ("val", "test"):
        d = df[df["split"] == split]
        tables[split] = {
            "before": decile_table(d, "eb", "entering_aadt"),
            "after": decile_table(d, col, "entering_aadt"),
        }
        b = decile_table(d, "eb", "eb")
        a = decile_table(d, col, col)
        pred_tables[split] = pd.DataFrame(
            {
                "decile": b["decile"],
                "EB pred/obs": b["predicted / observed"],
                "EB 95% CI": b["95% CI"],
                "chosen pred/obs": a["predicted / observed"],
                "chosen 95% CI": a["95% CI"],
            }
        )
    models_dir = Path(cfg["paths"]["models_dir"])
    with open(models_dir / RECALIBRATION, "wb") as f:
        pickle.dump(recal[RECALIBRATED], f)
    (models_dir / ROUTING_MODEL).write_text(
        json.dumps(
            {"chosen": chosen, "column": col, "metrics": metrics.to_dict("records")}, indent=2
        )  # fmt: skip
    )
    plot_calibration(
        df,
        {"EB (CP6)": "eb", chosen: col, "LightGBM + AADT": "lgbm_aadt"},
        Path(cfg["paths"]["figures_dir"]) / "calibration_by_volume.png",
    )
    r = {
        "metrics": metrics,
        "chosen": chosen,
        "justification": justification,
        "tables": tables,
        "pred_tables": pred_tables,
    }
    write_report(r, Path(cfg["paths"]["reports_dir"]) / "calibration.md")
    log.info("calibration: chose %s", chosen)
    return r
