"""Safety Performance Function: negative binomial (NB2) crashes per intersection-year.

    log mu = b0 + b1 log(major AADT) + b2 log(minor AADT) + static terms + year effect

Intersections with a single road (no minor approach) use log(entering AADT) and an
indicator instead of the major/minor pair. Fitted on clean intersections in the train label
years (2016-2019, 2022-2023) with statsmodels' NB2 (Var = mu + k mu^2). A pooled SPF is
compared with separate SPFs for signalised 4+ leg, signalised 3-leg and unsignalised
intersections by AIC and by Poisson deviance on validation 2024.

Year effects only exist for train years. For any year y the SPF level is set from the
citywide total instead: `spf_known` = base x (observed clean crashes in y / sum of base),
usable once y is over, and `spf_forecast` for year y uses the level of y - 1, so a forecast
never sees its own year. Outputs: data/processed/spf_predictions.parquet,
models/spf.json, reports/spf.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_poisson_deviance
from statsmodels.discrete.discrete_model import NegativeBinomial

from safepath.features.build import STATIC
from safepath.features.labels import LABELS
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.locations import markdown_table

log = logging.getLogger(__name__)

SPF_PREDICTIONS = "spf_predictions.parquet"
SPF_MODEL = "spf.json"
POOLED = "pooled"
GROUPS = ["signalised 4+ leg", "signalised 3-leg", "unsignalised"]
VOLUME_TERMS = ["log_major", "log_minor", "log_entering_single", "single_road"]
STATIC_TERMS = ["has_signal", "streetcar_20m", "min_leg_angle_10deg"]
CATEGORICAL_TERMS = ["legs", "class_combo"]
DEVIANCE_FLOOR = 0.05


def spf_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Every intersection-year 2014-2025 with crashes, exclusion flag, and SPF covariates."""
    processed = Path(cfg["paths"]["processed_dir"])
    labels = pd.read_parquet(processed / LABELS)
    static = pd.read_parquet(processed / STATIC)
    expo = pd.read_parquet(processed / INTERSECTION_EXPOSURE).set_index("intersection_id")
    df = labels.join(
        static[["n_legs", "has_signal", "streetcar_20m", "min_leg_angle_deg", "class_combo"]],
        on="intersection_id",
    ).join(expo, on="intersection_id")
    has_minor = df["minor_aadt"].notna()
    df["log_major"] = np.where(has_minor, np.log(df["major_aadt"]), 0.0)
    df["log_minor"] = np.where(has_minor, np.log(df["minor_aadt"].fillna(1)), 0.0)
    df["log_entering_single"] = np.where(has_minor, 0.0, np.log(df["entering_aadt"]))
    df["single_road"] = (~has_minor).astype(float)
    df["has_signal"] = df["has_signal"].astype(float)
    df["streetcar_20m"] = df["streetcar_20m"].astype(float)
    df["min_leg_angle_10deg"] = df["min_leg_angle_deg"] / 10
    # the four motorway combos are rare and some have no crashes at all within a group,
    # which leaves their coefficient unidentified; one "highway x any" level instead
    df["class_combo"] = df["class_combo"].where(
        ~df["class_combo"].str.startswith("highway"), "highway x any"
    )
    df["legs"] = pd.cut(df["n_legs"], [-1, 3, 4, 99], labels=["3-", "4", "5+"]).astype(str)
    df["group"] = np.select(
        [df["has_signal"].eq(1) & df["n_legs"].ge(4), df["has_signal"].eq(1)],
        GROUPS[:2],
        GROUPS[2],
    )
    return df


def design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Model matrix. At fit time (`columns` None) year dummies come from the rows given; at
    prediction time they are all zero, which gives the reference-year (2016) base, and the
    year level comes from `with_levels` instead."""
    X = df[VOLUME_TERMS + STATIC_TERMS].copy()
    cats = pd.get_dummies(df[CATEGORICAL_TERMS], drop_first=False, dtype=float)
    years = pd.get_dummies(df["year"].astype(str), prefix="year", dtype=float)
    X = pd.concat([X, cats, years], axis=1)
    X.insert(0, "const", 1.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
        X[[c for c in columns if c.startswith("year_")]] = 0.0
    return X


def reduce_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Drop constant columns and one reference level per categorical block (the most common
    one, so coefficients read as contrasts against the typical intersection; the first
    year for year effects). Returns the kept columns and the reference levels."""
    keep = [c for c in X.columns if c == "const" or X[c].std() > 0]
    out, refs = list(keep), []
    for prefix in ("legs_", "class_combo_", "year_"):
        block = [c for c in keep if c.startswith(prefix)]
        if block:
            ref = max(block, key=lambda c: X[c].sum()) if prefix != "year_" else min(block)
            out.remove(ref)
            refs.append(ref)
    return out, refs


def fit_nb(y: pd.Series, X: pd.DataFrame) -> Any:
    """NB2 by Newton from Poisson starting values (BFGS from the default start diverges)."""
    start = sm.GLM(y, X, family=sm.families.Poisson()).fit().params.to_numpy()
    res = NegativeBinomial(y, X, loglike_method="nb2").fit(
        start_params=np.r_[start, 0.5], method="newton", maxiter=200, disp=0
    )
    if not res.mle_retvals.get("converged", False):
        raise RuntimeError("NB2 SPF did not converge")
    return res


def fit_models(train: pd.DataFrame, grouped: bool) -> dict[str, dict[str, Any]]:
    parts = train.groupby("group") if grouped else [(POOLED, train)]
    out = {}
    for name, d in parts:
        X = design(d)
        cols, refs = reduce_columns(X)
        res = fit_nb(d["crash_count"], X[cols])
        ci = res.conf_int()
        out[name] = {
            "columns": cols,
            "params": res.params.drop("alpha").to_dict(),
            "ci_low": ci[0].drop("alpha").to_dict(),
            "ci_high": ci[1].drop("alpha").to_dict(),
            "k": float(res.params["alpha"]),
            "k_ci": [float(ci.loc["alpha", 0]), float(ci.loc["alpha", 1])],
            "reference": refs,
            "aic": float(res.aic),
            "n": int(len(d)),
        }
    return out


def predict_base(models: dict[str, dict[str, Any]], df: pd.DataFrame) -> pd.Series:
    """mu at the reference year for every row."""
    out = pd.Series(np.nan, index=df.index)
    for name, m in models.items():
        rows = df.index if name == POOLED else df.index[df["group"] == name]
        if len(rows) == 0:
            continue
        X = design(df.loc[rows], m["columns"])
        beta = pd.Series(m["params"]).reindex(m["columns"]).fillna(0.0)
        out.loc[rows] = np.exp(X.to_numpy() @ beta.to_numpy())
    return out


def k_of(models: dict[str, dict[str, Any]], df: pd.DataFrame) -> pd.Series:
    if POOLED in models:
        return pd.Series(models[POOLED]["k"], index=df.index)
    return df["group"].map({g: m["k"] for g, m in models.items()})


def year_levels(df: pd.DataFrame, base: pd.Series) -> pd.Series:
    """Citywide level per year: observed clean crashes / sum of base predictions."""
    clean = ~df["is_excluded"]
    obs = df.loc[clean].groupby("year")["crash_count"].sum()
    pred = base[clean].groupby(df.loc[clean, "year"]).sum()
    return obs / pred


def with_levels(df: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    levels = year_levels(df, base)
    out = df[["intersection_id", "year", "group"]].copy()
    out["spf_base"] = base
    out["spf_known"] = base * df["year"].map(levels)
    out["spf_forecast"] = base * df["year"].map(lambda y: levels.get(y - 1, np.nan))
    return out


def val_deviance(df: pd.DataFrame, pred: pd.DataFrame, year: int) -> float:
    m = (df["year"] == year) & ~df["is_excluded"]
    return float(
        mean_poisson_deviance(
            df.loc[m, "crash_count"], np.maximum(pred.loc[m, "spf_forecast"], DEVIANCE_FLOOR)
        )
    )


def coefficient_table(models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        for term in m["columns"]:
            if term.startswith("year_"):
                continue
            rows.append(
                {
                    "SPF": name,
                    "term": term,
                    "coef": m["params"][term],
                    "95% CI": f"[{m['ci_low'][term]:.3f}, {m['ci_high'][term]:.3f}]",
                }
            )
        rows.append(
            {
                "SPF": name,
                "term": "k (overdispersion)",
                "coef": m["k"],
                "95% CI": f"[{m['k_ci'][0]:.3f}, {m['k_ci'][1]:.3f}]",
            }
        )
    return pd.DataFrame(rows)


def elasticity_table(models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        row = {"SPF": name, "rows": m["n"], "AIC": m["aic"], "k": m["k"]}
        row["reference levels"] = ", ".join(r for r in m["reference"] if not r.startswith("year"))
        for term, label in (("log_major", "major"), ("log_minor", "minor")):
            b, lo, hi = m["params"][term], m["ci_low"][term], m["ci_high"][term]
            row[f"{label} elasticity [95% CI]"] = f"{b:.3f} [{lo:.3f}, {hi:.3f}]"
            row[f"{label} in 0.3-0.9"] = "yes" if 0.3 <= b <= 0.9 else "no"
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(r: dict[str, Any], path: Path) -> None:
    chosen = r["chosen"]
    lines = [
        "# Safety Performance Function (CP6)",
        "",
        "Negative binomial (NB2, statsmodels) on clean intersection-years in the train label "
        f"years ({r['train_rows']:,} rows). Volumes from CP5: major/minor road AADT from the "
        "two-way AADT of the approach segments, grouped by street name; single-road "
        "intersections use entering AADT with an indicator. Static terms: signal, legs, "
        "approach-class combo, streetcar within 20 m, minimum leg angle. Year fixed effects "
        "for the train years; other years get a citywide level (see `spf.py`).",
        "",
        "## Pooled vs grouped",
        "",
        markdown_table(r["selection"], floats="{:,.4f}"),
        "",
        f"Chosen: **{chosen}** ({r['reason']}).",
        "",
        "## Volume elasticities",
        "",
        "Literature range for intersection SPFs: roughly 0.3-0.9 per approach.",
        "",
        markdown_table(elasticity_table(r["models"]), floats="{:,.3f}"),
        "",
        "## Coefficients (chosen SPF)",
        "",
        "Categorical contrasts are against each group's most common level (the `reference "
        "levels` column above). Year effects omitted.",
        "",
        markdown_table(coefficient_table(r["models"]), floats="{:.3f}"),
        "",
        "## Citywide level by year",
        "",
        "Observed clean crashes / sum of base predictions. `spf_forecast` for year t uses "
        "the level of t - 1.",
        "",
        markdown_table(
            r["levels"].rename("level").reset_index().astype({"year": str}), floats="{:.3f}"
        ),
    ]
    if r.get("other"):
        lines += [
            "",
            "## The SPF not chosen",
            "",
            markdown_table(elasticity_table(r["other"]), floats="{:,.3f}"),
        ]
    path.write_text("\n".join(lines) + "\n")


def run_spf(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    df = spf_frame(cfg)
    train = df[(df["split"] == "train") & ~df["is_excluded"]]
    candidates = {POOLED: fit_models(train, grouped=False), "grouped": fit_models(train, True)}
    rows = []
    preds = {}
    for name, models in candidates.items():
        preds[name] = with_levels(df, predict_base(models, df))
        rows.append(
            {
                "SPF": name,
                "parameters": sum(len(m["columns"]) + 1 for m in models.values()),
                "AIC (train)": sum(m["aic"] for m in models.values()),
                "Poisson deviance val 2024": val_deviance(df, preds[name], 2024),
            }
        )
    selection = pd.DataFrame(rows)
    p, g = selection.set_index("SPF").loc[POOLED], selection.set_index("SPF").loc["grouped"]
    grouped_wins = (
        g["AIC (train)"] < p["AIC (train)"]
        and g["Poisson deviance val 2024"] <= p["Poisson deviance val 2024"]
    )
    chosen = "grouped" if grouped_wins else POOLED
    reason = (
        "lower AIC and validation deviance no worse"
        if grouped_wins
        else "grouping does not improve both AIC and validation deviance"
    )
    models = candidates[chosen]
    out = preds[chosen].assign(k=k_of(models, df).to_numpy())
    if out[["spf_base", "spf_known"]].isna().any().any():
        raise ValueError("SPF predictions contain NaN")
    out.to_parquet(processed / SPF_PREDICTIONS, index=False)
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / SPF_MODEL).write_text(json.dumps({"chosen": chosen, "models": models}, indent=2))
    r = {
        "chosen": chosen,
        "reason": reason,
        "models": models,
        "other": candidates["grouped" if chosen == POOLED else POOLED],
        "selection": selection,
        "train_rows": len(train),
        "levels": year_levels(df, out["spf_base"]),
    }
    write_report(r, Path(cfg["paths"]["reports_dir"]) / "spf.md")
    log.info("SPF: %s, val deviance %s", chosen, selection.to_dict(orient="records"))
    return r
