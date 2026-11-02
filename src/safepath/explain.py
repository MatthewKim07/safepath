"""Explanations: TreeSHAP for LightGBM + AADT, and per-intersection factors for routing.

- Global: exact TreeSHAP values (LightGBM's `pred_contrib`) for LightGBM + AADT on clean
  test-2025 intersections, as a beeswarm and a mean |SHAP| bar chart, plus the direction of
  each interpretable feature's effect.
- Per intersection: routing uses Empirical Bayes, so the factors shown in the app explain
  the EB number itself. log(EB) = log(SPF) + log(EB / SPF); log(SPF) is additive in the SPF
  terms, so each intersection gets exact log-space contributions relative to the citywide
  average: intersection type (its SPF group), traffic volume, legs, leg angle, streetcar,
  approach road classes, and its own crash history relative to that expectation.

Outputs: data/processed/intersection_factors.parquet, reports/shap.md,
reports/figures/shap_beeswarm.png, reports/figures/shap_importance.png.
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

from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.graph import INTERSECTIONS
from safepath.locations import markdown_table
from safepath.models import lgbm
from safepath.models.spf import SPF_MODEL, design, spf_frame
from safepath.models.train import categories, load_table
from safepath.risk import INTERSECTION_RISK

log = logging.getLogger(__name__)

INTERSECTION_FACTORS = "intersection_factors.parquet"
FACTOR_TERMS = {
    "traffic volume": ("log_major", "log_minor", "log_entering_single", "single_road"),
    "streetcar": ("streetcar_20m",),
    "leg angle": ("min_leg_angle_10deg",),
    "legs": ("legs_",),
    "road classes": ("class_combo_",),
}
# Expected sign of each feature's effect on crashes, for the sanity check.
EXPECTED = {
    "log_aadt_entering": ("+", "more traffic, more crashes"),
    "log_aadt_major": ("+", "busier main road"),
    "log_aadt_minor": ("+", "busier side road"),
    "has_signal": ("+", "signals mark busy, complex intersections"),
    "n_legs": ("+", "more conflict points"),
    "min_leg_angle_deg": ("-", "skewed intersections are worse"),
    "streetcar_20m": ("+", "tracks, stops and turning conflicts"),
    "dist_tram_m": ("-", "nearer streetcar, more crashes"),
    "max_maxspeed_kph": ("+", "faster approach roads"),
    "max_lanes": ("+", "wider roads, more conflicts"),
    "n_bus_stops_50m": ("+", "stops and pedestrians"),
    "n_pois_100m": ("+", "activity, pedestrians"),
    "prev3_crashes_mean": ("+", "crash history"),
    "dist_motorway_m": ("?", "ramp terminals vs local streets"),
}
GROUP_LABELS = {
    "signalised 4+ leg": "signalised 4+ leg intersection",
    "signalised 3-leg": "signalised T intersection",
    "unsignalised": "unsignalised intersection",
}
HAND_CHECKED = [
    "Dundas Street East / Dundas Street West / Yonge Street",
    "Ellesmere Road / Kennedy Road",
    "Jameson Avenue / Lake Shore Boulevard West",
    "Bloor Street West / Spadina Avenue / Spadina Road",
]


# --------------------------------------------------------------------------- SHAP


def shap_frame(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature values and TreeSHAP contributions (log scale, Poisson link) for clean
    test-2025 rows. The last contribution column is the expected value."""
    table = load_table(cfg)
    rows = table[(table["split"] == "test") & ~table["is_excluded"]].reset_index(drop=True)
    features = lgbm.FEATURE_SETS["aadt"]
    booster = lgb.Booster(model_file=str(Path(cfg["paths"]["models_dir"]) / "lgbm_aadt.txt"))
    X = lgbm.prepare(rows, features, categories(table))
    contrib = booster.predict(X, pred_contrib=True)
    shap = pd.DataFrame(contrib[:, :-1], columns=features)
    shap["expected_value"] = contrib[:, -1]
    return rows.assign(**{"_X_" + f: X[f] for f in features}), shap


def plot_beeswarm(rows: pd.DataFrame, shap: pd.DataFrame, path: Path, top: int = 15) -> list:
    feats = shap.drop(columns="expected_value").abs().mean().sort_values().tail(top).index
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    for i, f in enumerate(feats):
        x = shap[f].to_numpy()
        val = rows["_X_" + f]
        if isinstance(val.dtype, pd.CategoricalDtype):
            val = val.cat.codes
        color = pd.Series(val, dtype=float).rank(pct=True).fillna(0.5).to_numpy()
        sample = rng.choice(len(x), size=min(4000, len(x)), replace=False)
        y = i + rng.normal(0, 0.12, len(sample))
        sc = ax.scatter(x[sample], y, c=color[sample], cmap="coolwarm", s=4, alpha=0.6, lw=0)
    ax.axvline(0, color=TEXT_MUTED, lw=0.8)
    ax.set_yticks(range(len(feats)), feats, fontsize=8, color=TEXT)
    ax.set_xlabel("SHAP value (log expected crashes; > 0 raises the prediction)", fontsize=9,
                  color=TEXT_MUTED)  # fmt: skip
    ax.set_title("LightGBM + AADT, test 2025: TreeSHAP per intersection (colour = feature value)",
                 loc="left", fontsize=10, color=TEXT)  # fmt: skip
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("feature value (low → high, percentile)", fontsize=8, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return list(feats[::-1])


def plot_importance(shap: pd.DataFrame, path: Path, top: int = 20) -> pd.Series:
    imp = shap.drop(columns="expected_value").abs().mean().sort_values()
    shown = imp.tail(top)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    ax.barh(shown.index, shown.to_numpy(), color=BAR, height=0.65)
    ax.set_title("LightGBM + AADT: mean |SHAP| (test 2025, log scale)", loc="left", fontsize=10,
                 color=TEXT)  # fmt: skip
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return imp[::-1]


def effect_table(rows: pd.DataFrame, shap: pd.DataFrame, imp: pd.Series) -> pd.DataFrame:
    """Direction of each interpretable feature's effect: Spearman correlation between the
    feature's value and its SHAP value (+ means higher values raise predicted crashes)."""
    out = []
    for f, (sign, why) in EXPECTED.items():
        if f not in shap:
            continue
        val = pd.Series(rows["_X_" + f], dtype=float)
        rho = val.corr(shap[f], method="spearman")
        learned = "+" if rho > 0.1 else "-" if rho < -0.1 else "flat"
        verdict = (
            "as expected" if learned == sign or sign == "?"
            else "weak" if learned == "flat" else "counterintuitive"
        )  # fmt: skip
        out.append(
            {
                "feature": f,
                "mean |SHAP|": float(imp.get(f, np.nan)),
                "rank": int(list(imp.index).index(f)) + 1,
                "value-SHAP Spearman": float(rho),
                "learned": learned,
                "expected": f"{sign} ({why})",
                "verdict": verdict,
            }
        )
    return pd.DataFrame(out).sort_values("rank")


# --------------------------------------------------------------------------- EB factors


def spf_contributions(
    cfg: dict[str, Any], df: pd.DataFrame | None = None, models: dict | None = None
) -> pd.DataFrame:
    """Log-space contribution of each SPF factor per intersection, relative to the citywide
    average log SPF. Sums exactly to log(SPF_i) - mean log SPF. Another city passes its own
    covariate frame `df` and Toronto's SPF `models` (London)."""
    if models is None:
        models = json.loads((Path(cfg["paths"]["models_dir"]) / SPF_MODEL).read_text())["models"]
    df = spf_frame(cfg) if df is None else df
    df = df[df["year"] == df["year"].max()].set_index("intersection_id")
    parts = []
    for group, m in models.items():
        g = df[df["group"] == group] if group != "pooled" else df
        X = design(g.reset_index(), m["columns"]).set_index(g.index)
        beta = pd.Series(m["params"]).reindex(m["columns"]).fillna(0.0)
        terms = X.mul(beta, axis=1)
        mean = terms.mean()
        c = pd.DataFrame(index=g.index)
        for factor, prefixes in FACTOR_TERMS.items():
            cols = [col for col in terms if col.startswith(prefixes)]
            c[factor] = (terms[cols] - mean[cols]).sum(axis=1)
        c["_group_level"] = mean.sum()  # log SPF of the group's average profile
        c["group"] = group
        parts.append(c)
    out = pd.concat(parts)
    out["intersection type"] = (
        out["_group_level"] - (out[list(FACTOR_TERMS)].sum(axis=1) + out["_group_level"]).mean()
    )
    out = out.drop(columns="_group_level")
    keep = ["n_legs", "min_leg_angle_deg", "streetcar_20m", "class_combo"]
    return out.join(df[keep])


def factor_label(name: str, value: float, row: pd.Series) -> str:
    up = value > 0
    if name == "traffic volume":
        text = "high traffic volume" if up else "low traffic volume"
    elif name == "streetcar":
        text = "streetcar tracks" if row["streetcar_20m"] == 1 else "no streetcar tracks"
    elif name == "leg angle":
        text = "skewed angle" if row["min_leg_angle_deg"] < 70 else "near-right-angle legs"
    elif name == "legs":
        n = int(row["n_legs"])
        text = f"{n} legs" if n <= 4 else "5+ legs"
    elif name == "road classes":
        text = f"{row['class_combo']} roads"
    elif name == "intersection type":
        text = GROUP_LABELS.get(row["group"], row["group"])
    else:  # crash history
        text = (
            "more crashes than expected (2021-25)"
            if up
            else "fewer crashes than expected (2021-25)"
        )  # noqa: E501
    return f"{text} ({'+' if up else '−'}, ×{np.exp(value):.1f})"


def intersection_factors(cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    c = spf_contributions(cfg)
    risk = pd.read_parquet(processed / INTERSECTION_RISK).set_index("intersection_id")
    history = np.log(np.maximum(risk["eb_uncalibrated"], 1e-6) / np.maximum(risk["spf"], 1e-6))
    c["crash history"] = np.where(risk["is_proxy_fallback"].reindex(c.index), 0.0,
                                  history.reindex(c.index))  # fmt: skip
    return top_factors(c, [*FACTOR_TERMS, "intersection type", "crash history"])


def top_factors(c: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """The three largest log contributions per intersection, as app labels."""
    rows = []
    for iid, r in c.iterrows():
        top = r[factors].astype(float).abs().sort_values(ascending=False).index[:3]
        labels = [factor_label(f, float(r[f]), r) for f in top]
        rows.append({"intersection_id": iid, "factor_1": labels[0], "factor_2": labels[1],
                     "factor_3": labels[2]})  # fmt: skip
    out = pd.DataFrame(rows).merge(c[factors].reset_index(), on="intersection_id")
    return out


def hand_checks(
    cfg: dict[str, Any], factors: pd.DataFrame, rows: pd.DataFrame, shap: pd.DataFrame
) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    names = pd.read_parquet(processed / INTERSECTIONS).set_index("intersection_id")["street_names"]
    risk = pd.read_parquet(processed / INTERSECTION_RISK).set_index("intersection_id")
    f = factors.set_index("intersection_id")
    out = []
    for name in HAND_CHECKED:
        hits = names[names == name]
        if hits.empty:
            continue
        iid = int(hits.index[0])
        r = rows.index[rows["intersection_id"] == iid]
        shap_top = (
            ", ".join(
                f"{k} ({v:+.2f})"
                for k, v in shap.loc[r[0]]
                .drop("expected_value")
                .sort_values(key=np.abs)
                .tail(3)[::-1]
                .items()
            )  # fmt: skip
            if len(r)
            else "(excluded or not in test rows)"
        )
        out.append(
            {
                "intersection": name,
                "EB crashes/yr (2026)": risk.loc[iid, "risk"],
                "per million entering vehicles": risk.loc[iid, "risk_per_mev"],
                "EB top factors (app)": "; ".join(f.loc[iid, ["factor_1", "factor_2", "factor_3"]]),
                "LightGBM SHAP top 3 (test 2025)": shap_top,
            }
        )
    return pd.DataFrame(out)


def write_report(r: dict[str, Any], path: Path) -> None:
    eff = r["effects"]
    odd = eff[eff["verdict"] != "as expected"]
    lines = [
        "# Explanations (CP7)",
        "",
        "## LightGBM + AADT: TreeSHAP",
        "",
        f"Exact TreeSHAP (LightGBM `pred_contrib`) on {r['n_rows']:,} clean test-2025 "
        "intersections; values are on the log scale of expected crashes (Poisson link), so "
        "+0.69 doubles a prediction.",
        "",
        "![SHAP beeswarm](figures/shap_beeswarm.png)",
        "",
        "![Mean |SHAP|](figures/shap_importance.png)",
        "",
        f"History features take {r['history_share']:.0%} of total mean |SHAP|; the three AADT "
        f"features {r['aadt_share']:.0%}; static road features the rest.",
        "",
        "### Do the learned effects make sense?",
        "",
        "Direction = Spearman correlation between a feature's value and its SHAP value.",
        "",
        markdown_table(eff, floats="{:.3f}"),
        "",
    ]
    if len(odd):
        lines += ["Flagged:", ""]
        for _, row in odd.iterrows():
            lines.append(f"- `{row['feature']}`: learned {row['learned']}, expected "
                         f"{row['expected']} ({row['verdict']}).")  # fmt: skip
        lines += ["", *r["flag_notes"], ""]
    lines += [
        "## Routing model (EB): factor breakdown",
        "",
        "Routing uses EB, so the app explains the EB number. log(EB) = log(SPF) + "
        "log(EB / SPF): the SPF part splits exactly into its terms (coefficients in "
        "reports/spf.md), each measured against the citywide average intersection, and "
        "log(EB / SPF) is the intersection's own crash history relative to that "
        "expectation. The app shows the three largest, with the factor they multiply "
        "expected crashes per year by.",
        "",
        "Mean |log contribution| over all intersections:",
        "",
        markdown_table(r["factor_importance"], floats="{:.3f}"),
        "",
        "Traffic volume raises expected crashes per year, but with elasticities below 1 "
        "(reports/spf.md) it lowers crashes per vehicle: busier intersections are more "
        "dangerous places and less dangerous per car.",
        "",
        "## Hand-checked intersections",
        "",
        markdown_table(r["hand"], floats="{:,.2f}"),
        "",
        *r["hand_notes"],
    ]
    path.write_text("\n".join(lines) + "\n")


def run_explain(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    figures = Path(cfg["paths"]["figures_dir"])
    rows, shap = shap_frame(cfg)
    plot_beeswarm(rows, shap, figures / "shap_beeswarm.png")
    imp = plot_importance(shap, figures / "shap_importance.png")
    total = imp.sum()
    factors = intersection_factors(cfg)
    factors.to_parquet(processed / INTERSECTION_FACTORS, index=False)
    fi = (
        factors[[*FACTOR_TERMS, "intersection type", "crash history"]].abs().mean()
        .sort_values(ascending=False).rename("mean |log contribution|")
        .rename_axis("factor").reset_index()
    )  # fmt: skip
    hand = hand_checks(cfg, factors, rows, shap)
    flag_notes = flag_explanations(rows, shap)
    r = {
        "n_rows": len(rows),
        "effects": effect_table(rows, shap, imp),
        "history_share": imp[imp.index.isin(lgbm.HISTORY_FEATURES)].sum() / total,
        "aadt_share": imp[imp.index.isin(lgbm.AADT_FEATURES)].sum() / total,
        "factor_importance": fi,
        "hand": hand,
        "hand_notes": HAND_NOTES,
        "flag_notes": flag_notes,
    }
    write_report(r, Path(cfg["paths"]["reports_dir"]) / "shap.md")
    log.info("explanations for %d intersections", len(factors))
    return r


def flag_explanations(rows: pd.DataFrame, shap: pd.DataFrame) -> list[str]:
    """Where the flagged effects come from."""
    minor = rows["_X_log_aadt_minor"]
    single = minor.isna()
    top = minor >= minor.quantile(0.9)
    speed = rows["_X_max_maxspeed_kph"]
    sharp = rows["_X_min_leg_angle_deg"] <= 45
    flat = rows["_X_min_leg_angle_deg"] >= 179.9
    mid = ~sharp & ~flat
    rho_minor = shap.loc[~single & ~top, "log_aadt_minor"].corr(
        minor[~single & ~top], method="spearman"
    )
    return [
        "Where these come from:",
        "",
        f"- `log_aadt_minor`: {single.mean():.0%} of intersections have a single road (no "
        f"minor approach); their SHAP averages {shap.loc[single, 'log_aadt_minor'].mean():+.2f}, "
        "which is most of this feature's importance and is sensible (not a real crossing). "
        "Among real crossings the effect rises with minor-road volume "
        f"({rho_minor:+.2f} Spearman below the top decile) and drops in the top decile "
        f"({shap.loc[top, 'log_aadt_minor'].mean():+.2f}), where the 'minor' road is an "
        "expressway at a ramp or freeway cluster that TPS barely records.",
        f"- `max_maxspeed_kph`: the negative effect sits at 100 km/h approaches "
        f"(mean SHAP {shap.loc[speed >= 95, 'max_maxspeed_kph'].mean():+.3f} vs "
        f"{shap.loc[speed <= 50, 'max_maxspeed_kph'].mean():+.3f} at 50 km/h or less): "
        "ramp and expressway clusters, again OPP or few recorded crashes. Not evidence that "
        "speed is safe; the effect is small.",
        f"- `min_leg_angle_deg`: the low-angle end ({sharp.mean():.0%} of intersections at "
        f"45° or less) has mean SHAP {shap.loc[sharp, 'min_leg_angle_deg'].mean():+.2f}, but "
        "those are mostly carriageway and ramp merges (median entering AADT "
        f"{np.exp(rows.loc[sharp, '_X_log_aadt_entering']).median():,.0f} vs "
        f"{np.exp(rows.loc[~sharp, '_X_log_aadt_entering']).median():,.0f} elsewhere) where "
        "TPS records few crashes, not skewed street crossings. The other strongly negative "
        f"group is the {flat.mean():.0%} at exactly 180° (one-leg clusters, mean SHAP "
        f"{shap.loc[flat, 'min_leg_angle_deg'].mean():+.2f}), which are not real "
        "intersections. Between those ends the value-SHAP Spearman is "
        f"{shap.loc[mid, 'min_leg_angle_deg'].corr(rows.loc[mid, '_X_min_leg_angle_deg'], method='spearman'):+.2f}. "  # noqa: E501
        "The SPF, which holds legs, "
        "signal and volume fixed, gives the expected negative angle effect for unsignalised "
        "and 3-leg intersections but a positive one for signalised 4-leg ones "
        "(reports/spf.md), so the skew effect is not cleanly identified in this data.",
    ]


HAND_NOTES = [
    "Reading: the EB factors explain the number routing uses; LightGBM's SHAP is dominated "
    "by crash history at every busy intersection, which is right but uninformative.",
    "- Yonge / Dundas: a signalised 4+ leg arterial crossing with somewhat more crashes than "
    "its volume and design predict. Its per-vehicle risk is moderate because so much "
    "traffic passes through; much of its harm is to pedestrians, which per-vehicle routing "
    "does not price.",
    "- Ellesmere / Kennedy: among the city's highest crash counts; the factors say why: "
    "signalised 4+ leg, high volume, and about three times more crashes than that predicts.",
    "- Jameson / Lake Shore: before attribution it was the city's top intersection; after "
    "moving its excess to the Gardiner (reports/risk.md), it is an ordinary busy signalised "
    "crossing, and the factors no longer mention its history.",
    "- Bloor / Spadina: busy and signalised, but fewer crashes than expected over 2021-25, "
    "so EB pulls it below its SPF.",
]
