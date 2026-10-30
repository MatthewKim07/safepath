"""Empirical Bayes expected crashes per intersection-year.

For target year t and a history window of the W years before t:

    w    = 1 / (1 + k * sum_h SPF_h)
    EB_t = SPF_t * (w + (1 - w) * sum_h obs_h / sum_h SPF_h)

SPF_h = base x the citywide level of year h (known once h is over) and SPF_t = base x the
level of t - 1, so nothing from year t or later is used. This is the usual HSM form; when
the yearly level is flat it equals w * SPF_t + (1 - w) * (observed per year), and unlike
that simple average it does not mistake the 2020-21 dip for a safer intersection.

The window W is chosen on validation 2024 Poisson deviance. Outputs:
data/processed/eb_predictions.parquet (2016-2026), reports/eb.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from safepath.features.labels import LABELS
from safepath.locations import markdown_table
from safepath.models.spf import DEVIANCE_FLOOR, SPF_PREDICTIONS, year_levels

log = logging.getLogger(__name__)

EB_PREDICTIONS = "eb_predictions.parquet"
WINDOWS = [3, 5, 8]
TARGET_YEARS = range(2016, 2027)  # 2026 is the deployment year used for routing


def eb_for_year(
    base: pd.Series, k: pd.Series, labels: pd.DataFrame, levels: pd.Series, t: int, window: int
) -> pd.DataFrame:
    """EB for every intersection in `base` (indexed by intersection_id) for year t, using
    only labels and levels of years t - window .. t - 1."""
    hist_years = [y for y in range(t - window, t) if y in levels.index]
    hist = labels[labels["year"].isin(hist_years)]
    obs = hist.groupby("intersection_id")["crash_count"].sum().reindex(base.index).fillna(0)
    spf_hist = base * levels.reindex(hist_years).sum()
    spf_t = base * levels[t - 1]
    w = 1 / (1 + k * spf_hist)
    eb = spf_t * (w + (1 - w) * obs / spf_hist)
    return pd.DataFrame(
        {
            "intersection_id": base.index,
            "year": t,
            "spf": spf_t.to_numpy(),
            "eb": eb.to_numpy(),
            "weight_spf": w.to_numpy(),
            "obs_history": obs.to_numpy(),
            "history_years": len(hist_years),
        }
    )


def eb_all(
    base: pd.Series, k: pd.Series, labels: pd.DataFrame, years: range, window: int
) -> pd.DataFrame:
    """Levels are recomputed from labels each time, so year t's level never feeds EB_t."""
    base_rows = labels.join(base.rename("spf_base"), on="intersection_id")
    levels = year_levels(base_rows, base_rows["spf_base"])
    return pd.concat(
        [eb_for_year(base, k, labels, levels, t, window) for t in years], ignore_index=True
    )


def window_selection(
    base: pd.Series, k: pd.Series, labels: pd.DataFrame, clean: pd.Index
) -> pd.DataFrame:
    y = labels[labels["year"] == 2024].set_index("intersection_id")["crash_count"]
    y = y.reindex(clean)
    rows = []
    for window in WINDOWS:
        eb = eb_all(base, k, labels, range(2024, 2025), window).set_index("intersection_id")
        p = eb.loc[clean, "eb"]
        rows.append(
            {
                "history window (years)": window,
                "Poisson deviance val 2024": float(
                    mean_poisson_deviance(y, np.maximum(p, DEVIANCE_FLOOR))
                ),
                "Spearman val 2024": float(p.corr(y, method="spearman")),
                "median weight on SPF": float(eb.loc[clean, "weight_spf"].median()),
            }
        )
    return pd.DataFrame(rows)


def run_eb(cfg: dict[str, Any]) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    labels = pd.read_parquet(processed / LABELS)
    spf = pd.read_parquet(processed / SPF_PREDICTIONS).drop_duplicates("intersection_id")
    spf = spf.set_index("intersection_id")
    base, k = spf["spf_base"], spf["k"]
    clean = labels.loc[~labels["is_excluded"], "intersection_id"].unique()
    selection = window_selection(base, k, labels, pd.Index(clean))
    best = selection.loc[selection["Poisson deviance val 2024"].idxmin()]
    window = int(best["history window (years)"])
    out = eb_all(base, k, labels, TARGET_YEARS, window)
    if out["eb"].isna().any() or (out["eb"] < 0).any():
        raise ValueError("EB estimates contain NaN or negative values")
    out["window"] = window
    out.to_parquet(processed / EB_PREDICTIONS, index=False)
    w = out[out["year"] == 2026].set_index("intersection_id")["weight_spf"]
    lines = [
        "# Empirical Bayes (CP6)",
        "",
        "EB_t = SPF_t * (w + (1 - w) * observed / SPF over the history window), "
        "w = 1 / (1 + k * SPF summed over the window). History uses only years before t "
        "(tests/test_eb.py perturbs later years and checks EB_t is unchanged).",
        "",
        "## History window (chosen on validation 2024)",
        "",
        markdown_table(selection, floats="{:.4f}"),
        "",
        f"Chosen: **{window} years**.",
        "",
        "## Weight on the SPF, 2026 estimates",
        "",
        f"Median {w.median():.3f}; p10 {w.quantile(0.1):.3f}; p90 {w.quantile(0.9):.3f}. "
        "Busy intersections lean on their own history (small w); quiet ones on the SPF.",
    ]
    (Path(cfg["paths"]["reports_dir"]) / "eb.md").write_text("\n".join(lines) + "\n")
    log.info("EB with a %d-year window for %d intersection-years", window, len(out))
    return out
