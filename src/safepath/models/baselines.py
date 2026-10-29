"""Non-learned baselines for next-year intersection crash counts.

(a) prev3_mean  - mean count over the three years before the label year (history feature)
(b) persistence - long-run mean per intersection over the train label years
(c) class_combo - mean count per approach-class combo over train rows (no history at all)
"""

from __future__ import annotations

import pandas as pd


def fit_baselines(train: pd.DataFrame, target: str = "crash_count") -> dict[str, pd.Series]:
    return {
        "persistence": train.groupby("intersection_id")[target].mean(),
        "class_combo": train.groupby("class_combo")[target].mean(),
        "global": pd.Series({"mean": train[target].mean()}),
    }


def predict_baselines(df: pd.DataFrame, fitted: dict[str, pd.Series]) -> pd.DataFrame:
    fallback = fitted["global"]["mean"]
    return pd.DataFrame(
        {
            "baseline_prev3_mean": df["prev3_crashes_mean"].to_numpy(),
            # intersections absent from train (none for clean rows) fall back to the global mean
            "baseline_persistence": df["intersection_id"]
            .map(fitted["persistence"])
            .fillna(fallback)
            .to_numpy(),
            "baseline_class_combo": df["class_combo"]
            .map(fitted["class_combo"])
            .fillna(fallback)
            .to_numpy(),
        },
        index=df.index,
    )
