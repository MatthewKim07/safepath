"""Crash-history features per intersection-year, using only years strictly before the label year.

For label year Y the window is the (up to) three most recent years before Y that have data.
With every year published (Toronto) that is Y-3..Y-1; when a year is missing (Ottawa 2023)
the window reaches back past it, so for Ottawa 2024 it is 2020, 2021, 2022.
- prev1_*        counts in the most recent year before Y with data (Y-1 when present)
- prev3_*_mean   mean count over the window (only 2014-2015 exist for Toronto Y = 2016)
- *_share        the same as a share of that year's citywide intersection crashes, which
                 neutralises level shifts such as the 2020-2021 COVID drop
- years_since_last_crash  capped at YEARS_SINCE_CAP ("none recently")
- covid_in_window         1 if any year in the window is 2020 or 2021
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEARS_SINCE_CAP = 3
COVID_YEARS = {2020, 2021}


def _wide(labels: pd.DataFrame, col: str) -> pd.DataFrame:
    return labels.pivot(index="intersection_id", columns="year", values=col).sort_index(axis=1)


def history_features(labels: pd.DataFrame, label_years: list[int]) -> pd.DataFrame:
    """History features for every intersection and each year in `label_years`.

    `labels` needs intersection_id, year, crash_count, severe_count for every year that can
    appear in a history window. Only columns for years < Y are read when building year Y.
    """
    crashes = _wide(labels, "crash_count")
    severe = _wide(labels, "severe_count")
    available = [int(y) for y in crashes.columns]
    frames = []
    for year in label_years:
        past = [y for y in available if y < year][-3:]
        if not past:
            raise ValueError(f"no history available for label year {year}")
        c, s = crashes[past], severe[past]
        city_c, city_s = c.sum(axis=0), s.sum(axis=0)  # citywide totals of past years only
        last = past[-1]
        f = pd.DataFrame(index=crashes.index)
        f["prev1_crashes"] = c[last]
        f["prev1_severe"] = s[last]
        f["prev3_crashes_mean"] = c.mean(axis=1)
        f["prev3_severe_mean"] = s.mean(axis=1)
        f["prev1_crash_share"] = c[last] / city_c[last]
        f["prev1_severe_share"] = s[last] / city_s[last]
        f["prev3_crash_share_mean"] = (c / city_c).mean(axis=1)
        f["prev3_severe_share_mean"] = (s / city_s).mean(axis=1)

        before = [y for y in crashes.columns if y < year]
        had = crashes[before].to_numpy() > 0
        # index of the most recent year with a crash, or -1 if none
        last_idx = np.where(had.any(axis=1), had.shape[1] - 1 - np.argmax(had[:, ::-1], axis=1), -1)
        since = np.where(last_idx >= 0, year - np.array(before)[last_idx], YEARS_SINCE_CAP)
        f["years_since_last_crash"] = np.minimum(since, YEARS_SINCE_CAP)
        f["covid_in_window"] = int(bool(set(past) & COVID_YEARS))
        f["n_history_years"] = len(past)
        frames.append(f.assign(year=year).reset_index())
    return pd.concat(frames, ignore_index=True)
