"""reports/volumes.md: sources, AADT expansion factors, matching, coverage, and oddities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safepath.data.aadt import (
    COVID_YEARS,
    build_aadt,
    holdout_window_error,
    month_factor_ci,
)
from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.data.volumes import (
    MANIFEST,
    SVC_COUNTS,
    TMC_COUNTS,
    TMC_WINDOWS,
    raw_dir,
    run_volume_download,
)
from safepath.locations import markdown_table
from safepath.snapping import ASSIGNMENTS

log = logging.getLogger(__name__)

ORANGE = "#eb6834"
RECENCY_BINS = [0, 1999, 2009, 2014, 2019, 2021, 2023, 2100]
RECENCY_LABELS = ["before 2000", "2000-09", "2010-14", "2015-19", "2020-21", "2022-23", "2024-26"]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_profiles(days: pd.DataFrame, M: pd.DataFrame, path: Path) -> None:
    """Weekday hour-of-day share of daily volume (by era and road group), and month factors."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    hours = np.arange(24)
    styles = {
        ("pre2020", "major"): (BAR, "-"),
        ("2020on", "major"): (BAR, "--"),
        ("pre2020", "minor"): (ORANGE, "-"),
        ("2020on", "minor"): (ORANGE, "--"),
    }
    for (e, g), (color, ls) in styles.items():
        d = days[(days["era"] == e) & (days["group"] == g) & (days["daytype"] == "weekday")]
        share = d[[f"h{h}" for h in hours]].sum() / d["daily"].sum()
        a1.plot(hours + 0.5, 100 * share.to_numpy(), color=color, ls=ls, lw=2, label=f"{g}, {e}")
    for code, color in (("8R", GRID),):
        for start, end in TMC_WINDOWS[code]:
            h0 = int(start[:2]) + int(start[3:]) / 60
            h1 = int(end[:2]) + int(end[3:]) / 60
            a1.axvspan(h0, h1, color=color, alpha=0.6, lw=0)
    a1.set_title(
        "Weekday share of daily volume by hour (shaded: 8R TMC window)",
        fontsize=9,
        color=TEXT,
        loc="left",
    )
    a1.set_xlabel("hour", fontsize=8, color=TEXT_MUTED)
    a1.set_ylabel("% of daily volume", fontsize=8, color=TEXT_MUTED)
    a1.legend(fontsize=7, frameon=False, labelcolor=TEXT_MUTED)
    _style(a1)
    a2.bar(M["month"], M["M"], color=BAR, width=0.7)
    a2.errorbar(
        M["month"],
        M["M"],
        yerr=[M["M"] - M["M_lo"], M["M_hi"] - M["M"]],
        fmt="none",
        ecolor=TEXT_MUTED,
        lw=1,
        capsize=2,
    )
    a2.axhline(1, color=TEXT_MUTED, lw=0.8)
    a2.set_ylim(0.6, 1.2)
    a2.set_xticks(range(1, 13))
    a2.set_title(
        "Month factor M (weekday volume / annual weekday mean), 95% CI",
        fontsize=9,
        color=TEXT,
        loc="left",
    )
    _style(a2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def sources_table(cfg: dict[str, Any]) -> str:
    manifest = json.loads((raw_dir(cfg) / MANIFEST).read_text())
    rows = []
    for pkg in manifest.values():
        for r in pkg["resources"]:
            rows.append(
                {
                    "package": f"`{pkg['package_name']}`",
                    "resource": f"`{r['name']}`",
                    "resource id": f"`{r['id']}`",
                    "MB": round(int(r["size"]) / 1e6, 1) if r.get("size") else "",
                    "last modified": (r.get("last_modified") or "")[:10],
                }
            )
    return markdown_table(pd.DataFrame(rows))


def coverage_tables(
    r: dict[str, Any], assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    static, ci = r["static"], r["counted_intersections"].set_index("intersection_id")
    at = assignments[
        (assignments["unit_type"] == "intersection") & assignments["year"].between(2016, 2025)
    ]
    crashes = at.groupby(at["unit_id"].astype(int)).size()
    df = static[["class_combo"]].copy()
    df["crashes"] = crashes.reindex(df.index).fillna(0)
    df["counted"] = df.index.isin(ci.index)
    df["counted_2015on"] = df.index.map(ci["latest_year"]).to_numpy() >= 2015
    rows = []
    for combo, g in [*df.groupby("class_combo"), ("all", df)]:
        rows.append(
            {
                "class combo": combo,
                "intersections": len(g),
                "% counted": 100 * g["counted"].mean(),
                "% counted 2015+": 100 * g["counted_2015on"].mean(),
                "crashes 2016-25": int(g["crashes"].sum()),
                "% crashes at counted": 100
                * g.loc[g["counted"], "crashes"].sum()
                / max(g["crashes"].sum(), 1),
                "% crashes at counted 2015+": 100
                * g.loc[g["counted_2015on"], "crashes"].sum()
                / max(g["crashes"].sum(), 1),
            }
        )
    inter = pd.DataFrame(rows).sort_values("intersections", ascending=False)

    seg, cs = r["segments"], r["counted_segments"].set_index("segment_id")
    sd = seg[["segment_id", "highway", "length"]].copy()
    sd["counted"] = sd["segment_id"].isin(cs.index)
    rows = []
    for hw, g in [*sd.groupby("highway"), ("all", sd)]:
        rows.append(
            {
                "highway": hw,
                "segments": len(g),
                "km": g["length"].sum() / 1000,
                "counted segments": int(g["counted"].sum()),
                "% km counted": 100 * g.loc[g["counted"], "length"].sum() / g["length"].sum(),
            }
        )
    segs = pd.DataFrame(rows).sort_values("km", ascending=False)
    return inter, segs[segs["km"] >= 1]


def recency_table(ci: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    def bins(s: pd.Series) -> pd.Series:
        return pd.cut(s, RECENCY_BINS, labels=RECENCY_LABELS).value_counts().reindex(RECENCY_LABELS)

    return pd.DataFrame(
        {
            "latest count year": RECENCY_LABELS,
            "intersections": bins(ci["latest_year"]).to_numpy(),
            "segments": bins(cs["latest_year"]).to_numpy(),
        }
    )


def aadt_by_class(values: pd.Series, classes: pd.Series, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"aadt": values, label: classes}).dropna()
    g = df.groupby(label)["aadt"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "p10": g.quantile(0.1),
            "median": g.median(),
            "p90": g.quantile(0.9),
            "max": g.max(),
        }
    ).reset_index()
    return out.sort_values("median", ascending=False)


def oddities(r: dict[str, Any]) -> dict[str, pd.DataFrame | int]:
    t, s = r["tmc"], r["svc"]
    names = r["intersections"].set_index("intersection_id")["street_names"]
    combo = r["static"]["class_combo"]
    ci = r["counted_intersections"].copy()
    ci["intersection"] = ci["intersection_id"].map(names)
    ci["class combo"] = ci["intersection_id"].map(combo)
    top = ci.sort_values("latest_aadt", ascending=False).head(10)
    top = top.assign(latest_year=top["latest_year"].astype(str))
    local_busy = ci[(ci["class combo"] == "local x local") & (ci["latest_aadt"] > 20_000)]
    # counts at the same intersection since 2010 that disagree by more than 3x
    recent = t[t["year"] >= 2010].dropna(subset=["intersection_id"])
    spread = recent.groupby("intersection_id")["aadt"].agg(["min", "max", "size"])
    jumpy = spread[(spread["size"] >= 2) & (spread["max"] > 3 * spread["min"])]
    tiny_major = s[s["usable"] & s["highway"].isin(["primary", "secondary"]) & (s["aadt"] < 500)]
    return {
        "top": top,
        "local_busy": local_busy,
        "jumpy": len(jumpy),
        "jumpy_of": int((spread["size"] >= 2).sum()),
        "tiny_major": tiny_major,
    }


def write_report(cfg: dict[str, Any], r: dict[str, Any], path: Path) -> None:
    v = cfg["volumes"]["match"]
    t, s = r["tmc"], r["svc"]
    ci, cs = r["counted_intersections"], r["counted_segments"]
    processed = Path(cfg["paths"]["processed_dir"])
    assignments = pd.read_parquet(processed / ASSIGNMENTS)
    inter_cov, seg_cov = coverage_tables(r, assignments)
    odd = oddities(r)
    M = r["M"].merge(r["M_ci"], on="month")
    tmc_all = pd.read_parquet(processed / TMC_COUNTS)

    windows = pd.DataFrame(
        [
            {
                "code": code,
                "windows": ", ".join(f"{a}-{b}" for a, b in w),
                "counts": int((tmc_all["duration"] == code).sum()),
                "years": f"{tmc_all.loc[tmc_all['duration'] == code, 'date'].dt.year.min()}-"
                f"{tmc_all.loc[tmc_all['duration'] == code, 'date'].dt.year.max()}",
                "raw-checked": int(
                    tmc_all.loc[tmc_all["duration"] == code, "window_ok"].notna().sum()
                ),
                "% exact": 100
                * tmc_all.loc[tmc_all["duration"] == code, "window_ok"]
                .dropna()
                .astype(bool)
                .mean(),
            }
            for code, w in TMC_WINDOWS.items()
        ]
    )

    Rw = r["R"].copy()
    Rw = Rw[["era", "group", "daytype", "window", "R", "R_p10", "R_p90", "count_days", "counts"]]
    D = r["D"].reset_index()
    D["day"] = D["dow"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    D = D[["day", "D", "D_sd", "counts"]]

    tm = t["intersection_id"].notna()
    lines = [
        "# Traffic volumes and AADT (CP5)",
        "",
        "Exposure for the Phase 2 per-vehicle risk model. Every number here is regenerated by "
        "`python -m safepath volumes`.",
        "",
        "## Sources",
        "",
        "City of Toronto Open Data (CKAN API, packages found with `package_search` and an exact "
        "title match at run time). Licence: **Open Government Licence - Toronto**. Exact "
        "resources used (from `data/raw/volumes/manifest.json`):",
        "",
        sources_table(cfg),
        "",
        f"- **TMC** (Turning Movement Counts): {len(tmc_all):,} counts at "
        f"{tmc_all['centreline_id'].nunique():,} locations, "
        f"{tmc_all['date'].dt.year.min()}-{tmc_all['date'].dt.year.max()}. Volume = motor "
        "vehicles entering the intersection, summed over approaches (cars + trucks + buses; "
        "checked equal to the sum of approach volumes). "
        f"{int((tmc_all['centreline_type'] == 1).sum()):,} mid-block TMCs (centreline type 1) "
        "are not used.",
        f"- **SVC** (Speed / Volume / Classification, pneumatic tubes): {len(s):,} studies "
        f"after merging same-day duplicates, at {s['centreline_id'].nunique():,} locations. "
        "The City's `avg_weekday_daily_vol` (both directions, whole days) is the daily estimate "
        "used; the 2015+ raw 15-minute files supply profiles and directions.",
        "",
        "## TMC counted windows",
        "",
        "Checked bin-by-bin against every count in the raw 2010-2029 files:",
        "",
        markdown_table(windows),
        "",
        "The few 14-hour counts with missing bins have their window volume scaled up by "
        "expected / present bins.",
        "",
        "## AADT expansion",
        "",
        "AADT = average daily two-way (SVC) or entering (TMC) motor-vehicle volume over a year. "
        "All factors come from the City's own SVC data:",
        "",
        "- **R (window -> 24 h)**: from "
        f"{len(r['days']):,} complete count-days (every recorded direction has all 96 bins) "
        f"in {r['days']['count_id'].nunique():,} SVC counts, 2015-2026. R = sum of daily "
        "volume / sum of volume inside the window, by era (2015-19 vs 2020+), road group "
        "(major = primary/secondary/trunk/motorway; minor = everything else; TMCs use the "
        "intersection's highest approach class) and day type.",
        f"- **D (day of week)** from {int(r['D']['counts'].iloc[0])} counts covering all 7 days: "
        "a day's volume relative to that count's Mon-Fri mean. "
        f"**W** = mean of D over the week = **{r['W']:.3f}** (average day / average weekday).",
        f"- **M (month)**: location fixed-effects regression `log v = a_location + b_month + "
        f"c_year` on SVC weekday daily volumes ({cfg['volumes']['month_factor_years'][0]}-"
        f"{cfg['volumes']['month_factor_years'][1]}, 2020-21 excluded, {r['month_locations']:,} "
        "locations counted in at least two months), normalised to a day-weighted annual mean "
        "of 1. CIs: 200 bootstrap resamples of locations.",
        "",
        "TMC: `AADT = entering * R / D[dow] * W / M[month]`. "
        "SVC: `AADT = weekday daily * W / M[month]` (weekend-only studies are converted to a "
        "weekday equivalent with the mean Sat/Sun D).",
        "",
        "### Window ratios R",
        "",
        markdown_table(Rw, floats="{:.3f}"),
        "",
        "### Day-of-week factors D",
        "",
        markdown_table(D, floats="{:.3f}"),
        "",
        "### Month factors M",
        "",
        markdown_table(M.rename(columns={"counts": "SVC counts"}), floats="{:.3f}"),
        "",
        "![Hour-of-day profiles and month factors](figures/volume_profiles.png)",
        "",
        "### How good is the 24-hour expansion?",
        "",
        "Ratios fitted on even-numbered SVC counts, applied to the simulated TMC windows of "
        "odd-numbered counts' weekdays, compared with their true 24-hour volume:",
        "",
        markdown_table(holdout_window_error(r["days"])),
        "",
        "The window -> day step is accurate to a few percent. The month and day-of-week steps "
        "are not validated this way (no permanent counters in the City data) and are the "
        "larger uncertainty, along with count-to-count variation and staleness.",
        "",
        "## Matching to the graph",
        "",
        f"- **TMC -> consolidated intersection** (within {v['tmc_intersection_m']} m; among the "
        "5 nearest, the one sharing the most street names): "
        f"{int(tm.sum()):,} of {len(t):,} intersection TMCs matched "
        f"({100 * tm.mean():.1f}%) to {len(ci):,} intersections. Match distance median "
        f"{t.loc[tm, 'match_m'].median():.1f} m, p95 {t.loc[tm, 'match_m'].quantile(0.95):.1f} m. "
        f"{100 * (t.loc[tm, 'names_shared'] >= 2).mean():.1f}% share at least two street names "
        f"with the OSM intersection; {int((t.loc[tm, 'names_shared'] == 0).sum())} counts share "
        "none (unnamed OSM ramps). Unmatched counts are lanes, trails, and private roads "
        "with no drivable OSM intersection nearby "
        f"({t.loc[~tm, 'centreline_id'].nunique()} locations; nearest OSM intersection median "
        f"{t.loc[~tm, 'nearest_m'].median():.0f} m).",
        f"- **SVC -> undirected segment** (within {v['svc_segment_m']} m with the counted "
        f"street's name, else the nearest within {v['svc_unnamed_m']} m): "
        f"{100 * s['segment_id'].notna().mean():.1f}% matched, "
        f"{100 * s['name_match'].fillna(False).astype(bool).mean():.1f}% by name. Unmatched: "
        "laneways and park roads outside the drive graph.",
        f"- SVC studies not usable as segment AADT: {int(s['one_direction_on_two_way'].sum()):,} "
        "count one direction of a two-way street (raw files show a single direction); they "
        "are kept in the history but not used.",
        "",
        "## Coverage",
        "",
        "Crashes = TPS crashes assigned to intersections, 2016-2025.",
        "",
        markdown_table(inter_cov),
        "",
        "Segments (undirected) with a usable SVC count:",
        "",
        markdown_table(seg_cov),
        "",
        "## Count recency",
        "",
        "Most recent count year per counted unit. History is kept for every count "
        "(data/processed/tmc_aadt.parquet, svc_aadt.parquet).",
        "",
        markdown_table(recency_table(ci, cs)),
        "",
        f"{100 * ci['latest_year'].isin(COVID_YEARS).mean():.1f}% of counted intersections have "
        "their most recent count in 2020-21, when volumes were depressed; "
        "`latest_noncovid_*` columns keep the latest count outside those years.",
        "",
        "## AADT by class (most recent count)",
        "",
        "Intersections, entering AADT:",
        "",
        markdown_table(
            aadt_by_class(
                ci["latest_aadt"],
                ci["intersection_id"].map(r["static"]["class_combo"]),
                "class combo",
            ),
            floats="{:,.0f}",
        ),
        "",
        "Segments, two-way AADT:",
        "",
        markdown_table(
            aadt_by_class(
                cs["latest_aadt"],
                cs["segment_id"].map(r["segments"].set_index("segment_id")["highway"]),
                "highway",
            ),
            floats="{:,.0f}",
        ),
        "",
        "## Oddities",
        "",
        "Highest entering AADT (most recent count):",
        "",
        markdown_table(
            odd["top"][["intersection", "class combo", "latest_year", "latest_aadt", "n_counts"]],
            floats="{:,.0f}",
        ),
        "",
        f"- {len(odd['local_busy'])} counted local x local intersections exceed 20,000 entering "
        "AADT (OSM tags a busy road as residential, or the TMC sits at a consolidated "
        "cluster that OSM classes differently).",
        f"- {odd['jumpy']} of {odd['jumpy_of']:,} intersections counted at least twice since 2010 "
        "have counts more than 3x apart (construction detours, network changes, or a different "
        "count location under the same centreline id).",
        f"- {len(odd['tiny_major'])} usable SVC studies on primary/secondary segments give AADT "
        "below 500 (likely a service road or a misplaced tube; left as is).",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_volumes(cfg: dict[str, Any]) -> None:
    """`python -m safepath volumes`: download, parse, match, expand to AADT, and report."""
    run_volume_download(cfg)
    r = build_aadt(cfg)
    years = tuple(cfg["volumes"]["month_factor_years"])
    r["M_ci"] = month_factor_ci(pd.read_parquet(_svc_counts(cfg)), years, n_boot=200)
    figures = Path(cfg["paths"]["figures_dir"])
    plot_profiles(r["days"], r["M"].merge(r["M_ci"], on="month"), figures / "volume_profiles.png")
    write_report(cfg, r, Path(cfg["paths"]["reports_dir"]) / "volumes.md")


def _svc_counts(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / SVC_COUNTS
