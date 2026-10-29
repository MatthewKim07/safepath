"""Data report for the cleaned collision table: coverage, severity, drops, and a location map."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from safepath.data.clean import CLEAN_FILENAME, STATS_FILENAME, parse_records
from safepath.data.download import RAW_FILENAME

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def year_table(parsed: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Per-year counts. `parsed` is every raw row (typed); `clean` is the located subset."""
    raw_g = parsed.groupby("year")
    located = clean.groupby("year")
    t = pd.DataFrame(
        {
            "raw_rows": raw_g.size(),
            "months_covered": raw_g["month"].nunique(),
            "last_date": raw_g["occ_date"].max().dt.date,
            "located": located.size(),
            "injury_or_fatal": located["injury_or_fatal"].sum(),
            "fatal_events": located["is_fatal"].sum(),
            "vru": located["involves_vru"].sum(),
        }
    ).fillna(0)
    t["pct_located"] = 100 * t["located"] / t["raw_rows"]
    t["pct_injury_or_fatal"] = 100 * t["injury_or_fatal"] / t["located"]
    t["pct_vru"] = 100 * t["vru"] / t["located"]
    t["partial"] = t["months_covered"] < 12
    return t


def plot_per_year(t: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    years = t.index.astype(int)
    bars = ax.bar(years, t["located"], width=0.72, color=BAR, edgecolor=SURFACE, linewidth=2)
    for bar, partial in zip(bars, t["partial"], strict=True):
        if partial:
            bar.set_facecolor("#9ec5f4")
            bar.set_hatch("///")
            bar.set_edgecolor(BAR)
            bar.set_linewidth(0)
            ax.annotate(
                "partial year",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=TEXT_MUTED,
            )
    ax.set_title("Located collisions per year, Toronto", loc="left", color=TEXT, fontsize=12)
    ax.set_xticks(years)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_hexbin(clean: pd.DataFrame, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    panels = [
        ("All located collisions", clean),
        ("Injury or fatal collisions", clean[clean["injury_or_fatal"]]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), facecolor=SURFACE, layout="constrained")
    for ax, (title, d) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        hb = ax.hexbin(
            d["lon"], d["lat"], gridsize=110, cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0
        )
        ax.set_title(f"{title} (n = {len(d):,})", loc="left", color=TEXT, fontsize=11)
        ax.set_aspect(1 / 0.723)  # cos(43.7 deg): roughly equal-distance axes
        ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("collisions per hex (log scale)", color=TEXT_MUTED, fontsize=9)
        cb.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        cb.outline.set_visible(False)
    span = f"{clean['occ_date'].min():%b %Y} to {clean['occ_date'].max():%b %Y}"
    fig.suptitle(f"Toronto collision locations, {span}", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---:" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build_markdown(t: pd.DataFrame, stats: dict[str, Any], extra: dict[str, Any]) -> str:
    n_raw, n_clean = stats["n_raw"], stats["n_clean"]
    pct = lambda n: f"{100 * n / n_raw:.2f}%"  # noqa: E731
    shown = pd.DataFrame(
        {
            "year": t.index.astype(int),
            "raw rows": t["raw_rows"].astype(int).map("{:,}".format),
            "located": t["located"].astype(int).map("{:,}".format),
            "% located": t["pct_located"].map("{:.1f}".format),
            "% injury/fatal": t["pct_injury_or_fatal"].map("{:.1f}".format),
            "fatal events": t["fatal_events"].astype(int),
            "% VRU": t["pct_vru"].map("{:.1f}".format),
            "months": t["months_covered"].astype(int),
            "last date": t["last_date"].astype(str),
            "partial": t["partial"].map({True: "**yes**", False: ""}),
        }
    )
    sev = stats["missing_coords_rate_by_severity"]
    clean_sev = extra["severity"]
    lines = [
        "# Collision data report",
        "",
        "Source: Toronto Police Service, Traffic Collisions open data (ArcGIS FeatureServer), "
        "Open Government Licence - Toronto.",
        "",
        "## Row accounting",
        "",
        "| step | rows | share of raw |",
        "|---|---:|---:|",
        f"| raw download (matches service `returnCountOnly`) | {n_raw:,} | 100% |",
        f"| dropped: duplicate EVENT_UNIQUE_ID | {stats['n_dropped_duplicate']:,} | "
        f"{pct(stats['n_dropped_duplicate'])} |",
        f"| dropped: missing / zero coordinates | {stats['n_dropped_missing_coords']:,} | "
        f"{pct(stats['n_dropped_missing_coords'])} |",
        f"| dropped: outside Toronto bbox (non-zero) | {stats['n_dropped_outside_bbox']:,} | "
        f"{pct(stats['n_dropped_outside_bbox'])} |",
        f"| **clean, located events** | **{n_clean:,}** | **{pct(n_clean)}** |",
        "",
        "Duplicates: "
        f"{stats['n_duplicate_rows']:,} rows share an EVENT_UNIQUE_ID "
        f"({stats['n_event_ids_with_duplicates']:,} distinct IDs) after upper-casing and "
        "trimming. Rows that agree on date, hour, coordinates, and every flag but carry "
        f"different IDs: {extra['n_composite_dup_extra']:,} extra rows. These are kept: at "
        "intersection-level location and hour resolution they are indistinguishable from "
        "separate collisions at a busy intersection.",
        "",
        f"Every record with zero coordinates has HOOD_158 = NSA (no specified address): "
        f"{extra['zero_iff_nsa']}.",
        "",
        "## Per-year coverage",
        "",
        _md_table(shown),
        "",
        "VRU = pedestrian, cyclist, or motorcyclist involved. Partial years have fewer than "
        "12 months of records and must be excluded from full-year labels.",
        "",
        "## Severity (clean rows)",
        "",
        f"- injury or fatal: {clean_sev['injury_or_fatal']:,} "
        f"({100 * clean_sev['injury_or_fatal'] / n_clean:.1f}%)",
        f"- fatal events: {clean_sev['fatal_events']:,}; total fatalities "
        f"{clean_sev['fatalities']:,}",
        f"- fatal events without INJURY_COLLISIONS = YES: {clean_sev['fatal_without_injury']:,} "
        "(so severity is defined as injury OR fatalities > 0)",
        f"- road-user flags not recorded (N/R or empty): {clean_sev['vru_unknown']:,}",
        "",
        "## Things to watch",
        "",
        "- **Missing location is not random.** Share of records without coordinates: "
        f"{100 * sev['other']:.1f}% of property-damage / other collisions, "
        f"{100 * sev['injury_or_fatal']:.1f}% of injury-or-fatal, "
        f"{100 * sev['fatal']:.1f}% of fatal. Located data over-represents severe "
        "collisions relative to the full record.",
        f"- **Locations are heavily quantised.** {n_clean:,} located events sit on only "
        f"{extra['n_unique_points']:,} distinct coordinate pairs "
        f"({n_clean / extra['n_unique_points']:.1f} events per point; the busiest point has "
        f"{extra['max_per_point']:,}). This is consistent with offsetting to the nearest "
        "intersection and is checked against the road graph in CP2. The top 1,000 points "
        f"hold {extra['top1000_share']:.0%} of all located events. Some of the busiest points "
        "sit at shopping-centre parking lots (near Yorkdale and Scarborough Town Centre) with "
        "very low severity and high fail-to-remain shares; they may not lie on the drivable "
        "network at all.",
        "- **2020-2021 COVID dip.** Located counts fall sharply from April 2020 and recover "
        "through 2022.",
        "",
        "## Figures",
        "",
        "![Located collisions per year](figures/collisions_per_year.png)",
        "",
        "![Collision locations](figures/collisions_hexbin.png)",
        "",
    ]
    return "\n".join(lines)


def run_report(cfg: dict[str, Any]) -> Path:
    paths = cfg["paths"]
    raw = pd.read_parquet(Path(paths["raw_dir"]) / RAW_FILENAME)
    clean = pd.read_parquet(Path(paths["processed_dir"]) / CLEAN_FILENAME)
    stats = json.loads((Path(paths["processed_dir"]) / STATS_FILENAME).read_text())
    parsed = parse_records(raw)

    t = year_table(parsed, clean)
    zero = parsed["lon"].fillna(0).eq(0) | parsed["lat"].fillna(0).eq(0)
    points = clean.groupby(["lon", "lat"]).size()
    key = [
        "occ_date",
        "hour",
        "lon",
        "lat",
        "fatalities",
        "is_injury",
        "is_fail_to_remain",
        "is_property_damage",
        "involves_automobile",
        "involves_motorcycle",
        "involves_passenger",
        "involves_bicycle",
        "involves_pedestrian",
    ]
    extra = {
        "zero_iff_nsa": bool((zero == parsed["hood_158"].eq("NSA")).all()),
        "n_unique_points": len(points),
        "max_per_point": int(points.max()),
        "top1000_share": float(points.nlargest(1000).sum() / len(clean)),
        "n_composite_dup_extra": int(clean.duplicated(key).sum()),
        "severity": {
            "injury_or_fatal": int(clean["injury_or_fatal"].sum()),
            "fatal_events": int(clean["is_fatal"].sum()),
            "fatalities": int(clean["fatalities"].sum()),
            "fatal_without_injury": int(
                (clean["is_fatal"] & ~clean["is_injury"].fillna(False)).sum()
            ),
            "vru_unknown": int(clean["involves_vru"].isna().sum()),
        },
    }

    figures = Path(paths["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    plot_per_year(t, figures / "collisions_per_year.png")
    plot_hexbin(clean, figures / "collisions_hexbin.png")

    out = Path(paths["reports_dir"]) / "data_report.md"
    out.write_text(build_markdown(t, stats, extra))
    log.info("wrote %s and figures in %s", out, figures)
    return out
