"""Exposure sanity checks: what changes when crash counts become crash rates per vehicle.

reports/exposure.md:

1. Crashes per million entering vehicles (MEV) by intersection class combo, at intersections
   with a City count (count-based AADT only).
2. Top 20 intersections by crash count vs by crash rate, with minimum-volume and
   minimum-count filters so tiny denominators do not dominate.
3. Network-wide crashes per million vehicle-km (MVKM) by road class. Each intersection's
   crashes are split across the edges entering it in proportion to their directional AADT,
   and mid-block crashes stay on their segment, so every TPS crash is charged to the traffic
   that passed through it.
4. Expressways: MTO's collision rates on the 400-series vs TPS crashes divided by estimated
   vehicle-km on the DVP, Gardiner, and the 400-series themselves.

Crashes are TPS crashes in the non-pandemic label years 2016-2019 and 2022-2025, per year.
Proxy intersections (CP3) are left out of 1-3: their crashes happened elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from safepath.data.mto import MTO_SECTIONS
from safepath.features.build import PROXIES, STATIC
from safepath.graph import INTERSECTIONS, NODE_MAP, SEGMENTS
from safepath.locations import markdown_table
from safepath.models.volume import AADT_EDGES, AADT_INTERSECTIONS
from safepath.risk import INTERSECTION_RISK
from safepath.snapping import ASSIGNMENTS

log = logging.getLogger(__name__)

YEARS = [2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025]
MIN_COUNT_YEAR = 2010
TOP_MIN_AADT = 5_000  # entering vehicles per day
TOP_MIN_CRASHES = 5.0  # per year
EXPRESSWAYS = {
    r"^Highway 401\b": "401",
    r"^Highway 400\b": "400",
    r"^Highway 404\b": "404",
    r"^Highway 409\b": "409",
    r"^Highway 427\b": "427",
    r"^Queen Elizabeth Way\b": "QEW",
    r"^Don Valley Parkway\b": "DVP",
    r"^Gardiner Expressway\b": "Gardiner",
    r"^Allen Road\b": "Allen Road",
}
CLASS_GROUPS = {
    "motorway": "expressway (city)",
    "motorway_link": "expressway ramp",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "unclassified": "unclassified",
    "residential": "residential",
    "living_street": "residential",
}


def load(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    p = Path(cfg["paths"]["processed_dir"])
    a = pd.read_parquet(p / ASSIGNMENTS)
    a = a[a["year"].isin(YEARS)]
    n = len(YEARS)
    inter = a[a["unit_type"] == "intersection"]
    seg = a[a["unit_type"] == "segment"]
    return {
        "inter_crashes": inter.groupby(inter["unit_id"].astype(int)).size() / n,
        "seg_crashes": seg.groupby(seg["unit_id"].astype(int)).size() / n,
        "aadt_i": pd.read_parquet(p / AADT_INTERSECTIONS).set_index("intersection_id"),
        "edges": pd.read_parquet(p / AADT_EDGES),
        "static": pd.read_parquet(p / STATIC),
        "names": pd.read_parquet(p / INTERSECTIONS).set_index("intersection_id")["street_names"],
        "proxies": pd.read_parquet(p / PROXIES).set_index("intersection_id"),
        "risk": pd.read_parquet(p / INTERSECTION_RISK).set_index("intersection_id"),
        "segments": segment_table(p),
        "inter_xy": pd.read_parquet(
            p / INTERSECTIONS, columns=["intersection_id", "x", "y"]
        ).set_index("intersection_id"),
        "node_map": pd.read_parquet(p / NODE_MAP),
        "mto": pd.read_parquet(p / MTO_SECTIONS),
    }


def segment_table(processed: Path) -> pd.DataFrame:
    seg = gpd.read_parquet(
        processed / SEGMENTS, columns=["segment_id", "name", "length", "geometry"]
    )
    mid = seg.geometry.interpolate(0.5, normalized=True)
    return pd.DataFrame(
        {
            "segment_id": seg["segment_id"],
            "name": seg["name"],
            "length": seg["length"],
            "mid_x": mid.x,
            "mid_y": mid.y,
        }
    )


def intersection_rates(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per clean intersection: crashes/yr, count-based and final AADT, crashes per MEV."""
    a = d["aadt_i"]
    df = pd.DataFrame(index=a.index)
    df["crashes_per_year"] = d["inter_crashes"].reindex(df.index).fillna(0)
    df["class_combo"] = d["static"]["class_combo"].reindex(df.index)
    df["street_names"] = d["names"].reindex(df.index).replace("", "(unnamed)")
    df["aadt_count"] = a["aadt_count"].where(a["count_year"] >= MIN_COUNT_YEAR)
    df["count_year"] = a["count_year"]
    df["aadt"] = a["aadt"]
    df["rate_count"] = df["crashes_per_year"] / (df["aadt_count"] * 365 / 1e6)
    df["rate_all"] = df["crashes_per_year"] / (df["aadt"] * 365 / 1e6)
    proxy = d["proxies"]["exclusion_reason"].notna().reindex(df.index).fillna(False)
    return df[~proxy.to_numpy()]


def combo_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo, g in [*df.groupby("class_combo"), ("all", df)]:
        c = g[g["aadt_count"].notna()]
        mev = c["aadt_count"] * 365 / 1e6
        rows.append(
            {
                "class combo": combo,
                "counted": len(c),
                "crashes/yr": c["crashes_per_year"].sum(),
                "MEV/yr": mev.sum(),
                "pooled crashes per MEV": c["crashes_per_year"].sum() / mev.sum(),
                "median per MEV": c["rate_count"].median(),
                "median crashes/yr": c["crashes_per_year"].median(),
                "pooled per MEV, all (modelled AADT)": g["crashes_per_year"].sum()
                / (g["aadt"] * 365 / 1e6).sum(),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("pooled crashes per MEV", ascending=False)


def top_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    c = df[df["aadt_count"].notna()].copy()
    c["rank by count"] = c["crashes_per_year"].rank(ascending=False, method="min").astype(int)
    elig = c[(c["aadt_count"] >= TOP_MIN_AADT) & (c["crashes_per_year"] >= TOP_MIN_CRASHES)].copy()
    elig["rank by rate"] = elig["rate_count"].rank(ascending=False, method="min").astype(int)
    c["rank by rate"] = elig["rank by rate"].reindex(c.index)
    cols = {
        "street_names": "intersection",
        "class_combo": "class combo",
        "crashes_per_year": "crashes/yr",
        "aadt_count": "entering AADT",
        "count_year": "count year",
        "rate_count": "per MEV",
        "rank by count": "rank by count",
        "rank by rate": "rank by rate",
    }

    def fmt(t: pd.DataFrame) -> pd.DataFrame:
        t = t[list(cols)].rename(columns=cols)
        t["count year"] = t["count year"].astype(int).astype(str)
        t["rank by rate"] = t["rank by rate"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
        return t

    by_count = fmt(c.sort_values("crashes_per_year", ascending=False).head(20))
    by_rate = fmt(elig.sort_values("rate_count", ascending=False).head(20).assign(
        **{"rank by rate": elig["rank by rate"]}
    ))  # fmt: skip
    top_c = set(c.sort_values("crashes_per_year", ascending=False).head(20).index)
    top_r = set(elig.sort_values("rate_count", ascending=False).head(20).index)
    rho = spearmanr(elig["crashes_per_year"], elig["rate_count"]).statistic
    stats = {"overlap": len(top_c & top_r), "eligible": len(elig), "spearman": float(rho)}
    return by_count, by_rate, stats


def allocate(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Directed edges with MVKM/yr and TPS crashes/yr charged to them: each clean
    intersection's crashes split across its entering edges by directional AADT, plus
    mid-block crashes split across a segment's directed edges by directional AADT."""
    e = d["edges"].merge(d["segments"], on="segment_id")
    e["mvkm"] = e["aadt_dir"] * e["length"] / 1000 * 365 / 1e6
    cluster = d["node_map"].set_index("osmid")["intersection_id"]
    e["cu"], e["cv"] = e["u"].map(cluster), e["v"].map(cluster)
    entering = e["cv"].notna() & (e["cu"] != e["cv"])
    clean = d["inter_crashes"].drop(
        d["proxies"].index[d["proxies"]["exclusion_reason"].notna()], errors="ignore"
    )
    ent = e[entering]
    share = ent["aadt_dir"] / ent.groupby("cv")["aadt_dir"].transform("sum")
    e["inter_crashes"] = 0.0
    e.loc[ent.index, "inter_crashes"] = (
        ent["cv"].astype(int).map(clean).fillna(0) * share.fillna(0)
    ).to_numpy()
    seg_share = e["aadt_dir"] / e.groupby("segment_id")["aadt_dir"].transform("sum")
    e["midblock_crashes"] = e["segment_id"].map(d["seg_crashes"]).fillna(0) * seg_share
    e["crashes"] = e["inter_crashes"] + e["midblock_crashes"]
    e["expressway"] = None
    for pattern, label in EXPRESSWAYS.items():
        hit = e["name"].fillna("").str.contains(pattern, regex=True) & (e["highway"] == "motorway")
        e.loc[hit, "expressway"] = label
    e["group"] = e["highway"].map(CLASS_GROUPS).fillna("other")
    e.loc[e["mto_hwy"].notna(), "group"] = "400-series (OPP)"
    return e


def class_table(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, g in [*e.groupby("group"), ("all", e)]:
        km = g.drop_duplicates("segment_id")["length"].sum() / 1000
        rows.append(
            {
                "road class": group,
                "km (segments)": km,
                "MVKM/yr": g["mvkm"].sum(),
                "TPS crashes/yr": g["crashes"].sum(),
                "of which at intersections": g["inter_crashes"].sum(),
                "crashes/km/yr (CP4 view)": g["crashes"].sum() / km,
                "crashes per MVKM": g["crashes"].sum() / g["mvkm"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("crashes per MVKM", ascending=False)


def expressway_proxy_excess(d: dict[str, pd.DataFrame], e: pd.DataFrame) -> pd.Series:
    """Expressway-proxy intersections' crashes above their SPF expectation, per
    year, attributed to the nearest named expressway segment."""
    p = d["proxies"]
    ids = p.index[p["exclusion_reason"] == "expressway_proxy"]
    actual = d["inter_crashes"].reindex(ids).fillna(0)
    excess = (actual - d["risk"]["spf"].reindex(ids)).clip(lower=0)
    ex = e[e["expressway"].notna()].drop_duplicates("segment_id")
    tree = cKDTree(np.column_stack([ex["mid_x"], ex["mid_y"]]))
    _, idx = tree.query(d["inter_xy"].loc[ids, ["x", "y"]].to_numpy())
    return excess.groupby(ex["expressway"].to_numpy()[idx]).sum()


def gardiner_surface(processed: Path, d: dict[str, pd.DataFrame], max_m: float = 150) -> dict:
    """Lake Shore Blvd intersections under or beside the Gardiner and their crashes: TPS
    appears to place Gardiner crashes there, the way it places DVP crashes at nearby side
    streets (the expressway proxies), but the CP3 rule did not flag them."""
    seg = gpd.read_parquet(processed / SEGMENTS, columns=["name", "highway", "geometry"])
    gardiner = seg[
        seg["name"].fillna("").str.startswith("Gardiner Expressway")
        & (seg["highway"] == "motorway")
    ].union_all()
    inter = gpd.read_parquet(processed / INTERSECTIONS)
    near = inter[
        (inter.geometry.distance(gardiner) <= max_m)
        & inter["street_names"].str.contains("Lake Shore Boulevard")
    ]
    crashes = d["inter_crashes"].reindex(near["intersection_id"]).fillna(0)
    return {
        "max_m": max_m,
        "n": len(near),
        "crashes_per_year": float(crashes.sum()),
        "ids": set(near["intersection_id"]),
    }


def expressway_table(d: dict[str, pd.DataFrame], e: pd.DataFrame) -> pd.DataFrame:
    proxy = expressway_proxy_excess(d, e)
    mto = d["mto"][d["mto"]["mapped_km"] > 0]
    pooled = mto.groupby("hwy")[["collisions_per_year", "mvkm_per_year"]].sum()
    rows = []
    for label in EXPRESSWAYS.values():
        g = e[e["expressway"] == label]
        if g.empty:
            continue
        tps = g["crashes"].sum() + proxy.get(label, 0.0)
        row = {
            "expressway": label,
            "carriageway km": g["length"].sum() / 1000,
            "MVKM/yr (our AADT)": g["mvkm"].sum(),
            "TPS crashes/yr (mid-block + ramp merges + proxy excess)": tps,
            "TPS per MVKM": tps / g["mvkm"].sum(),
        }
        if label in pooled.index:
            row["MTO collisions/yr (2015-19)"] = pooled.loc[label, "collisions_per_year"]
            row["MTO per MVKM (2015-19)"] = (
                pooled.loc[label, "collisions_per_year"] / pooled.loc[label, "mvkm_per_year"]
            )
        else:
            model_mvkm = (g["aadt_dir_model"] * g["length"] / 1000 * 365 / 1e6).sum()
            row["TPS per MVKM if model AADT"] = tps / model_mvkm if model_mvkm > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(cfg: dict[str, Any], d: dict[str, pd.DataFrame], path: Path) -> dict[str, Any]:
    df = intersection_rates(d)
    combos = combo_table(df)
    by_count, by_rate, top = top_tables(df)
    e = allocate(d)
    classes = class_table(e)
    express = expressway_table(d, e)
    surface = gardiner_surface(Path(cfg["paths"]["processed_dir"]), d)
    in_top = int(by_rate.index.isin(surface["ids"]).sum())
    gx = express.set_index("expressway").loc["Gardiner"]
    surface["upper_rate"] = (
        gx["TPS crashes/yr (mid-block + ramp merges + proxy excess)"] + surface["crashes_per_year"]
    ) / gx["MVKM/yr (our AADT)"]
    cls = classes.set_index("road class")["crashes per MVKM"]
    ll = combos.set_index("class combo")["pooled crashes per MEV"]
    lines = [
        "# Exposure sanity checks (CP5)",
        "",
        "What happens to TPS crash counts when they are divided by traffic. Crashes: TPS, "
        f"label years {YEARS[0]}-2019 and 2022-{YEARS[-1]} (2020-21 left out), per year. "
        "Volumes: reports/volumes.md (counts), reports/volume_model.md (model), reports/mto.md "
        "(400-series). Proxy intersections (CP3) are excluded from sections 1-3.",
        "",
        "## 1. Crashes per million entering vehicles, counted intersections",
        "",
        f"Intersections with a City count from {MIN_COUNT_YEAR} or later (count-based AADT, no "
        "model). MEV = entering AADT x 365 / 1e6. The last column repeats the pooled rate over "
        "every intersection using the final (blended / modelled) AADT.",
        "",
        markdown_table(combos, floats="{:,.3f}"),
        "",
        "## 2. Does dividing by volume reorder the worst intersections?",
        "",
        f"Rate ranking among counted intersections with entering AADT >= {TOP_MIN_AADT:,} and "
        f">= {TOP_MIN_CRASHES:.0f} crashes/yr ({top['eligible']:,} intersections). "
        f"**{top['overlap']} of the top 20** by count are also in the top 20 by rate; "
        f"Spearman between crash count and crash rate among eligible intersections: "
        f"{top['spearman']:.2f}.",
        "",
        "Top 20 by crashes per year:",
        "",
        markdown_table(by_count, floats="{:,.2f}"),
        "",
        "Top 20 by crashes per million entering vehicles:",
        "",
        markdown_table(by_rate, floats="{:,.2f}"),
        "",
        "## 3. Crashes per million vehicle-km by road class (whole network)",
        "",
        "Each clean intersection's crashes are split across the edges entering it in "
        "proportion to their directional AADT; mid-block crashes stay on their segment. "
        "Vehicle-km use the final AADT on every directed edge (counts, model, MTO, "
        "lane-scaled). The per-km column is the CP4 view (no exposure) for contrast.",
        "",
        markdown_table(classes, floats="{:,.2f}"),
        "",
        "## 4. Expressways: MTO vs TPS",
        "",
        "TPS crashes on an expressway = mid-block crashes on its segments, plus intersection "
        "crashes charged to its edges at ramp-merge clusters, plus the excess (actual - "
        "SPF-expected) at expressway-proxy intersections attributed to the nearest "
        "expressway. MTO rates are reportable collisions per MVKM from MTO's own volumes "
        "(reports/mto.md). DVP and Gardiner vehicle-km use the lane-scaled AADT; the last "
        "column shows the rate if the uncorrected model AADT were used instead.",
        "",
        markdown_table(express, floats="{:,.3f}"),
        "",
        f"The Gardiner's TPS rate is implausibly low. {surface['n']} Lake Shore Blvd "
        f"intersections lie within {surface['max_m']:.0f} m of it and carry "
        f"{surface['crashes_per_year']:,.0f} crashes/yr; {in_top} of them are in the top 20 "
        "by rate above. TPS appears to place many Gardiner crashes at the surface intersections "
        "beneath it, as it places DVP crashes at side streets, but the CP3 expressway-proxy "
        "rule (residential corners far above their class median) does not catch arterial "
        "intersections. If every one of those crashes were a Gardiner crash the rate would be "
        f"{surface['upper_rate']:.2f} per MVKM; the truth lies between the two.",
        "",
        "## What this says",
        "",
    ]
    findings = conclusions(combos, classes, express, top, cls, ll, surface)
    lines.extend(findings)
    path.write_text("\n".join(lines) + "\n")
    return {"combos": combos, "classes": classes, "express": express, "top": top}


def conclusions(
    combos: pd.DataFrame,
    classes: pd.DataFrame,
    express: pd.DataFrame,
    top: dict[str, float],
    cls: pd.Series,
    ll: pd.Series,
    surface: dict[str, float],
) -> list[str]:
    x = express.set_index("expressway")
    out = [
        f"- **Residential vs arterial.** Per kilometre (CP4 view) primary roads carry "
        f"{classes.set_index('road class').loc['primary', 'crashes/km/yr (CP4 view)']:.0f} "
        "crashes/km/yr against "
        f"{classes.set_index('road class').loc['residential', 'crashes/km/yr (CP4 view)']:.1f} "
        "on residential streets. Per vehicle-km the gap shrinks to "
        f"{cls['primary']:.2f} vs {cls['residential']:.2f} crashes per MVKM "
        f"({cls['primary'] / cls['residential']:.1f}x instead of "
        f"{classes.set_index('road class').loc['primary', 'crashes/km/yr (CP4 view)'] / classes.set_index('road class').loc['residential', 'crashes/km/yr (CP4 view)']:.0f}x). "  # noqa: E501
        "Local x local intersections have "
        f"{ll.get('local x local', np.nan):.2f} crashes per MEV vs "
        f"{ll.get('arterial x arterial', np.nan):.2f} at arterial x arterial ones. A "
        "residential cut-through is therefore not the near-free option count-based risk "
        "made it look like.",
        f"- **Freeways.** MTO's 400-series rate in Toronto is "
        f"{x.loc[['401', '400', '404', '427'], 'MTO per MVKM (2015-19)'].min():.2f}-"
        f"{x.loc[['401', '400', '404', '427'], 'MTO per MVKM (2015-19)'].max():.2f} collisions "
        "per MVKM, while TPS records "
        f"{x.loc['401', 'TPS per MVKM']:.3f} per MVKM on the 401, i.e. about "
        f"{100 * x.loc['401', 'TPS per MVKM'] / x.loc['401', 'MTO per MVKM (2015-19)']:.0f}% of "
        "the collisions MTO counts. With MTO's rates the 401 is "
        f"{cls['primary'] / x.loc['401', 'MTO per MVKM (2015-19)']:.0f}x safer per vehicle-km "
        f"than primary arterials ({cls['primary']:.2f} per MVKM); CP4's per-km TPS risk put the "
        "gap at ~40x (reports/risk.md, risk per directed km 76 vs 1.9). Freeways are safer per "
        "kilometre driven, but not nearly free.",
        f"- **DVP** (TPS jurisdiction): {x.loc['DVP', 'TPS per MVKM']:.2f} crashes per MVKM "
        f"with lane-scaled AADT ({x.loc['DVP', 'TPS per MVKM if model AADT']:.2f} with the "
        "model's AADT), the same order as MTO's 400-series rates, mostly through the "
        "expressway-proxy excess. **Gardiner**: "
        f"{x.loc['Gardiner', 'TPS per MVKM']:.2f} as recorded, up to "
        f"{surface['upper_rate']:.2f} if its likely crashes at Lake Shore Blvd are counted "
        "(section 4). CP6 should attribute those, or give the city expressways an MTO-like "
        "rate.",
        f"- **Hotspot lists.** Normalising keeps {top['overlap']} of the top 20; the rate list "
        "adds lower-volume intersections with many crashes for their traffic. Rates at "
        "low-volume intersections are noisy, which is what the Empirical Bayes step in CP6 "
        "is for.",
    ]
    return out


def run_exposure(cfg: dict[str, Any]) -> dict[str, Any]:
    d = load(cfg)
    out = write_report(cfg, d, Path(cfg["paths"]["reports_dir"]) / "exposure.md")
    log.info("exposure report written")
    return out
