"""Per-vehicle crash risk for every directed edge of the routing graph (CP6).

Unit: expected crashes for one vehicle, stored per million vehicles so that a route's sum is
expected crashes per million trips.

- Intersections: Empirical Bayes expected crashes in 2026 / (entering AADT x 365), charged
  once per consolidated intersection on the edges entering it. Proxy intersections (CP3)
  use the SPF alone, since their own counts belong elsewhere.
- City segments: mid-block crashes per million vehicle-km (MVKM) by road class x length.
- 400-series: MTO's section collision rate (2015-2019) x length.
- Gardiner and DVP: TPS crashes attributed to them per MVKM x length. TPS places many
  expressway crashes at surface intersections beside or under the expressway, so for every
  surface intersection within ATTRIBUTION_M of the Gardiner or DVP mainline the crashes in
  excess of its SPF expectation (observed - expected, floored at 0) move to the expressway,
  and the intersection keeps the rest. CP3 expressway proxies add their excess the same way.
  Other city expressways (Allen Road, ...) use their own TPS crashes per MVKM.

Rates use the crash years RATE_YEARS (recent, after the 2020-21 dip) with the CP5 volumes.
Outputs: data/processed/routing_graph.pkl, intersection_risk.parquet, edge_risk.parquet,
risk_stats.json; reports/risk.md.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from safepath.data.mto import MTO_SEGMENTS
from safepath.features.build import PROXIES
from safepath.features.labels import LABELS
from safepath.features.roads import approaches
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.graph import EDGE_MAP, INTERSECTIONS, NODE_MAP, PROJ_GRAPH, SEGMENTS
from safepath.locations import markdown_table
from safepath.models.eb import EB_PREDICTIONS, eb_for_year
from safepath.models.spf import SPF_PREDICTIONS, year_levels
from safepath.models.volume import AADT_EDGES
from safepath.snapping import ASSIGNMENTS

log = logging.getLogger(__name__)

DEPLOY_YEAR = 2026
ROUTING_GRAPH = "routing_graph.pkl"
INTERSECTION_RISK = "intersection_risk.parquet"
EDGE_RISK = "edge_risk.parquet"
RISK_STATS = "risk_stats.json"
RATE_YEARS = [2022, 2023, 2024, 2025]
ATTRIBUTION_M = 150
ATTRIBUTED = {"Gardiner": r"^Gardiner Expressway\b", "DVP": r"^Don Valley Parkway\b"}
CITY_EXPRESSWAYS = {**ATTRIBUTED, "Allen Road": r"^Allen Road\b"}
MTO_RANGE = (0.5, 1.1)  # province's Toronto 400-series rates, per MVKM (reports/mto.md)
TOP_MIN_AADT = 5_000
PER_MILLION = 1e6


# --------------------------------------------------------------------------- attribution


def expressway_label(name: pd.Series, highway: pd.Series, patterns: dict[str, str]) -> pd.Series:
    out = pd.Series(None, index=name.index, dtype=object)
    for label, pattern in patterns.items():
        hit = name.fillna("").str.contains(pattern, regex=True) & (highway == "motorway")
        out[hit] = label
    return out


def surface_candidates(
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    node_map: pd.DataFrame,
    excluded: pd.Index,
    max_m: float = ATTRIBUTION_M,
) -> pd.DataFrame:
    """Surface intersections (no expressway-mainline approach, not a CP3 proxy) within
    `max_m` of the Gardiner or DVP mainline, with the nearer expressway."""
    seg = segments.assign(expressway=expressway_label(segments["name"], segments["highway"],
                                                      ATTRIBUTED))  # fmt: skip
    appr = approaches(segments, node_map)
    on_mainline = appr.loc[appr["highway"] == "motorway", "intersection_id"].unique()
    pts = intersections[
        ~intersections["intersection_id"].isin(on_mainline)
        & ~intersections["intersection_id"].isin(excluded)
    ]
    dist = pd.DataFrame(index=pts["intersection_id"])
    for label in ATTRIBUTED:
        mainline = seg.loc[seg["expressway"] == label].union_all()
        dist[label] = pts.geometry.distance(mainline).to_numpy()
    near = dist.min(axis=1) <= max_m
    out = pd.DataFrame({"expressway": dist[near].idxmin(axis=1), "dist_m": dist[near].min(axis=1)})
    return out.rename_axis("intersection_id").reset_index()


def attribute_excess(
    labels: pd.DataFrame,
    base: pd.Series,
    levels: pd.Series,
    candidates: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """Per candidate: observed and SPF-expected crashes per year over `years`, the excess
    moved to the expressway, and the share of its crashes the intersection keeps."""
    ids = candidates["intersection_id"]
    obs = (
        labels[labels["year"].isin(years) & labels["intersection_id"].isin(ids)]
        .groupby("intersection_id")["crash_count"]
        .sum()
        .reindex(ids)
        .fillna(0)
    )
    expected = base.reindex(ids) * levels.reindex(years).sum()
    excess = (obs - expected).clip(lower=0)
    out = candidates.set_index("intersection_id").copy()
    n = len(years)
    out["observed_per_year"] = obs / n
    out["spf_expected_per_year"] = expected / n
    out["reassigned_per_year"] = excess / n
    out["keep_share"] = np.where(obs > 0, 1 - excess / obs.where(obs > 0, 1), 1.0)
    return out.reset_index()


def adjusted_labels(labels: pd.DataFrame, attribution: pd.DataFrame) -> pd.DataFrame:
    """Crash counts with each attributed intersection scaled to the share it keeps."""
    keep = attribution.set_index("intersection_id")["keep_share"]
    out = labels.copy()
    share = out["intersection_id"].map(keep).fillna(1.0)
    out["crash_count"] = out["crash_count"] * share
    return out


def proxy_excess(
    labels: pd.DataFrame,
    base: pd.Series,
    levels: pd.Series,
    proxies: pd.DataFrame,
    intersections: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    years: list[int],
) -> pd.DataFrame:
    """CP3 expressway proxies: excess over the SPF per year, assigned to the nearest
    named expressway (400-series ones are dropped later: MTO counts those crashes)."""
    ids = proxies.index[proxies["exclusion_reason"] == "expressway_proxy"]
    patterns = {**CITY_EXPRESSWAYS, "400-series": r"^(?:Highway 4\d\d|Queen Elizabeth Way)\b"}
    seg = segments.assign(expressway=expressway_label(segments["name"], segments["highway"],
                                                      patterns))  # fmt: skip
    seg = seg[seg["expressway"].notna()]
    pts = intersections.set_index("intersection_id").loc[ids]
    nearest = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=pts.geometry.values, index=ids),
        seg[["expressway", "geometry"]],
    )
    nearest = nearest[~nearest.index.duplicated()]
    cand = pd.DataFrame({"intersection_id": ids, "expressway": nearest["expressway"].reindex(ids)})
    cand["dist_m"] = np.nan
    return attribute_excess(labels, base, levels, cand, years)


# --------------------------------------------------------------------------- rates


def edge_table(processed: Path, segments: gpd.GeoDataFrame) -> pd.DataFrame:
    edges = pd.read_parquet(processed / AADT_EDGES)
    seg = segments.set_index("segment_id")
    edges["length_m"] = edges["segment_id"].map(seg["length"])
    edges["name"] = edges["segment_id"].map(seg["name"])
    edges["mvkm"] = edges["aadt_dir"] * edges["length_m"] / 1000 * 365 / PER_MILLION
    edges["expressway"] = expressway_label(edges["name"], edges["highway"], CITY_EXPRESSWAYS)
    return edges


def midblock_per_year(assignments: pd.DataFrame, years: list[int]) -> pd.Series:
    mid = assignments[(assignments["unit_type"] == "segment") & assignments["year"].isin(years)]
    return mid.groupby(mid["unit_id"].astype(int)).size() / len(years)


def edge_rates(
    edges: pd.DataFrame,
    midblock: pd.Series,
    mto: pd.DataFrame,
    expressway_extra: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crashes per MVKM for every directed edge, and the table of rates by source.

    Mid-block crashes of a segment are shared between its directed edges by directional
    AADT. City classes pool crashes and MVKM by highway class; city expressways by name
    (plus their attributed and proxy excess); 400-series edges take their MTO section rate."""
    e = edges.copy()
    share = e["aadt_dir"] / e.groupby("segment_id")["aadt_dir"].transform("sum")
    e["midblock"] = e["segment_id"].map(midblock).fillna(0) * share
    is_mto = e["mto_lhrs"].notna()
    city_xway = e["expressway"].notna() & ~is_mto
    other_mw = (e["highway"] == "motorway") & ~is_mto & ~city_xway
    e["rate_group"] = np.select(
        [is_mto, city_xway, other_mw],
        ["400-series (MTO)", "city expressway: " + e["expressway"].astype(str), "other expressway"],
        e["highway"].astype(str),
    )
    groups = (
        e[~is_mto]
        .groupby("rate_group")
        .agg(
            km=("length_m", lambda x: x.sum() / 1000),
            midblock=("midblock", "sum"),
            mvkm=("mvkm", "sum"),
        )
    )
    groups["attributed"] = 0.0
    for label, extra in expressway_extra.items():
        key = f"city expressway: {label}"
        if key in groups.index:
            groups.loc[key, "attributed"] = extra
    groups["crashes_per_year"] = groups["midblock"] + groups["attributed"]
    groups["rate_per_mvkm"] = groups["crashes_per_year"] / groups["mvkm"]
    e["rate_per_mvkm"] = e["rate_group"].map(groups["rate_per_mvkm"])
    cr = mto.set_index("segment_id")["cr"]
    e.loc[is_mto, "rate_per_mvkm"] = e.loc[is_mto, "segment_id"].map(cr)
    mto_rows = e[is_mto].assign(crashes=lambda d: d["rate_per_mvkm"] * d["mvkm"])
    groups.loc["400-series (MTO)"] = {
        "km": mto_rows["length_m"].sum() / 1000,
        "midblock": np.nan,
        "mvkm": mto_rows["mvkm"].sum(),
        "attributed": np.nan,
        "crashes_per_year": mto_rows["crashes"].sum(),
        "rate_per_mvkm": mto_rows["crashes"].sum() / mto_rows["mvkm"].sum(),
    }
    e["rate_per_mvkm"] = e["rate_per_mvkm"].fillna(0.0)
    return e, groups.sort_values("rate_per_mvkm", ascending=False)


# --------------------------------------------------------------------------- intersections


def intersection_risk(
    labels: pd.DataFrame,
    attribution: pd.DataFrame,
    spf: pd.DataFrame,
    exposure: pd.DataFrame,
    window: int,
    levels: pd.Series,
) -> pd.DataFrame:
    """EB 2026 on attributed crash history; SPF alone for proxies; per million entering
    vehicles. Citywide levels come from the unadjusted labels: attribution moves crashes,
    it does not change how many happened."""
    base, k = spf["spf_base"], spf["k"]
    adj = adjusted_labels(labels, attribution)
    eb = eb_for_year(base, k, adj, levels, DEPLOY_YEAR, window).set_index("intersection_id")
    raw = eb_for_year(base, k, labels, levels, DEPLOY_YEAR, window).set_index("intersection_id")
    excluded = labels.groupby("intersection_id")["exclusion_reason"].first()
    out = pd.DataFrame(index=base.index.rename("intersection_id"))
    out["eb_unattributed"] = raw["eb"]
    out["exclusion_reason"] = excluded.reindex(out.index)
    out["is_proxy_fallback"] = out["exclusion_reason"].notna()
    out["risk"] = np.where(out["is_proxy_fallback"], eb["spf"], eb["eb"])  # crashes/yr
    out["spf"] = eb["spf"]
    out["entering_aadt"] = exposure["entering_aadt"].reindex(out.index)
    out["risk_per_mev"] = out["risk"] / (out["entering_aadt"] * 365 / PER_MILLION)
    out["attributed_share"] = 1 - attribution.set_index("intersection_id")["keep_share"].reindex(
        out.index
    ).fillna(1.0)
    return out.reset_index()


def attach_risk(
    G: nx.MultiDiGraph,
    node_map: pd.DataFrame,
    edge_map: pd.DataFrame,
    risk_per_mev: pd.Series,
    edge_rate: pd.DataFrame,
) -> pd.DataFrame:
    """Write intersection_risk (per million vehicles entering), segment_risk (rate x km, per
    million vehicles) and risk onto every directed edge of G."""
    cluster = node_map.set_index("osmid")["intersection_id"]
    edges = edge_map.merge(
        edge_rate[["u", "v", "key", "rate_per_mvkm", "rate_group"]],
        on=["u", "v", "key"],
        how="left",
    )
    edges["cu"] = edges["u"].map(cluster)
    edges["cv"] = edges["v"].map(cluster)
    enters = edges["cv"].notna() & (edges["cu"] != edges["cv"])
    edges["intersection_risk"] = np.where(enters, edges["cv"].map(risk_per_mev), 0.0)
    edges["entered_intersection"] = edges["cv"].where(enters)
    edges["length_m"] = [
        G.edges[u, v, k]["length"] for u, v, k in edges[["u", "v", "key"]].itertuples(index=False)
    ]
    edges["segment_risk"] = edges["rate_per_mvkm"].fillna(0.0) * edges["length_m"] / 1000
    edges["risk"] = edges["intersection_risk"] + edges["segment_risk"]
    if edges["risk"].isna().any():
        raise ValueError("edge risk has NaN values")
    for row in edges.itertuples(index=False):
        data = G.edges[row.u, row.v, row.key]
        data["intersection_risk"] = float(row.intersection_risk)
        data["segment_risk"] = float(row.segment_risk)
        data["risk"] = float(row.risk)
        data["entered_intersection"] = (
            int(row.entered_intersection) if not np.isnan(row.entered_intersection) else -1
        )
    return edges


# --------------------------------------------------------------------------- report


def top_rate_ids(risk: pd.DataFrame, column: str, n: int = 20) -> set[int]:
    r = risk[~risk["is_proxy_fallback"] & (risk["entering_aadt"] >= TOP_MIN_AADT)]
    rate = r[column] / (r["entering_aadt"] * 365 / PER_MILLION)
    return set(r.loc[rate.sort_values(ascending=False).index[:n], "intersection_id"])


def per_trip_km(edges: pd.DataFrame, highway: pd.Series) -> pd.DataFrame:
    """Risk a vehicle accumulates per km driven on each road class (each directed edge
    counted once, i.e. what the router sees), split into intersections and segments."""
    e = edges.assign(cls=highway.reindex(edges.index).to_numpy())
    e.loc[e["rate_group"] == "400-series (MTO)", "cls"] = "400-series"
    g = e.groupby("cls").agg(
        km=("length_m", lambda x: x.sum() / 1000),
        inter=("intersection_risk", "sum"),
        seg=("segment_risk", "sum"),
    )
    g = g[g["km"] >= 5]
    g["intersection risk per km"] = g["inter"] / g["km"]
    g["segment risk per km"] = g["seg"] / g["km"]
    g["total per km"] = g["intersection risk per km"] + g["segment risk per km"]
    g["intersections per km"] = (
        e[e["intersection_risk"] > 0].groupby("cls").size().reindex(g.index) / g["km"]
    )
    return (
        g.drop(columns=["inter", "seg"])
        .sort_values("total per km", ascending=False)
        .rename_axis("road class")
        .reset_index()
    )


def attribution_table(
    groups: pd.DataFrame, attribution: pd.DataFrame, proxies: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for label in CITY_EXPRESSWAYS:
        key = f"city expressway: {label}"
        if key not in groups.index:
            continue
        g = groups.loc[key]
        a = attribution[attribution["expressway"] == label]
        p = proxies[proxies["expressway"] == label]
        before = (g["midblock"] + p["reassigned_per_year"].sum()) / g["mvkm"]
        rows.append(
            {
                "expressway": label,
                "surface intersections within 150 m": len(a),
                "with excess": int((a["reassigned_per_year"] > 0).sum()),
                "reassigned crashes/yr": a["reassigned_per_year"].sum(),
                "proxy excess/yr": p["reassigned_per_year"].sum(),
                "mid-block crashes/yr": g["midblock"],
                "MVKM/yr": g["mvkm"],
                "rate before attribution": before,
                "rate after": g["rate_per_mvkm"],
                f"in MTO range {MTO_RANGE[0]}-{MTO_RANGE[1]}": "yes"
                if MTO_RANGE[0] <= g["rate_per_mvkm"] <= MTO_RANGE[1]
                else "no",
            }
        )
    return pd.DataFrame(rows)


def write_report(r: dict[str, Any], path: Path) -> None:
    risk, groups = r["risk"], r["groups"]
    names = r["names"]
    attr = r["attribution"]
    top = risk[~risk["is_proxy_fallback"] & (risk["entering_aadt"] >= TOP_MIN_AADT)].copy()
    top = top.sort_values("risk_per_mev", ascending=False).head(20)
    top["intersection"] = top["intersection_id"].map(names)
    city = groups[~groups.index.str.startswith(("city expressway", "400-series"))]
    lines = [
        "# Risk scoring (CP6: per-vehicle risk, 2026)",
        "",
        "Every directed edge carries the expected number of crashes for one vehicle that "
        "drives it, stored per million vehicles, so a route's total is expected crashes per "
        "million trips. Intersection risk is charged once, on the edge entering the "
        "consolidated intersection; segment risk is a rate per million vehicle-km (MVKM) x "
        f"length. Crash years for rates: {RATE_YEARS[0]}-{RATE_YEARS[-1]}.",
        "",
        "## Gardiner / DVP attribution",
        "",
        f"Surface intersections within {ATTRIBUTION_M} m of the Gardiner or DVP mainline "
        "(not expressway clusters, not CP3 proxies): crashes above the SPF expectation over "
        f"{RATE_YEARS[0]}-{RATE_YEARS[-1]} move to the expressway; the intersection keeps "
        "the rest (its EB is recomputed on the reduced history). CP3 expressway proxies add "
        "their excess over the SPF to the nearest expressway (excess at 400-series proxies is "
        "dropped: MTO already counts those crashes).",
        "",
        markdown_table(attribution_table(groups, attr, r["proxy_excess"]), floats="{:,.2f}"),
        "",
        f"Rate top 20 (EB per million entering vehicles, entering AADT >= {TOP_MIN_AADT:,}): "
        f"{r['attributed_in_top_before']} attributed intersections before attribution, "
        f"{r['attributed_in_top_after']} after; "
        f"{r['lake_shore_dropped']} Lake Shore Blvd intersections dropped out.",
        "",
        "Largest reassignments:",
        "",
        markdown_table(
            attr.sort_values("reassigned_per_year", ascending=False)
            .head(10)
            .assign(intersection=lambda d: d["intersection_id"].map(names))[
                [
                    "intersection",
                    "expressway",
                    "dist_m",
                    "observed_per_year",
                    "spf_expected_per_year",
                    "reassigned_per_year",
                ]
            ],  # fmt: skip
            floats="{:,.1f}",
        ),
        "",
        "## City segment rates (mid-block crashes per MVKM)",
        "",
        markdown_table(
            city.reset_index()[["rate_group", "km", "midblock", "mvkm", "rate_per_mvkm"]],
            floats="{:,.3f}",
        ),
        "",
        "TPS assigns 96% of crashes to intersections, so these mid-block rates are small "
        "next to intersection risk; they matter mainly on long links.",
        "",
        "## 400-series (MTO section rates, 2015-2019)",
        "",
        markdown_table(r["mto_by_hwy"], floats="{:,.3f}"),
        "",
        "## Risk per trip-km by road class",
        "",
        "Expected crashes per million vehicles per km driven on that class, with the "
        "intersections a vehicle enters along the way. This is the ordering the router sees.",
        "",
        markdown_table(r["per_km"], floats="{:,.3f}"),
        "",
        "## Intersections",
        "",
        f"- {len(risk):,} intersections; EB {DEPLOY_YEAR} expected crashes/yr total "
        f"{risk['risk'].sum():,.0f} (median {risk['risk'].median():.2f}). "
        f"{int(risk['is_proxy_fallback'].sum())} proxy intersections use the SPF alone.",
        f"- Crashes per million entering vehicles: median {risk['risk_per_mev'].median():.2f}, "
        f"p90 {risk['risk_per_mev'].quantile(0.9):.2f}, p99 "
        f"{risk['risk_per_mev'].quantile(0.99):.2f}.",
        "",
        "Highest per-vehicle risk (entering AADT >= 5,000), after attribution:",
        "",
        markdown_table(
            top[["intersection", "risk", "entering_aadt", "risk_per_mev", "attributed_share"]],
            floats="{:,.2f}",
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def run_score(cfg: dict[str, Any]) -> Path:
    processed = Path(cfg["paths"]["processed_dir"])
    labels = pd.read_parquet(processed / LABELS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    node_map = pd.read_parquet(processed / NODE_MAP)
    proxies = pd.read_parquet(processed / PROXIES).set_index("intersection_id")
    spf = pd.read_parquet(processed / SPF_PREDICTIONS).drop_duplicates("intersection_id")
    spf = spf.set_index("intersection_id")
    exposure = pd.read_parquet(processed / INTERSECTION_EXPOSURE).set_index("intersection_id")
    window = int(pd.read_parquet(processed / EB_PREDICTIONS, columns=["window"])["window"].iloc[0])
    levels = year_levels(labels, labels["intersection_id"].map(spf["spf_base"]))

    excluded = proxies.index[proxies["exclusion_reason"].notna()]
    cand = surface_candidates(intersections, segments, node_map, excluded)
    attribution = attribute_excess(labels, spf["spf_base"], levels, cand, RATE_YEARS)
    pex = proxy_excess(labels, spf["spf_base"], levels, proxies, intersections, segments,
                       RATE_YEARS)  # fmt: skip
    extra = (
        attribution.groupby("expressway")["reassigned_per_year"].sum()
        .add(pex.groupby("expressway")["reassigned_per_year"].sum(), fill_value=0)
    )  # fmt: skip
    extra = extra[extra.index.isin(CITY_EXPRESSWAYS)]

    risk = intersection_risk(labels, attribution, spf, exposure, window, levels)
    if risk["risk_per_mev"].isna().any() or (risk["risk_per_mev"] < 0).any():
        raise ValueError("intersection risk per vehicle has NaN or negative values")
    risk.to_parquet(processed / INTERSECTION_RISK, index=False)

    edges = edge_table(processed, segments)
    assignments = pd.read_parquet(processed / ASSIGNMENTS)
    e, groups = edge_rates(edges, midblock_per_year(assignments, RATE_YEARS),
                           pd.read_parquet(processed / MTO_SEGMENTS), extra)  # fmt: skip
    e[["u", "v", "key", "segment_id", "rate_group", "rate_per_mvkm", "mvkm"]].to_parquet(
        processed / EDGE_RISK, index=False
    )
    G = ox.load_graphml(processed / PROJ_GRAPH)
    attached = attach_risk(
        G, node_map, pd.read_parquet(processed / EDGE_MAP),
        risk.set_index("intersection_id")["risk_per_mev"], e,
    )  # fmt: skip
    out = processed / ROUTING_GRAPH
    with open(out, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    names = intersections.set_index("intersection_id")["street_names"].replace("", "(unnamed)")
    before = top_rate_ids(risk.assign(risk=risk["eb_unattributed"]), "risk")
    after = top_rate_ids(risk, "risk")
    attributed_ids = set(attribution.loc[attribution["reassigned_per_year"] > 0, "intersection_id"])
    lake_shore = {i for i in before - after if "Lake Shore" in str(names.get(i, ""))}
    mto_e = e[e["rate_group"] == "400-series (MTO)"]
    mto_by_hwy = (
        mto_e.assign(crashes=mto_e["rate_per_mvkm"] * mto_e["mvkm"])
        .groupby("mto_hwy")[["mvkm", "crashes"]].sum()
        .assign(rate_per_mvkm=lambda d: d["crashes"] / d["mvkm"])
        .rename(columns={"crashes": "MTO collisions/yr at our AADT"})
        .reset_index()
    )  # fmt: skip
    seg_class = segments.set_index("segment_id")["highway"]
    r = {
        "risk": risk,
        "groups": groups,
        "attribution": attribution,
        "proxy_excess": pex,
        "names": names,
        "attributed_in_top_before": len(before & attributed_ids),
        "attributed_in_top_after": len(after & attributed_ids),
        "lake_shore_dropped": len(lake_shore),
        "mto_by_hwy": mto_by_hwy,
        "per_km": per_trip_km(attached, attached["segment_id"].map(seg_class)),
    }
    write_report(r, Path(cfg["paths"]["reports_dir"]) / "risk.md")
    stats = {
        "rates_per_mvkm": groups["rate_per_mvkm"].to_dict(),
        "attributed_per_year": extra.to_dict(),
        "reassigned_intersections": int((attribution["reassigned_per_year"] > 0).sum()),
        "intersection_eb_total": float(risk["risk"].sum()),
        "per_trip_km": r["per_km"].to_dict(orient="records"),
    }
    (processed / RISK_STATS).write_text(json.dumps(stats, indent=2, default=float))
    log.info("per-vehicle risk on %d edges; wrote %s", len(attached), out)
    return out
