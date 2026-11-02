"""London risk, routing graph, 200-trip evaluation and the published top-10 check (CP10).

There are no public London collision records, so London's intersection risk is Toronto's
SPF (negative binomial on major / minor approach volume, signal, legs, leg angle, road
classes; no crash history, no EB) on London's OSM features and City volumes, at Toronto's
2025 citywide level. That is the setting that transferred best to Ottawa without local
crash data (count-free SPF recall@5% 0.619 vs 0.288 for a road-class baseline). Absolute
numbers inherit Toronto's reporting level and are not calibrated for London: the app shows
them as relative risk.

Per-vehicle risk is computed exactly as Toronto's: SPF expected crashes / (entering AADT x
365), charged on the edges entering each consolidated intersection; segments take
Toronto's crashes per million vehicle-km for their road class; Hwy 401 / 402 their MTO
section rate.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from safepath.data.aadt import street_key
from safepath.data.mto import map_sections
from safepath.features.build import STATIC
from safepath.graph import EDGE_MAP, INTERSECTIONS, NODE_MAP, PROJ_GRAPH, SEGMENTS
from safepath.london.volumes import entering_check, run_london_volumes
from safepath.models.spf import SPF_MODEL, SPF_PREDICTIONS, predict_base, spf_covariates
from safepath.risk import (
    EDGE_RISK,
    INTERSECTION_RISK,
    PER_MILLION,
    RISK_STATS,
    ROUTING_GRAPH,
    attach_risk,
)
from safepath.routing import (
    ROUTING_K,
    Router,
    calibrate_k,
    random_pairs,
    summarise,
    tradeoff_table,
)

log = logging.getLogger(__name__)

LONDON_HIGHWAYS = ["401", "402"]
MTO_SEGMENTS_FILE = "mto_segments.parquet"
MTO_PDF_CACHE = "mto_pdf_rows.parquet"
TOP10 = {
    2014: ["Wellington/Commissioners", "Highbury/Oxford", "Highbury/Hamilton", "Wonderland/Oxford",
           "Fanshawe Park/Adelaide", "Wharncliffe/Oxford", "Sarnia/Wonderland",
           "Southdale/Wonderland", "Oxford/Adelaide", "Oxford/Richmond"],
    2015: ["Wellington/Exeter", "Wellington/Commissioners", "Highbury/Oxford", "Wharncliffe/Oxford",
           "Hamilton/Highbury", "Adelaide/Oxford", "Fanshawe Park/Adelaide", "Wonderland/Oxford",
           "Sarnia/Wonderland", "Highbury/Huron"],
    2016: ["Wellington/Exeter", "Wellington/Commissioners", "Highbury/Oxford", "Wharncliffe/Oxford",
           "Oxford/Richmond", "Oxford/Wonderland", "Fanshawe Park/Adelaide", "Adelaide/Oxford",
           "Wonderland/Southdale", "Highbury/Huron", "Highbury/Hamilton"],
}  # fmt: skip


def toronto_rates(toronto: dict[str, Any]) -> dict[str, float]:
    stats = json.loads((Path(toronto["paths"]["processed_dir"]) / RISK_STATS).read_text())
    return stats["rates_per_mvkm"]


def spf_risk(cfg: dict[str, Any], toronto: dict[str, Any], expo: pd.DataFrame) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    spf = json.loads((Path(toronto["paths"]["models_dir"]) / SPF_MODEL).read_text())
    static = pd.read_parquet(processed / STATIC)
    rows = pd.DataFrame({"intersection_id": static.index, "year": 2025})  # year dummies are zeroed
    df = spf_covariates(rows, static, expo)
    base = predict_base(spf["models"], df)
    t = pd.read_parquet(Path(toronto["paths"]["processed_dir"]) / SPF_PREDICTIONS)
    level = float((t["spf_known"] / t["spf_base"])[t["year"] == 2025].iloc[0])
    out = df[["intersection_id", "group", "class_combo"]].copy()
    out["risk"] = base.to_numpy() * level  # expected crashes/yr at Toronto's 2025 level
    out["spf"] = out["risk"]
    out["entering_aadt"] = expo["entering_aadt"].reindex(out["intersection_id"]).to_numpy()
    out["risk_per_mev"] = out["risk"] / (out["entering_aadt"] * 365 / PER_MILLION)
    out["is_proxy_fallback"] = False
    if out["risk_per_mev"].isna().any():
        raise ValueError("London intersection risk has NaN values")
    return out


def edge_rates(edges: pd.DataFrame, rates: dict[str, float], mto: pd.DataFrame) -> pd.DataFrame:
    """Toronto's per-MVKM class rates; MTO section rates on 401 / 402; any other motorway
    (Highbury Avenue's freeway section) takes Toronto's 'other expressway' rate."""
    e = edges.copy()
    is_mto = e["mto_lhrs"].notna()
    other_mw = (e["highway"] == "motorway") & ~is_mto
    e["rate_group"] = np.select([is_mto, other_mw], ["400-series (MTO)", "other expressway"],
                                e["highway"].astype(str))  # fmt: skip
    e["rate_per_mvkm"] = e["rate_group"].map(rates).fillna(rates.get("other", 0.0))
    cr = mto.set_index("segment_id")["cr"]
    e.loc[is_mto, "rate_per_mvkm"] = e.loc[is_mto, "segment_id"].map(cr).to_numpy()
    return e


def key_pair(s: str) -> frozenset[str]:
    return frozenset(street_key(x) for x in s.split("/"))


def top10_check(risk: pd.DataFrame, intersections: pd.DataFrame) -> pd.DataFrame:
    """Each published London Police top-10 intersection, matched by street names to the
    consolidated intersection whose OSM names contain both streets (the busiest by entering
    AADT if several), with its percentile in the SPF ranking and in an entering-AADT-only
    ranking (100 = riskiest / busiest)."""
    names = intersections.set_index("intersection_id")["street_names"].fillna("")
    keys = names.map(lambda s: {street_key(x) for x in s.split(" / ") if x})
    r = risk.set_index("intersection_id")
    pct_spf = r["risk"].rank(pct=True) * 100
    pct_vol = r["entering_aadt"].rank(pct=True) * 100
    rows = []
    for year, lst in TOP10.items():
        for rank, pair in enumerate(lst, 1):
            want = key_pair(pair)
            hits = keys[keys.map(lambda k, w=want: w <= k)].index
            if len(hits):
                best = r.loc[hits, "entering_aadt"].idxmax()
                rows.append({"year": year, "rank": rank, "intersection": pair,
                             "OSM names": names[best], "intersection_id": int(best),
                             "SPF expected/yr": r.loc[best, "risk"],
                             "entering AADT": r.loc[best, "entering_aadt"],
                             "SPF percentile": pct_spf[best],
                             "volume-only percentile": pct_vol[best]})  # fmt: skip
            else:
                rows.append({"year": year, "rank": rank, "intersection": pair,
                             "OSM names": "(not found)"})  # fmt: skip
    return pd.DataFrame(rows)


def run_london_step(cfg: dict[str, Any], toronto: dict[str, Any]) -> dict[str, Any]:
    """`python -m safepath london`: graph + static features (Toronto code paths, cached),
    then volumes, risk, routing graph, evaluation and reports/london.md."""
    from safepath.features.build import build_static
    from safepath.graph import build_graph
    from safepath.parking import load_or_fetch_parking

    processed = Path(cfg["paths"]["processed_dir"])
    if not (processed / INTERSECTIONS).exists():
        build_graph(cfg)
    load_or_fetch_parking(cfg)
    if not (processed / STATIC).exists():
        build_static(cfg)
    return run_london(cfg, toronto)


def run_london(cfg: dict[str, Any], toronto: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    segments = gpd.read_parquet(processed / SEGMENTS)
    sections, segmap, located, rules, _ = map_sections(cfg, LONDON_HIGHWAYS, segments,
                                                       processed / MTO_PDF_CACHE)  # fmt: skip
    segmap.to_parquet(processed / MTO_SEGMENTS_FILE, index=False)
    vol = run_london_volumes(cfg, segmap)
    check = entering_check(toronto)
    expo = vol["exposure"].copy()
    # half the approach two-way AADT runs below Toronto's turning-count entering volume by a
    # median factor (check["median_ratio"]); scale so per-vehicle risk means the same thing
    expo["entering_aadt"] = expo["entering_aadt"] / check["median_ratio"]
    risk = spf_risk(cfg, toronto, expo)
    risk.to_parquet(processed / INTERSECTION_RISK, index=False)

    edges = pd.read_parquet(processed / "aadt_edges.parquet")
    seg = segments.set_index("segment_id")
    edges["length_m"] = edges["segment_id"].map(seg["length"])
    edges["mvkm"] = edges["aadt_dir"] * edges["length_m"] / 1000 * 365 / PER_MILLION
    e = edge_rates(edges, toronto_rates(toronto), segmap)
    e[["u", "v", "key", "segment_id", "rate_group", "rate_per_mvkm", "mvkm"]].to_parquet(
        processed / EDGE_RISK, index=False
    )
    G = ox.load_graphml(processed / PROJ_GRAPH)
    node_map = pd.read_parquet(processed / NODE_MAP)
    attached = attach_risk(G, node_map, pd.read_parquet(processed / EDGE_MAP),
                           risk.set_index("intersection_id")["risk_per_mev"], e)  # fmt: skip
    import pickle

    with open(processed / ROUTING_GRAPH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    junction = dict(zip(node_map["osmid"], node_map["intersection_id"], strict=True))
    router = Router(G, node_junction=junction)
    pairs = random_pairs(router, 200, min_km=3, max_km=20)
    calib = calibrate_k(router, pairs)
    (processed / ROUTING_K).write_text(json.dumps(calib, indent=2))
    trade = tradeoff_table(router, pairs, [0, 0.5, 1, 2])
    summary = summarise(trade)
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    top = top10_check(risk, intersections)
    r = {
        "sections": sections[sections["mapped_km"] > 0], "rules": rules, "located": located,
        "volumes": vol, "entering_check": check, "risk": risk, "edges": attached,
        "calibration": calib, "summary": summary, "trade": trade, "top10": top,
        "rate_groups": e.groupby("rate_group").agg(km=("length_m", lambda x: x.sum() / 1000),
                                                   rate_per_mvkm=("rate_per_mvkm", "median")),
    }  # fmt: skip
    from safepath.london.report import write_london_report

    write_london_report(cfg, r, Path(cfg["paths"]["reports_dir"]).parent / "london.md")
    log.info("london: k = %.1f; %d intersections", calib["k_seconds_per_crash"], len(risk))
    return r


def short(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()
