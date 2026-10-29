"""Score 2026 crash risk for every intersection and road segment and write it onto the
directed routing graph.

Units: expected police-reported crashes per year at that location. There is no exposure
(traffic volume) term yet, so this is where crashes happen, not a per-vehicle crash rate.

- Intersections: lgbm_main prediction for 2026 (history 2023-2025). Proxy intersections use
  the static-only deploy model instead of their own (misattributed) counts.
- Non-motorway segments: per-km rate by highway class from mid-block crash assignments over
  the train label years.
- Motorways: per-km rate from the excess crashes at expressway-proxy intersections (actual
  minus static-only expected) over the train label years, spread over all motorway km.

A directed edge carries `intersection_risk` when it enters a consolidated intersection (its
head node is in a different cluster from its tail node), so each intersection a route passes
through is charged exactly once. `segment_risk` = rate x length. `risk` is their sum.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import lightgbm as lgb
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from safepath.features.build import STATIC
from safepath.features.history import history_features
from safepath.features.labels import LABELS
from safepath.features.proxies import EXPRESSWAY
from safepath.graph import EDGE_MAP, INTERSECTIONS, NODE_MAP, PROJ_GRAPH, SEGMENTS
from safepath.locations import MOTORWAY_CLASSES, markdown_table
from safepath.models import lgbm
from safepath.models.train import STATIC_DEPLOY, categories
from safepath.snapping import ASSIGNMENTS

log = logging.getLogger(__name__)

DEPLOY_YEAR = 2026
ROUTING_GRAPH = "routing_graph.pkl"
INTERSECTION_RISK = "intersection_risk.parquet"
RISK_STATS = "risk_stats.json"


def intersection_risk(labels: pd.DataFrame, static: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """Predicted 2026 crashes per intersection, with the static fallback for proxies."""
    hist = history_features(labels, [DEPLOY_YEAR])
    rows = hist.merge(static, left_on="intersection_id", right_index=True, how="left")
    cats = categories(static.reset_index())
    main = lgb.Booster(model_file=str(models_dir / "lgbm_main.txt"))
    deploy = lgb.Booster(model_file=str(models_dir / STATIC_DEPLOY))
    rows["pred_main"] = lgbm.predict(main, rows, lgbm.FEATURE_SETS["main"], cats)
    rows["pred_static"] = lgbm.predict(deploy, rows, lgbm.FEATURE_SETS["static_only"], cats)

    proxy = labels.drop_duplicates("intersection_id").set_index("intersection_id")
    rows["exclusion_reason"] = rows["intersection_id"].map(proxy["exclusion_reason"])
    rows["is_proxy_fallback"] = rows["exclusion_reason"].notna()
    rows["risk"] = np.where(rows["is_proxy_fallback"], rows["pred_static"], rows["pred_main"])
    keep = [
        "intersection_id",
        "risk",
        "pred_main",
        "pred_static",
        "is_proxy_fallback",
        "exclusion_reason",
        "prev3_crashes_mean",
    ]
    return rows[keep]


def segment_rates(
    assignments: pd.DataFrame,
    segments: gpd.GeoDataFrame,
    risk: pd.DataFrame,
    labels: pd.DataFrame,
    train_years: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-km yearly crash rates by highway class, and the motorway rate from proxy excess."""
    n_years = len(train_years)
    seg = segments.set_index("segment_id")
    km = seg.groupby("highway")["length"].sum() / 1000
    mid = assignments[
        (assignments["unit_type"] == "segment") & assignments["year"].isin(train_years)
    ]
    mid_class = mid["unit_id"].astype(int).map(seg["highway"])
    per_class = mid_class.value_counts()
    rates = pd.DataFrame({"km": km, "train_midblock_crashes": per_class}).fillna(0)
    rates["crashes_per_km_year"] = rates["train_midblock_crashes"] / n_years / rates["km"]
    rates["source"] = "mid-block assignments"

    # Motorways: crashes TPS placed at nearby surface intersections, minus what an
    # intersection like that would be expected to have (static-only model).
    train = labels[labels["year"].isin(train_years)]
    proxy = train[train["exclusion_reason"] == EXPRESSWAY]
    actual = proxy.groupby("intersection_id")["crash_count"].sum() / n_years
    expected = risk.set_index("intersection_id").loc[actual.index, "pred_static"]
    excess = (actual - expected).clip(lower=0)
    mw_km = km[km.index.isin(MOTORWAY_CLASSES)].sum()
    mw_rate = float(excess.sum() / mw_km)
    mw_mid = rates.loc[rates.index.isin(MOTORWAY_CLASSES), "train_midblock_crashes"].sum()
    for cls in MOTORWAY_CLASSES & set(rates.index):
        rates.loc[cls, "crashes_per_km_year"] = mw_rate
        rates.loc[cls, "source"] = "expressway-proxy excess"
    detail = {
        "expressway_proxies": int(len(actual)),
        "proxy_actual_per_year": float(actual.sum()),
        "proxy_expected_per_year": float(expected.sum()),
        "proxy_excess_per_year": float(excess.sum()),
        "motorway_km": float(mw_km),
        "motorway_rate_per_km_year": mw_rate,
        "motorway_midblock_rate_per_km_year": float(mw_mid / n_years / mw_km),
    }
    return rates.sort_values("crashes_per_km_year", ascending=False), detail


def attach_risk(
    G: nx.MultiDiGraph,
    node_map: pd.DataFrame,
    edge_map: pd.DataFrame,
    segments: gpd.GeoDataFrame,
    risk: pd.DataFrame,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    """Write intersection_risk, segment_risk, and risk onto every directed edge of G."""
    cluster = node_map.set_index("osmid")["intersection_id"]
    risk_of = risk.set_index("intersection_id")["risk"]
    rate_of = rates["crashes_per_km_year"]
    seg_class = segments.set_index("segment_id")["highway"]

    edges = edge_map.copy()
    edges["cu"] = edges["u"].map(cluster)
    edges["cv"] = edges["v"].map(cluster)
    enters = edges["cv"].notna() & (edges["cu"] != edges["cv"])
    edges["intersection_risk"] = np.where(enters, edges["cv"].map(risk_of), 0.0)
    edges["entered_intersection"] = edges["cv"].where(enters)
    length = np.array(
        [G.edges[u, v, k]["length"] for u, v, k in edges[["u", "v", "key"]].itertuples(index=False)]
    )
    edges["length_m"] = length
    edges["highway"] = edges["segment_id"].map(seg_class)
    edges["segment_risk"] = edges["highway"].map(rate_of).fillna(0.0) * length / 1000
    edges["risk"] = edges["intersection_risk"] + edges["segment_risk"]
    for row in edges.itertuples(index=False):
        data = G.edges[row.u, row.v, row.key]
        data["intersection_risk"] = float(row.intersection_risk)
        data["segment_risk"] = float(row.segment_risk)
        data["risk"] = float(row.risk)
        data["entered_intersection"] = (
            int(row.entered_intersection) if not np.isnan(row.entered_intersection) else -1
        )
    return edges


def write_risk_report(
    risk: pd.DataFrame,
    rates: pd.DataFrame,
    detail: dict,
    edges: pd.DataFrame,
    names: pd.Series,
    path: Path,
) -> None:
    seg_total = rates["km"] * rates["crashes_per_km_year"]
    is_mw = rates.index.isin(MOTORWAY_CLASSES)
    by_kind = {
        "intersections": risk["risk"].sum(),
        "non-motorway segments": seg_total[~is_mw].sum(),
        "motorways": seg_total[is_mw].sum(),
    }
    total = sum(by_kind.values())
    # Risk per directed km by the class of the edge (intersection risk included)
    per_km = edges.groupby("highway").agg(
        km=("length_m", lambda x: x.sum() / 1000), risk=("risk", "sum"), seg=("segment_risk", "sum")
    )
    per_km["risk_per_km"] = per_km["risk"] / per_km["km"]
    per_km["segment_risk_per_km"] = per_km["seg"] / per_km["km"]
    per_km = per_km.sort_values("risk_per_km", ascending=False).reset_index()
    top = risk.sort_values("risk", ascending=False).head(15).copy()
    top["intersection"] = top["intersection_id"].map(names)
    lines = [
        "# Risk scoring (2026)",
        "",
        "Units are expected police-reported crashes per year at a location. There is no "
        "traffic-volume term yet, so busy roads look risky partly because they are busy.",
        "",
        "## Intersection risk",
        "",
        f"- {len(risk):,} intersections; lgbm_main prediction for {DEPLOY_YEAR} from "
        f"2023-2025 history. {int(risk['is_proxy_fallback'].sum())} proxy intersections use "
        "the static-only deploy model instead.",
        f"- Total {risk['risk'].sum():,.0f} crashes/yr; median {risk['risk'].median():.2f}, "
        f"p99 {risk['risk'].quantile(0.99):.1f}, max {risk['risk'].max():.1f}.",
        "",
        "Proxy fallbacks (own 2023-2025 mean vs the value used):",
        "",
        markdown_table(
            risk[risk["is_proxy_fallback"]]
            .assign(intersection=lambda d: d["intersection_id"].map(names))
            .sort_values("prev3_crashes_mean", ascending=False)[
                ["intersection", "exclusion_reason", "prev3_crashes_mean", "pred_main", "risk"]
            ],
            floats="{:.2f}",
        ),
        "",
        "## Segment rates (crashes per km per year, train label years)",
        "",
        markdown_table(rates.reset_index().rename(columns={"index": "highway"}), floats="{:.3f}"),
        "",
        "Motorway rate: "
        f"{detail['expressway_proxies']} expressway-proxy intersections average "
        f"{detail['proxy_actual_per_year']:.0f} crashes/yr in train years against "
        f"{detail['proxy_expected_per_year']:.1f} expected from static features, an excess of "
        f"{detail['proxy_excess_per_year']:.0f}/yr spread over "
        f"{detail['motorway_km']:.0f} km of motorway carriageway = "
        f"**{detail['motorway_rate_per_km_year']:.3f} crashes/km/yr**. For comparison, "
        "mid-block crashes TPS placed directly on motorway segments give "
        f"{detail['motorway_midblock_rate_per_km_year']:.3f}/km/yr; the two sources are "
        "separate crashes, so the rate used is a lower bound on motorway crashes that TPS "
        "located.",
        "",
        "## Total risk by source",
        "",
        "| source | crashes/yr | share |",
        "|---|---:|---:|",
        *[f"| {k} | {v:,.0f} | {100 * v / total:.1f}% |" for k, v in by_kind.items()],
        "",
        "Segment totals are rate x km over undirected segments (a two-way street counts "
        "once; each motorway carriageway is its own segment).",
        "",
        "## Risk per directed km by road class (sanity check: motorways are not free)",
        "",
        markdown_table(per_km, floats="{:.3f}"),
        "",
        "## Highest-risk intersections",
        "",
        markdown_table(
            top[["intersection", "risk", "prev3_crashes_mean", "is_proxy_fallback"]],
            floats="{:.1f}",
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def run_score(cfg: dict[str, Any]) -> Path:
    processed = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    labels = pd.read_parquet(processed / LABELS)
    static = pd.read_parquet(processed / STATIC)
    segments = gpd.read_parquet(processed / SEGMENTS)
    risk = intersection_risk(labels, static, models_dir)
    if risk["risk"].isna().any() or (risk["risk"] < 0).any():
        raise ValueError("intersection risk has NaN or negative values")
    risk.to_parquet(processed / INTERSECTION_RISK, index=False)

    assignments = pd.read_parquet(processed / ASSIGNMENTS)
    rates, detail = segment_rates(assignments, segments, risk, labels, cfg["splits"]["train"])
    G = ox.load_graphml(processed / PROJ_GRAPH)
    edges = attach_risk(
        G,
        pd.read_parquet(processed / NODE_MAP),
        pd.read_parquet(processed / EDGE_MAP),
        segments,
        risk,
        rates,
    )
    out = processed / ROUTING_GRAPH
    with open(out, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    names = pd.read_parquet(processed / INTERSECTIONS, columns=["intersection_id", "street_names"])
    names = names.set_index("intersection_id")["street_names"].replace("", "(unnamed)")
    write_risk_report(
        risk, rates, detail, edges, names, Path(cfg["paths"]["reports_dir"]) / "risk.md"
    )
    stats = {
        "rates": rates["crashes_per_km_year"].to_dict(),
        **detail,
        "intersection_risk_total": float(risk["risk"].sum()),
    }
    (processed / RISK_STATS).write_text(json.dumps(stats, indent=2))
    log.info("scored %d intersections and %d edges; wrote %s", len(risk), len(edges), out)
    return out
