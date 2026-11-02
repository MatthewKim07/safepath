"""Volume-model transfer: Toronto's AADT model on Ottawa's counted intersections.

1. Match each City count to the consolidated intersection within `ottawa.volumes.match_m`,
   preferring one that shares street names (Toronto's TMC matcher).
2. Target per counted intersection: log of the mean AADT in its most recent count year that
   is not a COVID year (2020-2021), as in Toronto.
3. Compare, on the same counted intersections and the same metrics as Toronto's CV:
   - Toronto's intersection model applied as is (no Ottawa data at all);
   - an Ottawa class-median baseline (5-fold 2 km spatial CV within Ottawa);
   - an Ottawa-trained LightGBM with the same features (same spatial CV): the upper bound.
4. Ottawa AADT for the crash models: the better of (Toronto model, Ottawa-trained model) on
   overall log-RMSE, blended with the latest count exactly as in Toronto (weights from
   Ottawa's own repeat counts). Segment AADT (for major / minor approach volumes and the GNN
   edge features) comes from Toronto's segment model; Ottawa publishes no segment counts in
   this set, so it is not validated here.

Betweenness is a path count, so it grows with the size of the graph: global betweenness
with n^2 and local (5-minute) betweenness with n. The Toronto model is scored with Ottawa's
values rescaled to Toronto's node count (`bt_scale`, the a-priori choice) and with the raw
values; the segment model uses whichever fits Ottawa's intersection counts better.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd

from safepath.data.aadt import match_tmc
from safepath.data.ottawa import VOLUME_COUNTS
from safepath.features.volume_features import (
    approach_volumes,
    edge_centrality,
    intersection_features,
    segment_features,
    write_intersection_exposure,
)
from safepath.graph import EDGE_MAP, GRAPH_STATS, INTERSECTIONS, NODE_MAP, SEGMENTS
from safepath.locations import markdown_table
from safepath.models.volume import (
    AADT_EDGES,
    AADT_INTERSECTIONS,
    INTERSECTION_FEATURES,
    PREDICT_YEAR,
    SEGMENT_CLASSES,
    SEGMENT_FEATURES,
    blend,
    latest_obs,
    metrics,
    prepare,
    repeat_count_variance,
    run_part,
)
from safepath.models.volume_report import VOLUME_BOOSTERS, VOLUME_CATEGORIES

log = logging.getLogger(__name__)

MATCHED_COUNTS = "counts_matched.parquet"
VOLUME_TRANSFER = "volume_transfer.json"
BT_GLOBAL = ["bt_global", "bt_global_in", "bt_global_in_max"]
BT_LOCAL = ["bt_local", "bt_local_in"]


def match_counts(cfg: dict[str, Any], intersections: gpd.GeoDataFrame) -> pd.DataFrame:
    counts = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / VOLUME_COUNTS)
    # "A ST @ B ST/C ST" -> "A ST/B ST/C ST" for the name matcher
    counts = counts.assign(location_name=counts["location_name"].str.replace(" @ ", "/"))
    out = match_tmc(counts, intersections, cfg["ottawa"]["volumes"]["match_m"])
    out["year"] = out["count_year"]
    out["date"] = out["count_date"]
    return out


def bt_scale(toronto_nodes: int, city_nodes: int) -> dict[str, float]:
    r = toronto_nodes / city_nodes
    return {**{c: r * r for c in BT_GLOBAL}, **{c: r for c in BT_LOCAL}}


def rescale(df: pd.DataFrame, scale: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, s in scale.items():
        if col in out:
            out[col] = out[col] * s
    return out


def metric_rows(data: pd.DataFrame, preds: dict[str, pd.Series]) -> pd.DataFrame:
    """Toronto's volume metrics per model, overall and per class combo (>= 20 counts)."""
    rows = []
    groups = [("all", data.index), *data.groupby("class_combo").groups.items()]
    for name, idx in groups:
        if len(idx) < 20:
            continue
        y = data.loc[idx, "log_aadt"].to_numpy()
        for model, p in preds.items():
            rows.append({"class combo": name, "model": model,
                         **metrics(y, p.loc[idx].astype(float).to_numpy())})  # fmt: skip
    return pd.DataFrame(rows)


COUNTFREE_EXPOSURE = "intersection_exposure_countfree.parquet"


def write_countfree_exposure(
    processed: Path,
    inter_s: pd.DataFrame,
    seg_s: pd.DataFrame,
    b_inter: lgb.Booster,
    b_seg: lgb.Booster,
    cats: dict[str, list],
) -> pd.DataFrame:
    """Exposure that uses no Ottawa traffic count at all: Toronto's intersection and segment
    models with the a-priori betweenness rescaling, no blending, no level correction. For
    the fully count-free SPF variant of group A."""
    entering = np.exp(b_inter.predict(prepare(inter_s.assign(count_year=PREDICT_YEAR),
                                              INTERSECTION_FEATURES, cats)))  # fmt: skip
    sx = seg_s.set_index("segment_id").assign(count_year=PREDICT_YEAR)
    seg = pd.Series(np.exp(b_seg.predict(prepare(sx, SEGMENT_FEATURES, cats))), index=sx.index)
    edges = pd.read_parquet(processed / EDGE_MAP)
    edges["aadt_segment"] = edges["segment_id"].map(seg)
    segments = pd.read_parquet(processed / SEGMENTS, columns=["segment_id", "name"])
    node_map = pd.read_parquet(processed / NODE_MAP)
    out = pd.DataFrame({"entering_aadt": entering}, index=inter_s.index.rename("intersection_id"))
    out = out.join(approach_volumes(edges, node_map, segments))
    out.reset_index().to_parquet(processed / COUNTFREE_EXPOSURE, index=False)
    return out


def run_ottawa_volumes(cfg: dict[str, Any], toronto: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    tor_models = Path(toronto["paths"]["models_dir"])
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)

    # counts -> intersections
    matched = match_counts(cfg, intersections)
    matched.to_parquet(processed / MATCHED_COUNTS, index=False)
    ok = matched["intersection_id"].notna()
    covid = set(cfg["ottawa"]["volumes"]["covid_years"])
    usable = matched[ok & ~matched["year"].isin(covid)].copy()
    usable["intersection_id"] = usable["intersection_id"].astype(int)
    match_stats = {
        "counts": len(matched),
        "matched": int(ok.sum()),
        "match_rate": float(ok.mean()),
        "median_match_m": float(matched.loc[ok, "match_m"].median()),
        "p95_match_m": float(matched.loc[ok, "match_m"].quantile(0.95)),
        "matched_sharing_a_name": float((matched.loc[ok, "names_shared"] > 0).mean()),
        "covid_counts_dropped": int((ok & matched["year"].isin(covid)).sum()),
        "counted_intersections": int(usable["intersection_id"].nunique()),
        "intersections": len(intersections),
    }

    # features (Ottawa graph), betweenness rescaled to Toronto's graph size
    cent = edge_centrality(cfg)
    inter = intersection_features(cfg, cent, intersections)
    seg = segment_features(cfg, cent, segments, inter)
    seg["highway"] = seg["highway"].where(seg["highway"].isin(SEGMENT_CLASSES), "other")
    t_nodes = json.loads((Path(toronto["paths"]["processed_dir"]) / GRAPH_STATS).read_text())
    o_nodes = json.loads((processed / GRAPH_STATS).read_text())
    scale = bt_scale(t_nodes["directed_nodes"], o_nodes["directed_nodes"])
    inter_s, seg_s = rescale(inter, scale), rescale(seg, scale)

    cats_all = json.loads((tor_models / VOLUME_CATEGORIES).read_text())
    cats = {k: cats_all[k] for k in ("class_combo", "highway")}
    unseen = sorted(set(inter["class_combo"]) - set(cats["class_combo"]))
    b_inter = lgb.Booster(model_file=str(tor_models / VOLUME_BOOSTERS["inter"]))
    b_seg = lgb.Booster(model_file=str(tor_models / VOLUME_BOOSTERS["seg"]))

    targets = latest_obs(usable, "intersection_id").rename(
        columns={"log_obs": "log_aadt", "obs_year": "count_year"}
    )
    data = inter.drop(columns="count_year", errors="ignore").join(targets, how="inner")
    data_s = inter_s.drop(columns="count_year", errors="ignore").join(targets, how="inner")
    p_toronto = pd.Series(b_inter.predict(prepare(data_s, INTERSECTION_FEATURES, cats)),
                          index=data.index)  # fmt: skip
    p_raw = pd.Series(b_inter.predict(prepare(data, INTERSECTION_FEATURES, cats)),
                      index=data.index)  # fmt: skip
    ottawa_cats = {"class_combo": sorted(inter["class_combo"].unique())}
    part = run_part(data, INTERSECTION_FEATURES, ottawa_cats, "class_combo")
    preds = {
        "Ottawa class median (spatial CV)": part["oof"]["baseline"],
        "Toronto model, betweenness rescaled": p_toronto,
        "Toronto model, raw betweenness": p_raw,
        "Ottawa-trained LightGBM (spatial CV, upper bound)": part["oof"]["lgbm"],
    }
    table = metric_rows(data, preds)
    overall = table[table["class combo"] == "all"].set_index("model")

    # choose the Ottawa AADT source for the crash models on overall log-RMSE
    use_ottawa = (overall.loc["Ottawa-trained LightGBM (spatial CV, upper bound)", "rmse_log"]
                  < overall.loc["Toronto model, betweenness rescaled", "rmse_log"])  # fmt: skip
    inter_s["count_year"] = PREDICT_YEAR
    if use_ottawa:
        pred_all = pd.Series(part["booster"].predict(
            prepare(inter.assign(count_year=PREDICT_YEAR), INTERSECTION_FEATURES, ottawa_cats)),
            index=inter.index)  # fmt: skip
        pred_all.loc[data.index] = part["oof"]["lgbm"].astype(float)  # honest for counted
        source = "Ottawa-trained intersection model"
    else:
        pred_all = pd.Series(b_inter.predict(prepare(inter_s, INTERSECTION_FEATURES, cats)),
                             index=inter.index)  # fmt: skip
        source = "Toronto intersection model"
    chosen = "Ottawa-trained LightGBM (spatial CV, upper bound)" if use_ottawa else (
        "Toronto model, betweenness rescaled")  # fmt: skip
    v_model = float(np.mean((preds[chosen].astype(float) - data["log_aadt"]) ** 2))
    var = repeat_count_variance(usable, "intersection_id")
    obs = latest_obs(usable, "intersection_id").reindex(inter.index)
    b = blend(pred_all, obs["log_obs"], obs["obs_year"], v_model, var)
    out_i = pd.DataFrame(
        {
            "intersection_id": inter.index,
            "aadt_model": np.exp(pred_all).to_numpy(),
            "aadt_count": np.exp(obs["log_obs"]).to_numpy(),
            "count_year": obs["obs_year"].to_numpy(),
            "weight_count": b["weight_count"].to_numpy(),
            "aadt": np.exp(b["log_aadt"]).to_numpy(),
            "source": np.where(obs["log_obs"].notna(), "blend", "model"),
            "class_combo": inter["class_combo"].to_numpy(),
        }
    )
    out_i.to_parquet(processed / AADT_INTERSECTIONS, index=False)

    # segments: Toronto's segment model, for approach volumes. No Ottawa segment counts in
    # this set, so two choices are borrowed from the intersection evidence on Ottawa counts:
    # the betweenness variant that fits better, and a level correction of exp(-bias) where
    # bias is the Toronto intersection model's mean log error on Ottawa counts.
    raw_better = (overall.loc["Toronto model, raw betweenness", "rmse_log"]
                  < overall.loc["Toronto model, betweenness rescaled", "rmse_log"])  # fmt: skip
    variant = "Toronto model, raw betweenness" if raw_better else (
        "Toronto model, betweenness rescaled")  # fmt: skip
    seg_bias = float(overall.loc[variant, "bias_log"])
    sx = (seg if raw_better else seg_s).set_index("segment_id").assign(count_year=PREDICT_YEAR)
    seg_raw = pd.Series(np.exp(b_seg.predict(prepare(sx, SEGMENT_FEATURES, cats))),
                        index=sx.index)  # fmt: skip
    seg_aadt = seg_raw * np.exp(-seg_bias)
    edges = pd.read_parquet(processed / EDGE_MAP).merge(
        sx[["highway"]].reset_index(), on="segment_id"
    )
    edges["aadt_segment"] = edges["segment_id"].map(seg_aadt)
    edges["source"] = "toronto segment model"
    edges.to_parquet(processed / AADT_EDGES, index=False)
    expo = write_intersection_exposure(processed, out_i, edges)
    write_countfree_exposure(processed, inter_s, seg_s, b_inter, b_seg, cats)

    by_class = (
        out_i.groupby("class_combo")["aadt"].agg(["size", "median"])
        .rename(columns={"size": "intersections", "median": "median AADT"})
        .sort_values("intersections", ascending=False).head(12).reset_index()
    )  # fmt: skip
    seg_by_class = (
        pd.DataFrame({"highway": sx["highway"], "uncorrected": seg_raw, "used": seg_aadt})
        .groupby("highway")[["uncorrected", "used"]].median().reset_index()
    )  # fmt: skip
    r = {
        "match": match_stats,
        "metrics": table.to_dict(orient="records"),
        "bt_scale": scale,
        "nodes": {"toronto": t_nodes["directed_nodes"], "ottawa": o_nodes["directed_nodes"]},
        "unseen_combos": unseen,
        "aadt_source": source,
        "segment_variant": variant,
        "segment_level_correction": float(np.exp(-seg_bias)),
        "blend": {**var, "v_model": v_model},
        "count_year_counts": usable.groupby("year").size().to_dict(),
        "entering_aadt_median": float(expo["entering_aadt"].median()),
    }
    (processed / VOLUME_TRANSFER).write_text(json.dumps(r, indent=2, default=float))
    write_volume_report(cfg, r, table, by_class, seg_by_class, reports / "volumes.md")
    log.info("ottawa volumes: %d counted intersections; AADT from %s",
             match_stats["counted_intersections"], source)  # fmt: skip
    return r


def write_volume_report(
    cfg: dict[str, Any],
    r: dict[str, Any],
    table: pd.DataFrame,
    by_class: pd.DataFrame,
    seg_by_class: pd.DataFrame,
    path: Path,
) -> None:
    m = r["match"]
    show = table.rename(columns={"rmse_log": "log-RMSE", "median_abs_pct_err": "median abs % err",
                                 "r2_log": "R²", "bias_log": "bias (log)"})  # fmt: skip
    lines = [
        "# Ottawa volume-model transfer",
        "",
        f"City counts: {m['counts']:,}; matched to a consolidated intersection within "
        f"{cfg['ottawa']['volumes']['match_m']} m: {m['matched']:,} ({100 * m['match_rate']:.1f}%"
        f", median {m['median_match_m']:.1f} m, p95 {m['p95_match_m']:.1f} m; "
        f"{100 * m['matched_sharing_a_name']:.0f}% share a street name with the OSM "
        f"intersection). Unmatched counts are mostly outside the study area. "
        f"{m['covid_counts_dropped']} matched 2021 counts dropped (COVID). Counted "
        f"intersections: **{m['counted_intersections']:,}** of {m['intersections']:,} "
        f"({100 * m['counted_intersections'] / m['intersections']:.1f}%; Toronto 26.5%).",
        "",
        "Target per counted intersection: log of the mean AADT in its latest non-COVID count "
        "year (2016-2025). Toronto's model sees `count_year` = that year.",
        "",
        f"Betweenness rescaled to Toronto's graph: Toronto {r['nodes']['toronto']:,} nodes, "
        f"Ottawa {r['nodes']['ottawa']:,}; global x{r['bt_scale']['bt_global']:.2f}, local "
        f"x{r['bt_scale']['bt_local']:.2f}. Class combos Toronto never saw: "
        f"{', '.join(r['unseen_combos']) or 'none'}.",
        "",
        "## Accuracy on Ottawa's counted intersections",
        "",
        "Same metrics as Toronto's spatial CV (Toronto CV, overall: R² 0.85, median abs % "
        "error 21%; class-median baseline R² 0.72, 33%).",
        "",
        markdown_table(show, floats="{:.3f}"),
        "",
        f"AADT used by the crash models: **{r['aadt_source']}**, chosen on overall log-RMSE, "
        "blended with the latest count where one exists: w = v_model / (v_model + s0² + "
        f"tau² x age), with v_model = {r['blend']['v_model']:.3f} (that model's squared error on "
        f"Ottawa counts), s0² = {r['blend']['s0_sq']:.4f}, tau² = {r['blend']['tau_sq']:.4f} "
        f"from {r['blend']['pairs']} Ottawa repeat-count pairs.",
        "",
        "Entering AADT by class combo (all intersections, after blending):",
        "",
        markdown_table(by_class, floats="{:,.0f}"),
        "",
        "Segment AADT, used for major / minor approach volumes, neighbour and GNN edge "
        "features: Toronto's segment model (Ottawa has no segment counts in this set). The "
        f"betweenness variant is the one that fits Ottawa's intersection counts better "
        f"({r['segment_variant'].removeprefix('Toronto model, ')}), and predictions are scaled "
        f"by x{r['segment_level_correction']:.2f}, the inverse of that intersection model's "
        "mean over-prediction on Ottawa counts. Medians before and after:",
        "",
        markdown_table(seg_by_class, floats="{:,.0f}"),
        "",
    ]
    path.write_text("\n".join(lines))
