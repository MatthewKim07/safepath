"""Run the traffic-volume model and write reports/volume_model.md with its figures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safepath.data.aadt import SVC_AADT, TMC_AADT
from safepath.data.mto import MTO_SEGMENTS
from safepath.data.report import BAR, GRID, SURFACE, TEXT, TEXT_MUTED
from safepath.features.volume_features import (
    CENTRALITY,
    edge_centrality,
    intersection_features,
    segment_features,
)
from safepath.graph import EDGE_MAP, INTERSECTIONS, SEGMENTS
from safepath.locations import markdown_table
from safepath.models.volume import (
    AADT_EDGES,
    AADT_INTERSECTIONS,
    EDGE_COLUMNS,
    INTERSECTION_FEATURES,
    MIN_TARGET_YEAR,
    PREDICT_YEAR,
    SEGMENT_CLASSES,
    SEGMENT_FEATURES,
    VOLUME_CV,
    blend,
    directional_split,
    distribution,
    lane_scaled_expressways,
    latest_obs,
    mto_check,
    prepare,
    repeat_count_variance,
    run_part,
    unit_targets,
)

log = logging.getLogger(__name__)


def plot_pred_vs_obs(panels: list[tuple[str, pd.DataFrame, pd.DataFrame]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.4), facecolor=SURFACE)
    for ax, (title, data, oof) in zip(axes, panels, strict=True):
        y = np.exp(data["log_aadt"])
        p = np.exp(oof["lgbm"].astype(float))
        hb = ax.hexbin(y, p, gridsize=45, xscale="log", yscale="log", bins="log", cmap="Blues",
                       mincnt=1, linewidths=0)  # fmt: skip
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=TEXT_MUTED, lw=1)
        for f in (0.5, 2):
            ax.plot([lo, hi], [lo * f, hi * f], color=GRID, lw=1, ls="--")
        ax.set_xlabel("observed AADT (count)", fontsize=8, color=TEXT_MUTED)
        ax.set_ylabel("predicted AADT (out-of-fold)", fontsize=8, color=TEXT_MUTED)
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        fig.colorbar(hb, ax=ax, shrink=0.8, label="counted locations")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(boosters: dict[str, lgb.Booster], path: Path) -> None:
    fig, axes = plt.subplots(1, len(boosters), figsize=(5.2 * len(boosters), 4), facecolor=SURFACE)
    for ax, (name, b) in zip(axes, boosters.items(), strict=True):
        imp = pd.Series(b.feature_importance("gain"), index=b.feature_name())
        imp = (imp / imp.sum()).sort_values().tail(12)
        ax.barh(imp.index, 100 * imp.to_numpy(), color=BAR, height=0.6)
        ax.set_title(f"{name}: % of total gain", fontsize=9, color=TEXT, loc="left")
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_MUTED, labelsize=7, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(cfg: dict[str, Any], r: dict[str, Any], path: Path) -> None:
    it, sg = r["inter"], r["seg"]
    lines = [
        "# Traffic volume model (CP5)",
        "",
        "Predicts AADT for every consolidated intersection (entering vehicles) and every road "
        "segment (two-way vehicles) where the City has no count. `python -m safepath "
        "volume-model` regenerates everything here.",
        "",
        "## Data",
        "",
        f"- Targets: log AADT of the most recent count since {MIN_TARGET_YEAR} (2020-21 left "
        f"out) at {len(it['data']):,} intersections (TMC) and "
        f"{int((sg['data']['target'] == 'svc').sum()):,} segments (SVC), plus MTO 2024 AADT on "
        f"{int((sg['data']['target'] == 'mto').sum())} 400-series mainline segments. "
        "`count_year` is a feature; predictions use "
        f"`count_year = {PREDICT_YEAR}`.",
        "- Features: the 27 static OSM intersection features from CP3, or segment attributes "
        "(class, lanes, speed limit, one-way / divided carriageway, length, endpoint classes "
        "and signals, POIs, bus stops, streetcar, motorway and downtown distance, local "
        "intersection density), plus sampled **edge betweenness** on free-flow travel time: "
        f"global ({CENTRALITY['bt_global']['n_sources']:,} random sources, all destinations) and "
        f"local ({CENTRALITY['bt_local']['n_sources']:,} sources, destinations within "
        f"{CENTRALITY['bt_local']['limit'] // 60} minutes). Intersections use the betweenness "
        "summed over entering edges. The implementation is checked exactly against "
        "networkx on a random graph (tests/test_centrality.py).",
        "- No coordinates or neighbourhood IDs.",
        "",
        "## Spatial-block cross-validation",
        "",
        "5 folds of 2 km blocks (as in CP3); each fold early-stops on the next fold and is "
        "scored on blocks it never saw. Baseline: median log AADT of the class "
        "(intersection class combo / segment highway class) in the training folds. "
        "`median abs % error` is |predicted / observed - 1|.",
        "",
        "### Intersections (entering AADT)",
        "",
        markdown_table(it["metrics"], floats="{:.3f}"),
        "",
        "### Segments (two-way AADT)",
        "",
        markdown_table(sg["metrics"], floats="{:.3f}"),
        "",
        "![Predicted vs observed](figures/volume_pred_vs_obs.png)",
        "",
        "![Feature importance](figures/volume_importance.png)",
        "",
        f"Early-stopping rounds per fold: intersections {it['iters']}, segments {sg['iters']}. "
        "Final models are refit on every counted location with the median round count.",
        "",
        "## Counted locations: count, model, or blend?",
        "",
        "Recent counts should win, but a single short count is itself noisy and old counts go "
        "stale. Both effects are measured from repeat counts at the same location (consecutive "
        "pairs, 2020-21 excluded), fitting `E[(log a2 - log a1)^2] = 2 s0^2 + tau^2 * gap`:",
        "",
        markdown_table(r["blend_table"], floats="{:.4f}"),
        "",
        "Final log AADT = w * log(latest count) + (1 - w) * out-of-fold prediction, with "
        "w = v_model / (v_model + s0^2 + tau^2 * age) and v_model the CV error variance. "
        "Count weight by age:",
        "",
        markdown_table(r["weights_by_age"], floats="{:.2f}"),
        "",
        "Counted units whose only counts are from 2020-21 use the model alone. "
        "400-series mainline segments use MTO's 2024 AADT (reports/mto.md); their ramps use "
        "the model; city expressways use the lane-scaled estimate below.",
        "",
        "## Expressways",
        "",
        "City counts on expressway mainline are unusable (reports/volumes.md), so the "
        "segment model's freeway targets are MTO's 2024 AADT on the 400-series mainline "
        f"({int((sg['data']['target'] == 'mto').sum())} OSM segments, inside the same spatial "
        "CV). Out-of-fold predictions on those segments, i.e. how well the model infers a "
        "freeway's volume from blocks it never saw; this is what the DVP and Gardiner rely "
        "on:",
        "",
        markdown_table(r["mto_check"], floats="{:,.2f}"),
        "",
        "The city expressways (Gardiner, DVP, Allen, ...) are a different matter: the model "
        "has seen no city-expressway volume, and it gives them 400-series volumes. They use a "
        f"lane-scaled estimate instead: the 400-series median of **{r['per_lane']:,.0f}** AADT "
        "per through lane (MTO AADT x directional share / OSM lanes) times each carriageway's "
        "lanes. The City's own expressway tube counts are shown for reference only; they mix "
        "single lanes, single directions, and full cross-sections. The DVP's 1997 counts "
        "(about 69-70k weekday each way, i.e. ~140k two-way) are the most complete and sit "
        "below the lane-scaled figure, so these volumes are probably on the high side. They "
        "are the least certain numbers in the exposure data.",
        "",
        markdown_table(r["expressways"], floats="{:,.0f}"),
        "",
        "## Final AADT by class (all units, after blending)",
        "",
        "Intersections, entering:",
        "",
        markdown_table(r["final_inter"], floats="{:,.0f}"),
        "",
        "Segments, two-way:",
        "",
        markdown_table(r["final_seg"], floats="{:,.0f}"),
        "",
        "## Limitations",
        "",
        "- Betweenness uses free-flow travel times and uniform trip ends, so it misses "
        "congestion-driven route choice and where people actually live and work.",
        "- Entering AADT at intersections and segment AADT are modelled separately and are not "
        "forced to agree.",
        "- Residential streets have few counts relative to their number, so their AADT rests "
        "mostly on the model; local streets that carry cut-through traffic are the hardest "
        "case.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_volume_model(cfg: dict[str, Any]) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    figures = Path(cfg["paths"]["figures_dir"])
    intersections = gpd.read_parquet(processed / INTERSECTIONS)
    segments = gpd.read_parquet(processed / SEGMENTS)
    cent = edge_centrality(cfg)
    inter = intersection_features(cfg, cent, intersections)
    seg = segment_features(cfg, cent, segments, inter)
    seg["highway"] = seg["highway"].where(seg["highway"].isin(SEGMENT_CLASSES), "other")
    cats = {
        "class_combo": sorted(inter["class_combo"].unique()),
        "highway": sorted(seg["highway"].unique()),
    }

    tmc = pd.read_parquet(processed / TMC_AADT)
    svc_all = pd.read_parquet(processed / SVC_AADT)
    svc = svc_all[svc_all["usable"]]
    ti = unit_targets(tmc, "intersection_id")
    ts = unit_targets(svc, "segment_id")
    di = inter.drop(columns="count_year", errors="ignore").join(ti, how="inner")
    # 400-series mainline targets from MTO, so the model learns freeway-scale volumes for
    # the uncounted city expressways (DVP, Gardiner)
    mto = pd.read_parquet(processed / MTO_SEGMENTS).set_index("segment_id")
    latest = cfg["mto"]["latest_year"]
    tm = pd.DataFrame(
        {"count_year": latest, "log_aadt": np.log(mto[f"aadt_{latest}"]), "target": "mto"}
    )
    ts = pd.concat([ts.assign(target="svc"), tm[~tm.index.isin(ts.index)]])
    ds = seg.set_index("segment_id").join(ts, how="inner")
    r: dict[str, Any] = {
        "inter": run_part(di, INTERSECTION_FEATURES, cats, "class_combo"),
        "seg": run_part(ds, SEGMENT_FEATURES, cats, "highway"),
    }

    var_i = repeat_count_variance(tmc, "intersection_id")
    var_s = repeat_count_variance(svc, "segment_id")
    r["blend_table"] = pd.DataFrame(
        [
            {"unit": "intersection (TMC)", **var_i, "v_model": r["inter"]["v_model"]},
            {"unit": "segment (SVC)", **var_s, "v_model": r["seg"]["v_model"]},
        ]
    )
    ages = [0, 2, 5, 10, 15, 20, 30]
    r["weights_by_age"] = pd.DataFrame(
        {
            "count age (years)": ages,
            "w intersection": [
                r["inter"]["v_model"]
                / (r["inter"]["v_model"] + var_i["s0_sq"] + var_i["tau_sq"] * a)
                for a in ages
            ],
            "w segment": [
                r["seg"]["v_model"] / (r["seg"]["v_model"] + var_s["s0_sq"] + var_s["tau_sq"] * a)
                for a in ages
            ],
        }
    )

    # ---- intersections: model for all, blended with the latest non-COVID count
    inter["count_year"] = PREDICT_YEAR
    pred_i = pd.Series(
        r["inter"]["booster"].predict(prepare(inter, INTERSECTION_FEATURES, cats)),
        index=inter.index,
    )
    pred_i.loc[di.index] = r["inter"]["oof"]["lgbm"].astype(float)  # honest for counted ones
    obs = latest_obs(tmc, "intersection_id").reindex(inter.index)
    b = blend(pred_i, obs["log_obs"], obs["obs_year"], r["inter"]["v_model"], var_i)
    out_i = pd.DataFrame(
        {
            "intersection_id": inter.index,
            "aadt_model": np.exp(pred_i).to_numpy(),
            "aadt_count": np.exp(obs["log_obs"]).to_numpy(),
            "count_year": obs["obs_year"].to_numpy(),
            "weight_count": b["weight_count"].to_numpy(),
            "aadt": np.exp(b["log_aadt"]).to_numpy(),
            "source": np.where(obs["log_obs"].notna(), "blend", "model"),
            "class_combo": inter["class_combo"].to_numpy(),
        }
    )
    out_i.to_parquet(processed / AADT_INTERSECTIONS, index=False)

    # ---- segments: model, blended with counts, MTO on 400-series mainline
    sx = seg.set_index("segment_id")
    sx["count_year"] = PREDICT_YEAR
    pred_s = pd.Series(
        r["seg"]["booster"].predict(prepare(sx, SEGMENT_FEATURES, cats)), index=sx.index
    )
    pred_s.loc[ds.index] = r["seg"]["oof"]["lgbm"].astype(float)
    obs_s = latest_obs(svc, "segment_id").reindex(sx.index)
    bs = blend(pred_s, obs_s["log_obs"], obs_s["obs_year"], r["seg"]["v_model"], var_s)
    seg_out = pd.DataFrame(
        {
            "aadt_model": np.exp(pred_s),
            "aadt_count": np.exp(obs_s["log_obs"]),
            "count_year": obs_s["obs_year"],
            "weight_count": bs["weight_count"].to_numpy(),
            "aadt": np.exp(bs["log_aadt"]).to_numpy(),
            "source": np.where(obs_s["log_obs"].notna(), "blend", "model"),
        },
        index=sx.index,
    )
    r["mto_check"] = mto_check(ds, r["seg"]["oof"], mto)
    seg_out.loc[mto.index, "aadt"] = mto[f"aadt_{latest}"]
    seg_out.loc[mto.index, "source"] = "mto"

    edge_map = pd.read_parquet(processed / EDGE_MAP)
    edges = edge_map.merge(
        sx[["highway", "name", "is_oneway", "is_divided", "lanes_n"]].reset_index(),
        on="segment_id",
    ).merge(seg_out[["aadt", "source"]].reset_index(), on="segment_id")
    edges = edges.merge(
        mto[["hwy", "lhrs"]].rename(columns={"hwy": "mto_hwy", "lhrs": "mto_lhrs"}).reset_index(),
        on="segment_id",
        how="left",
    )
    edges["dir_share"] = directional_split(edges)
    edges["aadt_dir"] = edges["aadt"] * edges["dir_share"]
    edges, r["per_lane"], r["expressways"] = lane_scaled_expressways(edges, svc_all, sx)
    seg_out.loc[r["expressways"]["segment_ids"], "aadt"] = (
        edges.drop_duplicates("segment_id").set_index("segment_id")["aadt"]
    ).reindex(r["expressways"]["segment_ids"])
    seg_out.loc[r["expressways"]["segment_ids"], "source"] = "lane-scaled"
    r["expressways"] = r["expressways"]["table"]
    edges = edges.rename(columns={"aadt": "aadt_segment"})
    edges[EDGE_COLUMNS].to_parquet(processed / AADT_EDGES, index=False)

    r["final_inter"] = distribution(out_i["aadt"], out_i["class_combo"], "class combo")
    r["final_seg"] = distribution(seg_out["aadt"], sx["highway"], "highway")
    plot_pred_vs_obs(
        [
            ("Intersections: entering AADT", di, r["inter"]["oof"]),
            ("Segments: two-way AADT", ds, r["seg"]["oof"]),
        ],
        figures / "volume_pred_vs_obs.png",
    )
    plot_importance(
        {"intersections": r["inter"]["booster"], "segments": r["seg"]["booster"]},
        figures / "volume_importance.png",
    )
    (processed / VOLUME_CV).write_text(
        json.dumps(
            {
                "intersections": r["inter"]["metrics"].to_dict(orient="records"),
                "segments": r["seg"]["metrics"].to_dict(orient="records"),
                "blend": r["blend_table"].to_dict(orient="records"),
            },
            indent=2,
        )
    )
    write_report(cfg, r, Path(cfg["paths"]["reports_dir"]) / "volume_model.md")
    log.info("AADT for %d intersections and %d directed edges", len(out_i), len(edges))
    return r
