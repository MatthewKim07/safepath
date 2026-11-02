"""reports/ottawa.md: the cross-city test write-up, from the Ottawa steps' outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safepath.data.ottawa import CLEAN_STATS, MANIFEST
from safepath.locations import markdown_table
from safepath.ottawa.build import BUILD_STATS
from safepath.ottawa.volumes import VOLUME_TRANSFER

PRIMARY = ["Spearman [95% CI]", "PR-AUC", "recall@1%", "recall@5% [95% CI]", "recall@10%"]
CALIB = ["dev. raw", "dev. level-adj.", "pred / obs raw", "level scalar", "CE raw",
         "CE level-adj."]  # fmt: skip


def _drop(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df if c.startswith("_")])


def _verdict(ci: tuple[float, float]) -> str:
    lo, hi = ci
    return "better" if lo > 0 else "worse" if hi < 0 else "no different"


def _spearman(table: pd.DataFrame, col: str) -> float:
    return float(table.loc[table["_col"] == col, "_spearman"].iloc[0])


def conclusions(r: dict[str, Any], vol: dict[str, Any]) -> list[str]:
    """Both views of the Ottawa result, each claim backed by a paired CI on test 2024."""
    from safepath.ottawa.transfer import MODELS, label

    t, p = r["test"]["table"], r["test"]["paired"].set_index("comparison")
    best_a, best_b = r["best"]["A"], r["best"]["B"]

    def pair(a: str, b: str, metric: str = "spearman") -> tuple[float, tuple[float, float]]:
        d, ci = p.loc[f"{label(a)} − {label(b)}", f"_{metric}"]
        return float(d), (float(ci[0]), float(ci[1]))

    def fmt(a: str, b: str, metric: str = "spearman") -> str:
        d, ci = pair(a, b, metric)
        return f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] ({_verdict(ci)})"

    m = pd.DataFrame(vol["metrics"])
    m = m[m["class combo"] == "all"].set_index("model")
    tor = m.loc["Toronto model, betweenness rescaled"]
    raw = m.loc["Toronto model, raw betweenness"]
    up = m.loc["Ottawa-trained LightGBM (spatial CV, upper bound)"]
    base = m.loc["Ottawa class median (spatial CV)"]
    sp = {c: _spearman(t, c) for c in MODELS}
    r5 = {row["_col"]: float(row["recall@5% [95% CI]"].split()[0]) for _, row in t.iterrows()}
    b_models = [c for c in MODELS if MODELS[c][0] == "B"]
    b_r5 = [pair(c, "upper_lgbm_aadt", "recall5") for c in b_models]
    b_r5_span = max(max(abs(ci[0]), abs(ci[1])) for _, ci in b_r5)
    b_r5_all_tie = all(ci[0] <= 0 <= ci[1] for _, ci in b_r5)
    b_pr = [pair(c, "upper_lgbm_aadt", "pr_auc")[0] for c in b_models]
    c = r["ceiling"].set_index("setting")
    t_obs = float(c.loc["Toronto 2025: LightGBM + AADT, observed", "Spearman"])
    thin = float(c[c.index.str.contains("thinned")].iloc[0]["Spearman"])
    sh = r["shift"][r["shift"]["SMD"].abs() > 0.2]
    shifts = ", ".join(f"{f} {v:+.2f}" for f, v in zip(sh["feature"], sh["SMD"], strict=True))
    rates = r["rates"].set_index(["city", "intersections"])["crashes per MEV"]
    return [
        "Two views of the same test year (2024), every difference with a paired 95% bootstrap "
        "CI over intersections. Spearman is dominated by ties here (83% of intersection-years "
        "have no crash), so hotspot recall@5% and PR-AUC are co-primary.",
        "",
        f"- **On Spearman, nothing beats Ottawa's own prev-3-year mean ({sp['prev3_mean']:.3f}).** "
        f"Best transferred model ({label(best_b)}) minus it: {fmt(best_b, 'prev3_mean')}; "
        f"the Ottawa-trained upper bound minus it: {fmt('upper_lgbm_aadt', 'prev3_mean')}.",
        "- **On hotspot recall, Toronto models fed Ottawa's crash history match the "
        "Ottawa-trained upper bound.** Recall@5% minus the upper bound for each group-B model: "
        + "; ".join(f"{MODELS[cc][1]} {fmt(cc, 'upper_lgbm_aadt', 'recall5')}" for cc in b_models)
        + (f" - every CI contains zero and lies within ±{b_r5_span:.3f}." if b_r5_all_tie else ".")
        + f" On PR-AUC they trail it ({min(b_pr):+.3f} to {max(b_pr):+.3f}, all CIs below "
        "zero), so the upper bound ranks the whole list better while the top 5% is the same.",
        f"- **EB vs persistence** is the same split: Spearman {fmt('eb', 'persistence')}, "
        f"recall@5% {fmt('eb', 'persistence', 'recall5')}, PR-AUC "
        f"{fmt('eb', 'persistence', 'pr_auc')}.",
        f"- **With no Ottawa data at all, the count-free SPF roughly doubles the road-class "
        f"baseline's hotspot recall**: {r5['spf_countfree']:.3f} vs {r5['class_combo_mean']:.3f} "
        f"(x{r5['spf_countfree'] / r5['class_combo_mean']:.1f}; difference "
        f"{fmt('spf_countfree', 'class_combo_mean', 'recall5')}; PR-AUC "
        f"{fmt('spf_countfree', 'class_combo_mean', 'pr_auc')}). On Spearman it is below the "
        f"class means ({fmt('spf_countfree', 'class_combo_mean')}): the class mean ties whole "
        "classes, which Spearman rewards on mostly-zero data, but it cannot pick out the busy "
        f"few. The best road-only model on validation Spearman, {label(best_a)}, shows the same "
        f"pattern: Spearman {fmt(best_a, 'class_combo_mean')}, recall@5% "
        f"{fmt(best_a, 'class_combo_mean', 'recall5')}.",
        f"- **Why Ottawa's numbers are lower than Toronto's: sparsity.** Ottawa has "
        f"{c.iloc[1, 0]:.2f} crashes per intersection-year against Toronto's {c.iloc[0, 0]:.2f}. "
        f"Thinning Toronto's own 2025 crashes to Ottawa's rate drops Toronto's LightGBM + AADT "
        f"from {t_obs:.3f} to {thin:.3f} Spearman with nothing else changed. Compare models "
        "within a city, never Ottawa's Spearman with Toronto's.",
        f"- **Does the graph help in a new city?** Not clearly. Road-only GraphSAGE minus "
        f"LightGBM: Spearman {fmt('road_sage', 'road_lgbm')}, recall@5% "
        f"{fmt('road_sage', 'road_lgbm', 'recall5')}; minus the graph-free MLP: Spearman "
        f"{fmt('road_sage', 'road_mlp')}, recall@5% {fmt('road_sage', 'road_mlp', 'recall5')}.",
        f"- **Volumes transfer badly**: Toronto's AADT model on Ottawa counts R² "
        f"{tor['r2_log']:.2f} (raw betweenness {raw['r2_log']:.2f}), over-predicting "
        f"x{np.exp(tor['bias_log']):.1f}; Ottawa class medians {base['r2_log']:.2f}; an "
        f"Ottawa-trained model {up['r2_log']:.2f}. Exposure for the count-using models is "
        f"Ottawa's ({vol['aadt_source']} blended with counts).",
        f"- **Feature shift** (|SMD| > 0.2): {shifts}. Crashes per million entering vehicles: "
        f"Ottawa {rates[('Ottawa', 'all')]:.2f}, Toronto {rates[('Toronto', 'all')]:.2f}; "
        "part of that is geocoding (TPS puts ~96% of crashes on intersections, Ottawa ~75%).",
        "- **Verdict:** for finding the top hotspots, Toronto's models transfer: with Ottawa "
        "history they find as many of the next year's crashes in their top 5% as a model "
        "trained on Ottawa, and with no Ottawa data at all the count-free SPF finds about "
        "twice what a road-class lookup does. On Spearman, Ottawa's own three-year average is "
        "still best and no Toronto model beats it (on PR-AUC the Ottawa-trained model is ahead "
        f"of that average, {fmt('upper_lgbm_aadt', 'prev3_mean', 'pr_auc')}).",
    ]


LIMITATIONS = [
    "- **2024 recording change.** Non-reportable records jump from under 100 a year to 2,358 "
    "(dropped), and the severe share falls from 16-20% to 8.8% with fatal and major-injury "
    "crashes steady: minor-injury crashes are recorded differently. The any-crash target "
    "looks consistent (2024 total in line with 2019), but the severe target is not compared "
    "and the test-year level may still be affected.",
    "- **2023 is unpublished.** Test 2024's 'previous year' is 2022, so persistence-style "
    "features are a year staler than in Toronto, and the level scalar comes from 2022.",
    "- **Different geocoding.** TPS puts almost every crash on an intersection; Ottawa keeps "
    "mid-block crashes mid-block. Toronto's intersection labels therefore include crashes "
    "Ottawa's do not, which shifts levels and what 'intersection risk' means.",
    "- **Exposure is Ottawa-derived for most models.** Only road-only LightGBM, road-only "
    "MLP and the count-free SPF use no Ottawa data at all. Segment AADT is Toronto's segment "
    "model with a betweenness variant and level correction chosen on Ottawa's intersection "
    "counts; Ottawa's mid-block counts were not used to validate it.",
    "- **Static features are today's OSM**, for every year, in both cities.",
    "- **GNN training on MPS is not bit-for-bit reproducible**; seed-averaged results move by "
    "about 0.0001 Spearman between runs.",
]


def write_ottawa_report(cfg: dict[str, Any], toronto: dict[str, Any], r: dict[str, Any]) -> None:
    processed = Path(cfg["paths"]["processed_dir"])
    raw = Path(cfg["paths"]["raw_dir"])
    manifest = json.loads((raw / MANIFEST).read_text())
    data = json.loads((processed / CLEAN_STATS).read_text())
    build = json.loads((processed / BUILD_STATS).read_text())
    vol = json.loads((processed / VOLUME_TRANSFER).read_text())
    clean = pd.read_parquet(processed / "collisions_clean.parquet",
                            columns=["year", "injury_or_fatal"])  # fmt: skip
    years = clean.groupby("year").agg(crashes=("year", "size"),
                                      pct_severe=("injury_or_fatal", "mean"))  # fmt: skip
    years["pct_severe"] *= 100
    years = years.reset_index().astype({"year": str}).rename(columns={"pct_severe": "% severe"})
    services = pd.DataFrame(
        [
            {"data": k, "service": s, "rows": v["rows"]}
            for k in ("collisions", "volumes")
            for s, v in manifest[k].items()
        ]  # fmt: skip
    )
    g = build["graph"]
    tol = str(cfg["graph"]["consolidation_tolerance_m"])
    loc = pd.DataFrame(build.get("location_summary", []))
    kinds = pd.DataFrame(build.get("precision", {}).get("by_kind", []))
    field = pd.DataFrame(build.get("precision", {}).get("by_field", []))
    conc = pd.DataFrame(build.get("precision", {}).get("conc", []))
    vm = pd.DataFrame(vol["metrics"])
    vm_all = vm[vm["class combo"] == "all"].drop(columns="class combo")
    keep = [
        "Toronto model, betweenness rescaled",
        "Ottawa-trained LightGBM (spatial CV, upper bound)",
    ]
    vm_cls = vm[(vm["class combo"] != "all") & vm["model"].isin(keep)]
    rename = {"rmse_log": "log-RMSE", "median_abs_pct_err": "median abs % err", "r2_log": "R²",
              "bias_log": "bias (log)"}  # fmt: skip
    asg = build["assignments"]
    n_asg = sum(asg.values())
    fc = build["freeway_check"]
    lines = [
        "# Cross-city test: Ottawa (CP9)",
        "",
        "Toronto-trained models scored on Ottawa intersections they never saw, with no Ottawa "
        "crash labels in any training set. Compared with Ottawa-native baselines and an "
        "Ottawa-trained upper bound. Ranking metrics are primary because the two cities "
        "record collisions differently; calibration is shown raw and after one level scalar "
        "per model fit on the prior year only. Everything below is regenerated by "
        "`python -m safepath ottawa`.",
        "",
        "Model groups and the Ottawa data their inputs use (no group trains on Ottawa crash "
        "labels except C and D):",
        "",
        "- **(A)** Toronto-trained, no Ottawa crash labels. Road-only LightGBM, road-only MLP "
        "and the count-free SPF use OSM only; the SPF with Ottawa AADT and road-only GraphSAGE "
        "(edge AADT) also use Ottawa traffic counts for exposure.",
        "- **(B)** Toronto-trained, Ottawa crash history as input features plus Ottawa counts "
        "for exposure; still no Ottawa training labels.",
        "- **(C)** Ottawa-native baselines from Ottawa crash labels (2016-2019) or history.",
        "- **(D)** LightGBM + AADT trained on Ottawa 2016-2019 labels: the upper bound.",
        "",
        "## Data",
        "",
        f"City of Ottawa open data ({manifest['licence']}), ArcGIS Online host "
        f"`{manifest['host']}`. Row counts match each service's `returnCountOnly`.",
        "",
        markdown_table(services, floats="{:.0f}"),
        "",
        "- `Traffic_Collision_Data` (the service named in the plan) does not exist on the "
        "City's host; the City's own yearly and 2017-2024 layers are used. **2023 is not "
        "published** and 2025 is not out, so the latest full year is 2024.",
        f"- Dropped: {sum(data['dropped_non_reportable_by_year'].values()):,} non-reportable "
        f"records (class 04; {data['dropped_non_reportable_by_year'].get('2024', 0):,} of them "
        f"in 2024 alone), {sum(data['dropped_outside_bbox_by_year'].values())} with zero or "
        f"out-of-city coordinates. **Freeway crashes removed: {data['freeway_removed']:,}** "
        "(location names only Hwy 416 / 417 mainline or ramps; ramp terminals on city "
        f"streets kept). OSM check inside the study area: {fc['removed_within_motorway_m']:,} "
        f"of {fc['removed_in_study_area']:,} removed crashes lie within {fc['motorway_m']} m of "
        f"a 416 / 417 edge; {fc['kept_on_4xx_only']:,} kept crashes sit on one and away from "
        "surface roads.",
        "- Severe share collapses in 2024 (fatal and major injuries steady, minor and minimal "
        "injuries halve): a recording change, so severity is not compared in the test year.",
        "",
        markdown_table(years),
        "",
        "Splits: Ottawa-train 2016-2019 (history from 2013), validation 2022, test 2024 "
        "(2020-2021 history only, as in Toronto; 2023 unpublished, so 2024's history window "
        "is 2020-2022 and its 'previous year' is 2022). 2021 gets feature rows only to fit "
        "validation's level scalar.",
        "",
        "## Study area and graph",
        "",
        f"{cfg['graph']['boundary']['name']} (`{cfg['graph']['boundary']['url']}`), "
        f"{build['study_area_km2']:.0f} km² of the 2,790 km² amalgamated city (Toronto: 630 "
        f"km²). {build['crashes_in_study_area']:,} non-freeway crashes inside, "
        f"{build['crashes_outside_study_area']:,} outside (rural, out of scope). OSM drive graph "
        f"{g['directed_nodes']:,} nodes / {g['directed_edges']:,} edges; consolidation at "
        f"{tol} m (as Toronto) gives **{g['consolidation'][tol]['intersections']:,} "
        "intersections** (Toronto 22,435).",
        "",
        "## Location precision: does Ottawa snap like TPS?",
        "",
        markdown_table(conc) if len(conc) else "",
        "",
        markdown_table(loc.rename(columns={"pct_crashes": "% crashes", "pct_points": "% points"}))
        if len(loc)
        else "",  # fmt: skip
        "",
        "Distance to the nearest consolidated intersection by the City's location string, % "
        "of crashes:",
        "",
        markdown_table(kinds) if len(kinds) else "",
        "",
        "By the City's own intersection field (2013-2016 layers):",
        "",
        markdown_table(field) if len(field) else "",
        "",
        f"Snapping with Toronto's thresholds: {asg.get('intersection', 0):,} crashes to "
        f"intersections ({100 * asg.get('intersection', 0) / n_asg:.1f}%; Toronto 96.3%), "
        f"{asg.get('segment', 0):,} mid-block ({100 * asg.get('segment', 0) / n_asg:.1f}%), "
        f"{asg.get('excluded', 0):,} excluded. Proxy rules: retail cannot fire (no "
        f"fail-to-remain field; {build.get('retail_candidates', 0)} busy low-severity "
        f"intersections near retail, listed in `reports/ottawa/build.md`); expressway rule "
        f"flags {build['proxies']['expressway_proxy']}.",
        "",
        "## Volume-model transfer",
        "",
        f"{vol['match']['counted_intersections']:,} counted intersections (latest non-COVID "
        f"count, 2016-2025; match rate {100 * vol['match']['match_rate']:.1f}%, median "
        f"{vol['match']['median_match_m']:.1f} m). Betweenness rescaled to Toronto's graph size "
        f"(global x{vol['bt_scale']['bt_global']:.2f}, local x{vol['bt_scale']['bt_local']:.2f}).",
        "",
        markdown_table(vm_all.rename(columns=rename), floats="{:.3f}"),
        "",
        "By class combo (>= 20 counts):",
        "",
        markdown_table(vm_cls.rename(columns=rename), floats="{:.3f}"),
        "",
        f"Crash models use **{vol['aadt_source']}** (lower overall log-RMSE), blended with the "
        "latest count as in Toronto. Segment AADT comes from Toronto's segment model "
        "(unvalidated in Ottawa).",
        "",
    ]
    for split in ("val", "test"):
        s = r[split]
        t = _drop(s["table"])
        lines += [
            f"## Crash transfer: {'validation' if split == 'val' else 'test'} {s['year']}",
            "",
            f"{s['rows']:,} clean intersections, {s['crashes']:,} crashes. Level scalar fit on "
            f"{s['prior']}. CE = mean |log predicted / observed| over entering-AADT deciles.",
            "",
            markdown_table(t[["group", "model", "Ottawa data", *PRIMARY]], floats="{:.3f}"),
            "",
            markdown_table(t[["group", "model", *CALIB]], floats="{:.3f}"),
            "",
            "Paired differences (same bootstrap resamples):",
            "",
            markdown_table(_drop(s["paired"])),
            "",
        ]
    zero = r["zero"]
    lines += [
        "Test, intersections with no crash in the prior three years:",
        "",
        markdown_table(zero, floats="{:.3f}"),
        "",
        "## Feature shift, Toronto 2025 vs Ottawa 2024 (clean intersections)",
        "",
        "SMD = (Ottawa mean - Toronto mean) / pooled sd; |SMD| > 0.2 is a noticeable shift.",
        "",
        markdown_table(r["shift"], floats="{:.3f}"),
        "",
        markdown_table(r["mix"], floats="{:.1f}"),
        "",
        "Crash levels:",
        "",
        markdown_table(r["rates"], floats="{:.3f}"),
        "",
        f"Class combos Ottawa has but Toronto never saw (LightGBM treats them as missing): "
        f"{', '.join(r['info']['unseen_class_combos']) or 'none'}.",
        "",
        "Per-vehicle calibration by entering-AADT decile, test (raw predictions):",
        "",
    ]
    from safepath.ottawa.transfer import label

    for m, dt in r["deciles"].items():
        lines += [f"{label(m)}:", "", markdown_table(dt, floats="{:.3f}"), ""]
    lines += [
        "## Noise ceiling and sparsity",
        "",
        "Thinning is the clean test of sparsity. The Poisson ceiling (a lower bound here, "
        "see the conclusions): if a model's expected counts were the true ones, the Spearman "
        "a perfect model reaches against one simulated year (200 draws). Expected counts: "
        "Ottawa, LightGBM + AADT trained on Ottawa; Toronto, its own LightGBM + AADT; both "
        "scaled to the observed total. Thinning keeps each Toronto crash with probability "
        "Ottawa rate / Toronto rate (50 draws). Ceilings are approximate: true risk is more "
        "spread out than any model's estimate, so the real ceiling is higher.",
        "",
        markdown_table(r["ceiling"], floats="{:.3f}"),
        "",
        "## Conclusions",
        "",
        *conclusions(r, vol),
        "",
        "## Limitations",
        "",
        *LIMITATIONS,
        "",
    ]
    Path(cfg["paths"]["reports_dir"]).parent.joinpath("ottawa.md").write_text("\n".join(lines))
