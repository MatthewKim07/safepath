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
    from safepath.ottawa.transfer import MODELS, label

    t, p = r["test"]["table"], r["test"]["paired"].set_index("comparison")
    best_a, best_cf, best_b = r["best"]["A"], r["best"]["A_countfree"], r["best"]["B"]

    def pair(a: str, b: str) -> tuple[float, tuple[float, float]]:
        row = p.loc[f"{label(a)} − {label(b)}"]
        return float(row["_d"]), tuple(row["_ci"])

    def fmt(d: float, ci: tuple[float, float]) -> str:
        return f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] ({_verdict(ci)})"

    m = pd.DataFrame(vol["metrics"])
    m = m[m["class combo"] == "all"].set_index("model")
    tor = m.loc["Toronto model, betweenness rescaled"]
    raw = m.loc["Toronto model, raw betweenness"]
    up = m.loc["Ottawa-trained LightGBM (spatial CV, upper bound)"]
    base = m.loc["Ottawa class median (spatial CV)"]
    sp = {c: _spearman(t, c) for c in MODELS}
    sp_r5 = {row["_col"]: float(row["recall@5% [95% CI]"].split()[0]) for _, row in t.iterrows()}
    sh = r["shift"][r["shift"]["SMD"].abs() > 0.2]
    shifts = ", ".join(f"{f} {v:+.2f}" for f, v in zip(sh["feature"], sh["SMD"], strict=True))
    rates = r["rates"].set_index(["city", "intersections"])["crashes per MEV"]
    o_rate, t_rate = rates[("Ottawa", "all")], rates[("Toronto", "all")]
    c = r["ceiling"].set_index("setting")
    o_ceil = float(c.loc["Ottawa 2024: noise ceiling", "Spearman"])
    t_ceil = float(c.loc["Toronto 2025: noise ceiling", "Spearman"])
    t_obs = float(c.loc["Toronto 2025: LightGBM + AADT, observed", "Spearman"])
    thin = c[c.index.str.contains("thinned")].iloc[0]
    lines = [
        f"- **Sparsity sets the scale.** Ottawa averages {c.iloc[1, 0]:.2f} crashes per "
        f"intersection-year against Toronto's {c.iloc[0, 0]:.2f} (Ottawa keeps mid-block "
        "crashes mid-block, and its level is lower). Thinning Toronto's own 2025 crashes to "
        f"Ottawa's rate drops Toronto's LightGBM + AADT from {t_obs:.3f} to "
        f"**{thin['Spearman']:.3f}** with nothing else changed, so about "
        f"{t_obs - thin['Spearman']:.2f} of the Toronto-Ottawa difference in headline "
        "Spearman is the sparser target, not the transfer. What remains (thinned Toronto "
        f"{thin['Spearman']:.3f} vs {sp[best_b]:.3f} for the best transferred model and "
        f"{sp['upper_lgbm_aadt']:.3f} for the Ottawa-trained one) mixes transfer loss with "
        "real differences between the cities. The model-based Poisson ceiling ("
        f"{o_ceil:.2f} in Ottawa, {t_ceil:.2f} in Toronto) is only a lower bound: Ottawa's "
        f"prev-3-year mean already reaches it ({sp['prev3_mean']:.3f}), so true risk is more "
        "spread out than the Ottawa-trained model's estimate. Compare models within a city, "
        "never Ottawa's Spearman with Toronto's.",
        f"- **Volumes transfer badly.** Toronto's AADT model on Ottawa's counted "
        f"intersections: R² {tor['r2_log']:.2f} (raw betweenness {raw['r2_log']:.2f}), median "
        f"abs error {tor['median_abs_pct_err']:.0f}%, over-predicting by "
        f"x{np.exp(tor['bias_log']):.1f} (Toronto's own CV: 0.85, 21%). Ottawa class medians "
        f"do better ({base['r2_log']:.2f} / {base['median_abs_pct_err']:.0f}%); an "
        f"Ottawa-trained model reaches {up['r2_log']:.2f} / {up['median_abs_pct_err']:.0f}%. "
        f"So the crash models' exposure is Ottawa's: {vol['aadt_source']} blended with counts.",
        *(
            [
                f"- **No Ottawa crash labels** (A): the best on validation is {label(best_a)}, "
                f"which uses no Ottawa data at all: test Spearman {sp[best_a]:.3f}; minus "
                f"Ottawa's class-combo means {fmt(*pair(best_a, 'class_combo_mean'))}."
            ]
            if best_a == best_cf
            else [
                f"- **No Ottawa crash labels, no Ottawa counts** (count-free part of A): best "
                f"on validation {label(best_cf)}, test Spearman {sp[best_cf]:.3f}; minus "
                f"class-combo means {fmt(*pair(best_cf, 'class_combo_mean'))}.",
                f"- **No Ottawa crash labels, Ottawa counts for exposure** (all of A): best on "
                f"validation {label(best_a)}, test Spearman {sp[best_a]:.3f}; minus "
                f"class-combo means {fmt(*pair(best_a, 'class_combo_mean'))}.",
            ]
        ),
        "  With road features alone Toronto knowledge does not beat a one-number-per-class "
        "Ottawa baseline on Spearman, though every road-only model beats it on recall@5% "
        f"(by {sp_r5['road_mlp'] - sp_r5['class_combo_mean']:.2f} to "
        f"{sp_r5['road_lgbm'] - sp_r5['class_combo_mean']:.2f}; the class mean cannot rank "
        "within a class). Ottawa counts barely matter here: the SPF with Ottawa AADT "
        f"{sp['spf']:.3f} vs count-free {sp['spf_countfree']:.3f}.",
        f"- **Does the graph help in a new city?** Road-only GraphSAGE minus road-only "
        f"LightGBM {fmt(*pair('road_sage', 'road_lgbm'))}; minus the graph-free MLP "
        f"{fmt(*pair('road_sage', 'road_mlp'))}. LightGBM minus MLP "
        f"{fmt(*pair('road_lgbm', 'road_mlp'))}. On Spearman the neural models transfer "
        "better than the trees and message passing does not add to the MLP; on recall@5% "
        "the order flips (LightGBM > GraphSAGE > MLP). No model family wins both, so the "
        "graph is not a clear help in Ottawa. Road-only GraphSAGE also sees Ottawa counts "
        "through its edge AADT, the MLP and LightGBM do not.",
        f"- **With Ottawa's crash history as features** (B, no Ottawa training labels) the "
        f"best on validation is {label(best_b)}, test Spearman {sp[best_b]:.3f}, but it "
        "trails the simplest "
        f"Ottawa-native baseline: minus persistence {fmt(*pair(best_b, 'persistence'))}; "
        f"prev-3-year mean {sp['prev3_mean']:.3f}. Toronto's trees and GNNs with history "
        f"({sp['lgbm_aadt']:.3f}-{sp['lgbm_aadt_nbr']:.3f}, GraphSAGE "
        f"{sp['sage_history']:.3f}) do worse than EB. A likely reason: they learned how much "
        "to trust history at Toronto's density of crashes and lean on the road / volume "
        "prior where Ottawa's own history is more informative; they also over-predict the "
        "level 1.5-2.4x raw.",
        f"- **Feature shift** (|SMD| > 0.2, Ottawa vs Toronto): {shifts}. Crashes per "
        f"million entering vehicles: Ottawa {o_rate:.2f}, Toronto {t_rate:.2f}. Part of that "
        "is geocoding (TPS puts ~96% of crashes on intersections, Ottawa's records put "
        "~75% within the snap radius); the data cannot separate the rest into reporting "
        "and real safety differences.",
        f"- **Upper-bound gap:** best transferred (B) minus LightGBM + AADT trained on "
        f"Ottawa 2016-2019 {fmt(*pair(best_b, 'upper_lgbm_aadt'))}. Even the Ottawa-trained "
        f"model ({sp['upper_lgbm_aadt']:.3f}) does not beat the prev-3-year mean on Spearman, "
        "though it is the best calibrated model per vehicle and leads on PR-AUC.",
        "- **Verdict:** Toronto's crash models carry over to Ottawa as rankings that are "
        "better than chance and useful for top-k hotspot lists (recall@5% about 0.69 with "
        "history, as good as Ottawa's own models), but not better than Ottawa's own crash "
        "history. The transferable part is the EB recipe (SPF + local history), not the "
        "trees or the GNNs. A city adopting this should bring its own counts and at least a "
        "few years of its own crash history.",
    ]
    return lines


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
