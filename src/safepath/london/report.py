"""reports/london.md: the London, Ontario demo write-up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from safepath.graph import GRAPH_STATS
from safepath.locations import markdown_table


def top10_summary(top: pd.DataFrame) -> pd.DataFrame:
    """Distinct intersections on any year's list, and how many land in each top band."""
    t = top[top["OSM names"] != "(not found)"].drop_duplicates("intersection_id")
    rows = []
    for col in ("SPF percentile", "volume-only percentile"):
        rows.append({"ranking": col.replace(" percentile", ""), "listed intersections": len(t),
                     "in top 1%": int((t[col] >= 99).sum()), "in top 5%": int((t[col] >= 95).sum()),
                     "in top 10%": int((t[col] >= 90).sum()),
                     "median percentile": float(t[col].median())})  # fmt: skip
    return pd.DataFrame(rows)


def write_london_report(cfg: dict[str, Any], r: dict[str, Any], path: Path) -> None:
    processed = Path(cfg["paths"]["processed_dir"])
    g = json.loads((processed / GRAPH_STATS).read_text())
    vs = r["volumes"]["stats"]
    cov = r["volumes"]["coverage"]
    chk = r["entering_check"]
    top = r["top10"]
    found = top[top["OSM names"] != "(not found)"]
    summ = top10_summary(top)
    distinct = found.drop_duplicates("intersection_id")
    s_better = int((distinct["SPF percentile"] > distinct["volume-only percentile"]).sum())
    trade = r["summary"]
    risk = r["risk"]
    names = pd.read_parquet(processed / "intersections.parquet",
                            columns=["intersection_id", "street_names"]).set_index(
        "intersection_id")["street_names"]  # fmt: skip
    top_spf = risk.sort_values("risk", ascending=False).head(15).assign(
        intersection=lambda d: d["intersection_id"].map(names))[
        ["intersection", "risk", "entering_aadt", "risk_per_mev", "class_combo"]]  # fmt: skip
    secs = r["sections"][["hwy", "lhrs", "description", "aadt_2024", "cr", "cr_source",
                          "mapped_km"]]  # fmt: skip
    lines = [
        "# London, Ontario demo (CP10)",
        "",
        "**No local crash data.** London publishes no collision records, so London risk is "
        "Toronto's Safety Performance Function (no crash history, no Empirical Bayes) on "
        "London's road features and the City's traffic volumes. In Ottawa this setting (the "
        "count-free SPF) had recall@5% 0.619 against 0.288 for a road-class baseline, with no "
        "Ottawa data at all, but it was below Ottawa's own crash history on every ranking "
        "metric. Absolute numbers carry Toronto's reporting level and are not calibrated for "
        "London: treat them as **relative** risk.",
        "",
        "## Graph",
        "",
        f'OSM drive graph for "{cfg["graph"]["place"]}" (the municipal boundary): '
        f"{g['directed_nodes']:,} nodes, {g['directed_edges']:,} directed edges, "
        f"{g['segments']:,} undirected segments; consolidation at "
        f"{cfg['graph']['consolidation_tolerance_m']} m (as Toronto) gives "
        f"**{g['consolidation'][str(cfg['graph']['consolidation_tolerance_m'])]['intersections']:,}"
        "** intersections. Static OSM features and context layers come from the same code as "
        "Toronto's (London has no on-street tram).",
        "",
        "## Traffic volumes",
        "",
        "City of London open data, *Traffic Volumes* "
        "(`OpenData_Transportation/MapServer/21`, Open Government Licence - City of London): "
        f"{vs['city_lines']:,} line segments, {vs['city_km']:.0f} km, AADT rounded to 500s. "
        f"Every row was last processed on {vs['last_edit']}; the City does not publish count "
        "dates, so the volumes' age is unknown (the layer is refreshed about yearly from its "
        "count programme). The companion *Intersection Types* layer (MapServer/16) only marks "
        "where volume segments start and end, so it is downloaded for reference but not used.",
        "",
        f"Matched to OSM (samples at 25/50/75% of each OSM segment within "
        f"{cfg['london']['volumes']['match_m']} m of a City line with the same street name, or "
        f"{cfg['london']['volumes']['match_unnamed_m']} m if unnamed; two of three samples must "
        f"agree): **{vs['city_lines_matched']:,} of {vs['city_lines']:,} City segments "
        f"({100 * vs['city_lines_matched'] / vs['city_lines']:.0f}%; "
        f"{100 * vs['city_km_matched'] / vs['city_km']:.0f}% of City km)** matched; "
        f"{vs['osm_segments_matched']:,} OSM segments carry a City AADT. Unmatched City "
        f"segments by City class: {vs['unmatched_by_class']}.",
        "",
        markdown_table(cov, floats="{:,.1f}"),
        "",
        "OSM segments without a City value take the median City AADT of matched segments of "
        "the same OSM class (Ottawa showed Toronto's volume model transfers poorly). "
        "Residential streets are mostly unmatched: the City counts few local streets, and the "
        "ones it counts are likely busier than typical, so local-street volumes are probably "
        f"overstated. {vs['mto_segments']} OSM segments on Hwy 401 / 402 take MTO 2024 AADT.",
        "",
        "Entering AADT per intersection = half the two-way AADT of each approach, summed. "
        "Checked on Toronto, where entering volume is measured by turning counts: on "
        f"{chk['n']:,} counted intersections this approximation has Spearman "
        f"{chk['spearman']:.2f} with the counts and runs low by a median factor of "
        f"{chk['median_ratio']:.2f} (median abs error {chk['median_abs_pct_err']:.0f}%), "
        "so London's entering volumes are divided by that factor. Major / minor approach "
        "volumes are defined exactly as Toronto's.",
        "",
        "## Hwy 401 / 402 (MTO)",
        "",
        "Sections mapped with Toronto's interchange method (naming rule per highway: "
        f"{r['rules']}); 2015-2019 pooled collision rates per million vehicle-km:",
        "",
        markdown_table(secs, floats="{:,.2f}"),
        "",
        "## Risk",
        "",
        "Intersection: Toronto SPF expected crashes/yr at Toronto's 2025 level / (entering "
        "AADT x 365), charged once per consolidated intersection. Segments: Toronto's mid-block "
        "crashes per million vehicle-km by class; 401 / 402 MTO section rates; Highbury "
        "Avenue's freeway section Toronto's 'other expressway' rate.",
        "",
        f"SPF expected crashes/yr: total {risk['risk'].sum():,.0f}, median "
        f"{risk['risk'].median():.2f}; per million entering vehicles median "
        f"{risk['risk_per_mev'].median():.2f}. Highest SPF expectation:",
        "",
        markdown_table(top_spf, floats="{:,.2f}"),
        "",
        "## Sanity check: London Police's published top-10 collision intersections",
        "",
        "London Police published yearly top-10 high-collision intersection lists for "
        "2014-2016 (from news coverage). They are not a rigorous label set: police-reported "
        "counts with no exposure adjustment, a decade old, and sometimes ranked by different "
        "criteria. Each is matched to the consolidated intersection whose OSM names contain "
        "both streets (the busiest if several). Percentile 100 = highest in London.",
        "",
        markdown_table(
            top.drop(columns=["intersection_id"], errors="ignore").astype({"year": str}),
            floats="{:,.1f}",
        ),  # fmt: skip
        "",
        markdown_table(summ, floats="{:.1f}"),
        "",
        f"Of {len(distinct)} distinct listed intersections, the SPF ranks {s_better} higher "
        f"than volume alone does; the SPF puts {int(summ.loc[0, 'in top 1%'])} and volume "
        f"alone {int(summ.loc[1, 'in top 1%'])} in London's top 1%. In a "
        "city without crash data, most of what the SPF knows about which intersections see "
        "the most crashes is how much traffic they carry, which is also what police counts "
        "track. The check shows the SPF is not wrong about London's known hotspots; it cannot "
        "show the SPF beats volume.",
        "",
        "## Routing (200 random trips, 3-20 km)",
        "",
        f"k recalibrated with Toronto's rule (at λ = 1 the median route's risk cost equals "
        f"its travel time): **{r['calibration']['k_seconds_per_crash']:.1f} s** per expected "
        "crash per million trips (Toronto 31.1).",
        "",
        markdown_table(trade, floats="{:.3f}"),
        "",
        "## Limitations",
        "",
        "- No London crash records: risk is a Toronto model's expectation, never checked "
        "against London outcomes beyond the decade-old police top-10 lists. Relative risk "
        "only.",
        "- Volume layer age unknown (single processing date, no count dates); AADT rounded to "
        "500; most local streets have no count and use a class median.",
        "- Entering volume is approximated from segment volumes and rescaled with a factor "
        "measured in Toronto.",
        "- Segment rates and the SPF level are Toronto's; London's own reporting level and "
        "road design are not reflected.",
        "- Static features are today's OSM; free-flow travel times.",
        "",
    ]
    path.write_text("\n".join(lines))
