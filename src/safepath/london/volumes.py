"""London traffic volumes onto the OSM graph (CP10).

Source: City of London open data, OpenData_Transportation/MapServer/21 "Traffic Volumes":
one polyline per volume segment with AADT (`VolumeCount`, rounded by the City to 500s),
street name and classification. The City says it is reprocessed about once a year; every
row carries the same LastEditDate, and count dates are not published.

Matching: each undirected OSM segment is sampled at 25/50/75% of its length. A sample takes
the nearest City line within `match_m` whose street name agrees with the OSM name (same
street key as Toronto's count matcher), or within `match_unnamed_m` when the OSM segment
has no name. A segment gets the City AADT voted by most of its samples (at least two).
Unmatched segments take the median City AADT of matched segments of the same OSM highway
class: Ottawa showed Toronto's volume model transfers poorly (R² 0.57 on Ottawa counts),
and London's matched segments are a direct local measure. Classes the City barely covers
(residential streets) are therefore estimated from the few local streets it does count,
which are likely busier than typical: documented in reports/london.md.

Intersections: entering AADT = half the sum of the two-way AADT of the segments entering it
(each approach carries about half its two-way volume towards the intersection), and major /
minor approach volumes exactly as Toronto's (busiest and second-busiest named road,
features/volume_features.approach_volumes). Toronto measures entering volume with turning
counts instead; the approximation is checked against Toronto's own counts in the report.

400-series: Hwy 401 and 402 edges inside the city get MTO's 2024 AADT and 2015-2019
collision rate per section (data/mto.py mapping, London interchanges).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from safepath.data.aadt import street_key
from safepath.data.download import LayerDownloader
from safepath.features.volume_features import approach_volumes
from safepath.graph import EDGE_MAP, NODE_MAP, SEGMENTS

log = logging.getLogger(__name__)

RAW_VOLUMES = "london_traffic_volumes.parquet"
RAW_TYPES = "london_intersection_types.parquet"
MATCHED = "segment_volumes.parquet"
LONDON_AADT_EDGES = "aadt_edges.parquet"  # same file names as Toronto, in London's dir
LONDON_EXPOSURE = "intersection_exposure.parquet"
VOLUME_STATS = "volume_stats.json"
SAMPLES = (0.25, 0.5, 0.75)


def download(cfg: dict[str, Any]) -> tuple[Path, Path]:
    v = cfg["london"]["volumes"]
    raw = Path(cfg["paths"]["raw_dir"])
    out = []
    for url, name in ((v["url"], RAW_VOLUMES), (v["intersection_types_url"], RAW_TYPES)):
        d = LayerDownloader(url, raw / "pages" / name.removesuffix(".parquet"), raw / name,
                            timeout=float(v["timeout_seconds"]), geometry=True)  # fmt: skip
        out.append(d.run())
    return out[0], out[1]


def city_lines(path: Path, crs: str) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
    geoms = [
        shapely.MultiLineString(json.loads(p)) if isinstance(p, str) else None
        for p in df["geom_paths"]
    ]
    g = gpd.GeoDataFrame(df.drop(columns="geom_paths"), geometry=geoms, crs="EPSG:4326")
    g = g[g.geometry.notna() & g["VolumeCount"].gt(0)].to_crs(crs)
    # the City abbreviates parkway as "Pky" (Veterans Memorial Pky); Toronto's keys know "Pkwy"
    g["key"] = g["StreetName"].str.replace(r"\bPky\b", "Pkwy", regex=True).map(street_key)
    return g.reset_index(drop=True)


def osm_key(name: Any) -> str:
    return street_key(name) if isinstance(name, str) and name else ""


def match_segments(
    segments: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, match_m: float, unnamed_m: float
) -> pd.DataFrame:
    """City line (row index) and AADT per OSM segment, or NaN when unmatched."""
    seg = segments.reset_index(drop=True)
    keys = seg["name"].map(osm_key).to_numpy()
    pts, owner = [], []
    for f in SAMPLES:
        pts.append(shapely.line_interpolate_point(seg.geometry.values, f, normalized=True))
        owner.append(np.arange(len(seg)))
    p = gpd.GeoDataFrame({"seg": np.concatenate(owner)}, geometry=np.concatenate(pts),
                         crs=seg.crs)  # fmt: skip
    cand = gpd.sjoin_nearest(p, lines[["key", "geometry"]], how="inner", max_distance=match_m,
                             distance_col="d")  # fmt: skip
    cand = cand.rename(columns={"index_right": "line"})
    k = keys[cand["seg"].to_numpy()]
    named_ok = (k != "") & (k == cand["key"].to_numpy())
    unnamed_ok = (k == "") & (cand["d"].to_numpy() <= unnamed_m)
    cand = cand[named_ok | unnamed_ok]
    votes = cand.groupby(["seg", "line"]).size().rename("votes").reset_index()
    votes = votes.sort_values(["seg", "votes"], ascending=[True, False]).drop_duplicates("seg")
    votes = votes[votes["votes"] >= 2]
    out = pd.DataFrame({"segment_id": seg["segment_id"], "line": np.nan, "aadt_city": np.nan})
    out.loc[votes["seg"].to_numpy(), "line"] = votes["line"].to_numpy()
    hit = out["line"].notna()
    out.loc[hit, "aadt_city"] = lines.loc[
        out.loc[hit, "line"].astype(int), "VolumeCount"
    ].to_numpy()
    return out


def coverage_table(seg: pd.DataFrame) -> pd.DataFrame:
    g = seg.groupby("highway")
    t = pd.DataFrame({
        "segments": g.size(),
        "km": g["length"].sum() / 1000,
        "% segments with City AADT": 100 * g["matched"].mean(),
        "% km with City AADT": 100 * g.apply(
            lambda d: d.loc[d["matched"], "length"].sum() / d["length"].sum(),
            include_groups=False),
        "median City AADT": g["aadt_city"].median(),
    })  # fmt: skip
    return t.sort_values("km", ascending=False).reset_index()


def run_london_volumes(cfg: dict[str, Any], mto_segments: pd.DataFrame) -> dict[str, Any]:
    """Segment AADT (City where matched, class median otherwise, MTO on 401/402), directed
    edge AADT, and intersection exposure. `mto_segments` maps segment_id -> MTO AADT."""
    processed = Path(cfg["paths"]["processed_dir"])
    v = cfg["london"]["volumes"]
    vol_path, _ = download(cfg)
    lines = city_lines(vol_path, cfg["graph"]["crs"])
    segments = gpd.read_parquet(processed / SEGMENTS)
    m = match_segments(segments, lines, v["match_m"], v["match_unnamed_m"])
    seg = segments[["segment_id", "highway", "name", "length", "oneway"]].merge(m, on="segment_id")
    seg["matched"] = seg["aadt_city"].notna()
    class_median = seg.groupby("highway")["aadt_city"].median()
    fallback = float(seg["aadt_city"].median())
    seg["aadt"] = seg["aadt_city"].fillna(seg["highway"].map(class_median)).fillna(fallback)
    seg["source"] = np.where(seg["matched"], "city", "class median")
    mto = mto_segments.set_index("segment_id")
    on_mto = seg["segment_id"].isin(mto.index)
    seg.loc[on_mto, "aadt"] = seg.loc[on_mto, "segment_id"].map(mto["aadt_2024"]).to_numpy()
    seg.loc[on_mto, "source"] = "mto"
    seg.to_parquet(processed / MATCHED, index=False)

    used = lines.index.isin(m["line"].dropna().astype(int))
    stats = {
        "city_lines": len(lines),
        "city_km": float(lines.length.sum() / 1000),
        "city_lines_matched": int(used.sum()),
        "city_km_matched": float(lines.loc[used].length.sum() / 1000),
        "osm_segments": len(seg),
        "osm_segments_matched": int(seg["matched"].sum()),
        "osm_km": float(seg["length"].sum() / 1000),
        "osm_km_matched": float(seg.loc[seg["matched"], "length"].sum() / 1000),
        "mto_segments": int(on_mto.sum()),
        "class_median": {k: float(x) for k, x in class_median.dropna().items()},
        "classification_counts": lines["StreetClassification"].value_counts().to_dict(),
        "last_edit": str(
            pd.to_datetime(pd.read_parquet(vol_path)["LastEditDate"].max(), unit="ms").date()
        ),  # fmt: skip
    }
    unmatched = lines.loc[~used]
    stats["unmatched_by_class"] = unmatched["StreetClassification"].value_counts().to_dict()

    # directed edges: two-way segment AADT, half per direction (all of it on one-way streets)
    edges = pd.read_parquet(processed / EDGE_MAP).merge(
        seg[["segment_id", "highway", "aadt", "source"]], on="segment_id"
    )
    edges = edges.rename(columns={"aadt": "aadt_segment"})
    oneway = edges["segment_id"].map(seg.set_index("segment_id")["oneway"].astype(str).eq("True"))
    edges["dir_share"] = np.where(oneway, 1.0, 0.5)
    edges["aadt_dir"] = edges["aadt_segment"] * edges["dir_share"]
    edges = edges.merge(
        mto.reset_index()[["segment_id", "hwy", "lhrs"]].rename(
            columns={"hwy": "mto_hwy", "lhrs": "mto_lhrs"}),
        on="segment_id", how="left",
    )  # fmt: skip
    edges.to_parquet(processed / LONDON_AADT_EDGES, index=False)

    node_map = pd.read_parquet(processed / NODE_MAP)
    cluster = node_map.set_index("osmid")["intersection_id"]
    e = edges.assign(cu=edges["u"].map(cluster), cv=edges["v"].map(cluster))
    ent = e[e["cv"].notna() & (e["cu"] != e["cv"])]
    # each undirected approach counted once: half of its two-way AADT enters the junction
    appr = ent.drop_duplicates(["cv", "segment_id"])
    entering = appr.groupby(appr["cv"].astype(int))["aadt_segment"].sum() / 2
    expo = pd.DataFrame({"entering_aadt": entering}).rename_axis("intersection_id")
    expo = expo.join(approach_volumes(edges, node_map, segments[["segment_id", "name"]]))
    ids = pd.read_parquet(processed / "intersections.parquet", columns=["intersection_id"])
    expo = expo.reindex(ids["intersection_id"])
    expo["entering_aadt"] = expo["entering_aadt"].fillna(expo["entering_aadt"].median())
    expo["major_aadt"] = expo["major_aadt"].fillna(expo["entering_aadt"])
    expo.reset_index().to_parquet(processed / LONDON_EXPOSURE, index=False)
    stats["intersections"] = len(expo)
    stats["intersections_with_city_approach"] = int(ent[ent["source"] == "city"]["cv"].nunique())
    stats["entering_median"] = float(expo["entering_aadt"].median())
    (processed / VOLUME_STATS).write_text(json.dumps(stats, indent=2, default=str))
    log.info("london volumes: %d of %d OSM segments matched", stats["osm_segments_matched"],
             len(seg))  # fmt: skip
    return {"stats": stats, "coverage": coverage_table(seg), "segments": seg, "exposure": expo}


def entering_check(toronto: dict[str, Any]) -> dict[str, float]:
    """How close is 'half the approach two-way AADT' to Toronto's measured entering volume
    (turning-movement counts, latest count per intersection)?"""
    processed = Path(toronto["paths"]["processed_dir"])
    edges = pd.read_parquet(processed / "aadt_edges.parquet",
                            columns=["u", "v", "segment_id", "aadt_segment"])  # fmt: skip
    cluster = pd.read_parquet(processed / NODE_MAP).set_index("osmid")["intersection_id"]
    e = edges.assign(cu=edges["u"].map(cluster), cv=edges["v"].map(cluster))
    ent = e[e["cv"].notna() & (e["cu"] != e["cv"])].drop_duplicates(["cv", "segment_id"])
    approx = ent.groupby(ent["cv"].astype(int))["aadt_segment"].sum() / 2
    ai = pd.read_parquet(processed / "aadt_intersections.parquet")
    counted = ai[ai["aadt_count"].notna()].set_index("intersection_id")["aadt_count"]
    j = pd.concat([counted.rename("tmc"), approx.rename("approx")], axis=1, join="inner")
    j = j[(j["tmc"] > 0) & (j["approx"] > 0)]
    err = np.log(j["approx"] / j["tmc"])
    return {"n": len(j), "median_ratio": float(np.exp(err.median())),
            "median_abs_pct_err": float(100 * np.median(np.abs(np.exp(err) - 1))),
            "spearman": float(j["approx"].corr(j["tmc"], method="spearman"))}  # fmt: skip
