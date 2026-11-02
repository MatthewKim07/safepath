"""City of Ottawa open data for the cross-city test: collisions and intersection volumes.

Sources (City of Ottawa ArcGIS Online, Open Government Licence - City of Ottawa); the exact
services are written to data/raw/ottawa/manifest.json and listed in reports/ottawa.md:

- Collisions: `YYYY_Tabular_Transportation_Collision_Data` (one layer per year 2013-2020,
  with an intersection / non-intersection field) and
  `Traffic_Collisions_by_Location_2017-2024_(excluding_2023)` (a newer edit of 2017-2024;
  2023 is not published). Label years 2017+ come from the combined layer, 2013-2016 from the
  yearly layers. Every layer is paginated by object id, cached page by page, and checked
  against returnCountOnly (the same downloader as the TPS data).
- Intersection volumes: `Transportation_Intersection_Volume(s)_YYYY` (and the 2022 layer
  `Intersection_Volume_2022_w_lat_long`). The City publishes factored 24-hour AADT; field
  names change from year to year (VOLUME_FIELDS). 2025 also lists counts the City did not
  factor (`AADT_Calculated` = 0); those are dropped.

Cleaning produces the Toronto schema (event_id, occ_date, year, lon, lat, injury_or_fatal,
is_fail_to_remain) plus Ottawa extras, and removes freeway crashes: Hwy 416 / 417 are OPP
roads whose collisions the City file includes but the TPS file does not.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safepath.data.download import LayerDownloader

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
CLEAN_FILENAME = "collisions_clean.parquet"  # same name as Toronto's, in Ottawa's dir
FREEWAY_CRASHES = "freeway_crashes.parquet"
VOLUME_COUNTS = "volume_counts.parquet"
CLEAN_STATS = "collisions_clean_stats.json"

# per-year volume layer fields: (location name, AADT, date)
VOLUME_FIELDS: dict[int, tuple[str, str, str]] = {
    2016: ("Intersecti", "All_Motori", "Data_Colle"),
    2017: ("Intersecti", "All_Motori", "Data_Colle"),
    2018: ("Intersecti", "All_Motori", "Data_Colle"),
    2019: ("Intersection", "All_Motorized_Vehicles_AADT_24Hour_Volume", "Date_Collected"),
    2021: ("Intersection", "All_Motorized_Vehicles_AADT_24H", "Date_Collected"),
    2022: ("Intersection", "All_Motorized_Vehicles_AADT_24_", "Date"),
    2023: ("Intersection", "All_Motorized_Vehicles_AADT_24_", "Date"),
    2024: ("Intersection", "All_Motori", "Study_Date"),
    2025: ("Intersection", "Total_Adjusted_Volume__24h_", "Study_Date"),
}
SEVERE_CLASSES = {"01", "02"}  # 01 fatal injury, 02 non-fatal injury; 03 P.D. only


def _layer_url(cfg: dict[str, Any], service: str) -> str:
    return f"{cfg['ottawa']['host']}/{service}/FeatureServer/0/query"


def _downloader(cfg: dict[str, Any], service: str, out: Path, geometry: bool) -> LayerDownloader:
    o = cfg["ottawa"]
    raw = Path(cfg["paths"]["raw_dir"])
    return LayerDownloader(
        _layer_url(cfg, service),
        raw / "pages" / service,
        out,
        timeout=float(o["timeout_seconds"]),
        max_retries=int(o["max_retries"]),
        backoff=float(o["backoff_seconds"]),
        workers=4,
        geometry=geometry,
    )


def collision_services(cfg: dict[str, Any]) -> list[str]:
    c = cfg["ottawa"]["collisions"]
    return [c["yearly"].format(year=y) for y in c["yearly_years"]] + [c["combined"]]


def layer_path(cfg: dict[str, Any], service: str) -> Path:
    return Path(cfg["paths"]["raw_dir"]) / "layers" / f"{service}.parquet"


def download_ottawa(cfg: dict[str, Any]) -> dict[str, Any]:
    """Download every collision and volume layer (cached page by page, one parquet per
    layer, row count checked against the service) and write the manifest."""
    raw = Path(cfg["paths"]["raw_dir"])
    manifest: dict[str, Any] = {
        "host": cfg["ottawa"]["host"],
        "licence": cfg["ottawa"]["licence"],
        "collisions": {},
        "volumes": {},
    }
    for service in collision_services(cfg):
        d = _downloader(cfg, service, layer_path(cfg, service), geometry=False)
        manifest["collisions"][service] = {"url": d.url, "rows": len(pd.read_parquet(d.run()))}
    for year, service in cfg["ottawa"]["volumes"]["services"].items():
        d = _downloader(cfg, service, layer_path(cfg, service), geometry=True)
        rows = len(pd.read_parquet(d.run()))
        manifest["volumes"][service] = {"url": d.url, "rows": rows, "year": int(year)}
    (raw / MANIFEST).write_text(json.dumps(manifest, indent=2))
    log.info(
        "ottawa raw: %d collision rows, %d volume rows",
        sum(v["rows"] for v in manifest["collisions"].values()),
        sum(v["rows"] for v in manifest["volumes"].values()),
    )
    return manifest


# --- collisions ----------------------------------------------------------------------------

_FREEWAY_ROAD = re.compile(r"^(HIGHWAY|HWY) ?41[67]\b")
_GEO_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def location_parts(location: str) -> tuple[str, list[str]]:
    """Split a City location string into its kind and the roads it names.

    "A btwn B & C" is a mid-block location on A ("btwn"); "A/B @ C" is an intersection of
    A, B and C ("at"); anything else is "other" with '/'-separated parts.
    """
    s = _GEO_SUFFIX.sub("", location or "").strip()
    if " btwn " in s:
        return "btwn", [s.split(" btwn ")[0].strip()]
    if " @ " in s:
        return "at", [p.strip() for side in s.split(" @ ") for p in side.split("/") if p.strip()]
    return "other", [p.strip() for p in s.split("/") if p.strip()]


def is_freeway_location(location: str) -> bool:
    """True when every road the location names is a Hwy 416 / 417 mainline or ramp.

    Mid-block crashes on the highway or a ramp, and junctions of two ramps, are freeway
    crashes. A ramp terminal on a city street ("BRONSON AVE @ HWY 417 ... IC121AR51") is a
    surface intersection, the kind TPS also records at 400-series interchanges, so it stays.
    """
    _, roads = location_parts(location)
    return bool(roads) and all(_FREEWAY_ROAD.match(r) for r in roads)


def _code(series: pd.Series) -> pd.Series:
    """'02 - Non-fatal injury' -> '02'."""
    return series.astype("string").str.extract(r"^\s*(\d+)", expand=False)


def parse_yearly(raw: pd.DataFrame) -> pd.DataFrame:
    """One yearly tabular layer (2013-2020) in the common schema."""
    date = pd.to_datetime(raw["Accident_Date"], errors="coerce")
    hour = pd.to_datetime(raw["Accident_Time"], format="%I:%M %p", errors="coerce").dt.hour
    return pd.DataFrame(
        {
            "source_id": raw["Anom_ID"].astype(str),
            "occ_date": date,
            "hour": hour.astype("Int8"),
            "lon": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "lat": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "classification": raw["Classification_of_Accident"],
            "location_text": raw["Location"],
            "geo_id": raw["Geo_ID"],
            "accident_location": raw["Accident_Location"],
            "traffic_control": raw["Traffic_Control"],
            "light": raw["Light"],
            "surface": raw["Road_Surface_Condition"],
            "impact_type": raw["Initial_Impact_Type"],
            "environment": raw["Environment_Condition"],
            "n_vehicles": pd.to_numeric(raw["No__of_Vehicles"], errors="coerce"),
        }
    )


def parse_combined(raw: pd.DataFrame) -> pd.DataFrame:
    """The 2017-2024 layer in the common schema (no time of day, no intersection field)."""
    return pd.DataFrame(
        {
            "source_id": raw["ID"].astype(str),
            "occ_date": pd.to_datetime(raw["Accident_Date"], errors="coerce"),
            "hour": pd.array([pd.NA] * len(raw), dtype="Int8"),
            "lon": pd.to_numeric(raw["Long"], errors="coerce"),
            "lat": pd.to_numeric(raw["Lat"], errors="coerce"),
            "classification": raw["Classification_Of_Accident"],
            "location_text": raw["Location"],
            "geo_id": raw["Geo_ID"],
            "accident_location": pd.Series(pd.NA, index=raw.index, dtype="string"),
            "traffic_control": raw["Traffic_Control"],
            "light": raw["Light"],
            "surface": raw["Road_1_Surface_Condition"],
            "impact_type": raw["Initial_Impact_Type"],
            "environment": raw["Environment_Condition"],
            "n_vehicles": pd.to_numeric(raw["num_of_vehicles"], errors="coerce"),
        }
    )


def load_records(cfg: dict[str, Any]) -> pd.DataFrame:
    """Every collision record for the label years it is used for, in the common schema."""
    c = cfg["ottawa"]["collisions"]
    cut = int(c["use_combined_from"])
    frames = []
    for year in c["yearly_years"]:
        service = c["yearly"].format(year=year)
        df = parse_yearly(pd.read_parquet(layer_path(cfg, service))).assign(source_service=service)
        frames.append(df[df["occ_date"].dt.year < cut])
    df = parse_combined(pd.read_parquet(layer_path(cfg, c["combined"])))
    frames.append(df[df["occ_date"].dt.year >= cut].assign(source_service=c["combined"]))
    return pd.concat(frames, ignore_index=True)


def clean_records(df: pd.DataFrame, bbox: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """Derive the Toronto columns, drop non-reportable and badly located records, and flag
    freeway crashes. Returns all kept records (freeway ones flagged) and drop counts."""
    stats: dict[str, Any] = {"raw_rows": len(df)}
    df = df[df["occ_date"].notna()].copy()
    stats["dropped_no_date"] = stats["raw_rows"] - len(df)
    df["year"] = df["occ_date"].dt.year.astype("int16")
    df["month"] = df["occ_date"].dt.month.astype("int8")
    df["dow"] = df["occ_date"].dt.day_of_week.astype("int8")
    df["event_id"] = df["year"].astype(str) + "-" + df["source_id"].str.split("--").str[-1]
    if df["event_id"].duplicated().any():
        raise ValueError("duplicate Ottawa event ids after combining layers")
    code = _code(df["classification"])
    # 04 "non-reportable" (below the reporting threshold) appears in 2017+ only, and in bulk
    # in 2024 (2,358 vs <100 a year before): dropped so years and cities stay comparable.
    nonrep = code.eq("04").fillna(False).to_numpy()
    stats["dropped_non_reportable_by_year"] = (
        df.loc[nonrep].groupby("year").size().astype(int).to_dict()
    )
    df, code = df[~nonrep], code[~nonrep]
    df["injury_or_fatal"] = code.isin(SEVERE_CLASSES).to_numpy()
    df["is_fatal"] = code.eq("01").fillna(False).to_numpy()
    df["is_fail_to_remain"] = pd.array([pd.NA] * len(df), dtype="boolean")  # not published
    inside = df["lon"].between(bbox["min_lon"], bbox["max_lon"]) & df["lat"].between(
        bbox["min_lat"], bbox["max_lat"]
    )
    stats["dropped_outside_bbox_by_year"] = df.loc[~inside].groupby("year").size().to_dict()
    stats["dropped_zero_coords"] = int(((df["lat"] == 0) | df["lat"].isna()).sum())
    df = df[inside].copy()
    df["location_kind"] = df["location_text"].fillna("").map(lambda s: location_parts(s)[0])
    df["is_freeway"] = df["location_text"].fillna("").map(is_freeway_location)
    return df.reset_index(drop=True), stats


# --- volumes -------------------------------------------------------------------------------


def _parse_dates(values: pd.Series) -> pd.Series:
    """Volume layers store dates as epoch ms, '2016-Jul-13', or '15-Jul-15' (yy-Mon-dd)."""
    num = pd.to_numeric(values, errors="coerce")
    out = pd.to_datetime(num, unit="ms", errors="coerce")
    text = values.where(num.isna()).astype("string")
    for fmt in ("%Y-%b-%d", "%y-%b-%d", "%Y-%m-%d"):
        miss = out.isna() & text.notna()
        out[miss] = pd.to_datetime(text[miss], format=fmt, errors="coerce")
    return out


def parse_volumes(cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per published intersection count: name, factored AADT, date, WGS84 point."""
    frames = []
    for year, service in cfg["ottawa"]["volumes"]["services"].items():
        raw = pd.read_parquet(layer_path(cfg, service))
        name, aadt, date = VOLUME_FIELDS[int(year)]
        df = pd.DataFrame(
            {
                "source_service": service,
                "layer_year": int(year),
                "location_name": raw[name].astype("string").str.strip(),
                "aadt": pd.to_numeric(
                    raw[aadt].astype("string").str.replace(",", ""), errors="coerce"
                ),
                "count_date": _parse_dates(raw[date]),
                "lon": pd.to_numeric(raw["geom_lon"], errors="coerce"),
                "lat": pd.to_numeric(raw["geom_lat"], errors="coerce"),
                "factored": True,
            }
        )
        if "AADT_Calculated" in raw:  # 2025 lists unfactored counts too
            df["factored"] = pd.to_numeric(raw["AADT_Calculated"], errors="coerce").eq(1)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["count_year"] = out["count_date"].dt.year.fillna(out["layer_year"]).astype(int)
    out.insert(0, "count_id", range(len(out)))
    return out


def clean_volumes(vol: pd.DataFrame, bbox: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    stats = {"rows": len(vol), "unfactored": int((~vol["factored"]).sum())}
    ok = vol["factored"] & vol["aadt"].gt(0)
    stats["no_aadt"] = int((vol["factored"] & ~vol["aadt"].gt(0)).sum())
    inside = vol["lon"].between(bbox["min_lon"], bbox["max_lon"]) & vol["lat"].between(
        bbox["min_lat"], bbox["max_lat"]
    )
    stats["outside_bbox"] = int((ok & ~inside).sum())
    out = vol[ok & inside].reset_index(drop=True)
    stats["kept"] = len(out)
    return out, stats


# --- step ----------------------------------------------------------------------------------


def run_ottawa_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """Download, clean, and report: collisions_clean.parquet (freeway crashes removed),
    freeway_crashes.parquet, volume_counts.parquet, reports/ottawa/data_report.md."""
    manifest = download_ottawa(cfg)
    processed = Path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    bbox = cfg["clean"]["bbox"]
    records, stats = clean_records(load_records(cfg), bbox)
    freeway = records[records["is_freeway"]]
    clean = records[~records["is_freeway"]].drop(columns="is_freeway")
    clean.to_parquet(processed / CLEAN_FILENAME, index=False)
    freeway.to_parquet(processed / FREEWAY_CRASHES, index=False)
    stats["freeway_removed"] = len(freeway)
    stats["clean_rows"] = len(clean)
    vol, vstats = clean_volumes(parse_volumes(cfg), bbox)
    vol.to_parquet(processed / VOLUME_COUNTS, index=False)
    stats["volumes"] = vstats
    (processed / CLEAN_STATS).write_text(json.dumps(stats, indent=2, default=int))
    write_data_report(cfg, manifest, records, stats, vol)
    log.info("ottawa clean: %d collisions (%d freeway removed), %d counts", len(clean),
             len(freeway), len(vol))  # fmt: skip
    return stats


def year_table(records: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    g = records.groupby("year")
    months = g["month"].nunique()
    out = pd.DataFrame(
        {
            "source": g["source_service"]
            .first()
            .map(lambda s: "yearly layer" if "Tabular" in s else "2017-2024 layer"),
            "records kept": g.size(),
            "freeway removed": g["is_freeway"].sum(),
            "used (non-freeway)": g.size() - g["is_freeway"].sum(),
            "% severe": 100 * g["injury_or_fatal"].mean(),
            "fatal": g["is_fatal"].sum(),
            "months present": months,
        }
    )
    nonrep = {int(k): v for k, v in stats["dropped_non_reportable_by_year"].items()}
    bbox = {int(k): v for k, v in stats["dropped_outside_bbox_by_year"].items()}
    out["non-reportable dropped"] = out.index.map(lambda y: nonrep.get(y, 0))
    out["bad coords dropped"] = out.index.map(lambda y: bbox.get(y, 0))
    out["partial?"] = np.where(months < 12, "PARTIAL", "")
    out = out.reset_index()
    out["year"] = out["year"].astype(str)
    return out


def write_data_report(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    records: pd.DataFrame,
    stats: dict[str, Any],
    vol: pd.DataFrame,
) -> None:
    from safepath.locations import markdown_table

    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    years = year_table(records, stats)
    fw = records[records["is_freeway"]]
    fw_roads = (
        fw["location_text"].fillna("").map(lambda s: location_parts(s)[1][0] if s else "")
        .str.replace(r"^HWY ?(41[67]).*", r"Hwy \1 ramp", regex=True)
        .str.replace(r"^HIGHWAY (41[67])$", r"Hwy \1 mainline", regex=True)
        .value_counts().rename_axis("road").reset_index(name="crashes")
    )  # fmt: skip
    kinds = records.groupby("location_kind").size().rename("records").reset_index()
    vby = vol.groupby("layer_year").agg(
        counts=("aadt", "size"),
        median_aadt=("aadt", "median"),
        p95_aadt=("aadt", lambda s: s.quantile(0.95)),
        first_date=("count_date", "min"),
        last_date=("count_date", "max"),
    )
    vby["first_date"] = vby["first_date"].dt.date.astype(str)
    vby["last_date"] = vby["last_date"].dt.date.astype(str)
    vby.index = vby.index.astype(str)
    vs = stats["volumes"]
    services = [
        {"kind": kind, "service": s, "rows": v["rows"]}
        for kind in ("collisions", "volumes")
        for s, v in manifest[kind].items()
    ]
    lines = [
        "# Ottawa collision and volume data",
        "",
        f"Source: City of Ottawa open data on ArcGIS Online ({manifest['host']}), "
        f"{manifest['licence']}. Every layer was paginated by object id and its row count "
        "checked against `returnCountOnly`.",
        "",
        markdown_table(pd.DataFrame(services), floats="{:.0f}"),
        "",
        "The service named `Traffic_Collision_Data` does not exist on the City's host (the "
        "only item with that title is a third-party copy whose URL returns 'Invalid URL'), so "
        "the City's own layers above are used. 2023 collisions are not published, and 2025 is "
        "not out yet.",
        "",
        "## Collisions per year",
        "",
        f"{stats['raw_rows']:,} records read for the years used; "
        f"{sum(stats['dropped_non_reportable_by_year'].values()):,} non-reportable records "
        "dropped (class 04, below the reporting threshold: under 100 a year until 2022, then "
        f"{stats['dropped_non_reportable_by_year'].get(2024, 0):,} in 2024, a recording "
        f"change); {sum(stats['dropped_outside_bbox_by_year'].values()):,} with zero or "
        "out-of-city coordinates dropped. Severe = fatal or non-fatal injury (classes 01-02).",
        "",
        markdown_table(years),
        "",
        "2024 looks different: its severe share is 8.8% against 16-20% before. Fatal and "
        "major-injury crashes are steady (26 and 118 in 2024 vs 23 and 112 in 2022) but "
        "crashes whose worst injury is minimal or minor drop from 1,887 to 1,104, and "
        "2,358 non-reportable records appear. This looks like a change in how minor-injury "
        "collisions are recorded in 2024, not a real drop, so the severe target is not "
        "comparable in the 2024 test year. The primary target (any crash) is unaffected "
        "as far as the totals show (12,693 non-freeway in 2024 vs 11,127 in 2022 and 14,800 "
        "in 2019).",
        "",
        f"Freeway crashes removed: **{stats['freeway_removed']:,}** "
        f"({100 * stats['freeway_removed'] / len(records):.1f}% of located records). Rule: "
        "every road the location string names is a Hwy 416 / 417 mainline or ramp (mid-block "
        "on the highway or a ramp, or a ramp-to-ramp junction). Ramp terminals on city "
        "streets are kept: they are surface intersections, the kind TPS also records at "
        "400-series interchanges. City expressways (Hwy 174, parkways) are kept, as the "
        "Gardiner and DVP are in Toronto.",
        "",
        markdown_table(fw_roads.head(8), floats="{:.0f}"),
        "",
        "Location string kinds (all located records): `at` = 'A @ B' intersection, `btwn` = "
        "'A btwn B & C' mid-block, `other` = blank or free text.",
        "",
        markdown_table(kinds, floats="{:.0f}"),
        "",
        "## Intersection volume counts",
        "",
        f"{vs['rows']:,} published counts; {vs['unfactored']} 2025 counts the City did not "
        f"factor to AADT dropped, {vs['no_aadt']} without a volume, {vs['outside_bbox']} "
        f"outside the city; {vs['kept']:,} kept. Values are the City's factored 24-hour AADT "
        "(all motorised vehicles entering the intersection). 2020 has no layer; 2021 counts "
        "are COVID-era and are never used. Matching to intersections and the "
        "one-count-per-intersection rule are in `reports/ottawa.md`.",
        "",
        markdown_table(vby.reset_index(), floats="{:.0f}"),
        "",
    ]
    (reports / "data_report.md").write_text("\n".join(lines))
