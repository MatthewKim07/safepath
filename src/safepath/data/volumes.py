"""Download and parse City of Toronto traffic counts from the open-data CKAN API.

Two packages are used, both found with `package_search` at run time (never by hardcoded ID):

- Turning Movement Counts (TMC): entering motor-vehicle volumes per intersection approach over
  8 non-continuous hours (before Sept 2023) or 14 continuous hours (after).
- Speed / Volume / Classification counts (SVC, also called ATR): 15-minute mid-block volumes
  per direction over 1-7 full days.

Every resource is cached under data/raw/volumes/ and checked against the size CKAN reports;
a manifest records the package, resource, licence, and modification time actually used.
Outputs in data/processed/:

- tmc_counts.parquet   one row per TMC: location, date, window type, entering volume by approach
- svc_counts.parquet   one row per SVC: location, dates, duration, City daily volumes, directions
- svc_15min.parquet    15-minute SVC volumes (both directions) from the raw files, for factors
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from safepath.fetch import FetchError, with_retries

log = logging.getLogger(__name__)

TMC_COUNTS = "tmc_counts.parquet"
SVC_COUNTS = "svc_counts.parquet"
SVC_15MIN = "svc_15min.parquet"
MANIFEST = "manifest.json"

# Counted time windows by TMC duration code, verified against every raw 2010-2026 count
# (reports/volumes.md): 8R and 8S are 8 hours in four or five blocks, 14 is 06:00-20:00.
TMC_WINDOWS: dict[str, list[tuple[str, str]]] = {
    "8R": [("07:30", "09:30"), ("10:00", "12:00"), ("13:00", "15:00"), ("16:00", "18:00")],
    "8S": [
        ("07:30", "09:30"),
        ("10:00", "11:00"),
        ("12:00", "13:30"),
        ("14:15", "15:45"),
        ("16:00", "18:00"),
    ],
    "14": [("06:00", "20:00")],
}
APPROACHES = ("n", "e", "s", "w")


def raw_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["raw_dir"]) / "volumes"


def _get_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise FetchError(f"CKAN call {url} failed: {body.get('error')}")
    return body["result"]


def find_package(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The package whose title matches `spec['title']` among `package_search` results."""
    v = cfg["volumes"]
    base = v["ckan_base"].rstrip("/")
    result = with_retries(
        lambda: _get_json(
            f"{base}/package_search", {"q": spec["query"], "rows": 20}, v["timeout_seconds"]
        ),
        what=f"CKAN package_search '{spec['query']}'",
        attempts=v["fetch_attempts"],
        backoff_seconds=v["backoff_seconds"],
    )
    matches = [p for p in result["results"] if p["title"].strip() == spec["title"]]
    if len(matches) != 1:
        titles = [p["title"] for p in result["results"]]
        raise FetchError(f"expected one package titled {spec['title']!r}, search gave {titles}")
    return matches[0]


def download_resource(cfg: dict[str, Any], resource: dict[str, Any], dest: Path) -> Path:
    """Stream a resource to `dest` (via a .part file); skip if a same-size copy is cached."""
    size = resource.get("size")
    if dest.exists() and (size is None or dest.stat().st_size == int(size)):
        return dest
    v = cfg["volumes"]

    def fetch() -> Path:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(resource["url"], stream=True, timeout=v["timeout_seconds"]) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        if size is not None and tmp.stat().st_size != int(size):
            raise FetchError(
                f"{resource['name']}: got {tmp.stat().st_size} bytes, CKAN says {size}"
            )
        tmp.replace(dest)
        return dest

    log.info("downloading %s", resource["name"])
    return with_retries(
        fetch,
        what=f"download of {resource['name']}",
        attempts=v["fetch_attempts"],
        backoff_seconds=v["backoff_seconds"],
    )


def download_counts(cfg: dict[str, Any]) -> dict[str, dict[str, Path]]:
    """Fetch the configured resources of each package; returns {package: {resource: path}}."""
    out_dir = raw_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    paths: dict[str, dict[str, Path]] = {}
    for key, spec in cfg["volumes"]["packages"].items():
        pkg = find_package(cfg, spec)
        by_name = {r["name"]: r for r in pkg["resources"]}
        missing = [n for n in spec["resources"] if n not in by_name]
        if missing:
            raise FetchError(f"package {pkg['name']} has no resources named {missing}")
        paths[key] = {}
        manifest[key] = {
            "package_name": pkg["name"],
            "package_id": pkg["id"],
            "title": pkg["title"],
            "license": pkg.get("license_title"),
            "refresh_rate": pkg.get("refresh_rate"),
            "resources": [],
        }
        for name in spec["resources"]:
            res = by_name[name]
            dest = out_dir / Path(res["url"]).name
            paths[key][name] = download_resource(cfg, res, dest)
            manifest[key]["resources"].append(
                {
                    "name": name,
                    "id": res["id"],
                    "url": res["url"],
                    "size": res.get("size"),
                    "last_modified": res.get("last_modified"),
                    "file": dest.name,
                }
            )
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return paths


def _minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def window_bins(duration: str) -> list[int]:
    """Start minute-of-day of every 15-minute bin a TMC of this duration code counts."""
    return [
        m for start, end in TMC_WINDOWS[duration] for m in range(_minutes(start), _minutes(end), 15)
    ]


def window_hours(duration: str) -> float:
    return len(window_bins(duration)) / 4


def parse_tmc(path: Path) -> pd.DataFrame:
    """One row per TMC. `entering` is the counted motor-vehicle volume summed over approaches
    (the City's total_vehicle; checked equal to the approach sum)."""
    raw = pd.read_csv(path, low_memory=False)
    raw["count_duration"] = raw["count_duration"].astype(str)
    unknown = set(raw["count_duration"]) - set(TMC_WINDOWS)
    if unknown:
        raise ValueError(f"unknown TMC duration codes {sorted(unknown)}")
    appr = raw[[f"{a}_appr_vehicle" for a in APPROACHES]].sum(axis=1)
    if not (appr == raw["total_vehicle"]).all():
        raise ValueError("TMC total_vehicle differs from the sum of approach volumes")
    out = pd.DataFrame(
        {
            "count_id": raw["count_id"],
            "location_name": raw["location_name"],
            "lon": raw["longitude"],
            "lat": raw["latitude"],
            "centreline_type": raw["centreline_type"],
            "centreline_id": raw["centreline_id"],
            "date": pd.to_datetime(raw["count_date"]),
            "duration": raw["count_duration"],
            "entering": raw["total_vehicle"],
            "heavy_pct": raw["total_heavy_pct"],
        }
    )
    for a in APPROACHES:
        out[f"{a}_entering"] = raw[f"{a}_appr_vehicle"]
    out["window_hours"] = out["duration"].map(window_hours)
    return out.sort_values("count_id").reset_index(drop=True)


def tmc_raw_windows(paths: list[Path]) -> pd.DataFrame:
    """15-minute bins (start minute-of-day) present per count in the raw TMC files."""
    frames = [pd.read_csv(p, usecols=["count_id", "start_time"]) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(raw["start_time"])
    raw["minute"] = t.dt.hour * 60 + t.dt.minute
    return raw[["count_id", "minute"]]


def attach_raw_windows(tmc: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Add `raw_bins` (bins in the raw file, NaN when the count has no raw data) and
    `window_ok` (raw bins are exactly the standard window for the duration code)."""
    sig = bins.groupby("count_id")["minute"].agg(lambda s: tuple(sorted(s)))
    out = tmc.copy()
    found = out["count_id"].map(sig)
    out["raw_bins"] = found.map(lambda s: len(s) if isinstance(s, tuple) else float("nan"))
    expected = out["duration"].map(lambda d: tuple(window_bins(d)))
    out["window_ok"] = pd.array(
        [(s == e) if isinstance(s, tuple) else pd.NA for s, e in zip(found, expected, strict=True)],
        dtype="boolean",
    )
    return out


def load_svc_15min(paths: list[Path]) -> pd.DataFrame:
    """15-minute volumes per (count, direction). Speed and classification files carry volume
    split into bins, so their bins are summed."""
    frames = []
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        if "volume_15min" in cols:
            f = pd.read_csv(p, usecols=["count_id", "time_start", "direction", "volume_15min"])
            f = f.rename(columns={"volume_15min": "volume"})
        else:
            bins = [c for c in cols if c.startswith("vol_")]
            f = pd.read_csv(p, usecols=["count_id", "time_start", "direction", *bins])
            f["volume"] = f[bins].sum(axis=1)
            f = f[["count_id", "time_start", "direction", "volume"]]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["time_start"] = pd.to_datetime(out["time_start"])
    dup = out.duplicated(["count_id", "time_start", "direction"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate SVC 15-minute rows across raw files")
    return out


def parse_svc(path: Path, raw15: pd.DataFrame) -> pd.DataFrame:
    """One row per SVC with the City's daily volumes; `directions` comes from the raw files
    (empty when the count's raw data was not downloaded)."""
    raw = pd.read_csv(path, low_memory=False)
    dirs = raw15.groupby("count_id")["direction"].agg(lambda s: ",".join(sorted(set(s))))
    out = pd.DataFrame(
        {
            "count_id": raw["count_id"],
            "count_type": raw["count_type"],
            "location_name": raw["location_name"],
            "lon": raw["longitude"],
            "lat": raw["latitude"],
            "centreline_id": raw["centreline_id"],
            "date_start": pd.to_datetime(raw["count_date_start"]),
            "date_end": pd.to_datetime(raw["count_date_end"]),
            "duration_h": raw["count_duration"].astype(int),
            "avg_daily": raw["avg_daily_vol"],
            "avg_weekday_daily": raw["avg_weekday_daily_vol"],
            "avg_weekend_daily": raw["avg_weekend_daily_vol"],
            "avg_speed": raw["avg_speed"],
            "heavy_pct": raw["avg_heavy_pct"],
        }
    )
    out["directions"] = out["count_id"].map(dirs).fillna("")
    out["n_directions"] = out["directions"].map(lambda s: len(s.split(",")) if s else 0)
    return out.sort_values("count_id").reset_index(drop=True)


def run_volume_download(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    paths = download_counts(cfg)
    processed = Path(cfg["paths"]["processed_dir"])
    tmc_files, svc_files = paths["tmc"], paths["svc"]
    tmc = parse_tmc(tmc_files["tmc_summary_data.csv"])
    tmc = attach_raw_windows(tmc, tmc_raw_windows([p for n, p in tmc_files.items() if "raw" in n]))
    raw15 = load_svc_15min([p for n, p in svc_files.items() if "raw" in n])
    svc = parse_svc(svc_files["svc_summary_data.csv"], raw15)
    tmc.to_parquet(processed / TMC_COUNTS, index=False)
    svc.to_parquet(processed / SVC_COUNTS, index=False)
    raw15.to_parquet(processed / SVC_15MIN, index=False)
    log.info(
        "parsed %d TMCs at %d locations, %d SVCs at %d locations, %d SVC 15-min rows",
        len(tmc),
        tmc["centreline_id"].nunique(),
        len(svc),
        svc["centreline_id"].nunique(),
        len(raw15),
    )
    return {"tmc": tmc, "svc": svc, "svc_15min": raw15}
