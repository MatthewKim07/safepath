"""Download ArcGIS FeatureServer layers; used for the TPS collision records and Ottawa's layers.

Pages are fetched as fixed OBJECTID windows (`lo < OBJECTID <= hi`, width = page size), so
every page is independent: each is cached as gzipped JSON, reruns skip cached windows, and an
interrupted download resumes where it stopped. If the service grows, the final window's upper
bound changes and that window is fetched again.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

RAW_FILENAME = "toronto_collisions.parquet"


class ArcGISError(RuntimeError):
    pass


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: float,
    max_retries: int,
    backoff: float,
) -> dict[str, Any]:
    """GET a JSON response, retrying with exponential backoff on network, HTTP, and ArcGIS errors.

    ArcGIS reports many failures as HTTP 200 with an `error` key, so the body is checked too.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params={**params, "f": "json"}, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if "error" in body:
                raise ArcGISError(body["error"])
            return body
        except (requests.RequestException, ValueError, ArcGISError) as exc:
            if attempt == max_retries:
                raise
            wait = backoff * 2**attempt
            log.warning(
                "request failed (%s); retry %d/%d in %.1fs", exc, attempt + 1, max_retries, wait
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def objectid_windows(min_id: int, max_id: int, size: int) -> list[tuple[int, int]]:
    """Half-open windows (lo, hi] of width `size` covering [min_id, max_id]."""
    return [(lo, min(lo + size, max_id)) for lo in range(min_id - 1, max_id, size)]


class LayerDownloader:
    """Download every record of one ArcGIS feature layer into a single parquet file.

    Works on any layer with an integer object-id field: `oid_field` and the page size default
    to the layer's own objectIdField and maxRecordCount. With `geometry=True` point
    geometries are requested in WGS84 and stored as `geom_lon` / `geom_lat`.
    """

    def __init__(
        self,
        url: str,
        page_dir: Path,
        out_path: Path,
        *,
        oid_field: str | None = None,
        page_size: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 6,
        backoff: float = 2.0,
        workers: int = 1,
        geometry: bool = False,
    ):
        self.url = url
        self.page_dir = Path(page_dir)
        self.out_path = Path(out_path)
        self.timeout, self.max_retries, self.backoff = timeout, max_retries, backoff
        self.workers = workers
        self.geometry = geometry
        self.session = requests.Session()
        if oid_field is None or page_size is None:
            meta = self._query_url(url.removesuffix("/query"))
            oid_field = oid_field or meta["objectIdField"]
            page_size = page_size or int(meta.get("maxRecordCount") or 1000)
        self.oid_field, self.page_size = oid_field, int(page_size)

    def _query_url(self, url: str, **params: Any) -> dict[str, Any]:
        return get_json(self.session, url, params, self.timeout, self.max_retries, self.backoff)

    def _query(self, **params: Any) -> dict[str, Any]:
        return self._query_url(self.url, **params)

    def count(self) -> int:
        return int(self._query(where="1=1", returnCountOnly="true")["count"])

    def objectid_range(self) -> tuple[int, int]:
        stats = [
            {"statisticType": s, "onStatisticField": self.oid_field, "outStatisticFieldName": s}
            for s in ("min", "max")
        ]
        attrs = self._query(where="1=1", outStatistics=json.dumps(stats))["features"][0][
            "attributes"
        ]
        return int(attrs["min"]), int(attrs["max"])

    def _page_path(self, lo: int, hi: int) -> Path:
        return self.page_dir / f"objectid_{lo + 1:07d}_{hi:07d}.json.gz"

    def fetch_window(self, lo: int, hi: int) -> Path:
        path = self._page_path(lo, hi)
        if path.exists():
            return path
        params: dict[str, Any] = {
            "where": f"{self.oid_field} > {lo} AND {self.oid_field} <= {hi}",
            "outFields": "*",
            "returnGeometry": "true" if self.geometry else "false",
            "orderByFields": f"{self.oid_field} ASC",
            "resultRecordCount": self.page_size,
        }
        if self.geometry:
            params["outSR"] = 4326
        body = self._query(**params)
        if body.get("exceededTransferLimit"):
            raise ArcGISError(f"window ({lo}, {hi}] exceeded the transfer limit")
        rows = []
        for feat in body.get("features", []):
            row = dict(feat["attributes"])
            if self.geometry:
                geom = feat.get("geometry") or {}
                row["geom_lon"], row["geom_lat"] = geom.get("x"), geom.get("y")
            rows.append(row)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(rows, f)
        tmp.replace(path)  # atomic, so an interrupted write never leaves a corrupt page
        return path

    def run(self) -> Path:
        self.page_dir.mkdir(parents=True, exist_ok=True)
        expected = self.count()
        min_id, max_id = self.objectid_range()
        windows = objectid_windows(min_id, max_id, self.page_size)
        todo = [w for w in windows if not self._page_path(*w).exists()]
        log.info(
            "%s: service reports %d records, %s %d..%d: %d pages (%d cached)",
            self.out_path.name,
            expected,
            self.oid_field,
            min_id,
            max_id,
            len(windows),
            len(windows) - len(todo),
        )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.fetch_window, lo, hi) for lo, hi in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="pages", unit="page"):
                fut.result()

        frames = []
        for lo, hi in windows:
            with gzip.open(self._page_path(lo, hi), "rt") as f:
                frames.append(pd.DataFrame.from_records(json.load(f)))
        df = pd.concat(frames, ignore_index=True)

        n_dup_oid = int(df[self.oid_field].duplicated().sum())
        if n_dup_oid:
            raise ArcGISError(f"{n_dup_oid} duplicate {self.oid_field}s across pages")
        if len(df) != expected:
            raise ArcGISError(f"downloaded {len(df)} rows but the service reports {expected}")
        log.info("downloaded %d rows, matches returnCountOnly", len(df))

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df = df.sort_values(self.oid_field).reset_index(drop=True)
        df.to_parquet(self.out_path, index=False)
        log.info("wrote %s", self.out_path)
        return self.out_path


class CollisionDownloader(LayerDownloader):
    """The TPS collision layer, configured from `download` in the config."""

    def __init__(self, cfg: dict[str, Any]):
        dl = cfg["download"]
        super().__init__(
            dl["url"],
            Path(cfg["paths"]["page_cache_dir"]),
            Path(cfg["paths"]["raw_dir"]) / RAW_FILENAME,
            oid_field="OBJECTID",
            page_size=int(dl["page_size"]),
            timeout=float(dl["timeout_seconds"]),
            max_retries=int(dl["max_retries"]),
            backoff=float(dl["backoff_seconds"]),
            workers=int(dl.get("workers", 1)),
        )


def download_collisions(cfg: dict[str, Any]) -> Path:
    return CollisionDownloader(cfg).run()
