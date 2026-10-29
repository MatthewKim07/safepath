"""Download Toronto Police Service collision records from the ArcGIS FeatureServer.

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


class CollisionDownloader:
    def __init__(self, cfg: dict[str, Any]):
        dl = cfg["download"]
        self.url = dl["url"]
        self.page_size = int(dl["page_size"])
        self.timeout = float(dl["timeout_seconds"])
        self.max_retries = int(dl["max_retries"])
        self.backoff = float(dl["backoff_seconds"])
        self.workers = int(dl.get("workers", 1))
        self.page_dir = Path(cfg["paths"]["page_cache_dir"])
        self.out_path = Path(cfg["paths"]["raw_dir"]) / RAW_FILENAME
        self.session = requests.Session()

    def _query(self, **params: Any) -> dict[str, Any]:
        return get_json(
            self.session, self.url, params, self.timeout, self.max_retries, self.backoff
        )

    def count(self) -> int:
        return int(self._query(where="1=1", returnCountOnly="true")["count"])

    def objectid_range(self) -> tuple[int, int]:
        stats = [
            {"statisticType": s, "onStatisticField": "OBJECTID", "outStatisticFieldName": s}
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
        body = self._query(
            where=f"OBJECTID > {lo} AND OBJECTID <= {hi}",
            outFields="*",
            returnGeometry="false",
            orderByFields="OBJECTID ASC",
            resultRecordCount=self.page_size,
        )
        if body.get("exceededTransferLimit"):
            raise ArcGISError(f"window ({lo}, {hi}] exceeded the transfer limit")
        rows = [feat["attributes"] for feat in body.get("features", [])]
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
            "service reports %d records, OBJECTID %d..%d: %d pages (%d cached)",
            expected,
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

        n_dup_oid = int(df["OBJECTID"].duplicated().sum())
        if n_dup_oid:
            raise ArcGISError(f"{n_dup_oid} duplicate OBJECTIDs across pages")
        if len(df) != expected:
            raise ArcGISError(f"downloaded {len(df)} rows but the service reports {expected}")
        log.info("downloaded %d rows, matches returnCountOnly", len(df))

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values("OBJECTID").reset_index(drop=True).to_parquet(self.out_path, index=False)
        log.info("wrote %s", self.out_path)
        return self.out_path


def download_collisions(cfg: dict[str, Any]) -> Path:
    return CollisionDownloader(cfg).run()
