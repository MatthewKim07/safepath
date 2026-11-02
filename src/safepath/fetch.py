"""Retry wrapper for network fetches that always fails loudly once retries run out."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import osmnx as ox

log = logging.getLogger(__name__)

T = TypeVar("T")


class FetchError(RuntimeError):
    """A network fetch failed after all retries, or returned nothing usable."""


def with_retries(
    fn: Callable[[], T], what: str, attempts: int = 3, backoff_seconds: float = 15.0
) -> T:
    """Call `fn` up to `attempts` times with exponential backoff; raise FetchError (chained to
    the last error) if every attempt fails. Never returns a partial or empty result silently."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # re-raised below as FetchError
            last = exc
            if attempt < attempts:
                wait = backoff_seconds * 2 ** (attempt - 1)
                log.warning(
                    "%s failed (attempt %d/%d: %s); retrying in %.0fs",
                    what,
                    attempt,
                    attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise FetchError(f"{what} failed after {attempts} attempts: {last}") from last


def require_rows(frame: Any, what: str) -> Any:
    """Raise if a fetched table is empty: an empty OSM layer means the fetch went wrong."""
    if len(frame) == 0:
        raise FetchError(f"{what} returned no rows")
    return frame


def configure_osmnx(cfg: dict[str, Any]) -> None:
    ox.settings.cache_folder = str(Path(cfg["paths"]["raw_dir"]) / "osmnx_cache")
    ox.settings.use_cache = True
    ox.settings.log_console = False


def fetch_settings(cfg: dict[str, Any]) -> dict[str, float]:
    osm = cfg.get("osm", {})
    return {
        "attempts": int(osm.get("fetch_attempts", 3)),
        "backoff_seconds": float(osm.get("backoff_seconds", 15.0)),
    }


STUDY_AREA = "study_area.geojson"


def study_area(cfg: dict[str, Any]) -> Any:
    """What OSM is fetched for: the `graph.place` name (Toronto), or the polygon of the
    `graph.boundary` ArcGIS layer (Ottawa's urban boundary), cached as GeoJSON in raw_dir.
    Returns a place string or a shapely (Multi)Polygon in WGS84."""
    g = cfg["graph"]
    if g.get("place"):
        return g["place"]
    boundary = g.get("boundary")
    if not boundary:
        raise ValueError("config needs graph.place or graph.boundary")
    path = Path(cfg["paths"]["raw_dir"]) / STUDY_AREA
    if not path.exists():
        import requests

        def get() -> bytes:
            resp = requests.get(
                f"{boundary['url']}/query",
                params={"where": "1=1", "outSR": 4326, "f": "geojson", "returnGeometry": "true"},
                timeout=120,
            )
            resp.raise_for_status()
            if b'"features"' not in resp.content:
                raise FetchError(f"boundary layer returned no features: {resp.text[:200]}")
            return resp.content

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(with_retries(get, "study-area boundary fetch", **fetch_settings(cfg)))
    import geopandas as gpd

    gdf = require_rows(gpd.read_file(path), "study-area boundary")
    return gdf.to_crs("EPSG:4326").union_all()


def osm_features(cfg: dict[str, Any], tags: dict[str, Any]) -> Any:
    """OSM features with `tags` inside the study area."""
    area = study_area(cfg)
    if isinstance(area, str):
        return ox.features_from_place(area, tags=tags)
    return ox.features_from_polygon(area, tags=tags)


def osm_drive_graph(cfg: dict[str, Any]) -> Any:
    area = study_area(cfg)
    kind = cfg["graph"]["network_type"]
    if isinstance(area, str):
        return ox.graph_from_place(area, network_type=kind, simplify=True)
    return ox.graph_from_polygon(area, network_type=kind, simplify=True)
