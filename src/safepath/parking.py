"""Off-street parking polygons from OSM, used to exclude parking-lot crashes from labels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox

from safepath.fetch import configure_osmnx, fetch_settings, require_rows, with_retries

log = logging.getLogger(__name__)

PARKING = "parking.parquet"
PARKING_TAGS = {"amenity": "parking", "parking": ["surface", "multi-storey"]}
# On-street parking polygons sit on the carriageway, and underground garages share a
# footprint with the buildings and streets above them; neither marks an off-road crash.
EXCLUDED_PARKING_TYPES = {"underground", "street_side", "lane", "on_kerb", "half_on_kerb"}


def filter_parking(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep off-street parking areas: polygons tagged amenity=parking or a surface /
    multi-storey parking type, minus on-street and underground parking."""
    f = features[features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    amenity = f.get("amenity")
    ptype = f.get("parking")
    is_lot = amenity.eq("parking") | (amenity.isna() & ptype.isin(PARKING_TAGS["parking"]))
    keep = is_lot & ~ptype.isin(EXCLUDED_PARKING_TYPES)
    cols = [c for c in ("amenity", "parking", "access", "name") if c in f]
    out = f.loc[keep, cols + ["geometry"]].reset_index()
    return out.rename(columns={"id": "osm_id", "element": "osm_type"})


def load_or_fetch_parking(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    path = Path(cfg["paths"]["processed_dir"]) / PARKING
    if path.exists():
        return gpd.read_parquet(path)
    configure_osmnx(cfg)
    raw = with_retries(
        lambda: ox.features_from_place(cfg["graph"]["place"], tags=PARKING_TAGS),
        what="OSM fetch of parking polygons",
        **fetch_settings(cfg),
    )
    parking = require_rows(filter_parking(raw), "OSM parking polygons").to_crs(cfg["graph"]["crs"])
    parking["area_m2"] = parking.area
    parking.to_parquet(path)
    log.info("kept %d of %d parking features as off-street polygons", len(parking), len(raw))
    return parking
