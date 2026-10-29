"""OSM context layers around intersections: retail areas, traffic control, transit, POIs.

Each layer is fetched once with `features_from_place`, projected, and cached as GeoParquet in
data/processed/. Point-like layers are reduced to points (polygon centroids).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox

from safepath.graph import configure_osmnx

log = logging.getLogger(__name__)

# Street furniture and parking are tagged amenity=* but say little about activity.
NON_ACTIVITY_AMENITIES = {
    "parking",
    "parking_space",
    "parking_entrance",
    "bicycle_parking",
    "motorcycle_parking",
    "bench",
    "waste_basket",
    "waste_disposal",
    "recycling",
    "post_box",
    "vending_machine",
    "drinking_water",
    "telephone",
    "charging_station",
    "bicycle_rental",
    "shelter",
    "toilets",
    "grit_bin",
    "fire_hydrant",
    "clock",
    "letter_box",
    "atm",
}
# shop=mall of any size, plus landuse=retail polygons at least this large (m^2).
LARGE_RETAIL_M2 = 20_000

LAYERS: dict[str, dict[str, Any]] = {
    "retail": {"shop": "mall", "landuse": "retail"},
    "traffic_control": {"highway": ["traffic_signals", "stop", "crossing"]},
    "tram": {"railway": "tram"},
    "bus_stops": {"highway": "bus_stop"},
    "pois": {"amenity": True, "shop": True},
}


def _filter(name: str, f: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if name == "retail":
        is_poly = f.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        mall = f.get("shop").eq("mall")
        big_retail = f.get("landuse").eq("retail") & is_poly & (f.area >= LARGE_RETAIL_M2)
        f = f[mall | big_retail]
        keep = ["shop", "landuse", "name"]
    elif name == "traffic_control":
        f = f[f.geometry.geom_type == "Point"]
        keep = ["highway", "crossing"]
    elif name == "tram":
        f = f[f.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        keep = ["railway", "name"]
    elif name == "bus_stops":
        f = f.assign(geometry=f.geometry.centroid)
        keep = ["highway", "name"]
    else:  # pois
        f = f[f.get("shop").notna() | ~f.get("amenity").isin(NON_ACTIVITY_AMENITIES)]
        f = f.assign(geometry=f.geometry.centroid)
        keep = ["amenity", "shop"]
    cols = [c for c in keep if c in f]
    return f[cols + ["geometry"]].reset_index()


def load_or_fetch_layer(cfg: dict[str, Any], name: str) -> gpd.GeoDataFrame:
    path = Path(cfg["paths"]["processed_dir"]) / f"osm_{name}.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    configure_osmnx(cfg)
    raw = ox.features_from_place(cfg["graph"]["place"], tags=LAYERS[name])
    layer = _filter(name, raw.to_crs(cfg["graph"]["crs"]))
    layer.to_parquet(path)
    log.info("osm layer %s: kept %d of %d features", name, len(layer), len(raw))
    return layer


def fetch_all(cfg: dict[str, Any]) -> dict[str, gpd.GeoDataFrame]:
    return {name: load_or_fetch_layer(cfg, name) for name in LAYERS}
