import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from safepath.parking import filter_parking


def test_filter_keeps_off_street_lots_only():
    lot = box(0, 0, 10, 10)
    features = gpd.GeoDataFrame(
        {
            "amenity": ["parking", "parking", "parking", None, "parking", "parking_space"],
            "parking": ["surface", "underground", "street_side", "multi-storey", None, None],
        },
        geometry=[lot, lot, lot, lot, Point(0, 0), lot],
        index=pd.MultiIndex.from_tuples([("way", i) for i in range(6)], names=["element", "id"]),
    )
    kept = filter_parking(features)
    assert list(kept["osm_id"]) == [0, 3]  # surface lot, untagged-amenity multi-storey
    assert set(kept.columns) >= {"osm_type", "osm_id", "geometry"}
