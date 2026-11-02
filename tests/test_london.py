import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from safepath.london.risk import key_pair, top10_check
from safepath.london.volumes import match_segments

CRS = "EPSG:32617"


def test_match_requires_name_agreement_or_closeness():
    segments = gpd.GeoDataFrame(
        {"segment_id": [1, 2, 3], "name": ["Oxford Street East", "Richmond Street", None]},
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 5), (100, 5)]),
            LineString([(0, 200), (100, 200)]),
        ],  # fmt: skip
        crs=CRS,
    )
    lines = gpd.GeoDataFrame(
        {"key": ["oxford", "unrelated"], "VolumeCount": [20000.0, 500.0]},
        geometry=[LineString([(0, 2), (100, 2)]), LineString([(0, 203), (100, 203)])],
        crs=CRS,
    )
    m = match_segments(segments, lines, match_m=15, unnamed_m=8).set_index("segment_id")
    assert m.loc[1, "aadt_city"] == 20000  # same street, 2 m away
    assert pd.isna(m.loc[2, "aadt_city"])  # close, but a different street
    assert m.loc[3, "aadt_city"] == 500  # unnamed, within 8 m


def test_top10_names_match_osm_full_names():
    assert key_pair("Fanshawe Park/Adelaide") == key_pair(
        "Fanshawe Park Road East/Adelaide Street North"
    )
    inter = pd.DataFrame({"intersection_id": [0, 1],
                          "street_names": ["Commissioners Road East / Wellington Road",
                                           "Oxford Street East / Richmond Street"]})  # fmt: skip
    risk = pd.DataFrame({"intersection_id": [0, 1], "risk": [5.0, 1.0],
                         "entering_aadt": [9e4, 7e4]})  # fmt: skip
    t = top10_check(risk, inter)
    hit = t[t["intersection"] == "Wellington/Commissioners"].iloc[0]
    assert hit["intersection_id"] == 0 and hit["SPF percentile"] == 100
