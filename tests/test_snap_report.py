import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from safepath.snap_report import arterial_check, intersection_stats

CRS = "EPSG:32617"


def test_intersection_stats_counts_and_flags():
    intersections = gpd.GeoDataFrame(
        {
            "intersection_id": [0, 1, 2],
            "street_names": ["Mall Road / Lot Lane", "Quiet Court / Side Street", "Finch / Main"],
            "highway_classes": ["residential", "residential", "secondary"],
            "n_osm_nodes": [1, 1, 4],
        },
        geometry=[Point(0, 0), Point(5000, 0), Point(10000, 0)],
        crs=CRS,
    )
    segments = gpd.GeoDataFrame(
        {"segment_id": [0], "highway": ["motorway"]},
        geometry=[LineString([(5000, 100), (6000, 100)])],
        crs=CRS,
    )
    rows = []
    # 600 low-severity, mostly hit-and-run crashes at node 0 (retail pattern)
    rows += [(0, i < 10, i < 300) for i in range(600)]
    # 600 ordinary crashes at a residential node 100 m from a motorway (expressway pattern)
    rows += [(1, i < 90, i < 60) for i in range(600)]
    rows += [(2, True, False)] * 3
    assignments = pd.DataFrame(rows, columns=["unit_id", "injury_or_fatal", "is_fail_to_remain"])
    assignments["unit_type"] = "intersection"
    assignments["event_id"] = range(len(assignments))

    stats = intersection_stats(assignments, intersections, segments)
    assert list(stats.index) == [0, 1, 2]
    assert stats.loc[2, "crashes"] == 3 and stats.loc[2, "pct_severe"] == 100
    assert stats.loc[0, "flag_retail_proxy"] and not stats.loc[0, "flag_expressway_proxy"]
    assert stats.loc[1, "flag_expressway_proxy"] and not stats.loc[1, "flag_retail_proxy"]
    assert not stats.loc[2, ["flag_retail_proxy", "flag_expressway_proxy"]].any()


def test_arterial_check_ranks():
    stats = pd.DataFrame({"street_names": ["Yonge / Bloor", "Finch / Keele", "Dundas / Finch"]})
    table = arterial_check(stats).set_index("street")
    assert table.loc["Finch", "best rank"] == 2
    assert table.loc["Finch", "intersections on street"] == 2
    assert table.loc["Dundas", "best rank"] == 3
    assert pd.isna(table.loc["Steeles", "best rank"])
