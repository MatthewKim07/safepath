import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from safepath.graph import consolidate, intersection_frame, segment_frames
from safepath.snapping import assign_points

CRS = "EPSG:32617"


@pytest.fixture
def network(toy_graph):
    Gc, node_map = consolidate(toy_graph, tolerance=10)
    intersections = intersection_frame(Gc, node_map, toy_graph)
    segments, _ = segment_frames(toy_graph)
    parking = gpd.GeoDataFrame(geometry=[box(280, -30, 320, -5), box(200, -12, 230, -4)], crs=CRS)
    return intersections, segments, parking


def _assign(network, xy):
    intersections, segments, parking = network
    pts = gpd.GeoDataFrame(
        {"point_id": range(len(xy))},
        geometry=gpd.points_from_xy([p[0] for p in xy], [p[1] for p in xy]),
        crs=CRS,
    )
    return assign_points(pts, intersections, segments, parking, intersection_m=30, edge_m=20)


def _row(out: pd.DataFrame, i: int) -> pd.Series:
    return out.set_index("point_id").loc[i]


def test_point_near_intersection_is_assigned_to_it(network):
    out = _row(_assign(network, [(203, 2)]), 0)
    assert out["unit_type"] == "intersection"
    assert out["unit_id"] == network[0]["intersection_id"].iloc[0]
    assert out["snap_distance_m"] == pytest.approx(((206 - 203) ** 2 + 2**2) ** 0.5)
    assert pd.isna(out["exclusion_reason"])


def test_mid_block_point_is_assigned_to_segment(network):
    out = _row(_assign(network, [(100, 5)]), 0)
    segments = network[1]
    expected = segments.loc[segments["name"].eq("Main Street") & (segments["length"] == 200)]
    expected = expected[expected.geometry.centroid.x < 150]["segment_id"].item()
    assert out["unit_type"] == "segment"
    assert out["unit_id"] == expected
    assert out["snap_distance_m"] == pytest.approx(5)


def test_parking_point_is_excluded_even_near_a_road(network):
    out = _assign(network, [(300, -10), (210, -8)])
    for i in (0, 1):  # 10 m from Main Street; and 8 m from the intersection
        assert _row(out, i)["unit_type"] == "excluded"
        assert _row(out, i)["exclusion_reason"] == "in_parking"
        assert pd.isna(_row(out, i)["unit_id"])


def test_far_point_is_excluded(network):
    out = _row(_assign(network, [(300, 80)]), 0)
    assert out["unit_type"] == "excluded"
    assert out["exclusion_reason"] == "beyond_snap_thresholds"
    assert out["snap_distance_m"] == pytest.approx(80)


def test_intersection_wins_over_segment_inside_threshold(network):
    # 25 m east of the cluster centre along Main Street: near both, intersection first
    out = _row(_assign(network, [(231, 1)]), 0)
    assert out["unit_type"] == "intersection"
