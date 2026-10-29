import numpy as np
import pytest

from safepath.features.static import approach_features, first_number, legs_and_min_angle
from safepath.graph import consolidate, segment_frames


@pytest.mark.parametrize(
    ("value", "expected"),
    [("50", 50), ("40;50", 40), ("['3', '2']", 3), ("30 mph", 30 * 1.609344), (None, np.nan)],
)
def test_first_number(value, expected):
    got = first_number(value)
    assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected)


def test_four_way_cross():
    assert legs_and_min_angle(np.array([0, 90, 180, 270])) == (4, 90.0)


def test_divided_carriageways_merge_into_one_leg():
    # two carriageways heading east (5 and 355 deg) plus a north and a west leg
    legs, angle = legs_and_min_angle(np.array([5, 355, 90, 180]))
    assert legs == 3
    assert angle == pytest.approx(90.0)


def test_skewed_y_intersection():
    legs, angle = legs_and_min_angle(np.array([0, 45, 180]))
    assert legs == 3 and angle == pytest.approx(45.0)


def test_single_leg():
    assert legs_and_min_angle(np.array([10.0])) == (1, 180.0)


def test_toy_graph_approaches(toy_graph):
    _, node_map = consolidate(toy_graph, tolerance=10)
    segments, _ = segment_frames(toy_graph)
    feats = approach_features(segments, node_map).iloc[0]
    # Main Street west and east, Cross Avenue north and south: a 4-leg crossing
    assert feats["n_approaches"] == 4
    assert feats["n_legs"] == 4
    assert feats["min_leg_angle_deg"] == pytest.approx(90, abs=1)
    assert feats["maxspeed_missing"]
