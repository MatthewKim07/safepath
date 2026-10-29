import pandas as pd

from safepath.locations import location_summary


def test_location_summary_is_crash_weighted():
    pts = pd.DataFrame(
        {
            "n_crashes": [90, 10],
            "dist_intersection": [2.0, 80.0],
            "dist_edge": [1.0, 5.0],
            "dist_motorway": [500.0, 10.0],
            "in_parking": [False, False],
        }
    )
    s = location_summary(pts).set_index("category")
    assert s.loc["within 5 m of an intersection", "pct_crashes"] == 90
    assert s.loc["within 5 m of an intersection", "pct_points"] == 50
    mid = "mid-block: <= 20 m from an edge, > 30 m from any intersection"
    assert s.loc[mid, "crashes"] == 10
    assert s.loc["within 25 m of a motorway and > 30 m from any intersection", "points"] == 1
