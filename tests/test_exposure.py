import numpy as np
import pandas as pd
import pytest

from safepath import exposure


def _inputs():
    # nodes 1-2 form intersection 10, node 3 is intersection 20, node 4 a dead end
    edges = pd.DataFrame(
        {
            "u": [3, 4, 1, 2, 1],
            "v": [1, 2, 3, 1, 2],
            "key": 0,
            "segment_id": [100, 101, 100, 102, 102],
            "highway": ["primary", "residential", "primary", "service", "service"],
            "aadt_dir": [9000.0, 1000.0, 9000.0, 50.0, 50.0],
            "mto_hwy": None,
        }
    )
    segments = pd.DataFrame(
        {
            "segment_id": [100, 101, 102],
            "name": ["Main Street", "Side Street", None],
            "length": [1000.0, 500.0, 10.0],
            "mid_x": 0.0,
            "mid_y": 0.0,
        }
    )
    node_map = pd.DataFrame({"osmid": [1, 2, 3], "intersection_id": [10, 10, 20]})
    return {
        "edges": edges,
        "segments": segments,
        "node_map": node_map,
        "inter_crashes": pd.Series({10: 10.0, 20: 4.0}),
        "seg_crashes": pd.Series({100: 2.0}),
        "proxies": pd.DataFrame({"exclusion_reason": [None, "retail_proxy"]}, index=[10, 20]),
    }


def test_allocation_splits_by_entering_volume_and_conserves_crashes():
    e = exposure.allocate(_inputs())
    # intersection 10 is entered by 3->1 (9000) and 4->2 (1000); 2->1 is internal
    assert e.loc[0, "inter_crashes"] == pytest.approx(9.0)
    assert e.loc[1, "inter_crashes"] == pytest.approx(1.0)
    assert e.loc[3, "inter_crashes"] == 0
    # intersection 20 is a retail proxy: its crashes are not charged to anyone
    assert e.loc[2, "inter_crashes"] == 0
    # the mid-block crashes on segment 100 split over its two directions
    assert e.loc[[0, 2], "midblock_crashes"].sum() == pytest.approx(2.0)
    assert e["crashes"].sum() == pytest.approx(12.0)
    mvkm = 9000 * 1.0 * 365 / 1e6
    assert e.loc[0, "mvkm"] == pytest.approx(mvkm)


def test_top_tables_apply_volume_and_count_filters():
    n = 30
    df = pd.DataFrame(
        {
            "street_names": [f"X{i}" for i in range(n)],
            "class_combo": "arterial x local",
            "crashes_per_year": np.linspace(1, 60, n),
            "aadt_count": np.r_[[500.0], np.full(n - 1, 20_000.0)],
            "count_year": 2024,
        }
    )
    df["rate_count"] = df["crashes_per_year"] / (df["aadt_count"] * 365 / 1e6)
    by_count, by_rate, stats = exposure.top_tables(df)
    # row 0 has the highest rate but only 500 vehicles/day and 1 crash/yr: filtered out
    assert "X0" not in set(by_rate["intersection"])
    assert by_count["intersection"].iloc[0] == f"X{n - 1}"
    assert stats["eligible"] == int(
        ((df["aadt_count"] >= 5000) & (df["crashes_per_year"] >= 5)).sum()
    )
