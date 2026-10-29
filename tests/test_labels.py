import pandas as pd

from safepath.features.labels import crash_counts, label_table, split_of
from safepath.features.proxies import EXPRESSWAY, RETAIL, flag_proxies
from safepath.features.roads import approaches, class_combo
from safepath.graph import consolidate, segment_frames

SPLITS = {"train": [2016, 2017, 2018, 2019, 2022, 2023], "val": [2024], "test": [2025]}


def test_split_assignment():
    assert [split_of(y, SPLITS) for y in (2015, 2016, 2020, 2023, 2024, 2025)] == [
        "history",
        "train",
        "history",
        "train",
        "val",
        "test",
    ]


def test_label_table_fills_zero_years_and_ignores_non_intersection_crashes():
    assignments = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "year": [2016, 2016, 2024, 2016],
            "unit_type": ["intersection", "intersection", "intersection", "segment"],
            "unit_id": pd.array([7, 7, 7, 7], dtype="Int64"),
            "injury_or_fatal": [True, False, False, True],
        }
    )
    labels = label_table(crash_counts(assignments), pd.Series([7, 9]), SPLITS)
    assert len(labels) == 2 * 12
    row = labels.set_index(["intersection_id", "year"])
    assert row.loc[(7, 2016), "crash_count"] == 2  # the segment crash is not counted
    assert row.loc[(7, 2016), "severe_count"] == 1
    assert row.loc[(7, 2024), "split"] == "val"
    assert row.loc[(9, 2016), "crash_count"] == 0 and not row.loc[(9, 2016), "any_crash"]


RULES = {
    "retail": {
        "min_train_crashes": 150,
        "max_severe_share": 0.08,
        "min_ftr_share": 0.25,
        "max_dist_retail_m": 150,
    },
    "expressway": {
        "max_dist_motorway_m": 200,
        "min_train_crashes": 150,
        "min_ratio_to_combo_median": 20,
    },
}


def test_proxy_rules():
    # rows 0-3 are the cases under test; rows 4-10 are ordinary intersections that set the
    # class-combo median to 10 train crashes
    n = 11
    ids = pd.Index(range(n))
    combo = pd.DataFrame({"class_combo": ["local x local"] * n}, index=ids)
    dist = pd.DataFrame(
        {
            "dist_motorway": [5000, 100, 5000, 100] + [5000] * 7,
            "dist_retail": [50, 5000, 500, 5000] + [5000] * 7,
        },
        index=ids,
    )
    stats = pd.DataFrame(
        {
            "train_crashes": [400, 400, 400, 100] + [10] * 7,
            "train_severe_share": [0.03, 0.15, 0.03, 0.15] + [0.1] * 7,
            "train_ftr_share": [0.40, 0.10, 0.40, 0.10] + [0.1] * 7,
        },
        index=ids,
    )
    out = flag_proxies(stats, dist, combo, RULES).set_index("intersection_id")
    assert out.loc[0, "exclusion_reason"] == RETAIL  # lot signature next to a mall
    assert out.loc[1, "exclusion_reason"] == EXPRESSWAY  # 40x the combo median, by a motorway
    assert pd.isna(out.loc[2, "exclusion_reason"])  # lot signature but no retail nearby
    assert pd.isna(out.loc[3, "exclusion_reason"])  # by a motorway but not busy enough
    assert out["exclusion_reason"].notna().sum() == 2


def test_approaches_skip_internal_links(toy_graph):
    _, node_map = consolidate(toy_graph, tolerance=10)
    segments, _ = segment_frames(toy_graph)
    appr = approaches(segments, node_map)
    # n2-n6 is inside the merged cluster; the four other segments approach it
    assert len(appr) == 4
    assert appr["intersection_id"].nunique() == 1
    combo = class_combo(appr)
    assert combo["class_combo"].iloc[0] == "arterial x arterial"
