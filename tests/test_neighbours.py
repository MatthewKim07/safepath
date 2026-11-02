import numpy as np
import pandas as pd
import pytest

from safepath.features.history import history_features
from safepath.features.neighbours import neighbour_features, neighbour_pairs


def _table(values: dict[str, list[float]], year: int = 2020) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    return pd.DataFrame({"intersection_id": range(n), "year": year, **values})


def test_path_graph_aggregates():
    # 0 - 1 - 2 - 3
    pairs = np.array([[0, 1, 2], [1, 2, 3]])
    hops = neighbour_pairs(pairs, 4)
    t = _table(
        {
            "log_aadt_entering": [1.0, 2.0, 3.0, 4.0],
            "prev3_crash_share_mean": [0.1, 0.2, 0.3, 0.4],
            "has_signal": [0, 1, 0, 0],
            "max_class_rank": [1, 4, 2, 1],
        }
    )
    f = neighbour_features(t, hops).set_index("intersection_id")
    assert f["degree"].tolist() == [1, 2, 2, 1]
    assert f.loc[1, "nb1_mean_log_aadt"] == pytest.approx(2.0)  # neighbours 0 and 2
    assert f.loc[0, "nb2_max_log_aadt"] == 3.0  # within two hops: 1 and 2
    assert f.loc[0, "nb2_mean_log_aadt"] == pytest.approx(2.5)
    assert f.loc[3, "nb1_max_signal"] == 0 and f.loc[0, "nb1_max_signal"] == 1
    assert f.loc[2, "nb2_max_class_rank"] == 4  # node 2 itself (rank 2) is excluded


def test_neighbour_crash_share_uses_only_past_years():
    pairs = np.array([[0, 1], [1, 2]])
    hops = neighbour_pairs(pairs, 3)
    rng = np.random.default_rng(0)
    rows = [{"intersection_id": i, "year": y, "crash_count": int(rng.poisson(3)),
             "severe_count": 0} for i in range(3) for y in range(2014, 2024)]  # fmt: skip
    labels = pd.DataFrame(rows)

    def year_features(lab):
        h = history_features(lab, [2020])
        h["log_aadt_entering"], h["has_signal"], h["max_class_rank"] = 1.0, 0, 1
        return neighbour_features(h, hops).set_index("intersection_id")

    before = year_features(labels)
    later = labels.copy()
    later.loc[later["year"] >= 2020, "crash_count"] += 50
    pd.testing.assert_frame_equal(before, year_features(later))
