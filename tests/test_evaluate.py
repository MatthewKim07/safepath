import numpy as np
import pytest

from safepath.evaluate import bootstrap, hotspot_recall, point_metrics


def test_hotspot_recall_simple():
    y = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    pred = np.arange(10, 0, -1, dtype=float)  # perfect ranking
    assert hotspot_recall(y, pred, 0.1) == 1.0
    assert hotspot_recall(y, pred[::-1], 0.1) == 0.0


def test_hotspot_recall_ties_get_fractional_credit():
    y = np.array([4, 0, 0, 0], float)
    pred = np.zeros(4)  # all tied: top 25% captures 1/4 of each tied crash in expectation
    assert hotspot_recall(y, pred, 0.25) == pytest.approx(0.25)
    pred = np.array([1, 1, 0, 0], float)  # 2 tied for 1 slot
    assert hotspot_recall(y, pred, 0.25) == pytest.approx(0.5)


def test_point_metrics_perfect_model():
    rng = np.random.default_rng(0)
    y = rng.poisson(1.5, 500).astype(float)
    m = point_metrics(y, y + 1e-3 * rng.random(500))
    assert m["spearman"] > 0.95 and m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_top10"] > 0.2


def test_bootstrap_paired_difference_is_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.poisson(1.0, 300).astype(float)
    p = y + rng.random(300)
    out = bootstrap(y, {"ref": p, "same": p.copy()}, "ref", n_boot=50, seed=0)
    assert out["same"]["spearman_diff_vs_ref"] == 0
    lo, hi = out["ref"]["spearman_ci"]
    assert lo <= hi
