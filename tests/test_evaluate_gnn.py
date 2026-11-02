import numpy as np
import pandas as pd

from safepath import evaluate_gnn


def test_choose_ensemble_picks_best_graph_model_and_weight():
    rng = np.random.default_rng(0)
    truth = rng.gamma(2.0, 1.0, 5000)
    y = rng.poisson(truth)
    rows = pd.DataFrame(
        {
            "split": "val",
            "crash_count": y,
            "gnn_sage": truth * np.exp(rng.normal(0, 0.3, 5000)),
            "gnn_gatv2": truth * np.exp(rng.normal(0, 0.8, 5000)),
            "lgbm_aadt": truth * np.exp(rng.normal(0, 0.3, 5000)),
        }
    )
    best, w, curve = evaluate_gnn.choose_ensemble(rows)
    assert best == "gnn_sage"
    # two equally noisy, independent predictors: averaging them helps, near 50/50
    assert 0.3 <= w <= 0.7
    assert curve["val deviance"].min() < curve.set_index("weight on GNN").loc[1.0, "val deviance"]
