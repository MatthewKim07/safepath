from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safepath import explain
from safepath.config import load_config


def test_factor_labels():
    row = pd.Series(
        {"streetcar_20m": 1, "min_leg_angle_deg": 40.0, "n_legs": 5,
         "class_combo": "arterial x local", "group": "signalised 3-leg"}
    )  # fmt: skip
    assert (
        explain.factor_label("traffic volume", np.log(2.0), row) == "high traffic volume (+, ×2.0)"
    )
    assert explain.factor_label("streetcar", 0.3, row).startswith("streetcar tracks (+")
    assert explain.factor_label("leg angle", 0.1, row).startswith("skewed angle (+")
    assert explain.factor_label("legs", 0.2, row).startswith("5+ legs")
    assert explain.factor_label("intersection type", 1.0, row).startswith("signalised T")
    assert explain.factor_label("crash history", -0.5, row) == (
        "fewer crashes than expected (2021-25) (−, ×0.6)"
    )


@pytest.mark.skipif(
    not (Path(load_config()["paths"]["models_dir"]) / "spf.json").exists(),
    reason="needs the fitted SPF",
)
def test_spf_contributions_add_up_to_log_spf():
    cfg = load_config()
    c = explain.spf_contributions(cfg)
    spf = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "spf_predictions.parquet")
    base = spf.drop_duplicates("intersection_id").set_index("intersection_id")["spf_base"]
    total = c[[*explain.FACTOR_TERMS, "intersection type"]].sum(axis=1)
    expected = np.log(base.reindex(c.index)) - np.log(base.reindex(c.index)).mean()
    assert np.allclose(total, expected, atol=1e-6)
