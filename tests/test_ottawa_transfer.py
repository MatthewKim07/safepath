import numpy as np
import pandas as pd

from safepath.ottawa.transfer import MODELS, level_scalars, poisson_ceiling, prior_year


def test_prior_year_skips_unpublished_year():
    assert prior_year([2021, 2022, 2024], 2024) == 2022
    assert prior_year([2021, 2022, 2024], 2022) == 2021


def test_level_scalar_uses_prior_year_only():
    rows = []
    for year, obs, pred in ((2022, 10, 5.0), (2024, 1000, 1.0)):
        rows.append({"year": year, "crash_count": obs, **{m: pred for m in MODELS}})
    scal = level_scalars(pd.DataFrame(rows), 2024)
    assert all(v == 2.0 for v in scal.values())  # 2024's own counts never enter


def test_poisson_ceiling_rises_with_crash_density():
    lam = np.random.default_rng(0).gamma(0.5, 1.0, 5000)
    sparse, _, _ = poisson_ceiling(lam * 0.25, seed=1, n_sim=20)
    dense, _, _ = poisson_ceiling(lam * 4, seed=1, n_sim=20)
    assert 0 < sparse < dense < 1
