from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from safepath.config import load_config  # noqa: E402
from safepath.features.labels import LABELS  # noqa: E402
from safepath.gnn import data  # noqa: E402

CFG = load_config()
HAVE_DATA = (Path(CFG["paths"]["processed_dir"]) / "intersection_exposure.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="needs the processed pipeline outputs")


@pytest.fixture(scope="module")
def graph():
    return data.build_graph(CFG)


@needs_data
def test_nodes_match_the_intersection_table(graph):
    ids = pd.read_parquet(Path(CFG["paths"]["processed_dir"]) / "intersections.parquet")
    assert len(graph.node_ids) == len(ids) == 22435
    assert np.array_equal(graph.node_ids, np.sort(ids["intersection_id"].to_numpy()))
    assert graph.x.shape[:2] == (len(graph.years), len(graph.node_ids))


@needs_data
def test_edges_symmetric_without_self_loops_and_match_segments(graph):
    ei = graph.edge_index.numpy()
    assert (ei[0] != ei[1]).all()
    fwd = set(zip(ei[0], ei[1], strict=True))
    assert all((b, a) in fwd for a, b in fwd)
    pairs, _ = data.build_edges(CFG)
    assert ei.shape[1] == 2 * pairs.shape[1]
    # one undirected edge per pair of intersections joined by at least one segment
    processed = Path(CFG["paths"]["processed_dir"])
    seg = pd.read_parquet(processed / "segments.parquet", columns=["u", "v"])
    cl = pd.read_parquet(processed / "node_to_intersection.parquet").set_index("osmid")
    a, b = seg["u"].map(cl["intersection_id"]), seg["v"].map(cl["intersection_id"])
    keep = a.notna() & b.notna() & (a != b)
    unique = pd.DataFrame({"lo": np.minimum(a[keep], b[keep]), "hi": np.maximum(a[keep], b[keep])})
    assert len(unique.drop_duplicates()) == pairs.shape[1]


@needs_data
def test_features_have_no_nans(graph):
    assert not torch.isnan(graph.x).any()
    assert not torch.isnan(graph.edge_attr).any()
    assert graph.clean.sum() == 22435 - 31  # proxies masked out


@needs_data
def test_node_features_for_year_t_ignore_years_from_t_on():
    labels = pd.read_parquet(Path(CFG["paths"]["processed_dir"]) / LABELS)
    t = 2022
    before = data.node_table(CFG, labels)
    perturbed = labels.copy()
    future = perturbed["year"] >= t
    rng = np.random.default_rng(0)
    perturbed.loc[future, "crash_count"] = rng.poisson(20, future.sum()).astype("int32")
    perturbed.loc[future, "severe_count"] = rng.poisson(5, future.sum()).astype("int32")
    after = data.node_table(CFG, perturbed)
    cats = sorted(before["class_combo"].unique())
    fb = data.feature_matrix(before[before["year"] == t], cats).reset_index(drop=True)
    fa = data.feature_matrix(after[after["year"] == t], cats).reset_index(drop=True)
    pd.testing.assert_frame_equal(fb, fa)
