import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from safepath.gnn.data import GraphData  # noqa: E402
from safepath.gnn.models import MODELS, NodeModel  # noqa: E402
from safepath.gnn.train import predict, train_model  # noqa: E402

CONFIG = {"hidden": 16, "layers": 2, "lr": 1e-2, "dropout": 0.1}


def tiny_graph(n=60, years=4, f=5, seed=0) -> GraphData:
    """A ring with chords; crashes depend on a node feature and on the neighbours' feature."""
    rng = np.random.default_rng(seed)
    src = np.arange(n)
    pairs = np.vstack([np.r_[src, src], np.r_[(src + 1) % n, (src + 7) % n]])
    edge_index = torch.tensor(np.hstack([pairs, pairs[::-1]]), dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 3, generator=torch.Generator().manual_seed(1))
    x = rng.normal(size=(years, n, f)).astype(np.float32)
    nb = np.zeros((years, n))
    np.add.at(nb, (slice(None), pairs[1]), x[:, pairs[0], 0])
    y = rng.poisson(np.exp(0.5 * x[:, :, 0] + 0.3 * nb / 2))
    return GraphData(
        node_ids=np.arange(n),
        edge_index=edge_index,
        edge_attr=edge_attr,
        years=list(range(2016, 2016 + years)),
        x=torch.tensor(x),
        y=torch.tensor(y, dtype=torch.float32),
        clean=torch.ones(n, dtype=torch.bool),
        split={"train": [0, 1], "val": [2], "test": [3]},
        feature_names=[f"f{i}" for i in range(f)],
        aadt=torch.ones(n),
    )


@pytest.mark.parametrize("kind", MODELS)
def test_training_is_deterministic_on_cpu(kind):
    g = tiny_graph()
    cpu = torch.device("cpu")
    m1, i1 = train_model(g, kind, CONFIG, cpu, seed=3, max_epochs=15, patience=50)
    m2, i2 = train_model(g, kind, CONFIG, cpu, seed=3, max_epochs=15, patience=50)
    assert np.array_equal(predict(m1, g, cpu), predict(m2, g, cpu))
    assert [h["val_deviance"] for h in i1["history"]] == [h["val_deviance"] for h in i2["history"]]


def test_training_learns_and_early_stopping_restores_best():
    g = tiny_graph()
    model, info = train_model(g, "sage", CONFIG, torch.device("cpu"), seed=0, max_epochs=60,
                              patience=10)  # fmt: skip
    hist = info["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert info["best_val_deviance"] == min(h["val_deviance"] for h in hist)
    assert info["epochs_run"] <= 60


def test_loss_masks_exclude_nodes():
    """Held-out nodes' labels never enter the loss: changing them leaves the fit unchanged."""
    g = tiny_graph()
    held = torch.zeros(len(g.node_ids), dtype=torch.bool)
    held[:20] = True
    cpu = torch.device("cpu")
    kw = dict(train_nodes=~held, stop_nodes=~held, max_epochs=10, patience=50)
    m1, _ = train_model(g, "sage", CONFIG, cpu, seed=1, **kw)
    g.y[:, :20] = 99.0
    m2, _ = train_model(g, "sage", CONFIG, cpu, seed=1, **kw)
    assert np.array_equal(predict(m1, g, cpu), predict(m2, g, cpu))


def test_head_starts_at_mean_rate():
    m = NodeModel("mlp", 4, 8, 1, 0.0, init_rate=2.5)
    assert float(m.head.bias) == pytest.approx(np.log(2.5), rel=1e-6)
