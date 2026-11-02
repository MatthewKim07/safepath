"""Train and evaluate the node models: `python -m safepath gnn` (runs this as a subprocess).

torch is imported before anything that loads LightGBM: on macOS the two ship different
OpenMP runtimes and torch crashes if LightGBM's starts first, so the CLI runs GNN training
in its own process.

Protocol (identical splits to the CP3/CP6 models):
- Each epoch takes one full-graph optimiser step per train-year snapshot (2016-19,
  2022-23), Poisson NLL on clean intersections; early stopping on validation-2024 Poisson
  deviance (patience PATIENCE). Hyperparameters chosen on validation only from GRID.
- Final configs are refit with 5 seeds; the model's prediction is the mean of the 5 seeds'
  expected counts.
- Road-only spatial holdout: road features only (no history, no AADT), CP3's 5-fold 2 km
  blocks. Held-out blocks' labels never enter the loss or early stopping; message passing
  sees every node's (label-free) features: transductive on features, inductive on labels.

Outputs: data/processed/gnn_predictions.parquet, gnn_spatial_oof.parquet,
models/gnn_summary.json, reports/figures/gnn_training_curves.png.
"""

# ruff: noqa: I001  (torch must be imported first; see the module docstring)
from __future__ import annotations

import torch

import argparse
import copy
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import mean_poisson_deviance

from safepath import tracking
from safepath.config import load_config
from safepath.data import report
from safepath.gnn.data import GraphData, load_graph
from safepath.gnn.device import pick_device
from safepath.gnn.models import MODELS, NodeModel
from safepath.models.lgbm import STATIC_FEATURES

log = logging.getLogger(__name__)

GNN_PREDICTIONS = "gnn_predictions.parquet"
GNN_SPATIAL_OOF = "gnn_spatial_oof.parquet"
GNN_SUMMARY = "gnn_summary.json"
GRID = {"hidden": [64, 128], "layers": [2, 3], "lr": [1e-3, 3e-3], "dropout": [0.1, 0.3]}
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 150
PATIENCE = 15
DEVIANCE_FLOOR = 0.05  # crashes/year, as in evaluate.py
WEIGHT_DECAY = 1e-5


def set_seed(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def deviance(y: torch.Tensor, log_mu: torch.Tensor) -> float:
    mu = np.maximum(torch.exp(log_mu).detach().cpu().numpy(), DEVIANCE_FLOOR)
    return float(mean_poisson_deviance(y.cpu().numpy(), mu))


def train_model(
    g: GraphData,
    kind: str,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    features: list[int] | None = None,
    train_nodes: torch.Tensor | None = None,
    stop_nodes: torch.Tensor | None = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    on_epoch: Any = None,
) -> tuple[NodeModel, dict[str, Any]]:
    """Fit one model; returns it at its best validation epoch and the training history.
    `features` selects node-feature columns; `train_nodes` / `stop_nodes` restrict which
    nodes' labels enter the loss and the early-stopping deviance (default: all clean)."""
    rng = set_seed(seed)
    cols = slice(None) if features is None else torch.tensor(features)
    xs = g.x[:, :, cols].to(device)
    ys = g.y.to(device)
    ei, ea = g.edge_index.to(device), g.edge_attr.to(device)
    train_mask = (g.clean if train_nodes is None else train_nodes & g.clean).to(device)
    stop_mask = (g.clean if stop_nodes is None else stop_nodes & g.clean).to(device)
    train_years = g.split["train"]
    rate = float(ys[train_years][:, train_mask].mean())
    model = NodeModel(kind, xs.shape[2], config["hidden"], config["layers"], config["dropout"],
                      edge_dim=ea.shape[1], init_rate=rate).to(device)  # fmt: skip
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    val = g.split["val"][0]
    best = (float("inf"), 0, None)
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for t in rng.permutation(train_years):
            opt.zero_grad()
            out = model(xs[t], ei, ea)
            loss = F.poisson_nll_loss(out[train_mask], ys[t][train_mask], log_input=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            dev = deviance(ys[val][stop_mask], model(xs[val], ei, ea)[stop_mask])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_deviance": dev})
        if on_epoch is not None:
            on_epoch(history[-1])
        if dev < best[0] - 1e-5:
            best = (dev, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= patience:
            break
    model.load_state_dict(best[2])
    seconds = time.time() - t0
    info = {
        "best_epoch": best[1],
        "best_val_deviance": best[0],
        "epochs_run": len(history),
        "seconds": seconds,
        "seconds_per_epoch": seconds / len(history),
        "history": history,
    }
    return model, info


@torch.no_grad()
def predict(model: NodeModel, g: GraphData, device: torch.device,
            features: list[int] | None = None) -> np.ndarray:  # fmt: skip
    """Expected crashes, (n_years, N)."""
    model.eval()
    cols = slice(None) if features is None else torch.tensor(features)
    ei, ea = g.edge_index.to(device), g.edge_attr.to(device)
    return np.stack([
        torch.exp(model(g.x[i, :, cols].to(device), ei, ea)).cpu().numpy()
        for i in range(len(g.years))
    ])  # fmt: skip


def grid_search(g: GraphData, device: torch.device, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    keys = list(GRID)
    for kind in MODELS:
        for values in itertools.product(*(GRID[k] for k in keys)):
            config = dict(zip(keys, values, strict=True))
            _, info = train_model(g, kind, config, device, seed=0)
            rows.append({"model": kind, **config, **{k: info[k] for k in
                         ("best_val_deviance", "best_epoch", "seconds")}})  # fmt: skip
            log.info("grid %s %s: val deviance %.4f at epoch %d (%.0fs)", kind, config,
                     info["best_val_deviance"], info["best_epoch"], info["seconds"])  # fmt: skip
    return pd.DataFrame(rows)


def best_configs(grid: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for kind, g in grid.groupby("model"):
        r = g.sort_values("best_val_deviance").iloc[0]
        out[kind] = {k: (int(r[k]) if k in ("hidden", "layers") else float(r[k])) for k in GRID}
    return out


def seed_metrics(g: GraphData, pred: np.ndarray) -> dict[str, float]:
    out = {}
    for split in ("val", "test"):
        i = g.split[split][0]
        m = g.clean.numpy()
        y, p = g.y[i].numpy()[m], pred[i][m]
        out[f"{split}_spearman"] = float(pd.Series(p).corr(pd.Series(y), method="spearman"))
        out[f"{split}_deviance"] = float(mean_poisson_deviance(y, np.maximum(p, DEVIANCE_FLOOR)))
    return out


def final_models(
    g: GraphData, device: torch.device, configs: dict[str, dict[str, Any]], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preds, summary = [], {}
    for kind, config in configs.items():
        seed_preds, per_seed = [], []
        for seed in SEEDS:
            with tracking.run(cfg, f"gnn-{kind}-seed{seed}", "gnn",
                              {**config, "model": kind, "seed": seed, "device": str(device),
                               "max_epochs": MAX_EPOCHS, "patience": PATIENCE}) as wb:  # fmt: skip
                model, info = train_model(g, kind, config, device, seed, on_epoch=wb.log)
                p = predict(model, g, device)
                metrics = seed_metrics(g, p)
                wb.summary({**metrics, **{k: info[k] for k in ("best_epoch", "epochs_run",
                            "seconds", "seconds_per_epoch")}})  # fmt: skip
            seed_preds.append(p)
            per_seed.append({"seed": seed, **metrics, "best_epoch": info["best_epoch"],
                             "seconds": info["seconds"], "history": info["history"]})  # fmt: skip
        mean_pred = np.mean(seed_preds, axis=0)  # average of the seeds' expected counts
        for i, year in enumerate(g.years):
            preds.append(pd.DataFrame({"intersection_id": g.node_ids, "year": year,
                                       "model": kind, "pred": mean_pred[i]}))  # fmt: skip
        summary[kind] = {"config": config, "seeds": per_seed,
                         "ensemble_of_seeds": seed_metrics(g, mean_pred)}  # fmt: skip
    return pd.concat(preds, ignore_index=True), summary


def road_only_columns(g: GraphData) -> list[int]:
    """Road features only: the CP3 static set (class combo one-hot), no history, no AADT."""
    keep = set(STATIC_FEATURES) - {"class_combo"}
    return [i for i, f in enumerate(g.feature_names) if f in keep or f.startswith("combo_")]


def spatial_holdout(
    g: GraphData, device: torch.device, configs: dict[str, dict[str, Any]], cfg: dict[str, Any]
) -> pd.DataFrame:
    """Road-only GNN and MLP on CP3's spatial blocks (fold k held out, fold k+1 for early
    stopping), predicting val and test years for the held-out nodes."""
    from safepath.models.train import N_FOLDS, spatial_folds

    path = Path(cfg["paths"]["processed_dir"]) / "intersections.parquet"
    xy = pd.read_parquet(path, columns=["intersection_id", "x", "y"]).set_index("intersection_id")
    folds = spatial_folds(xy.loc[g.node_ids].reset_index()).to_numpy()
    cols = road_only_columns(g)
    rows = []
    for kind in ("mlp", "sage"):
        for k in range(N_FOLDS):
            held = torch.tensor(folds == k)
            stop = torch.tensor(folds == (k + 1) % N_FOLDS)
            train = ~held & ~stop
            model, info = train_model(g, kind, configs[kind], device, seed=0, features=cols,
                                      train_nodes=train, stop_nodes=stop)  # fmt: skip
            p = predict(model, g, device, features=cols)
            for split in ("val", "test"):
                i = g.split[split][0]
                h = held.numpy()
                rows.append(pd.DataFrame({"intersection_id": g.node_ids[h],
                                          "year": g.years[i], "model": kind, "fold": k,
                                          "pred": p[i][h]}))  # fmt: skip
            log.info("spatial %s fold %d: best epoch %d", kind, k, info["best_epoch"])
    return pd.concat(rows, ignore_index=True)


def timing(g: GraphData, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Seconds per epoch and total for the final GraphSAGE config on every device present."""
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
        ["cuda"] if torch.cuda.is_available() else []
    )  # fmt: skip
    rows = []
    for name in devices:
        _, info = train_model(g, "sage", config, torch.device(name), seed=0)
        rows.append({"device": name, "epochs": info["epochs_run"],
                     "seconds per epoch": info["seconds_per_epoch"],
                     "total seconds": info["seconds"],
                     "best val deviance": info["best_val_deviance"]})  # fmt: skip
    return rows


def plot_curves(summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=report.SURFACE,
                             layout="constrained")  # fmt: skip
    colors = {"mlp": report.TEXT_MUTED, "sage": report.BAR, "gatv2": "#1baf7a"}
    for kind, s in summary.items():
        h = pd.DataFrame(s["seeds"][0]["history"])
        axes[0].plot(h["epoch"], h["train_loss"], color=colors[kind], lw=2, label=kind)
        axes[1].plot(h["epoch"], h["val_deviance"], color=colors[kind], lw=2, label=kind)
    titles = ("train Poisson NLL (seed 0)", "validation 2024 Poisson deviance (seed 0)")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_facecolor(report.SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=report.TEXT)
        ax.set_xlabel("epoch", fontsize=9, color=report.TEXT_MUTED)
        ax.tick_params(colors=report.TEXT_MUTED, labelsize=8)
        ax.grid(color=report.GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, labelcolor=report.TEXT)
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)
    plt.close(fig)


def run(cfg: dict[str, Any], device_name: str | None = None) -> dict[str, Any]:
    processed = Path(cfg["paths"]["processed_dir"])
    device = pick_device(device_name)
    g = load_graph(cfg)
    t0 = time.time()
    grid = grid_search(g, device, cfg)
    configs = best_configs(grid)
    with tracking.run(cfg, "gnn-grid", "gnn", {"grid": GRID, "device": str(device)}) as wb:
        wb.table("grid", grid)
    preds, summary = final_models(g, device, configs, cfg)
    preds.to_parquet(processed / GNN_PREDICTIONS, index=False)
    oof = spatial_holdout(g, device, configs, cfg)
    oof.to_parquet(processed / GNN_SPATIAL_OOF, index=False)
    times = timing(g, configs["sage"])
    with tracking.run(cfg, "gnn-timing", "gnn", {"config": configs["sage"]}) as wb:
        wb.table("timing", pd.DataFrame(times))
    plot_curves(summary, Path(cfg["paths"]["figures_dir"]) / "gnn_training_curves.png")
    out = {
        "device": str(device),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "torch": torch.__version__,
        "grid": grid.to_dict(orient="records"),
        "configs": configs,
        "models": summary,
        "timing": times,
        "total_seconds": time.time() - t0,
        "graph": {
            "nodes": len(g.node_ids),
            "undirected_edges": g.edge_index.shape[1] // 2,
            "node_features": g.x.shape[2],
            "edge_features": g.edge_attr.shape[1],
            "years": g.years,
        },  # fmt: skip
    }
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / GNN_SUMMARY).write_text(json.dumps(out, indent=2, default=float))
    log.info("GNN training done in %.0fs on %s", out["total_seconds"], device)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="safepath.gnn.train")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    args = parser.parse_args(argv)
    run(load_config(args.config), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
