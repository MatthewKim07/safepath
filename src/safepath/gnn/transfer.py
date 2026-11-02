"""Toronto-trained GNNs applied to Ottawa: `python -m safepath.gnn.transfer` (subprocess).

torch is imported first and LightGBM never loads here (see safepath.gnn.train).

1. Fit on Toronto (cached in models/gnn_transfer/): the CP8 configs, 5 seeds each,
   - GraphSAGE with crash history (the full LightGBM + AADT node features);
   - GraphSAGE and MLP with road features only (no history, no AADT), on every clean
     Toronto intersection (train years in the loss, validation 2024 for early stopping).
2. Build Ottawa's graph with Toronto's standardisation and class-combo encoding, run every
   saved model, and average each model's 5 seeds' expected counts. No Ottawa label enters
   any of this; history features are Ottawa's own past crashes, as for LightGBM.

Output: data/processed/ottawa/gnn_predictions.parquet (intersection_id, year, model, pred).
"""

# ruff: noqa: I001  (torch must be imported first)
from __future__ import annotations

import torch

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safepath import tracking
from safepath.config import load_config, ottawa_config
from safepath.gnn.data import build_graph, load_graph
from safepath.gnn.device import pick_device
from safepath.gnn.models import NodeModel
from safepath.gnn.train import GNN_SUMMARY, SEEDS, predict, road_only_columns, train_model

log = logging.getLogger(__name__)

TRANSFER_DIR = "gnn_transfer"
OTTAWA_GNN = "gnn_predictions.parquet"
VARIANTS = {  # name -> (architecture, road features only?)
    "sage_history": ("sage", False),
    "sage_road": ("sage", True),
    "mlp_road": ("mlp", True),
}


def checkpoint(models_dir: Path, name: str, seed: int) -> Path:
    return models_dir / TRANSFER_DIR / f"{name}_seed{seed}.pt"


def fit_toronto(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    models_dir = Path(cfg["paths"]["models_dir"])
    (models_dir / TRANSFER_DIR).mkdir(parents=True, exist_ok=True)
    configs = json.loads((models_dir / GNN_SUMMARY).read_text())["configs"]
    g = load_graph(cfg)
    road = road_only_columns(g)
    info: dict[str, Any] = {}
    for name, (kind, road_only) in VARIANTS.items():
        feats = road if road_only else None
        for seed in SEEDS:
            path = checkpoint(models_dir, name, seed)
            if path.exists():
                continue
            with tracking.run(cfg, f"gnn-transfer-{name}-seed{seed}", "gnn-transfer",
                              {**configs[kind], "model": name, "seed": seed,
                               "device": str(device)}) as wb:  # fmt: skip
                model, fit = train_model(g, kind, configs[kind], device, seed, features=feats,
                                         on_epoch=wb.log)  # fmt: skip
                wb.summary({k: fit[k] for k in ("best_epoch", "best_val_deviance", "seconds")})
            torch.save({"state": model.state_dict(), "kind": kind, "config": configs[kind],
                        "in_dim": len(feats) if feats else g.x.shape[2],
                        "edge_dim": g.edge_attr.shape[1], "features": feats,
                        "best_epoch": fit["best_epoch"]}, path)  # fmt: skip
            info[f"{name}_seed{seed}"] = fit["best_epoch"]
            log.info("toronto %s seed %d: best epoch %d", name, seed, fit["best_epoch"])
    return info


def load_model(path: Path, device: torch.device) -> tuple[NodeModel, list[int] | None]:
    ck = torch.load(path, weights_only=False)
    c = ck["config"]
    model = NodeModel(ck["kind"], ck["in_dim"], c["hidden"], c["layers"], c["dropout"],
                      edge_dim=ck["edge_dim"]).to(device)  # fmt: skip
    model.load_state_dict(ck["state"])
    return model, ck["features"]


def predict_ottawa(toronto: dict[str, Any], ottawa: dict[str, Any], device: torch.device) -> Path:
    reference = build_graph(toronto).stats  # Toronto's standardisation, rebuilt exactly
    g = build_graph(ottawa, reference=reference)
    if [f for f in g.feature_names] != list(reference["node_mean"].index):
        raise ValueError("Ottawa node features do not line up with Toronto's")
    models_dir = Path(toronto["paths"]["models_dir"])
    frames = []
    for name in VARIANTS:
        preds = []
        for seed in SEEDS:
            model, feats = load_model(checkpoint(models_dir, name, seed), device)
            preds.append(predict(model, g, device, features=feats))
        mean = np.mean(preds, axis=0)
        for i, year in enumerate(g.years):
            frames.append(pd.DataFrame({"intersection_id": g.node_ids, "year": year,
                                        "model": name, "pred": mean[i]}))  # fmt: skip
    out = Path(ottawa["paths"]["processed_dir"]) / OTTAWA_GNN
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    log.info("ottawa GNN predictions: %s years x %d nodes", g.years, len(g.node_ids))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="safepath.gnn.transfer")
    parser.add_argument("--config", default=None, help="Toronto config")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    toronto = load_config(args.config)
    device = pick_device(args.device)
    t0 = time.time()
    fit_toronto(toronto, device)
    predict_ottawa(toronto, ottawa_config(), device)
    log.info("GNN transfer done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
