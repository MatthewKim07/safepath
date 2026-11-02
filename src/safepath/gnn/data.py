"""The intersection graph for GNNs: one topology, one feature snapshot per label year.

- Nodes: the 22,435 consolidated intersections, indexed by intersection_id.
- Edges: road segments joining two different intersections, deduplicated to one undirected
  edge per intersection pair and stored in both directions. Parallel segments are merged
  (highest class, shortest length, most lanes, highest speed and AADT, any motorway).
- Edge features (standardised): class rank, length, lanes, max speed, log segment AADT,
  motorway flag.
- Node features for label year t: the LightGBM + AADT feature set (CP3 road features,
  crash history from years before t, log AADT), rebuilt here from the labels with the CP3
  history code so the leakage guarantee carries over. Standardised with clean train-year
  statistics; class combo one-hot; a missing-minor-road indicator replaces NaN.
- Masks: train (2016-19, 2022-23), val (2024), test (2025); proxy intersections are never
  in the loss or the headline evaluation.

Output: data/processed/gnn_graph.pt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from safepath.features.build import STATIC
from safepath.features.history import history_features
from safepath.features.labels import LABELS
from safepath.features.roads import class_rank
from safepath.features.static import first_number
from safepath.features.volume_features import INTERSECTION_EXPOSURE
from safepath.graph import INTERSECTIONS, NODE_MAP, SEGMENTS
from safepath.models import lgbm
from safepath.models.volume import AADT_EDGES

log = logging.getLogger(__name__)

GNN_GRAPH = "gnn_graph.pt"
EDGE_FEATURES = ["class_rank", "length_m", "lanes", "maxspeed", "log_aadt", "motorway"]


@dataclass
class GraphData:
    node_ids: np.ndarray  # intersection_id per node index
    edge_index: torch.Tensor  # (2, 2E), both directions
    edge_attr: torch.Tensor  # (2E, n_edge_features), standardised
    years: list[int]
    x: torch.Tensor  # (n_years, N, F), standardised
    y: torch.Tensor  # (n_years, N) crash counts
    clean: torch.Tensor  # (N,) bool: not a proxy intersection
    split: dict[str, list[int]]  # split name -> list of year positions
    feature_names: list[str]
    aadt: torch.Tensor  # (N,) entering AADT, for per-vehicle calibration

    def year_index(self, year: int) -> int:
        return self.years.index(year)


def build_edges(cfg: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    """Undirected intersection pairs (a < b) with merged, unstandardised edge features."""
    processed = Path(cfg["paths"]["processed_dir"])
    columns = ["segment_id", "u", "v", "length", "highway", "lanes", "maxspeed"]
    seg = pd.read_parquet(processed / SEGMENTS, columns=columns)
    cluster = pd.read_parquet(processed / NODE_MAP).set_index("osmid")["intersection_id"]
    aadt = pd.read_parquet(processed / AADT_EDGES, columns=["segment_id", "aadt_segment"])
    aadt = aadt.groupby("segment_id")["aadt_segment"].max()
    seg["a"], seg["b"] = seg["u"].map(cluster), seg["v"].map(cluster)
    seg = seg[seg["a"].notna() & seg["b"].notna() & (seg["a"] != seg["b"])].copy()
    lo, hi = np.minimum(seg["a"], seg["b"]), np.maximum(seg["a"], seg["b"])
    seg["a"], seg["b"] = lo.astype(int), hi.astype(int)
    seg["class_rank"] = class_rank(seg["highway"])
    seg["length_m"] = seg["length"]
    seg["lanes"] = seg["lanes"].map(first_number)
    seg["maxspeed"] = seg["maxspeed"].map(first_number)
    for col in ("lanes", "maxspeed"):  # impute by class, like the CP3 static features
        seg[col] = seg[col].fillna(seg.groupby("highway")[col].transform("median"))
        seg[col] = seg[col].fillna(seg[col].median())
    seg["log_aadt"] = np.log1p(seg["segment_id"].map(aadt).fillna(0))
    seg["motorway"] = seg["highway"].isin(["motorway", "motorway_link"]).astype(float)
    merged = seg.groupby(["a", "b"]).agg(
        class_rank=("class_rank", "max"),
        length_m=("length_m", "min"),
        lanes=("lanes", "max"),
        maxspeed=("maxspeed", "max"),
        log_aadt=("log_aadt", "max"),
        motorway=("motorway", "max"),
    )
    pairs = np.array(merged.index.tolist()).T
    return pairs, merged.reset_index(drop=True)


def node_table(cfg: dict[str, Any], labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Intersection-year rows for every label year with the LightGBM + AADT features, crash
    counts and the exclusion flag. History is rebuilt from `labels` (default: on disk)."""
    processed = Path(cfg["paths"]["processed_dir"])
    labels = pd.read_parquet(processed / LABELS) if labels is None else labels
    years = sorted(cfg["splits"]["train"] + cfg["splits"]["val"] + cfg["splits"]["test"])
    hist = history_features(labels, years)
    static = pd.read_parquet(processed / STATIC)
    expo = pd.read_parquet(processed / INTERSECTION_EXPOSURE).set_index("intersection_id")
    df = hist.join(static, on="intersection_id")
    df["log_aadt_entering"] = df["intersection_id"].map(np.log(expo["entering_aadt"]))
    df["log_aadt_major"] = df["intersection_id"].map(np.log(expo["major_aadt"]))
    df["log_aadt_minor"] = df["intersection_id"].map(np.log(expo["minor_aadt"]))
    df["entering_aadt"] = df["intersection_id"].map(expo["entering_aadt"])
    lab = labels[["intersection_id", "year", "crash_count", "is_excluded"]]
    return df.merge(lab, on=["intersection_id", "year"], how="left")


def feature_matrix(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """LightGBM + AADT features as numbers: booleans to 0/1, class combo one-hot, and a
    single-road indicator in place of the missing minor-road AADT."""
    X = df[lgbm.FEATURE_SETS["aadt"]].copy()
    combo = pd.Categorical(X.pop("class_combo"), categories=categories)
    X = X.join(pd.get_dummies(combo, prefix="combo", dtype=float).set_index(X.index))
    X["single_road"] = X["log_aadt_major"].notna() & X["log_aadt_minor"].isna()
    for col in ("log_aadt_major", "log_aadt_minor"):
        X[col] = X[col].fillna(X["log_aadt_entering"] if col == "log_aadt_major" else 0.0)
    return X.astype(float)


def standardise(X: pd.DataFrame, fit_rows: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    mean = X[fit_rows].mean()
    std = X[fit_rows].std().replace(0, 1.0)
    return (X - mean) / std, {"mean": mean, "std": std}


def build_graph(cfg: dict[str, Any], labels: pd.DataFrame | None = None) -> GraphData:
    processed = Path(cfg["paths"]["processed_dir"])
    ids = pd.read_parquet(processed / INTERSECTIONS, columns=["intersection_id"])
    node_ids = np.sort(ids["intersection_id"].to_numpy())
    if not np.array_equal(node_ids, np.arange(len(node_ids))):
        raise ValueError("intersection ids are expected to be 0..N-1")
    pairs, ef = build_edges(cfg)
    ef = (ef - ef.mean()) / ef.std().replace(0, 1.0)
    edge_index = torch.tensor(np.hstack([pairs, pairs[::-1]]), dtype=torch.long)
    edge_attr = torch.tensor(np.vstack([ef.to_numpy(), ef.to_numpy()]), dtype=torch.float32)

    df = node_table(cfg, labels).sort_values(["year", "intersection_id"]).reset_index(drop=True)
    categories = sorted(df["class_combo"].unique())
    X = feature_matrix(df, categories)
    train_years = cfg["splits"]["train"]
    fit = df["year"].isin(train_years) & ~df["is_excluded"]
    X, _ = standardise(X, fit)
    if X.isna().any().any():
        raise ValueError(f"NaN node features: {X.columns[X.isna().any()].tolist()}")
    years = sorted(int(t) for t in df["year"].unique())
    n = len(node_ids)
    x = torch.tensor(X.to_numpy().reshape(len(years), n, -1), dtype=torch.float32)
    y = torch.tensor(df["crash_count"].to_numpy().reshape(len(years), n), dtype=torch.float32)
    clean = ~torch.tensor(df.loc[df["year"] == years[0], "is_excluded"].to_numpy().astype(bool))
    split = {
        name: [years.index(t) for t in cfg["splits"][name]] for name in ("train", "val", "test")
    }
    aadt = torch.tensor(
        df.loc[df["year"] == years[0], "entering_aadt"].to_numpy(), dtype=torch.float32
    )
    return GraphData(node_ids, edge_index, edge_attr, years, x, y, clean, split,
                     list(X.columns), aadt)  # fmt: skip


def load_graph(cfg: dict[str, Any]) -> GraphData:
    path = Path(cfg["paths"]["processed_dir"]) / GNN_GRAPH
    if path.exists():
        return torch.load(path, weights_only=False)
    g = build_graph(cfg)
    torch.save(g, path)
    log.info(
        "GNN graph: %d nodes, %d undirected edges, %d node features x %d years",
        len(g.node_ids), g.edge_index.shape[1] // 2, g.x.shape[2], len(g.years),
    )  # fmt: skip
    return g
