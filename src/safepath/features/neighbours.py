"""Graph structure shared by the GNN and by LightGBM's neighbour features (torch-free).

- `build_edges`: one undirected edge per pair of consolidated intersections joined by at
  least one road segment, with merged segment attributes.
- `neighbour_features`: for every intersection-year, the mean and max over its 1-hop and
  2-hop neighbours of log entering AADT, prior-3-year crash share, signal presence and road
  class rank, plus its degree. Neighbours' crash shares come from their own history
  features, so they use only years before the label year.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, identity

from safepath.features.roads import class_rank
from safepath.features.static import first_number
from safepath.graph import NODE_MAP, SEGMENTS

EDGE_FEATURES = ["class_rank", "length_m", "lanes", "maxspeed", "log_aadt", "motorway"]
NEIGHBOUR_SOURCES = {
    "log_aadt": "log_aadt_entering",
    "crash_share": "prev3_crash_share_mean",
    "signal": "has_signal",
    "class_rank": "max_class_rank",
}
NEIGHBOUR_FEATURES = ["degree"] + [
    f"nb{hop}_{stat}_{name}"
    for hop in (1, 2)
    for name in NEIGHBOUR_SOURCES
    for stat in ("mean", "max")
]


def build_edges(cfg: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    """Undirected intersection pairs (a < b) with merged, unstandardised edge features."""
    from safepath.models.volume import AADT_EDGES  # here: models.lgbm imports this module

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


def neighbour_pairs(pairs: np.ndarray, n: int) -> dict[int, pd.DataFrame]:
    """(node, neighbour) rows for 1-hop and 2-hop neighbourhoods (2-hop = within two hops,
    excluding the node itself)."""
    a = coo_matrix((np.ones(pairs.shape[1] * 2), (np.r_[pairs[0], pairs[1]],
                    np.r_[pairs[1], pairs[0]])), shape=(n, n)).tocsr()  # fmt: skip
    a.data[:] = 1
    two = ((a @ a + a) > 0).astype(np.int8) - identity(n, dtype=np.int8, format="csr")
    two.eliminate_zeros()
    out = {}
    for hop, mat in ((1, a), (2, two.tocoo())):
        m = mat.tocoo()
        keep = m.data > 0
        out[hop] = pd.DataFrame({"node": m.row[keep], "nb": m.col[keep]})
    return out


def neighbour_features(table: pd.DataFrame, hops: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Neighbour aggregates for every (intersection_id, year) in `table`, which needs the
    NEIGHBOUR_SOURCES columns."""
    frames = []
    for year, t in table.groupby("year"):
        vals = t.set_index("intersection_id")[list(NEIGHBOUR_SOURCES.values())].astype(float)
        f = pd.DataFrame(index=vals.index)
        f["degree"] = hops[1].groupby("node").size().reindex(f.index).fillna(0)
        for hop, pairs in hops.items():
            joined = pairs.join(vals, on="nb")
            g = joined.groupby("node")
            for name, col in NEIGHBOUR_SOURCES.items():
                f[f"nb{hop}_mean_{name}"] = g[col].mean().reindex(f.index)
                f[f"nb{hop}_max_{name}"] = g[col].max().reindex(f.index)
        frames.append(f.assign(year=year).reset_index())
    return pd.concat(frames, ignore_index=True)


def neighbour_table(cfg: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    pairs, _ = build_edges(cfg)
    n = int(table["intersection_id"].max()) + 1
    return neighbour_features(table, neighbour_pairs(pairs, n))
