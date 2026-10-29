"""Approach roads of each consolidated intersection and their class hierarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Higher rank = more major road. Links rank with the road class they serve, except motorway
# and trunk links, which sit between the expressway and the arterials they feed.
CLASS_RANK = {
    "motorway": 7,
    "trunk": 6,
    "motorway_link": 5,
    "trunk_link": 5,
    "primary": 4,
    "primary_link": 4,
    "secondary": 3,
    "secondary_link": 3,
    "tertiary": 2,
    "tertiary_link": 2,
    "unclassified": 1,
    "residential": 1,
}


def class_rank(highway: pd.Series) -> pd.Series:
    """Rank of an OSM highway class; anything unlisted (living_street, busway, ...) is 0."""
    return highway.map(CLASS_RANK).fillna(0).astype(int)


def tier(rank: pd.Series | np.ndarray) -> pd.Series:
    """Collapse class ranks into four tiers used for class-combo categories."""
    return pd.Series(
        pd.cut(rank, bins=[-1, 1, 2, 4, 7], labels=["local", "collector", "arterial", "highway"])
    ).astype(str)


def approaches(segments: pd.DataFrame, node_map: pd.DataFrame) -> pd.DataFrame:
    """One row per (intersection, approach segment).

    A segment approaches an intersection when exactly one of its endpoints belongs to that
    intersection's cluster; segments inside a cluster (e.g. the short link between two
    carriageways) are internal and skipped. `at_u` records which end touches the
    intersection, so callers can orient the geometry outward.
    """
    cluster = node_map.set_index("osmid")["intersection_id"]
    seg = segments.copy()
    seg["cu"] = seg["u"].map(cluster)
    seg["cv"] = seg["v"].map(cluster)
    from_u = seg[seg["cu"].notna() & (seg["cu"] != seg["cv"])].assign(
        intersection_id=lambda d: d["cu"], at_u=True
    )
    from_v = seg[seg["cv"].notna() & (seg["cu"] != seg["cv"])].assign(
        intersection_id=lambda d: d["cv"], at_u=False
    )
    out = pd.concat([from_u, from_v], ignore_index=True)
    out["intersection_id"] = out["intersection_id"].astype(int)
    out["class_rank"] = class_rank(out["highway"])
    return out.drop(columns=["cu", "cv"])


def class_combo(appr: pd.DataFrame) -> pd.DataFrame:
    """Max / min approach class rank and the tier combo (e.g. 'arterial x local')."""
    g = appr.groupby("intersection_id")["class_rank"].agg(["max", "min"])
    g.columns = ["max_class_rank", "min_class_rank"]
    g["class_combo"] = (
        tier(g["max_class_rank"].to_numpy()).to_numpy()
        + " x "
        + tier(g["min_class_rank"].to_numpy()).to_numpy()
    )
    return g
