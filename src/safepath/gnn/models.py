"""Node-level crash models: MLP (no graph), GraphSAGE, GATv2 with edge features.

All three predict log expected crashes per intersection-year and share the same shape: an
input projection, `layers` residual blocks (layer -> LayerNorm -> ReLU -> dropout, added to
the block's input), and a linear head. The head bias starts at log(mean crashes) so training
begins from a calibrated constant.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv

MODELS = ("mlp", "sage", "gatv2")


class NodeModel(nn.Module):
    def __init__(
        self,
        kind: str,
        in_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        edge_dim: int | None = None,
        heads: int = 4,
        init_rate: float = 1.0,
    ):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind!r}")
        self.kind = kind
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            if kind == "mlp":
                layer = nn.Linear(hidden, hidden)
            elif kind == "sage":
                layer = SAGEConv(hidden, hidden, aggr="mean")
            else:
                layer = GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim)
            self.blocks.append(layer)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.head.bias.fill_(float(torch.log(torch.tensor(init_rate))))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        for layer, norm in zip(self.blocks, self.norms, strict=True):
            if self.kind == "mlp":
                z = layer(h)
            elif self.kind == "sage":
                z = layer(h, edge_index)
            else:
                z = layer(h, edge_index, edge_attr)
            h = h + self.drop(torch.relu(norm(z)))
        return self.head(h).squeeze(-1)  # log expected crashes
