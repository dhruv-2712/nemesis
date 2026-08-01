"""GraphSAGE model for NEMESIS — Phase 2 baseline.

Inductive GNN that learns structural embeddings on the Elliptic transaction
graph. Two SAGEConv layers aggregate 1- and 2-hop neighborhoods; the output of
the second layer is the node *embedding* (used for the t-SNE clustering proof),
and a small linear head maps embeddings to illicit/licit logits.

Input is 165 features per node — the raw 166 minus the time_step column
(feature 0), which is metadata for the temporal split, not a learnable signal.
Excluding it matches the canonical Elliptic protocol (Weber et al. 2019) and
stops the model from keying on the snapshot index as a spurious shortcut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from backend.graph.schema import NODE_FEATURE_DIM

# 166 raw features minus the time_step column (index 0) fed to the split, not the model.
MODEL_INPUT_DIM = NODE_FEATURE_DIM - 1  # 165


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE with an exposed embedding layer + classifier head."""

    def __init__(
        self,
        in_channels: int = MODEL_INPUT_DIM,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings (pre-classifier) — used for clustering/t-SNE."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, embeddings)."""
        emb = self.embed(x, edge_index)
        logits = self.classifier(F.relu(emb))
        return logits, emb
