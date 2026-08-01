"""GAT model for NEMESIS — Phase 2 stretch goal.

Same role as GraphSAGE (learn embeddings + classify illicit/licit) but with
multi-head *attention* aggregation: instead of mean-pooling neighbors, each node
learns how much weight to give each neighbor. On fraud rings this is the more
expressive choice — a mule node can learn to attend to the shared-flow neighbors
that make it look like part of a ring, rather than averaging over all neighbors
equally.

Drop-in compatible with train.py / evaluate.py: same (logits, embeddings)
forward signature and the same 165-feature input contract as GraphSAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from backend.models.graphsage import MODEL_INPUT_DIM


class GAT(nn.Module):
    """Two-layer GAT: multi-head attention -> single-head embedding + head."""

    def __init__(
        self,
        in_channels: int = MODEL_INPUT_DIM,
        hidden_channels: int = 32,
        embedding_dim: int = 64,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Layer 1: `heads` attention heads, concatenated -> hidden_channels * heads.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Layer 2: average a single head down to the embedding dimension.
        self.conv2 = GATConv(
            hidden_channels * heads, embedding_dim, heads=1, concat=False, dropout=dropout
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x, edge_index)
        logits = self.classifier(F.relu(emb))
        return logits, emb
