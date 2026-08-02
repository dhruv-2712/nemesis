"""Sandbox scoring — run a user-built transaction pattern through the pipeline.

The user builds a topology (nodes + directed edges, optional per-node illicit/
licit profile). We seed each node with real feature vectors sampled from that
profile (feature_bank), score every node with the trained GNN, summarize the
structure, and classify the laundering typology — the full GNN -> reasoning path,
live, on input that never existed in training.

Honesty note: node features are *sampled from* the illicit/licit distribution,
not derived from the drawn pattern (a drawn node has no real features). The API
returns that provenance so the UI can label it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from backend.models.feature_bank import sample_features
from backend.reasoning.cluster import illicit_probs, summarize_graph
from backend.reasoning.pipeline import classify_cluster

MAX_NODES = 100
MAX_EDGES = 400


def analyze(nodes: list[dict], edges: list[dict], seed: int | None = None) -> dict:
    """Score a hand-built pattern. nodes: [{id, profile?}], edges: [{source, target}]."""
    if not nodes:
        raise ValueError("no nodes provided")
    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise ValueError(f"pattern too large (max {MAX_NODES} nodes, {MAX_EDGES} edges)")

    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node ids")
    index = {nid: i for i, nid in enumerate(ids)}

    # Seed features per node from its profile (default illicit).
    rows = [sample_features(n.get("profile", "illicit"), 1, seed=seed)[0] for n in nodes]
    x = torch.tensor(np.stack(rows), dtype=torch.float)

    pairs = [
        (index[e["source"]], index[e["target"]])
        for e in edges
        if e["source"] in index and e["target"] in index
    ]
    edge_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs else torch.zeros((2, 0), dtype=torch.long)
    )

    data = Data(x=x, edge_index=edge_index)
    probs = illicit_probs(data)  # loads the committed checkpoint
    features = summarize_graph(list(range(len(ids))), pairs, probs, x[:, 0], "sandbox")
    verdict = classify_cluster(features)

    return {
        "nodes": [{"id": ids[i], "risk": round(probs[i].item(), 3)} for i in range(len(ids))],
        "edges": [{"source": e["source"], "target": e["target"]} for e in edges],
        "avg_risk": round(probs.mean().item(), 3),
        "max_risk": round(probs.max().item(), 3),
        "features": features.model_dump(),
        "feature_source": "sampled from illicit/licit distribution",
        "verdict": {
            "typology": verdict.typology.value,
            "confidence": verdict.confidence,
            "summary": verdict.summary,
            "reasoning_chain": verdict.reasoning_chain,
            "recommended_action": verdict.recommended_action,
        },
    }
