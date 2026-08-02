"""Generate committed artifacts for the flagship dashboard + embedding map.

One-off (needs the 135 MB graph + checkpoint). Produces small JSON files the API
serves so the deployed app needs neither the graph nor a live t-SNE:

  backend/api/seed_stats.json   — dataset totals + real model metrics + split
  backend/api/embedding_2d.json — subsampled t-SNE points (illicit/licit, risk,
                                  and the flagged cluster each point belongs to)

Run:  python -m scripts.build_flagship_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from backend.models.graphsage import GraphSAGE
from backend.models.splits import model_features, temporal_masks
from backend.api.detection import load_detection

GRAPH_PATH = Path("data/graphs/elliptic.pt")
CHECKPOINT_PATH = Path("backend/models/checkpoints/graphsage.pt")
STATS_PATH = Path("backend/api/seed_stats.json")
EMBEDDING_PATH = Path("backend/api/embedding_2d.json")

SUBSAMPLE_LICIT = 2000


def main() -> None:
    data = torch.load(GRAPH_PATH, weights_only=False)
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = GraphSAGE(in_channels=ckpt["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = model_features(data.x)
    with torch.no_grad():
        logits, emb = model(x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, 1]

    # ---- stats: real metrics on the temporal test split ----
    _, _, test_mask = temporal_masks(data.x, data.y)
    y_true = data.y[test_mask].numpy()
    y_pred = logits[test_mask].argmax(1).numpy()
    y_score = probs[test_mask].numpy()
    labeled = int((data.y >= 0).sum())
    illicit = int((data.y == 1).sum())
    stats = {
        "dataset": {
            "nodes": data.num_nodes, "edges": data.num_edges,
            "features": data.num_node_features, "labeled": labeled,
            "illicit": illicit, "illicit_rate": round(illicit / labeled, 4),
            "time_steps": int(data.x[:, 0].max()),
        },
        "model": {
            "architecture": "GraphSAGE",
            "roc_auc": round(float(roc_auc_score(y_true, y_score)), 3),
            "pr_auc": round(float(average_precision_score(y_true, y_score)), 3),
            "illicit_f1": round(float(f1_score(y_true, y_pred)), 3),
            "illicit_precision": round(float(precision_score(y_true, y_pred)), 3),
            "illicit_recall": round(float(recall_score(y_true, y_pred)), 3),
        },
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"stats -> {STATS_PATH}: {stats['model']}")

    # ---- embedding map: subsample, t-SNE, link to flagged clusters ----
    node_to_cluster = {}
    for c in load_detection()["clusters"]:
        for n in c["nodes"]:
            node_to_cluster[n["id"]] = c["cluster_id"]

    rng = np.random.default_rng(42)
    illicit_idx = (data.y == 1).nonzero(as_tuple=True)[0].numpy()
    licit_idx = (data.y == 0).nonzero(as_tuple=True)[0].numpy()
    licit_pick = rng.choice(licit_idx, size=min(SUBSAMPLE_LICIT, len(licit_idx)), replace=False)
    sel = np.concatenate([illicit_idx, licit_pick])
    rng.shuffle(sel)

    proj = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(
        emb[sel].detach().numpy()
    )
    points = []
    for k, node in enumerate(sel.tolist()):
        points.append({
            "x": round(float(proj[k, 0]), 2),
            "y": round(float(proj[k, 1]), 2),
            "label": int(data.y[node].item()),          # 1 illicit, 0 licit
            "risk": round(float(probs[node].item()), 3),
            "cluster_id": node_to_cluster.get(node),      # None if not in a flagged cluster
        })
    EMBEDDING_PATH.write_text(json.dumps({"points": points}), encoding="utf-8")
    print(f"embedding -> {EMBEDDING_PATH}: {len(points)} points "
          f"({int((data.y[sel] == 1).sum())} illicit)")


if __name__ == "__main__":
    main()
