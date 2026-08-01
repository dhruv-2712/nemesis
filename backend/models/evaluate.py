"""Evaluate a trained GraphSAGE checkpoint on the Elliptic test split.

Reports the metrics that matter for a rare-fraud detector — illicit-class
precision/recall/F1 and ROC-AUC / PR-AUC over the held-out temporal test band —
and exports node embeddings + labels so the notebook can show that illicit nodes
cluster in embedding space (the Phase 2 checkpoint goal).

Run:  python -m backend.models.evaluate
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from backend.models.graphsage import GraphSAGE
from backend.models.splits import temporal_masks, model_features

GRAPH_PATH = Path("data/graphs/elliptic.pt")
CHECKPOINT_PATH = Path("backend/models/checkpoints/graphsage.pt")
EMBEDDINGS_PATH = Path("data/graphs/embeddings.pt")


def load_model_and_graph():
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    data = torch.load(GRAPH_PATH, weights_only=False)
    model = GraphSAGE(in_channels=ckpt["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, data, ckpt


@torch.no_grad()
def evaluate() -> None:
    model, data, ckpt = load_model_and_graph()
    x = model_features(data.x)
    y = data.y
    _, _, test_mask = temporal_masks(data.x, y)

    logits, emb = model(x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, 1]  # P(illicit)

    y_true = y[test_mask].numpy()
    y_pred = logits[test_mask].argmax(dim=1).numpy()
    y_score = probs[test_mask].numpy()

    print(f"Test nodes: {len(y_true):,}  (illicit {int(y_true.sum()):,})\n")
    print(classification_report(y_true, y_pred, target_names=["licit", "illicit"], digits=3))
    print("confusion matrix [rows=true licit/illicit]:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"PR-AUC : {average_precision_score(y_true, y_score):.4f}")

    # Export embeddings for the t-SNE/UMAP clustering proof (labeled nodes only).
    labeled = (y >= 0)
    torch.save(
        {
            "embeddings": emb[labeled].detach(),
            "labels": y[labeled].detach(),
            "time_step": data.x[labeled, 0].detach(),
        },
        EMBEDDINGS_PATH,
    )
    print(f"\nsaved embeddings ({int(labeled.sum()):,} labeled nodes) -> {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    evaluate()
