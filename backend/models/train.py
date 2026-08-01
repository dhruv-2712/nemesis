"""Train the GraphSAGE fraud-ring detector on the Elliptic graph.

Full-batch training (the graph is small enough: 203k nodes / 234k edges fits in
CPU memory comfortably). Class imbalance — 9.76% illicit among labeled nodes —
is handled with a class-weighted cross-entropy so the minority illicit class is
not drowned out. Model selection is by illicit-class F1 on the temporal
validation band, with early stopping.

Run:  python -m backend.models.train                # GraphSAGE (baseline)
      python -m backend.models.train --model gat    # GAT (attention stretch)
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from backend.models.graphsage import GraphSAGE
from backend.models.gat import GAT
from backend.models.splits import temporal_masks, model_features

GRAPH_PATH = Path("data/graphs/elliptic.pt")
CHECKPOINT_DIR = Path("backend/models/checkpoints")

MODELS = {"graphsage": GraphSAGE, "gat": GAT}

EPOCHS = 300
LR = 0.01
WEIGHT_DECAY = 5e-4
PATIENCE = 30
SEED = 42


def class_weights(y: Tensor, train_mask: Tensor, beta: float = 0.5) -> Tensor:
    """Softened inverse-frequency weights over the training split.

    Full inverse-frequency (beta=1) weights illicit ~4.6x here, which over-flags:
    the Elliptic features are informative enough that such heavy reweighting
    trades away too much precision. beta=0.5 (sqrt inverse-frequency) keeps the
    minority class emphasized without collapsing precision.
    """
    y_train = y[train_mask]
    counts = torch.bincount(y_train, minlength=2).float()
    weights = (counts.sum() / (2.0 * counts.clamp(min=1))) ** beta
    return weights


@torch.no_grad()
def f1_illicit(logits: Tensor, y: Tensor, mask: Tensor) -> tuple[float, float, float]:
    """Precision, recall, F1 for the illicit class (label 1) over a mask."""
    pred = logits[mask].argmax(dim=1)
    true = y[mask]
    tp = ((pred == 1) & (true == 1)).sum().item()
    fp = ((pred == 1) & (true == 0)).sum().item()
    fn = ((pred == 0) & (true == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def train(model_name: str = "graphsage") -> None:
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    data = torch.load(GRAPH_PATH, weights_only=False)
    x = model_features(data.x).to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    train_mask, val_mask, test_mask = temporal_masks(data.x, y)
    train_mask, val_mask, test_mask = (
        train_mask.to(device),
        val_mask.to(device),
        test_mask.to(device),
    )
    print(
        f"nodes {x.size(0):,} | features {x.size(1)} | "
        f"train {int(train_mask.sum()):,} val {int(val_mask.sum()):,} "
        f"test {int(test_mask.sum()):,}"
    )

    model = MODELS[model_name](in_channels=x.size(1)).to(device)
    print(f"model: {model_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = class_weights(y, train_mask).to(device)
    print(f"class weights (licit, illicit): {weights.tolist()}")

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x, edge_index)
        _, _, val_f1 = f1_illicit(logits, y, val_mask)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            tr_p, tr_r, tr_f1 = f1_illicit(logits, y, train_mask)
            print(
                f"epoch {epoch:3d} | loss {loss.item():.4f} | "
                f"train F1 {tr_f1:.3f} | val F1 {val_f1:.3f} | best {best_val_f1:.3f}"
            )

        if epochs_no_improve >= PATIENCE:
            print(f"early stop at epoch {epoch} (no val improvement in {PATIENCE})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, _ = model(x, edge_index)
    p, r, f1 = f1_illicit(logits, y, test_mask)
    print(f"\nTEST (illicit) — precision {p:.3f} | recall {r:.3f} | F1 {f1:.3f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    torch.save(
        {
            "model": model_name,
            "state_dict": best_state,
            "in_channels": x.size(1),
            "test_precision": p,
            "test_recall": r,
            "test_f1": f1,
            "val_f1": best_val_f1,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NEMESIS GNN on Elliptic.")
    parser.add_argument("--model", choices=MODELS, default="graphsage")
    args = parser.parse_args()
    train(args.model)
