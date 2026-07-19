"""Load Elliptic Bitcoin dataset CSVs into tensors for graph construction.

Returns x (N x 166 float features), y (N labels), and tx_ids (node ordering)
in a single pass so build_graph.py can assemble the PyG Data object.
"""

from pathlib import Path
import pandas as pd
import torch
from backend.graph.schema import LABEL_MAP, NODE_FEATURE_DIM

ELLIPTIC_DIR = Path("data/raw/elliptic/elliptic_bitcoin_dataset")


def load_elliptic() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Load features and labels from the Elliptic dataset.

    Returns:
        x:       (N, 166) float tensor — time_step + local + agg features
        y:       (N,)     long tensor  — 1 illicit, 0 licit, -1 unknown
        tx_ids:  list of N transaction IDs, defining node index order
    """
    feats = pd.read_csv(ELLIPTIC_DIR / "elliptic_txs_features.csv", header=None)
    # col 0 = txId, cols 1-166 = features (matches NODE_FEATURE_DIM)
    tx_ids: list[int] = feats.iloc[:, 0].tolist()
    x = torch.tensor(feats.iloc[:, 1:].values, dtype=torch.float)
    assert x.shape[1] == NODE_FEATURE_DIM, (
        f"Expected {NODE_FEATURE_DIM} features, got {x.shape[1]}"
    )

    cls = pd.read_csv(ELLIPTIC_DIR / "elliptic_txs_classes.csv")
    cls_lookup = cls.set_index("txId")["class"].map(LABEL_MAP)
    y = torch.tensor(
        [cls_lookup.get(tid, -1) for tid in tx_ids], dtype=torch.long
    )

    return x, y, tx_ids
