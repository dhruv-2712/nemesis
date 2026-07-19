"""Assemble the Elliptic transaction graph and serialize it to disk.

Loads features/labels via features.py, maps txId pairs from the edgelist
to integer node indices, and saves a PyG Data object to data/graphs/.
Run directly to rebuild: python -m backend.graph.build_graph
"""

from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data
from backend.graph.features import load_elliptic, ELLIPTIC_DIR

OUTPUT_PATH = Path("data/graphs/elliptic.pt")


def build_elliptic_graph() -> Data:
    print("Loading features and labels...")
    x, y, tx_ids = load_elliptic()
    id_to_idx = {tid: i for i, tid in enumerate(tx_ids)}

    print("Building edge index...")
    edges = pd.read_csv(ELLIPTIC_DIR / "elliptic_txs_edgelist.csv")
    mask = edges["txId1"].isin(id_to_idx) & edges["txId2"].isin(id_to_idx)
    edges = edges[mask]
    src = [id_to_idx[tid] for tid in edges["txId1"]]
    dst = [id_to_idx[tid] for tid in edges["txId2"]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.validate(raise_on_error=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"  nodes: {data.num_nodes:,}  edges: {data.num_edges:,}")
    print(f"  features: {data.num_node_features}")
    labeled = (y >= 0).sum().item()
    illicit = (y == 1).sum().item()
    print(f"  labeled: {labeled:,}  illicit: {illicit:,} ({illicit/labeled*100:.1f}% of labeled)")
    return data


if __name__ == "__main__":
    build_elliptic_graph()
