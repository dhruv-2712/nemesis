"""Graph schema for NEMESIS: node/edge types and feature layout.

Dataset: Elliptic Bitcoin Dataset
  - 203,769 transaction nodes, 234,355 directed edges
  - Labels: 1 = illicit, 2 = licit, "unknown" = unlabeled (remapped to 0/1/-1)
  - 49 discrete time steps (temporal snapshots)

Unlike the original heterogeneous design (account/device/IP), Elliptic is a
*homogeneous* directed graph — every node is a Bitcoin transaction, every edge
is a Bitcoin flow (tx -> tx). PyTorch Geometric represents this as a plain
`Data` object (not `HeteroData`), which simplifies the model significantly.

Feature layout (166 features per node, order locked here):
  [0]       time_step          — integer 1-49, which temporal snapshot
  [1:94]    local_f0..f92      — 93 transaction-level features (anonymized:
                                 input/output counts, fee proxies, volume, etc.)
  [94:166]  agg_f0..f71        — 72 aggregated 1-hop neighborhood features
                                 (same feature types, averaged over neighbors)

`features.py` reads the raw CSVs and outputs tensors in this column order.
`build_graph.py` assembles x, edge_index, y into a PyG Data object.
Both must use the slices defined here — don't hardcode indices elsewhere.
"""

# Feature slice boundaries (end-exclusive, for use with tensor slicing)
TIME_STEP_IDX = 0
LOCAL_SLICE = slice(1, 94)    # 93 features
AGG_SLICE = slice(94, 166)    # 72 features
NODE_FEATURE_DIM = 166

# Label mapping: Elliptic raw -> internal
# raw "1" (illicit) -> 1, raw "2" (licit) -> 0, "unknown" -> -1
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}

# Column names in the raw CSVs
FEATURES_ID_COL = "txId"
EDGELIST_COLS = ("txId1", "txId2")
CLASSES_COLS = ("txId", "class")
