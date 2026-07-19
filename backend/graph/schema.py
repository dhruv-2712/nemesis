"""Graph schema for NEMESIS: node/edge types and their feature column order.

The transaction graph is *heterogeneous* — accounts, devices, and IPs are
different kinds of entities with different feature spaces, and PyTorch
Geometric represents that as a `HeteroData` object (one feature matrix per
node type, one edge-index matrix per edge type), rather than forcing
everything into a single node type with padded/shared features.

This module is the single source of truth for:
  - what node types and edge types exist
  - what order their feature columns appear in

`features.py` computes the feature values; `build_graph.py` assembles them
into tensors. Both must agree on column order, so it lives here once instead
of being duplicated (and drifting) in both places.
"""

from enum import Enum


class NodeType(str, Enum):
    ACCOUNT = "account"
    DEVICE = "device"
    IP = "ip"


class EdgeType(str, Enum):
    TRANSACTION = "transaction"       # account -> account (directed, weighted)
    SHARED_DEVICE = "shared_device"   # account -> device (undirected in practice)
    SHARED_IP = "shared_ip"           # account -> ip (undirected in practice)


# (source_node_type, edge_type, target_node_type) triples — this is the
# canonical PyG HeteroData edge-type key format.
EDGE_TYPE_TRIPLES: dict[EdgeType, tuple[NodeType, NodeType]] = {
    EdgeType.TRANSACTION: (NodeType.ACCOUNT, NodeType.ACCOUNT),
    EdgeType.SHARED_DEVICE: (NodeType.ACCOUNT, NodeType.DEVICE),
    EdgeType.SHARED_IP: (NodeType.ACCOUNT, NodeType.IP),
}

# Ordered feature columns per node type. Order matters: it defines the
# column index in the node feature tensor (x) that the GNN will consume.
NODE_FEATURES: dict[NodeType, list[str]] = {
    NodeType.ACCOUNT: [
        "account_age_days",
        "tx_velocity_1h",       # transactions initiated in the last hour
        "tx_velocity_24h",
        "avg_tx_amount",
        "std_tx_amount",
        "distinct_devices",     # count of distinct devices this account has used
        "distinct_ips",
    ],
    NodeType.DEVICE: [
        "distinct_accounts",    # count of distinct accounts seen on this device
        "first_seen_days_ago",
    ],
    NodeType.IP: [
        "distinct_accounts",
        "first_seen_days_ago",
    ],
}

# Ordered feature columns per edge type. These become edge_attr tensors.
EDGE_FEATURES: dict[EdgeType, list[str]] = {
    EdgeType.TRANSACTION: [
        "amount",
        "hours_since_epoch",    # timestamp, normalized to a continuous scale
        "tx_type_encoded",      # categorical transaction type, label-encoded
    ],
    EdgeType.SHARED_DEVICE: [],
    EdgeType.SHARED_IP: [],
}


def node_feature_dim(node_type: NodeType) -> int:
    return len(NODE_FEATURES[node_type])


def edge_feature_dim(edge_type: EdgeType) -> int:
    return len(EDGE_FEATURES[edge_type])
