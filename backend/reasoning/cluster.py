"""Extract flagged clusters from the graph + trained GNN scores.

The GNN scores every node with P(illicit). A single high-risk node is not a
story; a *connected group* of them is. This module:

  1. runs the trained model to get per-node illicit probabilities,
  2. keeps the high-risk nodes (prob >= threshold),
  3. finds weakly-connected components among them (a component = one cluster),
  4. summarizes each cluster's structure into a ClusterFeatures object.

Those features are what the reasoning layer (pipeline.py) hands to the LLM, and
the same extraction feeds the Phase 4 /detect endpoint. NetworkX is used for the
structural metrics — the clusters are small (tens of nodes), so it is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor

from backend.models.graphsage import GraphSAGE
from backend.models.splits import model_features
from backend.reasoning.schemas import ClusterFeatures

GRAPH_PATH = Path("data/graphs/elliptic.pt")
CHECKPOINT_PATH = Path("backend/models/checkpoints/graphsage.pt")

DEFAULT_THRESHOLD = 0.7
MIN_CLUSTER_NODES = 4
MAX_CLUSTER_NODES = 60  # keep clusters LLM-digestible; skip giant components


@dataclass
class FlaggedCluster:
    """A flagged cluster: LLM feature summary + the subgraph for visualization."""

    features: ClusterFeatures
    nodes: list[dict] = field(default_factory=list)  # {id, risk, time_step}
    edges: list[dict] = field(default_factory=list)  # {source, target}


@torch.no_grad()
def illicit_probs(data, checkpoint_path: Path = CHECKPOINT_PATH) -> Tensor:
    """Per-node P(illicit) from the trained GraphSAGE checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = GraphSAGE(in_channels=ckpt["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logits, _ = model(model_features(data.x), data.edge_index)
    return torch.softmax(logits, dim=1)[:, 1]


def _summarize(sub: nx.DiGraph, cluster_id: str, probs, time_steps) -> ClusterFeatures:
    """Compute structural features for one cluster subgraph."""
    n = sub.number_of_nodes()
    m = sub.number_of_edges()
    in_deg = [d for _, d in sub.in_degree()]
    out_deg = [d for _, d in sub.out_degree()]
    possible = n * (n - 1)

    # longest directed path — meaningful only on a DAG; laundering chains are
    # acyclic, and if a cycle exists we cap it at the node count as a proxy.
    try:
        longest_chain = nx.dag_longest_path_length(sub) if nx.is_directed_acyclic_graph(sub) else n
    except Exception:
        longest_chain = n

    node_probs = [probs[u].item() for u in sub.nodes]
    steps = [int(time_steps[u].item()) for u in sub.nodes]

    return ClusterFeatures(
        cluster_id=cluster_id,
        num_nodes=n,
        num_edges=m,
        density=(m / possible) if possible else 0.0,
        avg_illicit_prob=sum(node_probs) / n,
        max_illicit_prob=max(node_probs),
        mean_in_degree=sum(in_deg) / n,
        mean_out_degree=sum(out_deg) / n,
        max_in_degree=max(in_deg) if in_deg else 0,
        max_out_degree=max(out_deg) if out_deg else 0,
        longest_chain=int(longest_chain),
        reciprocity=nx.reciprocity(sub) or 0.0,
        num_sources=sum(1 for d in in_deg if d == 0),
        num_sinks=sum(1 for d in out_deg if d == 0),
        time_span_steps=(max(steps) - min(steps) + 1) if steps else 1,
    )


def summarize_graph(node_ids, edges, probs, time_steps, cluster_id: str = "sandbox") -> ClusterFeatures:
    """ClusterFeatures for an arbitrary graph (used by the sandbox).

    node_ids index into probs/time_steps; edges are (src_idx, dst_idx) pairs.
    Same structural summary as the flagged-cluster path, so a hand-built pattern
    is described with exactly the features the detector uses.
    """
    g = nx.DiGraph()
    g.add_nodes_from(node_ids)
    g.add_edges_from(edges)
    return _summarize(g, cluster_id, probs, time_steps)


def extract_flagged(
    data,
    probs: Tensor | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_nodes: int = MIN_CLUSTER_NODES,
    max_nodes: int = MAX_CLUSTER_NODES,
) -> list[FlaggedCluster]:
    """Return flagged clusters (features + viz subgraph), riskiest first."""
    if probs is None:
        probs = illicit_probs(data)

    time_steps = data.x[:, 0]
    high_risk = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
    high_risk_set = set(high_risk)

    # Build the directed subgraph induced on high-risk nodes only.
    g = nx.DiGraph()
    g.add_nodes_from(high_risk)
    src, dst = data.edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        if s in high_risk_set and d in high_risk_set:
            g.add_edge(s, d)

    flagged: list[FlaggedCluster] = []
    for i, comp in enumerate(nx.weakly_connected_components(g)):
        if not (min_nodes <= len(comp) <= max_nodes):
            continue
        sub = g.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        cid = f"cluster_{i:04d}"
        features = _summarize(sub, cid, probs, time_steps)
        nodes = [
            {"id": int(u), "risk": round(probs[u].item(), 3), "time_step": int(time_steps[u].item())}
            for u in sub.nodes
        ]
        edges = [{"source": int(s), "target": int(d)} for s, d in sub.edges]
        flagged.append(FlaggedCluster(features=features, nodes=nodes, edges=edges))

    flagged.sort(key=lambda fc: fc.features.avg_illicit_prob, reverse=True)
    return flagged


def extract_clusters(
    data,
    probs: Tensor | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_nodes: int = MIN_CLUSTER_NODES,
    max_nodes: int = MAX_CLUSTER_NODES,
) -> list[ClusterFeatures]:
    """Structural summaries only (LLM input). Thin wrapper over extract_flagged."""
    return [fc.features for fc in extract_flagged(data, probs, threshold, min_nodes, max_nodes)]


if __name__ == "__main__":
    data = torch.load(GRAPH_PATH, weights_only=False)
    clusters = extract_clusters(data)
    print(f"extracted {len(clusters)} flagged clusters (threshold {DEFAULT_THRESHOLD})")
    for c in clusters[:5]:
        print(
            f"  {c.cluster_id}: {c.num_nodes}n/{c.num_edges}e "
            f"avg_prob {c.avg_illicit_prob:.2f} chain {c.longest_chain} "
            f"max_in {c.max_in_degree} max_out {c.max_out_degree}"
        )
