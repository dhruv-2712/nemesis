"""GET endpoints — serve flagged clusters and their detail to the frontend.

Reads from the cached detection artifact (built by routes_detect / detection.py).
The list view is lightweight (no subgraph payload); the detail view carries the
node/edge subgraph the React force-directed graph draws, plus the full reasoning.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.detection import load_detection

router = APIRouter(prefix="/api/clusters", tags=["clusters"])

_LIST_FIELDS = (
    "cluster_id", "typology", "confidence", "num_nodes", "num_edges",
    "avg_risk", "max_risk", "summary", "time_span_steps",
)


@router.get("")
def list_clusters() -> dict:
    """Flagged-cluster summaries, riskiest first (no subgraph payload)."""
    detection = load_detection()
    summaries = [
        {k: c[k] for k in _LIST_FIELDS} for c in detection["clusters"]
    ]
    return {
        "generated_at": detection["generated_at"],
        "num_clusters": detection["num_clusters"],
        "clusters": summaries,
    }


@router.get("/{cluster_id}")
def get_cluster(cluster_id: str) -> dict:
    """Full detail for one cluster: subgraph (nodes/edges) + verdict + reasoning."""
    detection = load_detection()
    for c in detection["clusters"]:
        if c["cluster_id"] == cluster_id:
            return c
    raise HTTPException(status_code=404, detail=f"cluster '{cluster_id}' not found")
