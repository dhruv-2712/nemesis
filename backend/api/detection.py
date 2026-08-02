"""Detection orchestration — the graph+model+reasoning pipeline, cached to disk.

Running the GNN and the reasoning layer over the whole graph is expensive to do
per HTTP request, and the Elliptic snapshot is static, so we run detection once
and cache the result as JSON. The API then serves from that cache; POST /detect
rebuilds it. This keeps the API fast and lets the deployed demo boot without a
GPU or a live model pass on every call.

Artifact shape (data/graphs/detection.json):
  { generated_at, threshold, num_clusters, clusters: [ {summary + subgraph} ] }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from backend.reasoning.cache import VerdictCache, clear_cache
from backend.reasoning.cluster import extract_flagged, DEFAULT_THRESHOLD
from backend.reasoning.pipeline import classify_cluster_verbose

GRAPH_PATH = Path("data/graphs/elliptic.pt")
DETECTION_PATH = Path("data/graphs/detection.json")
# Committed fallback so a fresh deploy serves cluster data without the 135 MB
# graph or a model pass. Regenerate with `python -m backend.api.detection`.
SEED_DETECTION_PATH = Path("backend/api/seed_detection.json")

# Reason over the top-N riskiest clusters (keeps live LLM calls bounded).
MAX_CLUSTERS = 50


def _cluster_record(fc, verdict) -> dict:
    f = fc.features
    return {
        "cluster_id": f.cluster_id,
        "typology": verdict.typology.value,
        "confidence": verdict.confidence,
        "summary": verdict.summary,
        "recommended_action": verdict.recommended_action,
        "reasoning_chain": verdict.reasoning_chain,
        "num_nodes": f.num_nodes,
        "num_edges": f.num_edges,
        "avg_risk": round(f.avg_illicit_prob, 3),
        "max_risk": round(f.max_illicit_prob, 3),
        "time_span_steps": f.time_span_steps,
        "features": f.model_dump(),
        "nodes": fc.nodes,
        "edges": fc.edges,
    }


def run_detection(
    threshold: float = DEFAULT_THRESHOLD,
    max_clusters: int = MAX_CLUSTERS,
    refresh: bool = False,
) -> dict:
    """Run extraction + reasoning over the graph and cache the result to disk.

    Reasoning verdicts are served from a persistent fingerprint cache: identical
    structures reuse one verdict, so repeat runs make few or zero LLM calls. Pass
    refresh=True to discard the cache and regenerate every verdict from scratch.
    """
    if refresh:
        clear_cache()

    data = torch.load(GRAPH_PATH, weights_only=False)
    flagged = extract_flagged(data, threshold=threshold)[:max_clusters]

    cache = VerdictCache()
    clusters = []
    for fc in flagged:
        verdict = cache.get(fc.features)
        if verdict is None:
            verdict, used_llm = classify_cluster_verbose(fc.features)
            cache.put(fc.features, verdict, "llm" if used_llm else "heuristic")
        clusters.append(_cluster_record(fc, verdict))
    cache.flush()

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "num_clusters": len(clusters),
        "reasoning_calls": cache.misses,
        "cache_hits": cache.hits,
        "clusters": clusters,
    }

    DETECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETECTION_PATH.write_text(json.dumps(result), encoding="utf-8")
    return result


def load_detection() -> dict:
    """Return detection results, preferring the fresh cache, then seed, then a run.

    Order: runtime cache (from a prior /detect) -> committed seed (deploy-friendly)
    -> a live run if the graph is available. This lets the API serve anywhere
    while still recomputing when the full artifacts are present.
    """
    if DETECTION_PATH.exists():
        return json.loads(DETECTION_PATH.read_text(encoding="utf-8"))
    if SEED_DETECTION_PATH.exists():
        return json.loads(SEED_DETECTION_PATH.read_text(encoding="utf-8"))
    return run_detection()


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Run NEMESIS detection over the graph.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="discard the reasoning cache and regenerate every verdict (more LLM calls)",
    )
    args = parser.parse_args()

    res = run_detection(refresh=args.refresh)
    print(f"detected {res['num_clusters']} clusters -> {DETECTION_PATH}")
    print(f"reasoning calls: {res['reasoning_calls']} | cache hits: {res['cache_hits']}")
    by_typ = Counter(c["typology"] for c in res["clusters"])
    for typ, n in by_typ.most_common():
        print(f"  {typ}: {n}")
