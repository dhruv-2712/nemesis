"""Tests for the Phase 3 reasoning layer (offline / heuristic mode).

These run without a GROQ_API_KEY: they exercise the deterministic structural
heuristic and the LangGraph pipeline wiring, asserting that each canonical
cluster shape maps to the right typology with a well-formed, bounded verdict.
"""

import pytest

from backend.reasoning.schemas import ClusterFeatures, Typology, TypologyVerdict
from backend.reasoning.pipeline import classify_cluster, heuristic_verdict


def _features(**overrides) -> ClusterFeatures:
    """A neutral baseline cluster; override the fields under test."""
    base = dict(
        cluster_id="c_test",
        num_nodes=10,
        num_edges=9,
        density=0.1,
        avg_illicit_prob=0.95,
        max_illicit_prob=1.0,
        mean_in_degree=0.9,
        mean_out_degree=0.9,
        max_in_degree=1,
        max_out_degree=1,
        longest_chain=2,
        reciprocity=0.0,
        num_sources=1,
        num_sinks=1,
        time_span_steps=1,
    )
    base.update(overrides)
    return ClusterFeatures(**base)


def test_peeling_chain_signature():
    f = _features(num_nodes=10, longest_chain=9, max_out_degree=1)
    assert heuristic_verdict(f).typology is Typology.PEELING_CHAIN


def test_consolidation_signature():
    f = _features(max_in_degree=8, num_sinks=1)
    assert heuristic_verdict(f).typology is Typology.CONSOLIDATION_CASHOUT


def test_fanout_signature():
    f = _features(max_out_degree=9, num_sources=1)
    assert heuristic_verdict(f).typology is Typology.SCAM_PAYOUT_FANOUT


def test_mixing_signature():
    f = _features(density=0.3, max_in_degree=4, max_out_degree=4)
    assert heuristic_verdict(f).typology is Typology.MIXING_TUMBLING


def test_verdict_is_wellformed_and_bounded():
    v = classify_cluster(_features(longest_chain=9, max_out_degree=1))
    assert isinstance(v, TypologyVerdict)
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasoning_chain and all(isinstance(s, str) for s in v.reasoning_chain)
    assert v.summary and v.recommended_action


def test_reasoning_cites_structure():
    # The chain observation should reference the actual longest_chain value.
    v = classify_cluster(_features(num_nodes=10, longest_chain=9, max_out_degree=1))
    assert any("9" in step for step in v.reasoning_chain)


def test_extract_clusters_on_real_graph():
    """Integration: real graph -> clusters -> verdicts (skips if artifacts absent)."""
    torch = pytest.importorskip("torch")
    from pathlib import Path

    graph = Path("data/graphs/elliptic.pt")
    ckpt = Path("backend/models/checkpoints/graphsage.pt")
    if not (graph.exists() and ckpt.exists()):
        pytest.skip("graph/checkpoint artifacts not present")

    from backend.reasoning.cluster import extract_clusters

    data = torch.load(graph, weights_only=False)
    clusters = extract_clusters(data)
    assert clusters, "expected at least one flagged cluster"
    v = classify_cluster(clusters[0])
    assert isinstance(v, TypologyVerdict)
    assert v.typology in Typology
