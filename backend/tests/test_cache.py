"""Tests for the reasoning verdict cache — the LLM-call reduction mechanism."""

import pytest

from backend.reasoning.schemas import ClusterFeatures, Typology, TypologyVerdict
from backend.reasoning import cache as cache_mod


def _features(**overrides) -> ClusterFeatures:
    base = dict(
        cluster_id="c", num_nodes=6, num_edges=5, density=0.16,
        avg_illicit_prob=0.95, max_illicit_prob=1.0, mean_in_degree=0.8,
        mean_out_degree=0.8, max_in_degree=1, max_out_degree=1, longest_chain=5,
        reciprocity=0.0, num_sources=1, num_sinks=1, time_span_steps=1,
    )
    base.update(overrides)
    return ClusterFeatures(**base)


def _verdict() -> TypologyVerdict:
    return TypologyVerdict(
        typology=Typology.PEELING_CHAIN, confidence=0.9, summary="s",
        reasoning_chain=["a"], recommended_action="act",
    )


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_PATH", tmp_path / "cache.json")
    return cache_mod


def test_identical_structure_shares_fingerprint(temp_cache):
    # Same structural facts, different cluster_id -> same fingerprint (id is not part of it).
    f1 = _features(cluster_id="cluster_0001")
    f2 = _features(cluster_id="cluster_9999")
    assert temp_cache.cluster_fingerprint(f1) == temp_cache.cluster_fingerprint(f2)


def test_hit_avoids_recompute(temp_cache):
    c = temp_cache.VerdictCache()
    assert c.get(_features(cluster_id="a")) is None  # miss
    c.put(_features(cluster_id="a"), _verdict(), "llm")
    hit = c.get(_features(cluster_id="b"))  # different id, same shape -> hit
    assert hit is not None and hit.typology is Typology.PEELING_CHAIN
    assert c.hits == 1 and c.misses == 1


def test_persistence_across_instances(temp_cache):
    c1 = temp_cache.VerdictCache()
    c1.put(_features(), _verdict(), "llm")
    c1.flush()
    # A fresh cache instance loads the flushed entry -> zero-call repeat runs.
    c2 = temp_cache.VerdictCache()
    assert c2.get(_features()) is not None
    assert c2.misses == 0


def test_different_structure_misses(temp_cache):
    c = temp_cache.VerdictCache()
    c.put(_features(max_in_degree=1), _verdict(), "llm")
    assert c.get(_features(max_in_degree=8)) is None  # consolidation shape -> distinct
