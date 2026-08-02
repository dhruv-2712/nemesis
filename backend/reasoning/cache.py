"""Persistent verdict cache — avoid re-paying for LLM calls on the same structure.

Every detection run otherwise fires one LLM call per cluster (50+), which burns
through token quotas fast, especially since many flagged clusters are *identical*
in shape (dozens of 4-node peel chains). This cache keys a verdict by the
cluster's structural fingerprint, so:

  - within a run, structurally identical clusters share ONE call, and
  - across runs, a warm cache serves repeat detections with ZERO new calls
    (the Elliptic graph is static, so the same shapes recur).

Keying on the exact structural tuple keeps it honest: two clusters with the same
fingerprint cite the same numbers, so a reused narration is still accurate.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.reasoning.schemas import ClusterFeatures, TypologyVerdict

CACHE_PATH = Path("data/graphs/reasoning_cache.json")


def cluster_fingerprint(f: ClusterFeatures) -> str:
    """Stable key over the structural facts that drive typology + narration."""
    parts = (
        f.num_nodes, f.num_edges, f.max_in_degree, f.max_out_degree,
        f.longest_chain, round(f.density, 2), f.num_sources, f.num_sinks,
        round(f.reciprocity, 2),
    )
    return "|".join(str(p) for p in parts)


def _load() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


class VerdictCache:
    """Load once, look up / store by fingerprint, flush to disk on demand."""

    def __init__(self) -> None:
        self._cache = _load()
        self.hits = 0
        self.misses = 0

    def get(self, features: ClusterFeatures) -> TypologyVerdict | None:
        entry = self._cache.get(cluster_fingerprint(features))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return TypologyVerdict.model_validate(entry["verdict"])

    def put(self, features: ClusterFeatures, verdict: TypologyVerdict, source: str) -> None:
        self._cache[cluster_fingerprint(features)] = {
            "verdict": verdict.model_dump(),
            "source": source,
        }

    def flush(self) -> None:
        _save(self._cache)


def clear_cache() -> None:
    """Delete the on-disk cache (use to force fresh LLM verdicts)."""
    CACHE_PATH.unlink(missing_ok=True)
