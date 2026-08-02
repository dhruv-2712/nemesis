"""Dashboard stats — dataset totals + model metrics + live detection summary.

Static facts (dataset size, temporal-test metrics) come from the committed
seed_stats.json so the dashboard works without the graph; the typology breakdown
and risk histogram are computed live from the current detection run.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from backend.api.detection import load_detection

SEED_STATS_PATH = Path("backend/api/seed_stats.json")
RISK_BINS = 10


def _histogram(values: list[float], bins: int = RISK_BINS) -> list[int]:
    counts = [0] * bins
    for v in values:
        counts[min(int(v * bins), bins - 1)] += 1
    return counts


def get_stats() -> dict:
    stats = json.loads(SEED_STATS_PATH.read_text(encoding="utf-8"))
    det = load_detection()
    clusters = det["clusters"]
    stats["detection"] = {
        "num_flagged": det["num_clusters"],
        "typology_breakdown": dict(Counter(c["typology"] for c in clusters)),
        "risk_histogram": _histogram([c["avg_risk"] for c in clusters]),
    }
    return stats
