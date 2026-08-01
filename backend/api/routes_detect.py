"""POST /api/detect — (re)run the detection pipeline and persist results.

Rebuilds the cached detection artifact (GNN scoring -> cluster extraction ->
typology reasoning) and writes the verdicts to SQLite for audit. Returns a
compact run summary; the frontend then fetches details via /api/clusters.
"""

from __future__ import annotations

import os
from collections import Counter

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.detection import run_detection
from backend.db.models import persist_detection
from backend.reasoning.cluster import DEFAULT_THRESHOLD

router = APIRouter(prefix="/api/detect", tags=["detect"])


class DetectRequest(BaseModel):
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


@router.post("")
def detect(req: DetectRequest | None = None) -> dict:
    threshold = (req or DetectRequest()).threshold
    result = run_detection(threshold=threshold)
    used_llm = bool(os.getenv("GROQ_API_KEY"))
    persist_detection(result["clusters"], used_llm=used_llm)
    return {
        "generated_at": result["generated_at"],
        "threshold": threshold,
        "num_clusters": result["num_clusters"],
        "reasoning_source": "llm" if used_llm else "heuristic",
        "typology_breakdown": dict(
            Counter(c["typology"] for c in result["clusters"])
        ),
    }
