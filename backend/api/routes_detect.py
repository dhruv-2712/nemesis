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
    # Discard the reasoning cache and regenerate every verdict (costs more LLM
    # calls). Default False: reuse cached verdicts for identical structures.
    refresh: bool = False


@router.post("")
def detect(req: DetectRequest | None = None) -> dict:
    req = req or DetectRequest()
    result = run_detection(threshold=req.threshold, refresh=req.refresh)
    used_llm = bool(os.getenv("GROQ_API_KEY"))
    persist_detection(result["clusters"], used_llm=used_llm)
    return {
        "generated_at": result["generated_at"],
        "threshold": req.threshold,
        "num_clusters": result["num_clusters"],
        "reasoning_calls": result["reasoning_calls"],
        "cache_hits": result["cache_hits"],
        "reasoning_source": "llm" if used_llm else "heuristic",
        "typology_breakdown": dict(
            Counter(c["typology"] for c in result["clusters"])
        ),
    }
