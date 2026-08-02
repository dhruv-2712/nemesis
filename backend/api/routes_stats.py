"""GET /api/stats — dataset totals, model metrics, live detection summary."""

from fastapi import APIRouter

from backend.api.stats import get_stats

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("")
def stats() -> dict:
    return get_stats()
