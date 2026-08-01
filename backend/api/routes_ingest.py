"""POST /api/ingest — accept new transaction rows.

Scope note: NEMESIS trains and detects on a prebuilt Elliptic graph snapshot, so
this endpoint validates and acknowledges submitted transactions but does not yet
splice them into the live graph. Streaming ingestion (incremental graph updates +
re-scoring) is on the roadmap; this endpoint fixes the contract the frontend and
future pipeline will use, and rejects malformed payloads up front.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


class Transaction(BaseModel):
    tx_id: str
    source: str = Field(..., description="sending tx/address id")
    target: str = Field(..., description="receiving tx/address id")
    amount: float = Field(..., ge=0)
    time_step: int = Field(..., ge=1)


class IngestRequest(BaseModel):
    transactions: list[Transaction]


@router.post("", status_code=202)
def ingest(req: IngestRequest) -> dict:
    return {
        "accepted": len(req.transactions),
        "status": "queued",
        "note": "Validated and acknowledged. Live graph splicing is on the roadmap; "
        "current detection runs on the prebuilt Elliptic snapshot.",
    }
