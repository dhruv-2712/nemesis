"""POST /api/sandbox — score a user-built transaction pattern live."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.sandbox import analyze

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


class SandboxNode(BaseModel):
    id: str
    profile: Literal["illicit", "licit"] = "illicit"


class SandboxEdge(BaseModel):
    source: str
    target: str


class SandboxRequest(BaseModel):
    nodes: list[SandboxNode] = Field(..., min_length=1)
    edges: list[SandboxEdge] = []


@router.post("")
def sandbox(req: SandboxRequest) -> dict:
    try:
        return analyze(
            [n.model_dump() for n in req.nodes],
            [e.model_dump() for e in req.edges],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
