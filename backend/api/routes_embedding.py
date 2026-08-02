"""GET /api/embedding — subsampled t-SNE points for the embedding map."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/embedding", tags=["embedding"])

EMBEDDING_PATH = Path("backend/api/embedding_2d.json")


@router.get("")
def embedding() -> dict:
    if not EMBEDDING_PATH.exists():
        raise HTTPException(status_code=404, detail="embedding artifact not built")
    return json.loads(EMBEDDING_PATH.read_text(encoding="utf-8"))
