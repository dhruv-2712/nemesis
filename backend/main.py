"""NEMESIS FastAPI application entrypoint.

Mounts the ingest / detect / clusters routers and locks CORS to the frontend
origin (configurable via FRONTEND_ORIGIN — default the Vite dev server). The DB
schema is initialized on startup so the first /detect can persist immediately.

Run:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes_clusters, routes_detect, routes_ingest
from backend.db.models import init_db

app = FastAPI(
    title="NEMESIS — Fraud Network Intelligence",
    description="GNN-flagged transaction clusters with LLM typology reasoning.",
    version="0.4.0",
)

# CORS: allow only the known frontend origin(s); comma-separate for multiple.
_origins = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(routes_ingest.router)
app.include_router(routes_detect.router)
app.include_router(routes_clusters.router)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "nemesis"}
