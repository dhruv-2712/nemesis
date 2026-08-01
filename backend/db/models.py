"""SQLite schema + persistence helpers for detection results.

Two tables:
  flagged_clusters — one row per flagged cluster (latest verdict summary)
  reasoning_logs   — the reasoning chain + provenance (LLM vs heuristic) per run

The API persists a detection run here so verdicts survive restarts and can be
audited (who/what flagged this cluster, and the reasoning that justified it).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.db.session import get_connection

SCHEMA = """
CREATE TABLE IF NOT EXISTS flagged_clusters (
    cluster_id          TEXT PRIMARY KEY,
    typology            TEXT NOT NULL,
    confidence          REAL NOT NULL,
    num_nodes           INTEGER NOT NULL,
    num_edges           INTEGER NOT NULL,
    avg_risk            REAL NOT NULL,
    max_risk            REAL NOT NULL,
    summary             TEXT NOT NULL,
    recommended_action  TEXT NOT NULL,
    detected_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reasoning_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id      TEXT NOT NULL REFERENCES flagged_clusters(cluster_id) ON DELETE CASCADE,
    reasoning_chain TEXT NOT NULL,
    used_llm        INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def persist_detection(clusters: list[dict], used_llm: bool = False) -> None:
    """Replace stored clusters with the latest run and append reasoning logs."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("DELETE FROM flagged_clusters")  # cascades to reasoning_logs
        for c in clusters:
            conn.execute(
                """INSERT INTO flagged_clusters
                   (cluster_id, typology, confidence, num_nodes, num_edges,
                    avg_risk, max_risk, summary, recommended_action, detected_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    c["cluster_id"], c["typology"], c["confidence"], c["num_nodes"],
                    c["num_edges"], c["avg_risk"], c["max_risk"], c["summary"],
                    c["recommended_action"], now,
                ),
            )
            conn.execute(
                """INSERT INTO reasoning_logs (cluster_id, reasoning_chain, used_llm, created_at)
                   VALUES (?,?,?,?)""",
                (c["cluster_id"], json.dumps(c["reasoning_chain"]), int(used_llm), now),
            )
        conn.commit()


def count_clusters() -> int:
    init_db()
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM flagged_clusters").fetchone()[0]
