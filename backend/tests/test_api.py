"""API tests — detect -> persist -> list -> detail, plus ingest validation.

Uses a temp SQLite file (stdlib sqlite gives each connection its own :memory:
db, so a file is required for the connection-per-call layer). Skips if the graph
artifacts are absent, since /detect runs the real pipeline.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

ARTIFACTS = Path("data/graphs/elliptic.pt"), Path("backend/models/checkpoints/graphsage.pt")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMESIS_DB_PATH", str(tmp_path / "test.db"))
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_validates(client):
    ok = client.post(
        "/api/ingest",
        json={"transactions": [{"tx_id": "t1", "source": "a", "target": "b", "amount": 5.0, "time_step": 3}]},
    )
    assert ok.status_code == 202 and ok.json()["accepted"] == 1
    assert client.post("/api/ingest", json={"transactions": [{"tx_id": "t1"}]}).status_code == 422


@pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="graph artifacts absent")
def test_detect_then_serve(client):
    from backend.db.models import count_clusters

    det = client.post("/api/detect", json={"threshold": 0.7})
    assert det.status_code == 200
    assert det.json()["num_clusters"] > 0
    assert count_clusters() == det.json()["num_clusters"]

    listing = client.get("/api/clusters").json()
    assert listing["num_clusters"] == det.json()["num_clusters"]

    cid = listing["clusters"][0]["cluster_id"]
    detail = client.get(f"/api/clusters/{cid}").json()
    assert detail["nodes"] and detail["edges"] and detail["reasoning_chain"]
    assert client.get("/api/clusters/does-not-exist").status_code == 404
