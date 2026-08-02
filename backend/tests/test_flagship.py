"""Tests for the flagship endpoints — sandbox, stats, embedding."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

GRAPH = Path("data/graphs/elliptic.pt")
CKPT = Path("backend/models/checkpoints/graphsage.pt")
BANK = Path("backend/models/feature_bank.npz")
needs_model = pytest.mark.skipif(
    not (CKPT.exists() and BANK.exists()), reason="checkpoint/feature bank absent"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMESIS_DB_PATH", str(tmp_path / "t.db"))
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


def test_stats_shape(client):
    s = client.get("/api/stats").json()
    assert s["dataset"]["nodes"] > 0
    assert 0 <= s["model"]["roc_auc"] <= 1
    assert "typology_breakdown" in s["detection"]
    assert len(s["detection"]["risk_histogram"]) == 10


def test_embedding_served(client):
    if not Path("backend/api/embedding_2d.json").exists():
        pytest.skip("embedding artifact absent")
    e = client.get("/api/embedding").json()
    assert e["points"] and {"x", "y", "label", "risk"} <= set(e["points"][0])


@needs_model
def test_sandbox_peel_chain_scores_and_classifies(client):
    nodes = [{"id": f"n{i}", "profile": "illicit"} for i in range(8)]
    edges = [{"source": f"n{i}", "target": f"n{i+1}"} for i in range(7)]
    r = client.post("/api/sandbox", json={"nodes": nodes, "edges": edges})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"]["typology"] == "peeling_chain"
    assert body["avg_risk"] > 0.5
    assert len(body["nodes"]) == 8 and all("risk" in n for n in body["nodes"])


@needs_model
def test_sandbox_consolidation(client):
    nodes = [{"id": f"s{i}", "profile": "illicit"} for i in range(8)] + [{"id": "hub"}]
    edges = [{"source": f"s{i}", "target": "hub"} for i in range(8)]
    body = client.post("/api/sandbox", json={"nodes": nodes, "edges": edges}).json()
    assert body["verdict"]["typology"] == "consolidation_cashout"
    assert body["features"]["max_in_degree"] == 8


def test_sandbox_rejects_empty(client):
    assert client.post("/api/sandbox", json={"nodes": [], "edges": []}).status_code == 422


@needs_model
def test_feature_bank_sampling():
    from backend.models.feature_bank import sample_features

    rows = sample_features("illicit", 5)
    assert rows.shape == (5, 166)
