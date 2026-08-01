import { useEffect, useState, useCallback } from "react";
import NetworkGraph from "./components/NetworkGraph.jsx";
import ClusterInspector from "./components/ClusterInspector.jsx";
import RiskLegend from "./components/RiskLegend.jsx";
import {
  listClusters,
  getCluster,
  runDetection,
  typologyLabel,
  riskColor,
} from "./api/client.js";

export default function App() {
  const [clusters, setClusters] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const data = await listClusters();
      setClusters(data.clusters);
      if (data.clusters.length && !selectedId) setSelectedId(data.clusters[0].cluster_id);
    } catch (e) {
      setError("Backend unreachable. Start it with: uvicorn backend.main:app");
    }
  }, [selectedId]);

  useEffect(() => {
    refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedId) return;
    setDetailLoading(true);
    getCluster(selectedId)
      .then(setDetail)
      .catch(() => setError("Failed to load cluster detail."))
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  const onDetect = async () => {
    setStatus("running");
    setError(null);
    try {
      await runDetection();
      setSelectedId(null);
      await refresh();
    } catch (e) {
      setError("Detection run failed.");
    } finally {
      setStatus("idle");
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◈</span>
          <div>
            <h1>NEMESIS</h1>
            <p>Fraud Network Intelligence — GNN structure + LLM typology reasoning</p>
          </div>
        </div>
        <div className="topbar-right">
          <RiskLegend />
          <button className="detect-btn" onClick={onDetect} disabled={status === "running"}>
            {status === "running" ? "Running…" : "Run detection"}
          </button>
        </div>
      </header>

      {error && <div className="banner error">{error}</div>}

      <main className="layout">
        <nav className="cluster-list">
          <div className="list-header">
            Flagged clusters <span className="count">{clusters.length}</span>
          </div>
          <ul>
            {clusters.map((c) => (
              <li
                key={c.cluster_id}
                className={c.cluster_id === selectedId ? "selected" : ""}
                onClick={() => setSelectedId(c.cluster_id)}
              >
                <div className="row1">
                  <span className="tid">{c.cluster_id.replace("cluster_", "#")}</span>
                  <span className="typ">{typologyLabel(c.typology)}</span>
                </div>
                <div className="row2">
                  <span className="risk-dot" style={{ background: riskColor(c.avg_risk) }} />
                  <span className="meta">
                    {c.num_nodes} tx · risk {c.avg_risk.toFixed(2)} · {Math.round(c.confidence * 100)}%
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </nav>

        <section className="graph-col">
          <NetworkGraph cluster={detail} />
        </section>

        <ClusterInspector cluster={detail} loading={detailLoading} />
      </main>
    </div>
  );
}
