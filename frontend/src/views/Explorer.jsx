import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import NetworkGraph from "../components/NetworkGraph.jsx";
import ClusterInspector from "../components/ClusterInspector.jsx";
import RiskLegend from "../components/RiskLegend.jsx";
import {
  listClusters,
  getCluster,
  typologyLabel,
  riskColor,
  typologyColor,
} from "../api/client.js";

const SORTS = {
  risk: (a, b) => b.avg_risk - a.avg_risk,
  size: (a, b) => b.num_nodes - a.num_nodes,
  confidence: (a, b) => b.confidence - a.confidence,
};

export default function Explorer() {
  const { clusterId } = useParams();
  const navigate = useNavigate();
  const [clusters, setClusters] = useState([]);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);

  const [query, setQuery] = useState("");
  const [typology, setTypology] = useState("all");
  const [sort, setSort] = useState("risk");

  useEffect(() => {
    listClusters()
      .then((data) => {
        setClusters(data.clusters);
        if (!clusterId && data.clusters.length) {
          navigate(`/explorer/${data.clusters[0].cluster_id}`, { replace: true });
        }
      })
      .catch(() => setError("Backend unreachable. Start it with: uvicorn backend.main:app"));
  }, []); // eslint-disable-line

  useEffect(() => {
    if (!clusterId) return;
    setDetailLoading(true);
    getCluster(clusterId)
      .then(setDetail)
      .catch(() => setError("Failed to load cluster."))
      .finally(() => setDetailLoading(false));
  }, [clusterId]);

  const typologies = useMemo(
    () => ["all", ...new Set(clusters.map((c) => c.typology))],
    [clusters]
  );

  const shown = useMemo(() => {
    return clusters
      .filter((c) => typology === "all" || c.typology === typology)
      .filter((c) => !query || c.cluster_id.toLowerCase().includes(query.toLowerCase()))
      .sort(SORTS[sort]);
  }, [clusters, query, typology, sort]);

  return (
    <div className="view explorer-view">
      {error && <div className="banner error">{error}</div>}
      <div className="explorer-layout">
        <nav className="cluster-list">
          <div className="list-controls">
            <input
              className="search"
              placeholder="Search cluster id…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="filters">
              <select value={typology} onChange={(e) => setTypology(e.target.value)}>
                {typologies.map((t) => (
                  <option key={t} value={t}>{t === "all" ? "All typologies" : typologyLabel(t)}</option>
                ))}
              </select>
              <select value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="risk">Sort: risk</option>
                <option value="size">Sort: size</option>
                <option value="confidence">Sort: confidence</option>
              </select>
            </div>
            <div className="list-count">{shown.length} clusters</div>
          </div>
          <ul>
            {shown.map((c) => (
              <li
                key={c.cluster_id}
                className={c.cluster_id === clusterId ? "selected" : ""}
                onClick={() => navigate(`/explorer/${c.cluster_id}`)}
              >
                <div className="row1">
                  <span className="tid">{c.cluster_id.replace("cluster_", "#")}</span>
                  <span className="typ" style={{ color: typologyColor(c.typology) }}>
                    {typologyLabel(c.typology)}
                  </span>
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
          <div className="graph-legend-float"><RiskLegend /></div>
          <NetworkGraph cluster={detail} />
        </section>

        <ClusterInspector cluster={detail} loading={detailLoading} />
      </div>
    </div>
  );
}
