import { useEffect, useState } from "react";
import { getStats } from "../api/client.js";
import StatTile from "../components/StatTile.jsx";
import TypologyDonut from "../components/TypologyDonut.jsx";
import RiskHistogram from "../components/RiskHistogram.jsx";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => setError("Backend unreachable."));
  }, []);

  if (error) return <div className="view"><div className="banner error">{error}</div></div>;
  if (!stats) return <div className="view"><p className="muted">Loading…</p></div>;

  const d = stats.dataset;
  const m = stats.model;
  const det = stats.detection;

  return (
    <div className="view dashboard">
      <div className="view-head">
        <h2>Command center</h2>
        <p className="muted">
          Structural fraud detection on the Elliptic Bitcoin graph — GNN scoring
          fused with LLM typology reasoning.
        </p>
      </div>

      <div className="stat-row">
        <StatTile value={d.nodes.toLocaleString()} label="Transactions" sub="graph nodes" />
        <StatTile value={d.edges.toLocaleString()} label="Value flows" sub="directed edges" />
        <StatTile
          value={`${(d.illicit_rate * 100).toFixed(1)}%`}
          label="Illicit rate"
          sub={`${d.illicit.toLocaleString()} of ${d.labeled.toLocaleString()} labeled`}
          accent="#ef4444"
        />
        <StatTile
          value={det.num_flagged}
          label="Flagged clusters"
          sub="current detection run"
          accent="#38bdf8"
        />
      </div>

      <div className="dash-grid">
        <section className="panel">
          <h3>Typology distribution</h3>
          <TypologyDonut breakdown={det.typology_breakdown} />
        </section>

        <section className="panel">
          <h3>Flagged-cluster risk</h3>
          <RiskHistogram bins={det.risk_histogram} />
        </section>

        <section className="panel metrics">
          <h3>Model — GraphSAGE (temporal test split)</h3>
          <div className="metric-grid">
            <div><span className="mv">{m.roc_auc}</span><span className="ml">ROC-AUC</span></div>
            <div><span className="mv">{m.pr_auc}</span><span className="ml">PR-AUC</span></div>
            <div><span className="mv">{m.illicit_f1}</span><span className="ml">Illicit F1</span></div>
            <div><span className="mv">{m.illicit_precision}</span><span className="ml">Precision</span></div>
            <div><span className="mv">{m.illicit_recall}</span><span className="ml">Recall</span></div>
          </div>
          <p className="metric-note">
            Leakage-free temporal split (train early steps, test later). The
            illicit class is a {(d.illicit_rate * 100).toFixed(1)}% minority.
          </p>
        </section>
      </div>
    </div>
  );
}
