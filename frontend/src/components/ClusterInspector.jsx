import { typologyLabel, riskColor } from "../api/client.js";

// Right-hand panel: the LLM/heuristic verdict for the selected cluster —
// typology, confidence, the ordered reasoning chain, and recommended action.
export default function ClusterInspector({ cluster, loading }) {
  if (loading) return <aside className="inspector"><p className="muted">Loading…</p></aside>;
  if (!cluster) {
    return (
      <aside className="inspector">
        <p className="muted">No cluster selected.</p>
      </aside>
    );
  }

  const f = cluster.features || {};
  const conf = Math.round((cluster.confidence ?? 0) * 100);

  return (
    <aside className="inspector">
      <div className="verdict-head">
        <span className="typology-badge">{typologyLabel(cluster.typology)}</span>
        <div className="confidence">
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${conf}%` }} />
          </div>
          <span>{conf}% confidence</span>
        </div>
      </div>

      <p className="summary">{cluster.summary}</p>

      <section>
        <h4>Reasoning chain</h4>
        <ol className="reasoning">
          {(cluster.reasoning_chain || []).map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      </section>

      <section>
        <h4>Recommended action</h4>
        <p className="action">{cluster.recommended_action}</p>
      </section>

      <section>
        <h4>Structure</h4>
        <dl className="features">
          <div><dt>Transactions</dt><dd>{cluster.num_nodes}</dd></div>
          <div><dt>Flows (edges)</dt><dd>{cluster.num_edges}</dd></div>
          <div>
            <dt>Avg risk</dt>
            <dd style={{ color: riskColor(cluster.avg_risk) }}>{cluster.avg_risk?.toFixed(2)}</dd>
          </div>
          <div><dt>Longest chain</dt><dd>{f.longest_chain}</dd></div>
          <div><dt>Max fan-in</dt><dd>{f.max_in_degree}</dd></div>
          <div><dt>Max fan-out</dt><dd>{f.max_out_degree}</dd></div>
          <div><dt>Density</dt><dd>{f.density?.toFixed(3)}</dd></div>
          <div><dt>Time span</dt><dd>{cluster.time_span_steps} step(s)</dd></div>
        </dl>
      </section>
    </aside>
  );
}
