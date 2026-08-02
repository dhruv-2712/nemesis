import { useMemo, useRef, useState, useEffect } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { analyzeSandbox, riskColor, typologyLabel, typologyColor } from "../api/client.js";

let _uid = 0;
const nid = () => `n${_uid++}`;

// Topology presets — profile is applied per node when analyzed.
function preset(kind, size) {
  const n = Math.max(3, Math.min(15, size));
  const nodes = [];
  const edges = [];
  const ids = Array.from({ length: n }, () => nid());
  ids.forEach((id) => nodes.push({ id }));
  if (kind === "chain") {
    for (let i = 0; i < n - 1; i++) edges.push({ source: ids[i], target: ids[i + 1] });
  } else if (kind === "consolidation") {
    for (let i = 0; i < n - 1; i++) edges.push({ source: ids[i], target: ids[n - 1] });
  } else if (kind === "fanout") {
    for (let i = 1; i < n; i++) edges.push({ source: ids[0], target: ids[i] });
  } else if (kind === "mixer") {
    const mid1 = ids[0], mid2 = ids[1];
    for (let i = 2; i < n; i++) {
      edges.push({ source: ids[i], target: mid1 });
      edges.push({ source: mid2, target: ids[i] });
    }
    edges.push({ source: mid1, target: mid2 });
    edges.push({ source: mid2, target: mid1 });
  }
  return { nodes, edges };
}

export default function Sandbox() {
  const [graph, setGraph] = useState(() => preset("chain", 6));
  const [profile, setProfile] = useState("illicit");
  const [size, setSize] = useState(6);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [pending, setPending] = useState(null); // source id for click-to-connect
  const fgRef = useRef(null);

  const loadPreset = (kind) => {
    setResult(null);
    setPending(null);
    setGraph(preset(kind, size));
  };

  const addNode = () => {
    setResult(null);
    setGraph((g) => ({ ...g, nodes: [...g.nodes, { id: nid() }] }));
  };

  const clearAll = () => {
    setResult(null);
    setPending(null);
    setGraph({ nodes: [], edges: [] });
  };

  const onNodeClick = (node) => {
    if (pending === null) {
      setPending(node.id);
    } else if (pending !== node.id) {
      setGraph((g) => ({
        ...g,
        edges: [...g.edges, { source: pending, target: node.id }],
      }));
      setResult(null);
      setPending(null);
    } else {
      setPending(null);
    }
  };

  const analyze = async () => {
    if (!graph.nodes.length) return;
    setAnalyzing(true);
    try {
      const nodes = graph.nodes.map((n) => ({ id: n.id, profile }));
      const edges = graph.edges.map((e) => ({
        source: typeof e.source === "object" ? e.source.id : e.source,
        target: typeof e.target === "object" ? e.target.id : e.target,
      }));
      setResult(await analyzeSandbox(nodes, edges));
    } catch {
      setResult({ error: "Analysis failed (need ≥1 node; backend must be running)." });
    } finally {
      setAnalyzing(false);
    }
  };

  const riskById = useMemo(() => {
    const m = {};
    if (result?.nodes) result.nodes.forEach((n) => (m[n.id] = n.risk));
    return m;
  }, [result]);

  const graphData = useMemo(
    () => ({
      nodes: graph.nodes.map((n) => ({ ...n })),
      links: graph.edges.map((e) => ({
        source: typeof e.source === "object" ? e.source.id : e.source,
        target: typeof e.target === "object" ? e.target.id : e.target,
      })),
    }),
    [graph]
  );

  useEffect(() => {
    if (fgRef.current) fgRef.current.d3ReheatSimulation();
  }, [graphData]);

  const v = result && !result.error ? result.verdict : null;

  return (
    <div className="view sandbox-view">
      <aside className="sandbox-controls">
        <h2>Sandbox</h2>
        <p className="muted small">
          Build a transaction pattern; NEMESIS seeds each node with features from
          the <b>{profile}</b> profile, scores it with the GNN, and classifies the
          typology. Click one node then another to connect them.
        </p>

        <div className="control-group">
          <label>Presets</label>
          <div className="preset-btns">
            <button onClick={() => loadPreset("chain")}>Peel chain</button>
            <button onClick={() => loadPreset("consolidation")}>Consolidation</button>
            <button onClick={() => loadPreset("fanout")}>Scam fan-out</button>
            <button onClick={() => loadPreset("mixer")}>Mixer</button>
          </div>
        </div>

        <div className="control-group">
          <label>Preset size: {size}</label>
          <input type="range" min="3" max="15" value={size}
                 onChange={(e) => setSize(Number(e.target.value))} />
        </div>

        <div className="control-group">
          <label>Node profile</label>
          <div className="toggle">
            {["illicit", "licit"].map((p) => (
              <button key={p} className={profile === p ? "on" : ""}
                      onClick={() => { setProfile(p); setResult(null); }}>
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group build-actions">
          <button onClick={addNode}>+ Node</button>
          <button onClick={clearAll}>Clear</button>
        </div>

        <button className="analyze-btn" onClick={analyze} disabled={analyzing || !graph.nodes.length}>
          {analyzing ? "Analyzing…" : "Analyze →"}
        </button>

        <div className="build-stat">
          {graph.nodes.length} nodes · {graph.edges.length} edges
          {pending && <span className="pending"> · connecting from {pending}…</span>}
        </div>
      </aside>

      <section className="graph-col">
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          backgroundColor="rgba(0,0,0,0)"
          nodeRelSize={6}
          nodeColor={(n) => (n.id in riskById ? riskColor(riskById[n.id]) : (n.id === pending ? "#38bdf8" : "#64748b"))}
          nodeLabel={(n) => (n.id in riskById ? `risk ${riskById[n.id].toFixed(2)}` : "click to connect")}
          linkColor={() => "rgba(148,163,184,0.4)"}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkDirectionalParticles={result ? 2 : 0}
          linkDirectionalParticleColor={() => "rgba(239,68,68,0.7)"}
          onNodeClick={onNodeClick}
          cooldownTicks={80}
        />
      </section>

      <aside className="inspector sandbox-verdict">
        {result?.error && <div className="banner error">{result.error}</div>}
        {!result && <p className="muted">Build a pattern and hit Analyze to see the live verdict.</p>}
        {v && (
          <>
            <div className="verdict-head">
              <span className="typology-badge" style={{ borderColor: typologyColor(result.verdict.typology), color: typologyColor(result.verdict.typology) }}>
                {typologyLabel(v.typology)}
              </span>
              <div className="confidence">
                <div className="confidence-bar">
                  <div className="confidence-fill" style={{ width: `${Math.round(v.confidence * 100)}%` }} />
                </div>
                <span>{Math.round(v.confidence * 100)}% confidence</span>
              </div>
            </div>
            <div className="sandbox-risk">
              <span>GNN risk</span>
              <b style={{ color: riskColor(result.avg_risk) }}>{result.avg_risk.toFixed(2)}</b>
              <span className="risk-src">avg · features {result.feature_source}</span>
            </div>
            <p className="summary">{v.summary}</p>
            <section>
              <h4>Reasoning chain</h4>
              <ol className="reasoning">
                {v.reasoning_chain.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
            </section>
            <section>
              <h4>Recommended action</h4>
              <p className="action">{v.recommended_action}</p>
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
