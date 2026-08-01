import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { riskColor } from "../api/client.js";

// Force-directed view of one flagged cluster. Nodes are transactions colored by
// GNN risk; directed links show the flow of value (arrows + moving particles).
export default function NetworkGraph({ cluster }) {
  const wrapRef = useRef(null);
  const fgRef = useRef(null);
  const [dims, setDims] = useState({ width: 800, height: 600 });

  // Track container size so the canvas fills its column responsively.
  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const update = () =>
      setDims({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fresh copies each cluster — react-force-graph mutates node objects in place.
  const graphData = useMemo(() => {
    if (!cluster) return { nodes: [], links: [] };
    return {
      nodes: cluster.nodes.map((n) => ({ ...n })),
      links: cluster.edges.map((e) => ({ ...e })),
    };
  }, [cluster]);

  useEffect(() => {
    if (fgRef.current) fgRef.current.d3ReheatSimulation();
  }, [graphData]);

  if (!cluster) {
    return (
      <div className="graph-empty" ref={wrapRef}>
        <p>Select a flagged cluster to inspect its transaction structure.</p>
      </div>
    );
  }

  return (
    <div className="graph-wrap" ref={wrapRef}>
      <ForceGraph2D
        ref={fgRef}
        width={dims.width}
        height={dims.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)"
        nodeRelSize={5}
        nodeVal={(n) => 1 + n.risk * 3}
        nodeColor={(n) => riskColor(n.risk)}
        nodeLabel={(n) => `tx ${n.id} — risk ${n.risk.toFixed(2)} (t=${n.time_step})`}
        linkColor={() => "rgba(148, 163, 184, 0.35)"}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={1.6}
        linkDirectionalParticleColor={() => "rgba(239, 68, 68, 0.7)"}
        cooldownTicks={120}
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node, ctx) => {
          // Halo on high-risk nodes so illicit hotspots pop.
          if (node.risk >= 0.8) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 6 + node.risk * 3, 0, 2 * Math.PI);
            ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }}
      />
    </div>
  );
}
