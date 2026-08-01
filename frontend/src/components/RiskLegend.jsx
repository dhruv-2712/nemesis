import { riskColor } from "../api/client.js";

// Continuous risk color scale legend (green -> amber -> red).
export default function RiskLegend() {
  const stops = [0, 0.25, 0.5, 0.75, 1];
  const gradient = `linear-gradient(90deg, ${stops
    .map((s) => `${riskColor(s)} ${s * 100}%`)
    .join(", ")})`;
  return (
    <div className="legend">
      <span className="legend-label">GNN risk</span>
      <div className="legend-bar" style={{ background: gradient }} />
      <div className="legend-ends">
        <span>licit 0.0</span>
        <span>1.0 illicit</span>
      </div>
    </div>
  );
}
