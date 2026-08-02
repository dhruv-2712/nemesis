import { riskColor } from "../api/client.js";

// Bar chart of flagged-cluster average-risk distribution (10 bins, 0..1).
export default function RiskHistogram({ bins }) {
  const data = bins || [];
  const max = Math.max(1, ...data);
  return (
    <div className="histogram">
      <div className="hist-bars">
        {data.map((count, i) => {
          const mid = (i + 0.5) / data.length;
          return (
            <div key={i} className="hist-col">
              <div className="hist-count">{count || ""}</div>
              <div
                className="hist-bar"
                style={{
                  height: `${(count / max) * 100}%`,
                  background: riskColor(mid),
                }}
                title={`risk ${(i / data.length).toFixed(1)}–${((i + 1) / data.length).toFixed(1)}: ${count}`}
              />
            </div>
          );
        })}
      </div>
      <div className="hist-axis">
        <span>0.0</span>
        <span>avg cluster risk</span>
        <span>1.0</span>
      </div>
    </div>
  );
}
