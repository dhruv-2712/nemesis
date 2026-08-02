import { typologyColor, typologyLabel } from "../api/client.js";

// SVG donut of the typology breakdown (no chart-lib dependency).
export default function TypologyDonut({ breakdown }) {
  const entries = Object.entries(breakdown || {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, n]) => s + n, 0) || 1;

  const R = 70;
  const C = 2 * Math.PI * R;
  let offset = 0;
  const segments = entries.map(([typ, n]) => {
    const frac = n / total;
    const seg = {
      typ,
      n,
      color: typologyColor(typ),
      dash: frac * C,
      gap: C - frac * C,
      offset: -offset * C,
    };
    offset += frac;
    return seg;
  });

  return (
    <div className="donut-card">
      <svg viewBox="0 0 180 180" className="donut">
        <g transform="translate(90,90) rotate(-90)">
          {segments.map((s) => (
            <circle
              key={s.typ}
              r={R}
              fill="none"
              stroke={s.color}
              strokeWidth="24"
              strokeDasharray={`${s.dash} ${s.gap}`}
              strokeDashoffset={s.offset}
            />
          ))}
        </g>
        <text x="90" y="86" className="donut-total">{total}</text>
        <text x="90" y="104" className="donut-total-label">flagged</text>
      </svg>
      <ul className="donut-legend">
        {segments.map((s) => (
          <li key={s.typ}>
            <span className="swatch" style={{ background: s.color }} />
            <span className="dl-label">{typologyLabel(s.typ)}</span>
            <span className="dl-count">{s.n}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
