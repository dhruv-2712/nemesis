import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getEmbedding } from "../api/client.js";

const ILLICIT = "#ef4444";
const LICIT = "#3b82f6";

export default function EmbeddingMap() {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [points, setPoints] = useState(null);
  const [error, setError] = useState(null);
  const [hover, setHover] = useState(null);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const navigate = useNavigate();
  const transform = useRef({ scale: 1, ox: 0, oy: 0 });

  useEffect(() => {
    getEmbedding().then((d) => setPoints(d.points)).catch(() => setError("Embedding unavailable."));
  }, []);

  // Map data coords -> canvas pixels, fit to the container with padding.
  const computeTransform = useCallback((w, h) => {
    if (!points?.length) return;
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 30;
    const scale = Math.min((w - 2 * pad) / (maxX - minX), (h - 2 * pad) / (maxY - minY));
    transform.current = {
      scale,
      ox: pad - minX * scale + (w - 2 * pad - (maxX - minX) * scale) / 2,
      oy: pad - minY * scale + (h - 2 * pad - (maxY - minY) * scale) / 2,
    };
  }, [points]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !points) return;
    const ctx = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    ctx.clearRect(0, 0, w, h);
    const { scale, ox, oy } = transform.current;
    for (const p of points) {
      const flagged = !!p.cluster_id;
      if (onlyFlagged && !flagged) continue;
      const px = p.x * scale + ox, py = p.y * scale + oy;
      ctx.beginPath();
      ctx.arc(px, py, flagged ? 3 : 2, 0, 2 * Math.PI);
      ctx.fillStyle = p.label === 1 ? ILLICIT : LICIT;
      ctx.globalAlpha = flagged ? 1 : 0.45;
      ctx.fill();
      if (flagged) {
        ctx.globalAlpha = 0.6;
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }, [points, onlyFlagged]);

  useEffect(() => {
    const wrap = wrapRef.current, canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const resize = () => {
      canvas.width = wrap.clientWidth;
      canvas.height = wrap.clientHeight;
      computeTransform(canvas.width, canvas.height);
      draw();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [computeTransform, draw]);

  useEffect(() => { draw(); }, [draw]);

  const nearest = (clientX, clientY) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mx = clientX - rect.left, my = clientY - rect.top;
    const { scale, ox, oy } = transform.current;
    let best = null, bestD = 64; // px^2 threshold
    for (const p of points) {
      if (onlyFlagged && !p.cluster_id) continue;
      const px = p.x * scale + ox, py = p.y * scale + oy;
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bestD) { bestD = d; best = { p, px, py }; }
    }
    return best;
  };

  const onMove = (e) => { if (points) setHover(nearest(e.clientX, e.clientY)); };
  const onClick = (e) => {
    const hit = points && nearest(e.clientX, e.clientY);
    if (hit?.p.cluster_id) navigate(`/explorer/${hit.p.cluster_id}`);
  };

  if (error) return <div className="view"><div className="banner error">{error}</div></div>;

  return (
    <div className="view embedding-view">
      <div className="view-head embed-head">
        <div>
          <h2>Embedding map</h2>
          <p className="muted">
            Learned GNN embeddings projected to 2D (t-SNE). Illicit transactions
            cluster apart from licit — evidence the model learned structure. Click a
            gold-ringed point (in a flagged cluster) to open it in the Explorer.
          </p>
        </div>
        <div className="embed-legend">
          <span><i className="dot" style={{ background: ILLICIT }} /> illicit</span>
          <span><i className="dot" style={{ background: LICIT }} /> licit</span>
          <span><i className="dot ring" /> flagged</span>
          <label className="flag-toggle">
            <input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} />
            flagged only
          </label>
        </div>
      </div>

      <div className="embed-canvas-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          onClick={onClick}
          style={{ cursor: hover?.p.cluster_id ? "pointer" : "default" }}
        />
        {hover && (
          <div className="embed-tooltip" style={{ left: hover.px + 12, top: hover.py + 12 }}>
            {hover.p.label === 1 ? "illicit" : "licit"} · risk {hover.p.risk.toFixed(2)}
            {hover.p.cluster_id && <> · {hover.p.cluster_id.replace("cluster_", "#")}</>}
          </div>
        )}
      </div>
    </div>
  );
}
