// Thin fetch wrappers for the NEMESIS backend.
// Default is same-origin "" — local dev proxies /api via Vite, production proxies
// /api via vercel.json to the Render backend. Override with VITE_API_BASE if you
// want to hit an absolute backend URL directly (CORS must then allow the origin).
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function get(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
  return res.json();
}

export const listClusters = () => get("/api/clusters");
export const getCluster = (id) => get(`/api/clusters/${id}`);
export const runDetection = (threshold = 0.7) => post("/api/detect", { threshold });
export const getStats = () => get("/api/stats");
export const getEmbedding = () => get("/api/embedding");
export const analyzeSandbox = (nodes, edges) => post("/api/sandbox", { nodes, edges });

// Risk -> color on a green (safe) -> amber -> red (illicit) scale.
export function riskColor(risk) {
  const r = Math.max(0, Math.min(1, risk));
  if (r < 0.5) {
    // green -> amber
    const t = r / 0.5;
    return `rgb(${Math.round(34 + t * 211)}, ${Math.round(197 - t * 39)}, ${Math.round(94 - t * 83)})`;
  }
  // amber -> red
  const t = (r - 0.5) / 0.5;
  return `rgb(${Math.round(245 - t * 6)}, ${Math.round(158 - t * 90)}, ${Math.round(11 + t * 57)})`;
}

// Human-friendly typology label.
export function typologyLabel(t) {
  return (t || "unknown").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Stable color per laundering typology (used across charts, badges, legends).
export const TYPOLOGY_COLORS = {
  peeling_chain: "#f59e0b",
  consolidation_cashout: "#ef4444",
  scam_payout_fanout: "#a855f7",
  mixing_tumbling: "#ec4899",
  layering: "#3b82f6",
  unknown_suspicious: "#64748b",
};

export const typologyColor = (t) => TYPOLOGY_COLORS[t] || "#64748b";
