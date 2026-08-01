// Thin fetch wrappers for the NEMESIS backend.
// Base URL is configurable so the same build works locally and when deployed.
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

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
