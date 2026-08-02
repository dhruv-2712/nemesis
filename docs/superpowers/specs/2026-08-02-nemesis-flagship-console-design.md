# NEMESIS Flagship Console — Design

**Date:** 2026-08-02
**Goal:** Turn NEMESIS from a single-screen demo into a flagship, multi-view fraud-intelligence console with an interactive input (sandbox), a command-center dashboard, cluster search/filter, an interactive embedding map, and a cohesive UI.

## Decisions

- **Routing:** `react-router-dom` — four views with their own URLs (shareable in a demo).
- **Deployability:** commit two small artifacts so the sandbox/embedding work without the 135 MB graph:
  - the trained checkpoint `graphsage.pt` (239 KB) — un-gitignored,
  - a **feature bank** `backend/models/feature_bank.npz` (~400 illicit + 400 licit real feature vectors) for seeding sandbox nodes.
  - `embedding_2d.json` and `seed_stats.json` committed for the map/dashboard.
- Torch stays a backend dependency (the sandbox scores live). The 135 MB graph stays gitignored.

## Views

```
Top nav:  ◈ NEMESIS   Dashboard · Explorer · Sandbox · Embedding Map
```

1. **Dashboard** — headline stat tiles (203,769 tx · 234,355 flows · 9.8% illicit · 50 flagged clusters), a typology-distribution donut, a risk histogram of flagged clusters, and a model-metrics card (ROC-AUC 0.879, illicit F1 0.53, PR-AUC 0.49).
2. **Explorer** — the existing cluster list + force-directed graph + inspector, now with **search** (by id) and **filter/sort** (by typology, risk, size).
3. **Sandbox** — build a transaction pattern (add nodes/edges, or pick a preset: peel chain / consolidation / mixer / fan-out). On "Analyze", each node is seeded with features sampled from the **illicit** profile (labeled as such), the GNN scores every node, and the reasoning layer returns a typology verdict. Renders the graph colored by live GNN risk + the verdict panel.
4. **Embedding Map** — interactive 2D t-SNE scatter of the learned embeddings (illicit vs licit); hovering shows risk; clicking a point that belongs to a flagged cluster navigates to it in Explorer.

## Backend

New/changed modules:

- `backend/models/feature_bank.py` — build/load the committed feature bank; `sample_features(profile, n)` returns seeded feature rows (166-dim) drawn from real illicit/licit transactions.
- `backend/api/sandbox.py` + `routes_sandbox.py` — `POST /api/sandbox`:
  - **in:** `{ nodes: [{id, profile?}], edges: [{source, target}] }`
  - builds a PyG `Data`, seeds each node from the feature bank by profile (default illicit), runs the checkpoint → per-node `risk`, computes structural `ClusterFeatures`, runs `classify_cluster` → verdict.
  - **out:** `{ nodes: [{id, risk}], edges, features, verdict }`
  - Presets are built client-side (topology only); the server seeds + scores.
- `backend/api/stats.py` + `routes_stats.py` — `GET /api/stats`: merges committed `seed_stats.json` (dataset totals + model metrics) with the live typology breakdown + risk histogram from `load_detection()`.
- `backend/api/embedding.py` + `routes_embedding.py` — `GET /api/embedding`: serves committed `embedding_2d.json` (subsampled points: `{x, y, label, risk, cluster_id?}`).
- `scripts/build_flagship_artifacts.py` — one-off: regenerates `feature_bank.npz`, `embedding_2d.json`, `seed_stats.json` from the graph + checkpoint.

## Frontend

- `frontend/src/App.jsx` — `<BrowserRouter>` + `<TopNav>` + `<Routes>`.
- `components/TopNav.jsx`, `views/Dashboard.jsx`, `views/Explorer.jsx`, `views/Sandbox.jsx`, `views/EmbeddingMap.jsx`.
- Reusable: `StatTile`, `TypologyDonut`, `RiskHistogram`, `MetricsCard`, `EmbeddingScatter`, `SandboxBuilder`, plus existing `NetworkGraph`, `ClusterInspector`, `RiskLegend`.
- Charts drawn with inline SVG/canvas (no chart lib dependency) to stay self-contained.
- `api/client.js` — add `getStats`, `getEmbedding`, `analyzeSandbox`.
- Visual system: keep the dark AML-console theme; add nav, stat tiles, chart styles, transitions, loading/empty states, responsive grid.

## Testing / verification

- Backend: unit tests for `sample_features` (shape/profile), `POST /api/sandbox` (peel-chain topology → high risk + `peeling_chain`), `GET /api/stats` and `/api/embedding` shape. Keep the suite green.
- Frontend: `npm run build` clean; then run the full stack and click through all four views + a live sandbox analysis in the browser; screenshot each; fix issues.

## Non-goals (YAGNI)

- No auth/multi-user, no persistence of sandbox sessions, no real-time streaming ingestion (still roadmap).
- No chart library; no drag-to-draw node editor beyond click-to-add + presets (keeps the builder simple and robust).
