# NEMESIS — Autonomous Fraud Network Intelligence System

## Project Overview
NEMESIS detects fraud rings by analyzing transaction *structure*, not individual transactions. It combines a Graph Neural Network (GNN) that learns structural embeddings from transaction graphs with an LLM reasoning layer that narrates why a flagged cluster looks like a specific fraud typology (mule network, synthetic identity farm, card testing ring, etc).

This is the ML-depth counterpart to prior projects (ARGUS, AVRA, HYDRA, ARIA) — those relied on LLM orchestration over multi-source data; NEMESIS proves applied ML (GNN training, embedding evaluation) fused with the same LLM-reasoning pattern used in ARGUS's SPECTER pipeline.

## Architecture

```
Raw transactions (IEEE-CIS / PaySim)
        ↓
Graph construction (accounts, devices, IPs as nodes; transfers/shared-attrs as edges)
        ↓
GNN (GraphSAGE → GAT) — learns embeddings, flags structurally anomalous clusters
        ↓
LLM reasoning layer (LangGraph) — narrates WHY a cluster looks fraudulent, classifies typology, confidence score
        ↓
FastAPI backend ← → React frontend (force-directed graph viz, click-to-inspect reasoning)
```

## Tech Stack
- **Graph/ML**: PyTorch, PyTorch Geometric (GraphSAGE, GAT), NetworkX (prototyping)
- **Data**: IEEE-CIS Fraud Detection (primary), PaySim (synthetic augmentation)
- **Reasoning**: LangGraph, Groq llama-3.3-70b (consistent with AVRA/ARGUS pattern)
- **Backend**: FastAPI, SQLite
- **Frontend**: React + react-force-graph or d3 for network visualization
- **Deploy**: Docker, Render

## Graph Schema (defined early — do not change casually)
- **Nodes**: account, device, IP address
- **Edges**: transaction (weighted, directed, account→account), shared_device (account—device), shared_ip (account—IP)
- Node features: account age, transaction velocity, avg amount, etc.
- Edge features: amount, timestamp, transaction type

---

## File Structure

```
nemesis/
├── CLAUDE.md                      # This file — project context for Claude Code
├── README.md                      # Public-facing writeup: problem, architecture, validation, roadmap (AVRA-style)
├── docker-compose.yml             # Orchestrates backend + frontend containers for local/deploy
├── .env.example                   # Template for required env vars (API keys, DB path) — never commit real .env
├── .gitignore                     # Excludes data/, models/, .env, __pycache__, node_modules
│
├── data/
│   ├── raw/                       # Untouched IEEE-CIS / PaySim CSVs as downloaded
│   ├── processed/                 # Cleaned, joined transaction tables ready for graph construction
│   └── graphs/                    # Serialized PyG Data objects (.pt files) — built graphs, not re-generated each run
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory analysis on raw transaction data — distributions, fraud rate, missingness
│   ├── 02_graph_sanity_check.ipynb # Visualize small subgraphs, confirm schema produces sensible clusters
│   └── 03_embedding_eval.ipynb     # t-SNE/UMAP projections of GNN embeddings — Phase 2 checkpoint proof
│
├── backend/
│   ├── main.py                     # FastAPI app entrypoint, mounts routers
│   ├── requirements.txt
│   ├── Dockerfile                  # Non-root user, minimal base image (consistent with AVRA hygiene)
│   │
│   ├── graph/
│   │   ├── schema.py                # Node/edge type definitions (account, device, IP; transaction, shared_device, shared_ip)
│   │   ├── build_graph.py           # Raw transaction rows → NetworkX graph → PyG Data object
│   │   └── features.py              # Node/edge feature engineering (account age, tx velocity, avg amount, etc.)
│   │
│   ├── models/
│   │   ├── graphsage.py             # GraphSAGE model definition (Phase 2 baseline)
│   │   ├── gat.py                   # GAT model definition (Phase 2 stretch goal, attention-based)
│   │   ├── train.py                 # Training loop, handles class imbalance (weighted loss / oversampling)
│   │   ├── evaluate.py              # Precision/recall/F1 on held-out fraud labels, embedding clustering quality
│   │   └── checkpoints/             # Saved model weights (.pt) — gitignored, large files
│   │
│   ├── reasoning/
│   │   ├── pipeline.py              # LangGraph pipeline definition — SPECTER-style pattern reused from ARGUS
│   │   ├── prompts.py               # System prompts for typology classification (mule network, synthetic ID, card testing)
│   │   └── schemas.py               # Pydantic models for structured LLM output (typology, confidence, reasoning chain)
│   │
│   ├── api/
│   │   ├── routes_ingest.py         # POST endpoint: accept new transaction data
│   │   ├── routes_detect.py         # POST endpoint: run detection pipeline, return flagged clusters
│   │   └── routes_clusters.py       # GET endpoint: fetch cluster details + reasoning for frontend inspection
│   │
│   ├── db/
│   │   ├── models.py                 # SQLite schema — accounts, transactions, flagged_clusters, reasoning_logs
│   │   └── session.py                 # DB connection/session handling
│   │
│   └── tests/
│       ├── test_graph_build.py
│       ├── test_reasoning_pipeline.py
│       └── test_api.py
│
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   ├── src/
│   │   ├── App.jsx                   # Root component, routing
│   │   ├── components/
│   │   │   ├── NetworkGraph.jsx      # Force-directed graph viz (react-force-graph/d3), color-coded by risk score
│   │   │   ├── ClusterInspector.jsx  # Click-to-inspect panel — shows LLM reasoning, typology, confidence
│   │   │   └── RiskLegend.jsx        # Color scale legend for the graph
│   │   ├── api/
│   │   │   └── client.js             # Fetch wrappers for backend endpoints
│   │   └── styles/
│   │       └── theme.css
│   └── public/
│
├── validation/
│   ├── case_study.md                 # Documented real-world mule network / laundering case being reconstructed
│   └── replay_test.py                # Feeds validation case data through pipeline, confirms NEMESIS flags it
│
└── scripts/
    ├── download_data.sh              # Pulls IEEE-CIS/PaySim datasets
    ├── build_all_graphs.py           # Batch graph construction from processed data
    └── deploy_render.sh              # Deployment helper script
```

### What each top-level piece is for
- **data/** — never committed except `.gitkeep` placeholders; raw and processed data stay local/gitignored given dataset size and licensing.
- **notebooks/** — exploratory/checkpoint work, not production code. `02` and `03` are the Phase 1 and Phase 2 proof points — keep as evidence for README/interview writeup.
- **backend/graph/** — turns tabular fraud data into the graph object the GNN trains on. Phase 1.
- **backend/models/** — the actual ML: GraphSAGE first, GAT as stretch upgrade. Phase 2, highest-risk/highest-value.
- **backend/reasoning/** — LLM layer, structurally identical to ARGUS's SPECTER pipeline, re-prompted for fraud typology. Phase 3.
- **backend/api/** — thin FastAPI layer connecting graph+model+reasoning to frontend. Phase 4.
- **frontend/** — the demo centerpiece. `NetworkGraph.jsx` is what people will screenshot; invest real time here.
- **validation/** — Phase 5, the "Galwan Valley" proof. Keep `case_study.md` well-documented, strongest interview talking point.
- **scripts/** — one-off automation so setup isn't repeated manually each session.

---

## Complete Roadmap (Phased)

### Phase 0 — Foundation
- Acquire datasets: IEEE-CIS Fraud Detection (primary), PaySim (synthetic augmentation)
- Set up environment: PyTorch, PyTorch Geometric, FastAPI, React
- Finalize graph schema (nodes: account/device/IP; edges: transaction/shared_device/shared_ip) — do not revisit casually once set
- Scaffold repo per file structure above
- Files touched: `scripts/download_data.sh`, `.env.example`, `.gitignore`, base repo skeleton

### Phase 1 — Graph construction
- Transform tabular transaction data into graph structure
- Build ingestion pipeline: raw rows → NetworkX graph (prototyping) → PyG Data objects (training)
- Sanity check: visualize a small subgraph, confirm schema produces sensible clusters before scaling
- Files touched: `backend/graph/schema.py`, `build_graph.py`, `features.py`, `notebooks/01_eda.ipynb`, `02_graph_sanity_check.ipynb`

### Phase 2 — GNN model (highest-risk phase)
- Start with GraphSAGE (inductive, easier to train) before attempting GAT (attention-based, harder to tune, better results)
- Handle class imbalance — fraud is always a small minority class; use weighted loss or oversampling
- Budget for multiple failed training runs and hyperparameter iteration — this phase has the highest time-overrun risk
- Checkpoint goal: embeddings visibly cluster fraud rings under t-SNE/UMAP projection — this is the proof the GNN learned real structure
- Files touched: `backend/models/graphsage.py`, `gat.py`, `train.py`, `evaluate.py`, `notebooks/03_embedding_eval.ipynb`

### Phase 3 — Reasoning layer
- GNN flags a suspicious cluster → feed structural features (shared devices, tx velocity, circular flow, account age) into an LLM agent
- LLM narrates why the cluster looks fraudulent, classifies likely typology (mule network / synthetic identity farm / card testing ring), assigns confidence with visible reasoning chain
- Directly reuses the SPECTER-pattern LangGraph pipeline from ARGUS — should be the fastest phase given prior experience
- Files touched: `backend/reasoning/pipeline.py`, `prompts.py`, `schemas.py`

### Phase 4 — API + visualization
- FastAPI endpoints: ingest transactions, run detection, return flagged clusters + reasoning
- React frontend: force-directed graph viz, color-coded by risk score, click-to-inspect reasoning panel
- This is the demo centerpiece — invest real time, it's what interviewers will actually look at
- Files touched: `backend/api/routes_ingest.py`, `routes_detect.py`, `routes_clusters.py`, `backend/db/`, all of `frontend/src/`

### Phase 5 — Validation case
- Find a documented real-world mule network / laundering typology (FinCEN advisories, published academic research)
- Reconstruct the pattern in test/synthetic data, show NEMESIS would have flagged it
- This is the "Galwan Valley" moment — the difference between "I trained a model on Kaggle data" and a defensible interview case study; do not skip
- Files touched: `validation/case_study.md`, `validation/replay_test.py`

### Phase 6 — Polish + deploy
- Docker (non-root, minimal base image), deploy to Render — consistent with AVRA/ARGUS deployment pattern
- Write README in AVRA's structure: problem, architecture, validation, roadmap
- Add an "S-tier roadmap" section showing what's next even if unbuilt (e.g., temporal graph modeling, cross-institution federated detection)
- Files touched: `README.md`, `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `scripts/deploy_render.sh`

---

## Conventions
- Follow existing project conventions from ARGUS/AVRA/HYDRA: LangGraph for reasoning, FastAPI backend, Docker for deploy, Render for hosting.
- Keep reasoning layer output structured (JSON/Pydantic) so the frontend renders confidence scores and typology labels consistently.
- Non-root Docker, CORS lockdown, no secrets committed — same security hygiene as AVRA.

## Current Status
Planning complete. Phase 0 not yet started.