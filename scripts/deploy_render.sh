#!/usr/bin/env bash
# Deploy helper for Render (https://render.com).
#
# NEMESIS deploys as two services:
#   - nemesis-api      (Docker, backend/Dockerfile)  — FastAPI
#   - nemesis-frontend (Docker, frontend/Dockerfile) — nginx static
#
# The read API serves from the committed backend/api/seed_detection.json, so the
# deployed backend needs neither the 135 MB Elliptic graph nor a GPU. To enable
# live re-detection (/detect) or LLM narration, provide the artifacts / key below.
#
# This script does not call the Render API directly; it prints the settings to
# use in the Render dashboard (or render.yaml), keeping secrets out of the repo.
set -euo pipefail

cat <<'INSTRUCTIONS'
NEMESIS — Render deploy settings
================================

1) Backend service (Web Service, Docker)
   - Root directory:   .
   - Dockerfile path:  backend/Dockerfile
   - Health check:     /health
   - Environment:
       FRONTEND_ORIGIN = https://<your-frontend>.onrender.com
       NEMESIS_DB_PATH = /app/backend/db/nemesis.db
       GROQ_API_KEY    = <optional; enables LLM narration>

2) Frontend service (Web Service or Static Site, Docker)
   - Root directory:   .
   - Dockerfile path:  frontend/Dockerfile
   - Build arg:        VITE_API_BASE = https://<your-backend>.onrender.com

3) After first deploy, set the backend's FRONTEND_ORIGIN to the real frontend
   URL and redeploy so CORS permits it.

Notes:
   - Serving works out of the box from the committed seed detection artifact.
   - /detect (live model pass) requires data/graphs/elliptic.pt + the checkpoint,
     which are gitignored; attach them via a Render disk or bake them in.
INSTRUCTIONS
