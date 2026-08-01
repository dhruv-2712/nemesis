# NEMESIS backend — FastAPI + CPU PyTorch, non-root, slim base.
FROM python:3.11-slim

# System deps kept minimal; slim base + no build tools needed for wheels.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for layer caching. Use the CPU-only PyTorch index so
# the image stays lean (no CUDA), then the rest of the requirements.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r backend/requirements.txt

# App code (data/ and model checkpoints are gitignored; the committed
# seed_detection.json lets the read API serve without them).
COPY backend ./backend

# Run as an unprivileged user (AVRA/ARGUS hygiene).
RUN useradd --create-home --uid 10001 nemesis \
    && mkdir -p backend/db && chown -R nemesis:nemesis /app
USER nemesis

EXPOSE 8000

# Render/other PaaS inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
