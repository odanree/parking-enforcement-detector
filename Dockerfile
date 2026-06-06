# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:24-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# vite.config.ts sets outDir: '../frontend-dist', so output lands at /app/frontend-dist
RUN npm run build


# ── Stage 2: Python runtime ────────────────────────────────────────────────────
# GPU acceleration (YOLO on the RTX 3090) uses the CUDA torch wheels below; the
# wheels bundle the CUDA/cuDNN runtime, so the slim base needs no system CUDA —
# only the host NVIDIA driver + nvidia-container-runtime (already in place).
FROM python:3.11-slim

WORKDIR /app

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — install CUDA torch first so the later requirements.txt
# resolve doesn't pull a different (CPU) build. cu124 matches host driver 595.79.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY src/ ./src/
COPY config/ ./config/
# Maintenance/ingestion CLIs (run via `docker compose run` / `docker exec`)
COPY scripts/ ./scripts/
COPY models/ ./models/

# Built React frontend from stage 1
COPY --from=frontend-builder /app/frontend-dist/ ./frontend-dist/

# Runtime directories (populated at runtime via volume mounts)
RUN mkdir -p snapshots logs dataset data/vectors

EXPOSE 8000
CMD ["python", "-m", "src.main_web"]
