# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:24-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# vite.config.ts sets outDir: '../frontend-dist', so output lands at /app/frontend-dist
RUN npm run build


# ── Stage 2: Python runtime ────────────────────────────────────────────────────
# For GPU acceleration (YOLO), replace the base image with:
#   nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# and install python3.11 + pip manually.
FROM python:3.11-slim

WORKDIR /app

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY src/ ./src/
COPY config/ ./config/

# Built React frontend from stage 1
COPY --from=frontend-builder /app/frontend-dist/ ./frontend-dist/

# Runtime directories (populated at runtime via volume mounts)
RUN mkdir -p snapshots logs

EXPOSE 8000
CMD ["python", "-m", "src.main_web"]
