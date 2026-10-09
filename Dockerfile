# ====================================================
# ForensicLens v2.1 — Production Docker Image
# ====================================================
# Runs the FastAPI server with the forensic model.
# CPU-only (no CUDA needed for inference).
# Model checkpoint is baked into the image.
# ====================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies (OpenCV, Pillow, etc.) ───────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────
# Install requirements first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────
# Copy code directories (NOT node_modules, tests, etc — handled by .dockerignore)
COPY config.py .
COPY predict.py .
COPY train_v2.py .

COPY models/ ./models/
COPY data/ ./data/
COPY deploy/ ./deploy/
COPY utils/ ./utils/
COPY monitoring/ ./monitoring/
COPY scripts/ ./scripts/

# ── Copy checkpoints (model + calibrators + thresholds) ──
COPY checkpoints/ ./checkpoints/

# ── Copy frontend build ──────────────────────────────
COPY frontend-app/dist/ ./frontend-app/dist/

# ── Copy test images ─────────────────────────────────
COPY test_images/ ./test_images/

# ── Create runtime directories ───────────────────────
RUN mkdir -p /app/cases /app/logs /app/exports

# ── Health check ─────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Launch ───────────────────────────────────────────
# Use deploy.api_server directly (it runs uvicorn internally)
CMD ["python", "deploy/api_server.py"]
