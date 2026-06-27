# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build deps (LightGBM needs cmake/libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime lib only (LightGBM needs libgomp at runtime too)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY data/eval_v2/index.json ./data/index.json

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Uvicorn with 2 workers; tune via WORKERS env var
CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}"]