# ============================================================
# Backend Dockerfile — FastAPI + RAG pipeline
# ============================================================
# Multi-stage build để image final nhỏ + không leak pip cache.

FROM python:3.11-slim AS builder

WORKDIR /build

# System deps cho sentence-transformers / pyarrow / etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements + install vào virtualenv
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ============================================================
# Runtime image — gọn hơn (no build-essential)
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps (libgomp cho torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv từ builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy code
COPY src ./src
COPY scripts ./scripts
COPY pyproject.toml ./

# Non-root user
RUN useradd -m -u 1000 app && \
    mkdir -p /app/data && \
    chown -R app:app /app
USER app

EXPOSE 8000

# Healthcheck — Cloudflared tunnel kiểm tra /health.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
