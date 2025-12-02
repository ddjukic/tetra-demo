# Tetra v1 - Scientific Knowledge Graph Agent
# Multi-stage build for efficient image size

FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (before copying source for better caching)
RUN uv sync --frozen --no-dev

# Copy project files
COPY ml/ ./ml/
COPY clients/ ./clients/
COPY extraction/ ./extraction/
COPY agent/ ./agent/
COPY models/ ./models/
COPY frontend/ ./frontend/
COPY main.py ./

# Create models directory if not exists
RUN mkdir -p models

# Expose ports
# 8501 = Streamlit frontend
# 8000 = FastAPI (if added later)
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit frontend
CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
