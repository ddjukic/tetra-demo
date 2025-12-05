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

# Expose port (Cloud Run uses PORT env var, default to 8080)
EXPOSE 8080

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check (uses PORT env var for flexibility)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Default command: run Streamlit frontend with dynamic port (Cloud Run sets PORT)
CMD uv run streamlit run frontend/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
