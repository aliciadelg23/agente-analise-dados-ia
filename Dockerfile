# syntax=docker/dockerfile:1.7-labs

FROM python:3.12-slim AS base

# Copy the uv binary from the official image so we do not run curl|sh.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# curl is used by the compose healthchecks; the rest keep the image small.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

# Install dependencies first so the layer cache survives source-code changes.
COPY --chown=app:app pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the source tree (uses .dockerignore).
COPY --chown=app:app . /app

# Install the project itself (no-op for uv package=false, kept for parity).
RUN uv sync --frozen --no-dev

RUN chown -R app:app /app

USER app

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides it for the dashboard.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
