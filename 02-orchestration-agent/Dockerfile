# syntax=docker/dockerfile:1.9
# Multi-stage build using uv (mirrors 00/01). The runtime image contains only the
# virtualenv + app + knowledge base, runs as a non-root user, and carries no secrets.

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app

ENTRYPOINT ["orchestration-agent"]
CMD ["--help"]
