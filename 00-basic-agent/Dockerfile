# syntax=docker/dockerfile:1.9
# ---------------------------------------------------------------------------- #
# Multi-stage build using uv. The builder resolves dependencies against a locked
# environment; the runtime stage contains only the virtualenv and the app, runs
# as a non-root user, and never carries build tooling or secrets.
# ---------------------------------------------------------------------------- #

FROM python:3.12-slim-bookworm AS builder

# Pull the uv binary from its official image (pinned) — no curl-pipe-to-shell.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first, in their own cached layer, without the project. This
# layer is only invalidated when the lockfile or pyproject changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Now copy the source and install the project itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------- #

FROM python:3.12-slim-bookworm AS runtime

# Non-root user for the running process.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app

# `basic-agent` is the console script defined in pyproject. Override CMD to pass a
# question, e.g. `docker run --rm --env-file .env IMAGE ask "..."`.
ENTRYPOINT ["basic-agent"]
CMD ["--help"]
