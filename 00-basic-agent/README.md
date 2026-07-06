# 00 — Basic Agent

The first step of the platform: a **single tool-calling agent** that answers questions
about a GitHub user's public repositories.

```
You: "What are octocat's most popular Python projects?"
 └─▶ LLM decides to call the tool
      └─▶ list_github_repositories(username="octocat")  ──▶ GitHub REST API
      ◀─ formatted repo list
 ◀─ grounded, natural-language answer (streamed)
```

## Architecture

| Module | Responsibility |
| ------ | -------------- |
| [`config.py`](src/basic_agent/config.py) | Typed settings via `pydantic-settings`; secrets from env / `.env`. |
| [`logging.py`](src/basic_agent/logging.py) | Structured logging (`structlog`); console or JSON. |
| [`github_client.py`](src/basic_agent/github_client.py) | Narrow, typed `httpx` client + Pydantic repo model. |
| [`tools.py`](src/basic_agent/tools.py) | The `list_github_repositories` LangChain tool (built via an injectable factory). |
| [`agent.py`](src/basic_agent/agent.py) | LangGraph `create_react_agent` wired to `ChatNVIDIA`, wrapped by `BasicAgent`. |
| [`cli.py`](src/basic_agent/cli.py) | `Typer` entry point: `basic-agent ask "<question>"`. |

**Design choices worth noting**

- **Dependency injection everywhere** — the tool takes a `client_factory` and `BasicAgent`
  takes an optional `graph`, so the whole thing is unit-testable without the network or LLM.
- **Secrets are `SecretStr`** and only unwrapped at the call site; they never appear in
  `repr`/logs.
- **The LLM is instructed to ground answers in tool output** to reduce hallucination.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) and Python 3.12
- An NVIDIA AI endpoints API key ([build.nvidia.com](https://build.nvidia.com/))

## Setup

```bash
uv sync                 # create the venv and install deps (incl. dev group)
cp .env.example .env    # then edit .env and set NVIDIA_API_KEY + GITHUB_USERNAME
```

## Run

```bash
# Streamed answer (default)
uv run basic-agent ask "What are my most starred repositories?"

# Non-streamed
uv run basic-agent ask --no-stream "Which languages do I use most?"
```

## Develop

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # static types (strict)
uv run pytest              # tests
uv run pytest --cov        # tests with coverage
```

## Docker

The image reads configuration from environment variables at runtime — no secrets are baked in.

```bash
docker build -t basic-agent .
docker run --rm --env-file .env basic-agent ask "What are my most popular projects?"
```

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)). Highlights:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `NVIDIA_API_KEY` | _(required)_ | NVIDIA AI endpoints key. |
| `NVIDIA_MODEL` | `z-ai/glm-5.2` | Model served by NVIDIA. |
| `GITHUB_USERNAME` | _(none)_ | Default account inspected when a question names no user. |
| `GITHUB_TOKEN` | _(none)_ | Optional; raises the API rate limit to 5000/hour. |
| `LOG_JSON` | `false` | Emit JSON logs for production aggregation. |

> **Security:** `.env` is git-ignored. Never commit real keys. If a key is ever exposed,
> rotate it immediately in the provider dashboard.
