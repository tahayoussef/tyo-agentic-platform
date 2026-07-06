# tyo-agentic-platform

A learning-oriented, **production-grade** playground for building agentic systems with
[LangChain](https://python.langchain.com/) and its ecosystem. The repository grows from a
single, simple agent toward progressively more complex, multi-agent architectures.

Every step is meant to reflect senior-level engineering practice: typed code, Pydantic
models, unit tests, linting, static typing, and reproducible Docker builds.

## Roadmap

| Step | Project | What it teaches |
| ---- | ------- | --------------- |
| `00` | [`00-basic-agent`](./00-basic-agent) | A single tool-calling agent that answers questions about a user's **public GitHub repositories**. Establishes the project template: `uv`, Pydantic settings, structured logging, tests, and a hardened Dockerfile. |
| …    | _(coming)_ | Memory, retrieval, multi-tool routing, multi-agent orchestration. |

## Conventions

- Each numbered folder is a **self-contained `uv` project** (own `pyproject.toml`, lockfile,
  and `Dockerfile`) so each learning step stays isolated and reproducible.
- Configuration and secrets are supplied via environment variables / `.env` files —
  **never hardcoded**. See each project's `.env.example`.

## Getting started

```bash
cd 00-basic-agent
uv sync
cp .env.example .env   # then fill in your secrets
uv run basic-agent ask "What are my most popular repositories?"
```

See [`00-basic-agent/README.md`](./00-basic-agent/README.md) for full details.
