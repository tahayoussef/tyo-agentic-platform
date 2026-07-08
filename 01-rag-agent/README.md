# 01 — RAG Agent

The second step of the platform: an **agentic RAG** system that answers questions about a set
of GitHub repositories by combining two sources of truth:

- a **static knowledge base** (rich repo docs) indexed in a **Qdrant** vector database, and
- the **live GitHub API** (current stars, languages, dates).

The agent retrieves from the knowledge base *and* checks live data, then reconciles the two —
surfacing conflicts where the static docs have gone stale.

> **Status:** Phase 0 (infra) + Phase 1 (ingestion) are implemented. The retrieval tool and
> the reconciling agent (`ask`) arrive in later phases. See the blueprint for the full plan.

## Why "agentic" RAG (do tools fit RAG?)

Two RAG styles exist. **Vanilla RAG** always retrieves, then answers — a fixed pipeline.
**Agentic RAG** exposes retrieval as a *tool* the LLM chooses to call, so it can also call
other tools (here, the live GitHub API) and synthesize across them. Because this agent must
consult two sources and reconcile them, it needs the agentic style — which is just
`00-basic-agent`'s tool-calling loop plus a `search_knowledge_base` tool.

## The intentional divergence (what makes this tangible)

The knowledge base is deliberately **different** from live GitHub, so reconciliation is real:

- **Narrative vs. metrics** — the docs hold architecture/rationale the API can't return.
- **Staleness** — each doc has a frozen `Snapshot (Q1 2025)` block; live counts differ.
- **A planted conflict** — `carthage-architecture-center` claims its primary language is
  **Python**, while the live repo is **HCL (Terraform)**. A good answer flags this.

## Architecture

```
INGEST (offline):  github_repos_readmes/*.md ─▶ chunk ─▶ embed(passage) ─▶ Qdrant
QUERY  (online):   question ─▶ agent ─┬─ search_knowledge_base (Qdrant, static)
                                      └─ list_github_repositories (API, live)   [later phase]
```

| Module | Responsibility |
| ------ | -------------- |
| [`config.py`](src/rag_agent/config.py) | Typed settings (embedding model, Qdrant, chunking, retrieval). |
| [`embeddings.py`](src/rag_agent/embeddings.py) | NVIDIA embeddings; asymmetric query/passage handling. |
| [`vector_store.py`](src/rag_agent/vector_store.py) | Qdrant client, collection lifecycle, dimension probing. |
| [`ingest.py`](src/rag_agent/ingest.py) | Load → chunk → embed → upsert. |
| [`retriever.py`](src/rag_agent/retriever.py) | Embed query → nearest-chunk search. |
| [`cli.py`](src/rag_agent/cli.py) | `ingest` and `search` commands. |

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/), Python 3.12, Docker
- An NVIDIA AI endpoints API key (for embeddings)

## Setup

```bash
docker compose up -d          # start Qdrant  (dashboard: http://localhost:6333/dashboard)
uv sync
cp .env.example .env          # then set NVIDIA_API_KEY
```

## Index the knowledge base

```bash
uv run rag-agent ingest --recreate
# → Ingested N chunks from 5 files into 'github_repos' (vector dim = 1024).
```

Open the Qdrant dashboard to *see* the points, vectors, and payloads you just created.

## Inspect retrieval

```bash
uv run rag-agent search "which project uses dbt and a medallion architecture?"
uv run rag-agent search "what language is carthage written in?"
```

## Develop

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest          # runs fully offline (in-memory Qdrant + fake embeddings)
```

## Key learning levers (for the optimization phase)

- **Asymmetric embeddings** — passage vs. query `input_type` (handled in `embeddings.py`).
- **Chunking** — `CHUNK_SIZE` / `CHUNK_OVERLAP`.
- **Vector dimension & distance** — probed from the model; cosine metric.
- Later: MMR, metadata filtering, hybrid (dense+sparse), reranking, and a recall@k eval set.

## Configuration

See [`.env.example`](.env.example). All settings are environment variables; secrets are
`SecretStr` and never logged.
