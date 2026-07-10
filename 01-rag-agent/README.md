# 01 — RAG Agent

The second step of the platform: an **agentic RAG** system that answers questions about a set
of GitHub repositories by combining two sources of truth:

- a **static knowledge base** (rich repo docs) indexed in a **Qdrant** vector database, and
- the **live GitHub API** (current stars, languages, dates).

The agent retrieves from the knowledge base *and* checks live data, then reconciles the two —
surfacing conflicts where the static docs have gone stale.

> **Status:** Phases 0–4 are implemented — infra, ingestion, the two tools, the reconciling
> agent (`ask`), **repository metadata filtering**, and a generic **eval harness** (`eval`).
> What remains is *using* the harness to drive optimization experiments (chunking, hybrid,
> reranking). See the blueprint for the full plan.
>
> **New to this?** Start with the concept series in
> [`src/rag_agent/accompany_docs/`](src/rag_agent/accompany_docs/README.md) — a junior-friendly,
> ordered tour of *what* we built and *why* (embeddings → chunking → the DB → ingestion →
> retrieval → filtering → agentic RAG → evaluation).

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
                                      └─ list_github_repositories (API, live)
                              └─▶ reconcile the two ─▶ concise answer
```

| Module | Responsibility |
| ------ | -------------- |
| [`config.py`](src/rag_agent/config.py) | Typed settings (embedding model, Qdrant, chunking, retrieval). |
| [`embeddings.py`](src/rag_agent/embeddings.py) | NVIDIA embeddings; asymmetric query/passage handling. |
| [`vector_store.py`](src/rag_agent/vector_store.py) | Qdrant client, collection lifecycle, dimension probing. |
| [`ingest.py`](src/rag_agent/ingest.py) | Load → chunk → embed → upsert. |
| [`retriever.py`](src/rag_agent/retriever.py) | Embed query → nearest-chunk search, with optional repo pre-filter. |
| [`github_client.py`](src/rag_agent/github_client.py) | Typed live GitHub REST client (ported from 00). |
| [`tools.py`](src/rag_agent/tools.py) | `search_knowledge_base` (static) + `list_github_repositories` (live). |
| [`agent.py`](src/rag_agent/agent.py) | ReAct agent + reconciliation prompt (`RagAgent` facade). |
| [`tracing.py`](src/rag_agent/tracing.py) | Callback handler that prints tool calls/results (`--show-trace`). |
| [`evaluation.py`](src/rag_agent/evaluation.py) | Generic retrieval eval: hit rate, recall@k, MRR, precision. |
| [`cli.py`](src/rag_agent/cli.py) | `ingest`, `search`, `ask`, and `eval` commands. |

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

## Ask the agent (fuses both sources)

```bash
uv run rag-agent ask "What is gobekli-tepe and how is it doing on GitHub right now?"
uv run rag-agent ask "What language is carthage-architecture-center written in?"
```

The second question exercises the planted conflict: the knowledge base says **Python**, the
live API says **HCL** — a good answer flags the discrepancy and prefers the live fact.

Add `--show-trace` to print every tool call and its full result to **stderr** — the fastest
way to see whether the agent consulted both sources and what each returned:

```bash
uv run rag-agent ask --show-trace "What language is carthage-architecture-center written in?"
```

## Evaluate retrieval (measure before you optimize)

The eval harness scores the **current** retriever against a fixed gold set
([`eval/gold.yaml`](eval/gold.yaml)) — so when you later change chunking or add reranking, you
re-run it and compare numbers instead of guessing.

```bash
uv run rag-agent ingest --recreate     # make sure the index reflects current settings
uv run rag-agent eval
```

```
Retrieval eval — k=4, 11 cases

  hit  recall     rr   prec  query
------------------------------------------------
 1.00    1.00   1.00   0.50  What language is carthage-architecture-cent...
 ...
------------------------------------------------
 0.82    0.71   0.68   0.39  AGGREGATE      ← the scorecard to move
```

Relevance is judged at the **repo level** (stable across re-chunking). Metrics: **hit rate**
(found it at all?), **recall@k** (found *all* the relevant repos?), **MRR** (how high up?),
**precision** (how little noise?). The gold set mixes question shapes (exact / fuzzy / multi /
KB-only) on purpose — a change that only helps one shape shows up as a flat aggregate, which
is the guard against fitting to a single demo question.

## Develop

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest          # runs fully offline (in-memory Qdrant + fake embeddings)
```

## Key learning levers (for the optimization phase)

Each is a knob; the `eval` harness tells you which ones actually pay:

- **Asymmetric embeddings** — passage vs. query `input_type` (handled in `embeddings.py`).
- **Chunking** — `CHUNK_SIZE` / `CHUNK_OVERLAP`.
- **Vector dimension & distance** — probed from the model; cosine metric.
- **Metadata filtering** — repo pre-filter (done). Next: entity resolution so it survives
  fuzzy references, not just exact slugs.
- **Still to try:** MMR (diversity), hybrid (dense + sparse) search, and reranking.

## Configuration

See [`.env.example`](.env.example). All settings are environment variables; secrets are
`SecretStr` and never logged.
