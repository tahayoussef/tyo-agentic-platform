# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

A learning-oriented but **production-grade** playground for agentic systems (LangChain / LangGraph + NVIDIA AI endpoints). Each numbered folder (`00-basic-agent`, `01-rag-agent`, …) is a **self-contained uv project** with its own `pyproject.toml`, `uv.lock`, `.env`, and `Dockerfile`. Always `cd` into the project folder before running any command — nothing installs or runs from the repo root.

Projects build on each other: `00` establishes the template (typed settings, structured logging, DI, tests, Docker); `01` adds RAG on top of the same agent shape. Even though this is a learning repo, the quality bar is senior/production level: strict mypy, ruff, Pydantic models, unit tests for everything.

## Commands

Run from inside the relevant project directory (e.g. `cd 01-rag-agent`):

```bash
uv sync                          # install deps (incl. dev group)
uv run pytest                    # all tests — fully offline (in-memory Qdrant, fake embeddings/LLM)
uv run pytest tests/test_retriever.py::test_name   # single test
uv run ruff check . && uv run ruff format --check . # lint / format check
uv run mypy                      # strict static typing (covers src + tests)
```

01-rag-agent additionally needs infra and an index for live runs (not for tests):

```bash
docker compose up -d             # Qdrant (dashboard: http://localhost:6333/dashboard)
uv run rag-agent ingest --recreate   # (re)build the index — required after chunking/retrieval-mode changes
uv run rag-agent search "..."        # inspect raw retrieval
uv run rag-agent ask --show-trace "..."  # full agent; trace prints tool calls to stderr
uv run rag-agent eval                # score retrieval vs eval/gold.yaml (hit/recall@k/MRR/precision)
```

`00-basic-agent` has the same shape with the `basic-agent ask` entry point.

02-orchestration-agent needs no infra (its knowledge base is local Markdown):

```bash
uv run orchestration-agent ask --mode router "..."      # or --mode supervisor
uv run orchestration-agent route "..."                  # show the routing decision only
uv run orchestration-agent eval                         # routing accuracy vs eval/gold.yaml
uv run orchestration-agent eval --mode supervisor       # delegation coverage
```

## Architecture

### Common agent shape (both projects)

- LangGraph `create_react_agent` wrapped in a typed facade class (`BasicAgent` / `RagAgent`) exposing `ask()` and `stream()`. The facade accepts an optional pre-built `graph` for testing.
- **Dependency injection everywhere**: tools are built via factories that take clients/embeddings as arguments (`build_tools(settings, embeddings=..., qdrant_client=...)`), so tests run with no network, no LLM, and no real DB. Follow this pattern for any new tool or component.
- Configuration: `pydantic-settings` `Settings` class in `config.py`; every knob is an env var (field `x` ↔ env `X`), secrets are `SecretStr` unwrapped only at call sites. Never hardcode config.
- Structured logging via `structlog` (`logging.py`); CLI via Typer (`cli.py`).

### 01-rag-agent specifics

Two sources of truth that the agent must **reconcile**:

1. Static KB: `github_repos_readmes/*.md` → chunk → embed (asymmetric passage/query `input_type`, `embeddings.py`) → Qdrant (`ingest.py`, `vector_store.py`) → exposed as the `search_knowledge_base` tool.
2. Live GitHub REST API (`github_client.py`) → `list_github_repositories` tool.

The system prompt in `agent.py` (`RECONCILIATION_SYSTEM_PROMPT`) is the core of the design: consult both, flag disagreements, prefer live data for present-day facts. The KB is **intentionally stale/wrong** — notably `carthage-architecture-center` claims Python while the live repo is HCL — so reconciliation is testable. Don't "fix" that divergence.

`retriever.py` layers three orthogonal, opt-in features over basic dense search: repo metadata pre-filtering, hybrid search (dense + BM25 sparse via FastEmbed, `RETRIEVAL_MODE=hybrid`, requires `ingest --recreate`), and two-stage reranking (`USE_RERANKER=true`, fetch `rerank_fetch_k` then cross-encode down to `top_k`).

**Measure before optimizing**: any retrieval change should be judged by re-running `uv run rag-agent eval` against `eval/gold.yaml` (repo-level relevance, stable across re-chunking), not by eyeballing one query. The gold set deliberately mixes question shapes so single-shape improvements show up as a flat aggregate.

### 02-orchestration-agent specifics

One specialist team (`specialists.py`: `github` = live API ReAct agent, `docs` = local-KB ReAct agent, `general` = bare LLM call), two orchestration styles behind one `OrchestrationAgent` facade (`mode="router" | "supervisor"`):

- **Router** (`router.py`): structured-output classification — `RouteDecision` with a `Literal` route field constrains the model by schema; runs on an optional cheaper model (`NVIDIA_ROUTER_MODEL`) at temperature 0. Dispatches to exactly one specialist.
- **Supervisor** (`supervisor.py`): agents-as-tools — each specialist wrapped as a `consult_<name>_specialist` tool; sub-agents run in isolated contexts (delegations pass a question, return only the final answer). The `general` specialist deliberately gets no delegation tool.

The `SPECIALIST_DESCRIPTIONS` dict is the shared source of truth for both modes' prompts — it's the main tuning knob. The docs specialist's keyword searcher (`knowledge_base.py`) is **deliberately primitive** (retrieval was 01's lesson); don't upgrade it to embeddings without being asked — it hides behind the `DocsSearcher` protocol precisely so that swap stays isolated. Orchestration changes are judged by `orchestration-agent eval` (router accuracy) and `eval --mode supervisor` (delegation coverage — a trajectory metric recorded from tool calls), against `eval/gold.yaml` whose cross-source probe questions are the analog of 01's planted conflict.

### accompany_docs convention

Each project keeps a junior-friendly concept series in `src/<pkg>/accompany_docs/` (numbered, ordered markdown explaining *what* was built and *why*). When adding a significant feature, add or update the corresponding doc and its index in `accompany_docs/README.md`.

## Testing conventions

- Tests must stay fully offline: `QdrantClient(location=":memory:")`, fake embeddings/sparse/reranker classes, `respx` for httpx mocking, and `FakeGraph` for the agent facade.
- `pytest` runs with `--strict-markers`; coverage is configured (`uv run pytest --cov`).
