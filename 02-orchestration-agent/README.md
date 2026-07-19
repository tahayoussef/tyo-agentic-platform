# 02 — Orchestration Agent

The third step of the platform: **one team of specialist agents, two orchestration
styles** over it — a **router** (classify, dispatch to exactly one specialist) and a
**supervisor** (an LLM coordinator that can consult several specialists and synthesize).
Same facade, same CLI, same eval set: the point is to *compare* the two styles, not to
crown one.

> **Router vs orchestrator — are they the same?** No: routing is the simplest special
> case of orchestration. A router makes one classification decision and forwards one
> specialist's answer untouched; a supervisor runs a ReAct loop whose tools are *other
> agents*, so it can consult many and merge what they say. Start with
> [`accompany_docs/00-routing-vs-orchestration.md`](src/orchestration_agent/accompany_docs/00-routing-vs-orchestration.md)
> for the full taxonomy (pipeline → router → supervisor → handoffs → hierarchical).
>
> **New to this?** The concept series in
> [`src/orchestration_agent/accompany_docs/`](src/orchestration_agent/accompany_docs/README.md)
> is the ordered tour: routing vs orchestration → specialists → the router → the
> supervisor → evaluating orchestration.

## The team

| Specialist | Source | Implementation |
| ---------- | ------ | -------------- |
| `github` | LIVE GitHub API | ReAct agent, one tool (`list_github_repositories`) |
| `docs` | Curated project docs (may be stale) | ReAct agent, one tool (`search_project_docs`) |
| `general` | None (small talk / meta) | A bare LLM call — no tools, so no graph |

The docs specialist searches with **deliberately simple keyword overlap**, not embeddings:
retrieval was 01's lesson, orchestration is 02's. The searcher hides behind a protocol, so
01's Qdrant stack could drop in later without either orchestrator noticing.

## The probe question (what makes the difference tangible)

The knowledge base still contains 01's planted conflict (the docs claim
`carthage-architecture-center` is **Python**; live GitHub says **HCL**). So ask:

> *"What language do the docs say carthage-architecture-center is written in, and what
> does live GitHub report today?"*

- **Router** — must pick ONE specialist, so the answer is structurally incomplete, no
  matter how good the model is.
- **Supervisor** — consults both specialists, flags the disagreement, prefers live data.

Same lesson as 01, one level up: there an *agent* reconciled two tools; here an
*orchestrator* reconciles two agents.

## Architecture

```
ROUTER mode:      question ─▶ classify (structured output, cheap model, temp 0)
                              └─▶ exactly ONE of: github | docs | general ─▶ answer

SUPERVISOR mode:  question ─▶ supervisor (ReAct loop)
                              ├─ consult_github_specialist ─▶ github agent (own context)
                              ├─ consult_docs_specialist   ─▶ docs agent   (own context)
                              └─▶ synthesize + attribute + flag conflicts ─▶ answer
```

| Module | Responsibility |
| ------ | -------------- |
| [`config.py`](src/orchestration_agent/config.py) | Typed settings (mode, router model, docs KB, GitHub). |
| [`llm.py`](src/orchestration_agent/llm.py) | Chat-model factories: reasoning model vs deterministic router model. |
| [`knowledge_base.py`](src/orchestration_agent/knowledge_base.py) | Heading-split sections + keyword search (deliberately simple). |
| [`github_client.py`](src/orchestration_agent/github_client.py) | Typed live GitHub REST client (ported from 00/01). |
| [`specialists.py`](src/orchestration_agent/specialists.py) | The team: prompts, tools, and the `Runner` facade. |
| [`router.py`](src/orchestration_agent/router.py) | Structured-output classification → `RouteDecision`. |
| [`supervisor.py`](src/orchestration_agent/supervisor.py) | Specialists wrapped as `consult_*` tools + supervisor prompt. |
| [`agent.py`](src/orchestration_agent/agent.py) | `OrchestrationAgent` facade — one surface, two modes. |
| [`tracing.py`](src/orchestration_agent/tracing.py) | Tool/delegation trace (`--show-trace`). |
| [`evaluation.py`](src/orchestration_agent/evaluation.py) | Routing accuracy + delegation coverage vs the gold set. |
| [`cli.py`](src/orchestration_agent/cli.py) | `ask`, `route`, and `eval` commands. |

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) and Python 3.12
- An NVIDIA AI endpoints API key ([build.nvidia.com](https://build.nvidia.com/))

No Docker infra needed — the knowledge base is local Markdown.

## Setup

```bash
uv sync
cp .env.example .env    # then set NVIDIA_API_KEY + GITHUB_USERNAME
```

## Run

```bash
# Router mode (default): one classification, one specialist
uv run orchestration-agent ask "How many stars does gobekli-tepe have right now?"
uv run orchestration-agent ask "According to the docs, why does carthage-architecture-center exist?"

# See the routing decision without running anyone
uv run orchestration-agent route "What does the documentation say about machu-pichu?"

# Supervisor mode: multi-specialist synthesis — try the probe question in both modes
uv run orchestration-agent ask --mode router     "What language do the docs say carthage-architecture-center is written in, and what does live GitHub report today?"
uv run orchestration-agent ask --mode supervisor "What language do the docs say carthage-architecture-center is written in, and what does live GitHub report today?"
```

Add `--show-trace` to watch orchestration happen on stderr. In supervisor mode the trace
shows the **delegations** (and *not* the specialists' inner tool calls — the trace
boundary is the context-isolation boundary); in router mode it shows the chosen
specialist's internal tool calls.

## Evaluate (measure before you optimize)

Two metrics for two failure modes ([`eval/gold.yaml`](eval/gold.yaml) is the shared gold set):

```bash
uv run orchestration-agent eval                     # router: routing accuracy
uv run orchestration-agent eval --mode supervisor   # supervisor: delegation coverage
```

- **Routing accuracy** — did the classifier pick an acceptable specialist per question?
- **Delegation coverage** — did the supervisor *actually consult* every specialist the
  question needs (recorded from its tool-call trajectory), and how many unnecessary
  delegations did it make? A supervisor that skips a needed source still *sounds*
  complete — the miss is invisible in the prose and obvious in the trajectory.

The knobs these numbers respond to: the specialist descriptions (shared by both modes),
the router/supervisor prompts, and `NVIDIA_ROUTER_MODEL` (routing is classification — try
a small model and see if accuracy holds).

## Develop

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest          # runs fully offline (fake LLMs, fake graphs, respx for HTTP)
```

## Docker

```bash
docker build -t orchestration-agent .
docker run --rm --env-file .env orchestration-agent ask "What can you do?"
```

## Configuration

See [`.env.example`](.env.example). All settings are environment variables; secrets are
`SecretStr` and never logged. Highlights: `ORCHESTRATION_MODE` (default `router`),
`NVIDIA_ROUTER_MODEL` (optional cheap routing model), `KNOWLEDGE_BASE_DIR`, `DOCS_TOP_K`.
