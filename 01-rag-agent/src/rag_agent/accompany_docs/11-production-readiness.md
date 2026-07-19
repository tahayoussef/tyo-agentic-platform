# 11 · Production readiness — what "good code" still isn't

This is the capstone. The previous files taught how the agent *works*; this one is about the
gap between **a well-built learning agent** and **a system you'd put in front of real users**.

## The key distinction: craft vs. operations

This project already has production-grade **craft**: typed code with `mypy --strict`, a real
test suite, dependency injection, structured logging, secrets as `SecretStr`, validated config,
and a hardened Docker build. That's necessary — but it is *not* what makes something
production-ready. Production readiness is the **operational envelope** around the code: how it
serves traffic, survives failure, stays safe, controls cost, and is watched over time. That
envelope is what's missing, and it's mostly independent of code quality.

Below, grouped by how badly you'd regret skipping them.

## Tier 1 — don't ship without these

**It's a CLI, not a service.** A production agent is a concurrent HTTP service (health checks,
graceful shutdown, request-id correlation, connection pooling), not a one-shot command. Every
other concern here assumes that shape.

**No reliability/resilience.** There are no retries, backoff, or timeouts on the LLM,
embedding, reranker, or GitHub calls — one transient `429`/`5xx` fails the whole request. There
is no **cap on the agent's reasoning loop**, so a confused model can iterate and burn tokens
without bound, and no per-request timeout. And no graceful degradation (reranker down → fall
back to dense; knowledge base down → answer from live data only).

**Security & safety — the agent-specific gap.**
- **Prompt injection.** Retrieved documents and tool outputs are *untrusted text* fed straight
  into the model. A document containing "ignore your instructions and…" can hijack the agent.
  Production needs input handling, output guardrails, and scoped tool permissions.
- **Access control.** Our repo filter is *optional* — the model decides whether to apply it. In
  a multi-tenant product, scoping (by tenant/user) must be a **hard, server-enforced boundary**,
  never the model's choice, or one user can read another's data.
- **Secrets.** Env + `SecretStr` is fine for local dev; production wants a secrets manager with
  rotation.

## Tier 2 — needed for a trustworthy system

**Evaluation depth.** We measure *retrieval*, not *answers*. Production also needs
**answer-quality** evaluation — faithfulness/groundedness, answer relevance, correctness — via
gold answers or an LLM-as-judge, plus negative and "I don't know" cases and injection tests, all
running **in CI** as a quality gate so regressions are caught before release.

**Data & index lifecycle.** Ingestion today is a manual full wipe-and-rebuild (`--recreate`).
Production needs incremental/scheduled reindexing, **upserts by stable document id**,
**deletion** (removed docs, right-to-be-forgotten), change detection, and **blue-green
reindex** so users never hit a half-built index.

**Cost & latency controls.** No token or iteration budgets, no caching (embedding cache,
semantic response cache), no cheap-model routing. Every query pays full price from scratch.

## Tier 3 — operational maturity

**Infra & ops.** Qdrant is a single container on a local volume — no high availability,
replication, or backups (production = a managed/clustered deployment with snapshots). There's
no monitoring, alerting, or SLOs for latency, error rate, cost, or **retrieval-quality drift**.
Observability is a local `--show-trace`; production wants a persistent tracing backend
(LangSmith / Langfuse) with dashboards. Add CI/CD, infrastructure-as-code, autoscaling, and
**model-version pinning + a migration plan** (models get deprecated — recall the
`z-ai/glm-5.2` tool-support warning).

**Model & agent robustness.** A model with reliable tool-calling, validation of structured
tool calls, handling of malformed calls, and provider fallbacks.

## Product concerns (for a chat product)

The agent is **single-turn** — each question starts fresh. A conversational product needs
memory: session history, context-window management, and a user-feedback loop (thumbs up/down)
that feeds back into the eval set. (This is its own topic — memory is a natural next step.)

## How to read this list

Don't try to do all of it at once. The honest first move to productionize *this* agent is Tier
1: wrap it in a service with retries/timeouts, cap the reasoning loop, and enforce access
scoping. The rest is a roadmap, and several items (a serving layer, conversation memory,
multi-agent orchestration) are large enough to be their own steps in the platform's
simple→complex arc — not afterthoughts to bolt on.

The meta-lesson: **"the tests pass and mypy is happy" is the start line, not the finish line.**
Production is about what happens on the worst day, not the average one.
