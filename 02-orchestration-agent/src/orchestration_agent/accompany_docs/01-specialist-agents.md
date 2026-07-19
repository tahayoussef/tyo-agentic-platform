# 01 · Specialist agents — why not one agent with all the tools?

Project 01's agent had two tools and one prompt telling it how to use both. That works at
two tools. It stops working as tools multiply, for reasons that are really *prompt* and
*context* problems:

- **Prompt dilution.** One system prompt must explain every tool, every edge case, every
  reconciliation rule. Each instruction competes for the model's attention; the more jobs
  the prompt covers, the worse it does each one.
- **Tool confusion.** With many similar tools, models start picking the wrong one or
  spraying calls. A specialist that only *has* the GitHub tool cannot call the wrong tool.
- **Context pollution.** Every tool result lands in the one shared conversation. A long
  docs dump sits in context while the model reasons about star counts.
- **Untestable behavior.** "Does the mega-agent handle docs questions well?" is one
  entangled question. "Does the docs specialist answer from its search results?" is a
  small, isolated one.

## What a specialist looks like here

A specialist = **a name + a description + a runner** (see `Specialist` in
`../specialists.py`):

- `github` — a ReAct agent with exactly one tool (`list_github_repositories`) and a prompt
  that says: ground everything in the live API, and admit you don't know history.
- `docs` — a ReAct agent with one tool (`search_project_docs`) and a prompt that says:
  answer from the docs, *attribute* claims to the docs, and admit the docs may be stale.
- `general` — **no tools at all**, so it isn't even a graph: it's a single LLM call behind
  the same interface (`LlmRunner`). A specialist is defined by its narrow job, not by
  being an "agent".

The `description` field matters more than it looks: it is the **single source of truth**
both orchestrators read. The router pastes the descriptions into its classification
prompt; the supervisor pastes them into its delegation-tool descriptions. Writing crisp,
contrastive descriptions ("RIGHT NOW" vs "may be out of date") *is* the tuning knob for
orchestration quality.

## The deliberately-dumb searcher

The docs specialist searches with plain keyword overlap (`../knowledge_base.py`), not
embeddings — a big step "down" from project 01. Deliberate: retrieval was 01's lesson;
orchestration is 02's. One new hard thing per project. The searcher hides behind a
`DocsSearcher` protocol, so swapping in 01's Qdrant retrieval later is a drop-in change
that neither orchestrator would notice.

## Where this lives

`../specialists.py` — the team, its prompts, its tools, and the `Runner` protocol that
makes graph-backed and LLM-backed specialists interchangeable.
