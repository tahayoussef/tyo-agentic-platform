# 07 · Agentic RAG — letting the agent search and reconcile

## Two ways to do RAG

- **Vanilla RAG:** a fixed pipeline. Every question *always* runs a search, the results get
  pasted into the prompt, the model answers. Simple, but rigid — it always retrieves, from one
  source, once.
- **Agentic RAG:** retrieval is a **tool** the model can *choose* to call, like any other tool.
  The model decides whether to search, what to search for, and can also call *other* tools —
  then reason over everything.

We use agentic RAG because our whole point is combining **two** sources: the static knowledge
base (via a `search_knowledge_base` tool) and the **live** GitHub API (via a
`list_github_repositories` tool). Only an agent that can call both and think about the results
can *reconcile* them. (This is the same tool-calling loop as `00-basic-agent`, with a retrieval
tool added — which is why agent 01 builds directly on agent 00.)

## Reconciliation: the actual goal

The knowledge base is rich but can be stale; the live API is current but shallow. The agent is
instructed to consult both, prefer live data for present-day facts, and **flag disagreements**.
We deliberately planted one: the KB says carthage's primary language is Python; live GitHub
says HCL. A good answer surfaces that conflict instead of silently picking one.

## Why this file sits *after* the retrieval files

Because the agent is only as good as what retrieval feeds it. Our reconciliation demo first
**failed** — the agent confidently said "consistent," because the search never returned the
contradicting chunk. The bug wasn't the model's reasoning; it was retrieval quality (see
[05](05-retrieval.md), [06](06-metadata-filtering.md)). The fix was on the retrieval side
(filtering), not the prompt. Lesson: **when an agent gives a wrong answer, suspect its inputs
before its reasoning.**

## Seeing inside the loop

Because the agent decides things autonomously, you need to *observe* what it did: which tools
it called, with what arguments, and what each returned. We added a trace (the `--show-trace`
flag) that prints exactly that. It's the same idea as production observability tools
(LangSmith / Langfuse) — and it's what turned "it didn't work" into a precise diagnosis.

## Where this lives

`../tools.py` (both tools), `../agent.py` (the reconciliation prompt and the ReAct loop), and
`../tracing.py` (the trace handler).
