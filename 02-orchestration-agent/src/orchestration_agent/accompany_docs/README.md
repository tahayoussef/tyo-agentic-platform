# Orchestration concepts — read in order

A junior-friendly tour of **what** we built for the orchestration agent and **why** — the
ideas, not the syntax. Each file is one concept; read them in number order.

| # | Concept | The one-line idea |
|---|---------|-------------------|
| [00](00-routing-vs-orchestration.md) | Routing vs orchestration | A router picks ONE agent; an orchestrator can use MANY and synthesize — routing is the simplest special case. |
| [01](01-specialist-agents.md) | Specialist agents | Several narrow agents beat one agent with every tool — focus, isolation, testability. |
| [02](02-the-router.md) | The router | Turn "who should answer?" into a classification task with structured output. |
| [03](03-the-supervisor.md) | The supervisor | An LLM coordinator that delegates via tools, reads answers, and synthesizes. |
| [04](04-evaluating-orchestration.md) | Evaluating orchestration | Routing accuracy and delegation coverage — score the *trajectory*, not just the answer. |

The code these describe lives one directory up (`../specialists.py`, `../router.py`,
`../supervisor.py`, `../agent.py`, `../evaluation.py`).
