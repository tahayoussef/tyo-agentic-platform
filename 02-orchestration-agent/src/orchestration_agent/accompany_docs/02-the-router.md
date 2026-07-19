# 02 · The router — "who should answer?" as a classification task

## The idea

Don't ask the LLM to *answer* the query. Ask it a much smaller question: **which of these
three specialists should answer it?** That's classification, and it changes everything
about how you build and run the call:

- **Structured output.** We force the reply into a Pydantic schema (`RouteDecision`) whose
  `route` field is a `Literal["github", "docs", "general"]`. The schema is sent to the
  model as a contract — an invalid destination is impossible *by construction*, not
  caught by an if-statement afterwards. This is the single most useful trick in the file.
- **A cheaper model.** Classification doesn't need the big reasoning model. The
  `NVIDIA_ROUTER_MODEL` setting lets routing run on a small model while specialists keep
  the big one — in production this is often the difference that pays for the router.
- **Temperature 0.** A routing decision should be deterministic. There is nothing creative
  about picking a destination (`build_router_llm` pins temperature to 0).

## What the decision carries

`RouteDecision` has three fields, each earning its place:

- `route` — the verdict; the only field the dispatcher needs.
- `confidence` — self-reported, so treat it as a *signal*, not a probability. Its real use
  is operational: log it, and a threshold ("below 0.6, escalate to the supervisor") is the
  classic router→supervisor hybrid. We log it but don't threshold yet.
- `reason` — one sentence. Costs a few tokens, and turns every misroute in the eval from a
  mystery into a diagnosis.

## The router's ceiling

The router returns the chosen specialist's answer **untouched** — there is no step where
anything could merge two sources. So on cross-source questions (see doc 00's probe) it is
structurally partial. The eval makes this visible instead of anecdotal: cross-source gold
cases accept *either* route as "correct" for the router, yet the answer still can't be
complete. When accuracy is high and answers are still unsatisfying, the architecture — not
the prompt — is the bottleneck. That's your cue to reach for the supervisor.

## Where this lives

`../router.py` (schema, prompt, classifier) and `../agent.py` (`_dispatch` — classify,
resolve, forward). `orchestration-agent route "<q>"` shows a decision without running
anyone.
