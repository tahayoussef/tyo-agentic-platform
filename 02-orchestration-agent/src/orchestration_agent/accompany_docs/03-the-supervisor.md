# 03 · The supervisor — agents as tools

## The idea

The supervisor is *just the ReAct loop from projects 00/01 with a twist*: its "tools"
are **other agents**. Each specialist gets wrapped in a `consult_<name>_specialist` tool
that takes a question and returns the specialist's final answer. The supervisor LLM then
does what any ReAct agent does — call a tool, read the result, decide what's next — except
every call is a whole sub-agent run.

That inherited loop is exactly the ability the router lacked: consult the docs specialist,
*read the answer*, realize live data is also needed, consult the github specialist, then
write **one synthesized reply** that attributes claims and flags disagreements. (The
reconciliation instinct from project 01 lives on in the supervisor's prompt — one level
up: reconciling *agents* instead of *tools*.)

## Context isolation — the big design choice

When the supervisor consults a specialist, it sends a question and gets back a paragraph.
It never sees the specialist's inner life: not its tool calls, not its intermediate
reasoning, not the 3 doc sections it retrieved. Each specialist runs in a **fresh, private
context**.

- Cost: the specialist can't see the conversation either — so the supervisor's prompt
  insists on *self-contained* sub-questions ("it cannot see this conversation").
- Benefit: no context pollution, and each side stays small and debuggable.

The main alternative is the **handoff** pattern (LangGraph's swarm style), where agents
share one message history and pass control around. Shared history means no re-explaining,
but every agent pays to read everything, and one agent's noise becomes everyone's noise.
Agents-as-tools is the simpler default; reach for handoffs when agents genuinely need each
other's full working state.

You can *see* the isolation: run `ask --mode supervisor --show-trace` and the trace shows
delegations and their answers — but not the specialists' internal tool calls. Run the same
question through `--mode router --show-trace` and you see the chosen specialist's
internals instead. The trace boundary *is* the context boundary.

## What supervision costs

Every delegation is a full sub-agent run (its own LLM calls, maybe its own tool calls),
plus the supervisor's own reasoning turns around it. A three-delegation answer can cost
5–10× a routed answer, and it's slower. That's the honest trade: synthesis and coverage
vs. tokens and latency — and it's why the router still exists (doc 00).

Note the `general` specialist gets **no** delegation tool: the supervisor handles small
talk itself rather than paying a hop for it. Delegation should buy capability, not
ceremony.

## Where this lives

`../supervisor.py` — the delegation tools, the tool-name mapping (which the eval reuses to
recover trajectories), and the supervisor prompt.
