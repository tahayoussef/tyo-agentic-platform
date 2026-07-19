# 00 · Routing vs orchestration — are they the same thing?

Short answer: **no, but they're related**. Routing is the simplest special case of
orchestration. This project builds both, over the same team, so you can feel the
difference instead of taking it on faith.

## The spectrum of multi-agent shapes

Once you have more than one agent (or one agent whose jobs pull in different directions),
you need to decide *who handles what*. The common shapes, simplest first:

1. **Fixed pipeline** — no decision at all: agent A always runs, then B, then C. (Vanilla
   RAG from project 01 is a fixed pipeline: always retrieve, then answer.)
2. **Router** — one *classification* decision up front: read the query, send it to exactly
   **one** specialist, return that specialist's answer untouched. One hop, no follow-up.
3. **Supervisor** (what people usually mean by "orchestrator") — an LLM in a loop that can
   call specialists like tools, *read their answers*, decide it needs another one, and
   finally write a **synthesized** reply from everything it collected.
4. **Handoffs / swarm** — no central coordinator: agents pass control (and the shared
   conversation) directly to each other.
5. **Hierarchical** — supervisors of supervisors, for when one team gets too big.

"Orchestration" is the umbrella activity — deciding which agents run, in what order, with
what inputs, and how their outputs combine. A router *orchestrates* in the most degenerate
way possible: one decision, one delegate, no combination step.

## Why the difference matters (the probe question)

Ask: *"What language do the docs say carthage is written in, and what does live GitHub
report today?"*

- The **router** must pick one specialist. Whichever it picks, the answer is **structurally
  incomplete** — not because the model was dumb, but because the architecture only allows
  one source. No prompt engineering fixes this.
- The **supervisor** consults the docs specialist, then the github specialist, notices the
  disagreement, and reports both sides.

This is the same lesson as 01's planted conflict, one level up: there the *agent* had to
reconcile two tools; here the *orchestrator* has to reconcile two agents.

## So why ever use a router?

Cost and latency. A router is one cheap classification call (we run it on a smaller model
at temperature 0); a supervisor is a full ReAct loop where every delegation is an entire
sub-agent run. Most real traffic ("how many stars does X have?") is single-source, and a
router handles it with a fraction of the tokens. Production systems often *combine* them:
route the easy 90%, escalate the ambiguous 10% to a supervisor.

## Where this lives

`../agent.py` — one facade, `mode="router" | "supervisor"`, so the two styles are
swappable and comparable. The router is `../router.py`; the supervisor is
`../supervisor.py`.
