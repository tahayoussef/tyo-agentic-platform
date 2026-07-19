# 04 · Evaluating orchestration — score the trajectory, not just the answer

Project 01 taught: measure before you optimize, against a fixed gold set. Orchestration
needs that discipline too, but the *thing to measure* changes. Answer text is hard to
grade automatically — whether the right agents ran is easy to grade, and it's where
orchestration actually fails.

## Two metrics for two modes

**Router accuracy** — classification accuracy. Each gold case lists its acceptable
`routes`; the router is correct if its pick is in the set. Cross-source questions list
several acceptable routes, because *any* single pick is defensible for a router (and all
of them are incomplete — the metric scores the router against its own job description,
not against the supervisor's).

**Delegation coverage** — a *trajectory* metric. For the supervisor we record which
specialists it consulted (a callback collects tool calls; the `consult_*` naming scheme
maps them back to specialist names) and check that every specialist in the case's `needs`
was actually consulted. We also count **extra** delegations — consultations the question
didn't need. Coverage catches confident-but-underinformed answers; extras catch the
opposite failure, an over-eager supervisor burning tokens.

Why trajectory instead of judging the final prose? Because a supervisor that never
consulted the github specialist can still *sound* complete — the miss is invisible in the
text and obvious in the trajectory. (Grading the prose itself needs an LLM-as-judge,
which is a later project's lesson.)

## The gold set is the contract

`eval/gold.yaml` mixes clearly-live, clearly-docs, small-talk, and cross-source shapes on
purpose — a prompt tweak that only helps one shape shows up as a flat aggregate, same
guard against overfitting as 01. `needs` defaults to `routes` minus `general`: a question
routable to a data specialist should see that specialist consulted; small talk should
trigger **no** delegation (and any delegation it does trigger counts as an extra).

## What you tune with this

The levers are almost all *words*: the specialist descriptions (shared by both modes —
doc 01), the router prompt, the supervisor prompt, and the router model choice. Change
one, re-run `eval`, compare numbers. The harness takes callables/protocols rather than
concrete agents, so tests exercise it with fakes and the CLI plugs in the real thing —
same "generic harness" philosophy as 01.

## Where this lives

`../evaluation.py` (gold loading, both evaluators, the delegation recorder) and
`../cli.py` (`eval` command, report tables).
