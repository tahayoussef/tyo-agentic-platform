# 08 · Evaluating retrieval — turning "better" into a number

## Why this exists

We just added metadata filtering and it fixed our one test question. But *did* it make the
system better, or did we just fit the code to the demo? You genuinely cannot tell by eyeballing
one query. The only honest answer is to **measure** — on a set of varied questions, with a
number that goes up or down when you change something.

That discipline — *measure before you optimize, and after* — is what separates a production RAG
system from a pile of plausible-looking tricks.

## What "correct retrieval" means here

For each test question we write down which **repo(s)** *should* show up in the results. We
judge relevance at the **repo level** (not exact chunk) on purpose: chunk boundaries move every
time you change chunk size, but "the answer is in the carthage docs" stays true. That makes the
same test set valid across chunking changes, reranking, hybrid search — whatever we try next.

## The metrics (in plain words)

Given the ranked chunks a search returns for a question:

- **Hit rate @k** — did *any* correct repo appear in the top-k? (Did we find it at all?)
- **Recall @k** — of the repos that *should* appear, how many did? (Did we find *all* of them?)
- **MRR** (mean reciprocal rank) — how *high up* was the first correct result? Rank 1 scores 1.0,
  rank 2 scores 0.5, and so on. (Rewards good ordering, which is what reranking improves.)
- **Precision @k** — of the k returned, how many were relevant? (Punishes noise.)

Average each across all questions and you get a scorecard for the *current* retriever.

## Why the harness is generic

The harness doesn't know or care *how* retrieval works — it just runs the current `search` and
scores the repos that come back. So when you later add reranking, or change chunk size, or turn
on hybrid search, you **re-run the same harness** and compare scorecards. It validates whatever
retrieval configuration exists at the time. The test set is the fixed yardstick; the retriever
is the thing being measured.

## How this guards against overfitting

Put varied question shapes in the test set — exact names, fuzzy references, multi-repo,
no-repo. If a change only helps the one question you were staring at and the aggregate barely
moves, you overfit. If the aggregate rises across shapes, the improvement is real. That's the
empirical answer to "production-grade or fit-to-demo?" from [06](06-metadata-filtering.md).

## The levers you'll be measuring

Chunk size/overlap · query-vs-passage embedding · top-k · metadata filtering · hybrid
(dense + keyword) search · reranking. Each is a knob; the eval tells you which ones actually pay.

## Where this lives

`../evaluation.py` (metrics + runner), `eval/gold.yaml` (the test questions), and the `eval`
command in `../cli.py`.
