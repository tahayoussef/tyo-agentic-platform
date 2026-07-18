# 09 · Reranking — a second, smarter opinion

## The problem

Vector search ranks chunks by **similarity** — but similarity isn't quite the same as
**relevance**. An embedding is a *lossy summary* of a whole chunk squeezed into one vector, so
the ordering it produces is approximate. The genuinely best chunk sometimes lands at position
5 while a merely-related one sits at position 1.

We can't fix this by "embedding harder," because the embedding was computed **before** we ever
saw the query — the query and the chunk were turned into vectors *separately*, then compared.
The model never got to look at them *together*.

## Bi-encoder vs. cross-encoder (the key distinction)

- **Bi-encoder** (what embeddings do): encode the query and each chunk *independently*, then
  compare the vectors. **Fast** — you embed the corpus once, and each search is just vector
  math — but approximate, because the model judged each side in isolation.
- **Cross-encoder** (what a reranker is): feed the query **and** a chunk into the model
  *together*, and it outputs a single relevance score having seen how they interact. **Far more
  accurate** — but **slow**, because it's a full model pass for every (query, chunk) pair. You
  could never run it over a million chunks per query.

## The fix: two-stage retrieval

Combine their strengths:

1. **Stage 1 — retrieve wide (bi-encoder).** Use fast vector search to fetch a *generous*
   candidate set (we fetch `RERANK_FETCH_K`, e.g. 20) — optimized for **recall** ("get the
   right chunk *somewhere* in the candidates").
2. **Stage 2 — rerank narrow (cross-encoder).** Run the reranker over just those ~20
   candidates, re-score each, reorder, and keep the top-k — optimized for **precision** ("put
   the best ones first").

Cheap-and-wide narrows the field; expensive-and-accurate picks the winners. That's the whole
idea, and it's one of the highest-leverage quality upgrades in real RAG.

## What we built

`search(..., reranker=...)` fetches `rerank_fetch_k` candidates, hands them to an
`NVIDIARerank` cross-encoder, and returns its top-k. It's opt-in: `USE_RERANKER=true`, with
`RERANKER_MODEL` and `RERANK_FETCH_K` configurable.

## The trade-off

An extra model call per query, over `fetch_k` documents — real **latency and cost**. You pay
it when precision matters. And a lesson from *our* project: the eval showed our 5-doc corpus is
too easy for reranking to help (there's no bad ordering to fix). That doesn't make reranking
wrong — it makes our test corpus small. On a real corpus, this is often the single biggest win.

## Where this lives

`../retriever.py` (`_rerank`, the `reranker` argument to `search`) and `build_reranker` in
`../embeddings.py`. It composes with hybrid search ([10](10-hybrid-search.md)): hybrid widens
the candidates, reranking sharpens the final order.
