# 10 · Hybrid search — meaning *and* exact words

## The blind spot in pure semantic search

Dense embedding search ([01](01-embeddings.md)) is brilliant at **meaning**: it matches
"medallion architecture" to "bronze, silver, gold layers" even with no shared words. But that
same strength is a weakness for **exact terms**. Rare, out-of-vocabulary tokens — a model id
like `nv-embedqa-e5-v5`, a function name, an error code, a product SKU, a person's surname —
get blurred into the semantic average and can rank *low*, because the embedding doesn't have a
crisp notion of "this exact string appears here."

Old-school **keyword search** (BM25) is the mirror image: superb at exact-term matching,
useless at synonyms and meaning. Each is strong exactly where the other is weak.

## The fix: run both, fuse the results

**Hybrid search** does dense (semantic) *and* sparse (keyword) retrieval, then merges the two
ranked lists. You get semantic recall **and** exact-match precision.

### Dense vs. sparse vectors

- **Dense vector:** ~1024 floating-point numbers, *all* filled in — a compressed
  representation of *meaning*.
- **Sparse vector:** mostly zeros with a handful of non-zero entries, each keyed to a specific
  word/token and weighted by importance — essentially a smart, weighted bag-of-words. **BM25**
  is the classic recipe for those weights (it rewards rare, discriminating terms). We compute
  these locally with FastEmbed.

So every chunk gets stored **twice**: once as a dense "meaning" vector and once as a sparse
"keywords" vector.

### Fusion: combining two rankings

Dense scores and sparse scores live on totally different scales, so you can't just add them. The
standard trick is **Reciprocal Rank Fusion (RRF)**: ignore the raw scores and use each list's
*ranks* — a document's fused score adds up `1 / (k + rank)` from each list. It's scale-free and
needs no tuning: a document that ranks well in *either* list bubbles up, and one that ranks well
in *both* wins.

## What we built

A collection created with **two** named vectors — a dense one and a sparse one — plus
`FastEmbedSparse` (BM25) for the sparse side, and Qdrant's `RetrievalMode.HYBRID`, which runs
both searches and fuses them with RRF. Opt-in via `RETRIEVAL_MODE=hybrid`. (Switching modes
changes the collection schema, so it needs a `--recreate` re-ingest.)

## The trade-off

More storage (two vectors per chunk), a sparse model to run at ingest and query time, and a
more complex collection. Worth it for **technical corpora** — code, docs full of identifiers,
names, acronyms — where exact terms carry meaning that embeddings smear over.

## Where this lives

`../vector_store.py` (dual-vector collection schema, `build_vector_store` hybrid branch) and
`build_sparse_embeddings` in `../embeddings.py`. It **stacks** with reranking
([09](09-reranking.md)): hybrid improves *which* candidates you fetch, reranking improves how
they're *ordered*.
