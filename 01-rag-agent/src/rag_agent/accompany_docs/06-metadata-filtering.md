# 06 · Metadata & filtering — precision beyond similarity

This is the technique we added most recently, and the one worth understanding deeply.

## The problem it solves

Similarity search answers "what's *closest in meaning*?" But sometimes closeness isn't enough.
We hit this exactly: asked *"what language is carthage written in?"*, the search returned
carthage's overview plus unrelated chunks from **other** repos — and the specific chunk that
held the answer never made the top-k. The one fact we needed was crowded out by
similar-looking noise from documents we didn't even care about.

## The fix: filter first, then rank by similarity

Every chunk carries metadata (its `repo`, from [02](02-chunking.md)). **Filtering** means:
*before* comparing vectors, throw away every chunk that doesn't match a condition — here,
`repo == "carthage-architecture-center"`. Now similarity search runs only over that one repo's
chunks, so the answer can't be drowned out by others. (To make this fast at scale, we also add
a small **index** on the `repo` field — the database's equivalent of an index in a book.)

## Why this is a *real* production technique — not a demo hack

Filtering is foundational, not a workaround. The clearest example: **multi-tenancy**. In a real
product with many customers, you filter every search by `customer_id` — not to be tidy, but as
a hard **security boundary**: you must never return one customer's data to another. Scoping by
document, by date, by access level — same mechanism. Our `repo` filter is structurally
identical to a tenant filter, which is about as production-grade as it gets.

## …but be honest about the part that *is* naive

Here's the nuance worth internalizing. Two things about *our* version are entry-level:

1. **Who decides the filter value?** Right now the agent has to produce the *exact* repo slug
   `carthage-architecture-center`. That worked because the question contained it verbatim and
   there are only five repos. In the real world people say "my Carthage project" or "the
   architecture repo," and with thousands of documents the model can't know the canonical name.
   The production answer is **entity resolution**: map a fuzzy reference to a canonical id via a
   lookup/alias table, and validate the filter against known values — don't trust the model to
   guess it.
2. **Filtering masked a weaker retriever.** It didn't *fix* the bad ranking that put a diagram
   above the answer; it just shrank the search space so the bad ranking couldn't hurt. For
   questions where you can't scope to one entity, that weakness is still there. A fuller system
   also improves the chunks and adds reranking so retrieval is precise *without* leaning on the
   filter.

## The takeaway

Metadata filtering: keep it, it's real and load-bearing. Just remember it's **one lever among
several**, and that "it works on the question I tested" isn't proof it generalizes — which is
precisely what evaluation ([08](08-evaluating-retrieval.md)) is for.

## Where this lives

`../retriever.py` (`_repo_filter`, and the `repo` argument to `search`), the `repo` field on
the `search_knowledge_base` tool in `../tools.py`, and the payload index in `../vector_store.py`.
