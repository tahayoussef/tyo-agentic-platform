# 05 · Retrieval — searching by meaning

## The query-time flow

Everything so far was preparation. **Retrieval** is the moment it pays off. Given a question:

1. **Embed the question** into a vector (same model as ingestion).
2. Ask Qdrant for the **k nearest** chunk-vectors to it.
3. Return those chunks (their text + metadata + a similarity score).

That's it. The returned chunks are the "relevant snippets" from [00](00-what-is-rag.md) that
get handed to the model. `k` (how many to return, "top-k") is a knob: too few and you might
miss the answer; too many and you dilute the prompt with irrelevant text.

## The subtle part: questions and answers are embedded *differently*

Retrieval-tuned models are **asymmetric**. They embed a short question one way ("query" mode)
and a full passage another ("passage" mode), specifically so a question lands near the passage
that answers it. If you accidentally embed your query in "passage" mode, accuracy quietly
craters — the vectors just don't line up.

Our code avoids this by always using the model's dedicated "embed a query" method at search
time and "embed documents" method at ingest time. You don't set anything; you just have to
know *not* to bypass those methods. It's one of the highest-leverage, least-visible details in
all of RAG.

## Similarity isn't the whole story

Nearest-by-meaning is a great first pass, but it's not always enough. On a small corpus we
watched a query for one repo's language pull back an unrelated repo's diagram, ranked above the
chunk we actually needed. Pure similarity has no notion of "only look inside *this* project."
That gap is what the next file — filtering — fixes, and it's why real systems layer more
techniques on top of raw similarity.

## Where this lives

`../retriever.py` (`search`) embeds the query and runs the Qdrant search; the `search` command
in `../cli.py` lets you try queries by hand and see the scores.
