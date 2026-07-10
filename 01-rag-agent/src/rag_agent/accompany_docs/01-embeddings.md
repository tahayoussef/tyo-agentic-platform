# 01 · Embeddings — meaning as numbers

## The core idea

An **embedding** is a way to turn a piece of text into a list of numbers (a **vector**) that
captures its *meaning*. The magic property: **texts that mean similar things get vectors that
are close together**, and unrelated texts get vectors that are far apart.

So "a platform for transforming raw data into analytics" and "an ELT pipeline with
bronze/silver/gold layers" land near each other — even though they share almost no words.
That's the superpower embeddings give you: **search by meaning, not by keyword.** A keyword
search for "medallion architecture" would miss a doc that says "bronze, silver, gold layers";
an embedding search finds it, because the *meanings* are close.

## A mental picture

Imagine a giant map where every possible sentence is a point. Sentences about dbt sit in one
neighborhood; sentences about Terraform sit in another. An embedding model is the thing that
places text on that map. "Similarity" is just distance on the map.

(The map isn't 2-D — our vectors have **1024** dimensions. You can't picture 1024-D space, but
the intuition "close = similar" still holds. The dimension count is simply how long each
vector is; it's a property of the model we chose.)

## Why a *retrieval-tuned* model matters

Not all embedding models are equal. We use one built specifically for search
(`nvidia/nv-embedqa-e5-v5`). These models are trained so that a **question** and the
**passage that answers it** land close together — which is exactly what retrieval needs.

That leads to one subtle but important point you'll meet again in [05](05-retrieval.md): these
models embed a *question* slightly differently from a *document*. Handling that split
correctly is one of the biggest levers on accuracy — and it's why the code always goes
through the model's "embed a document" vs "embed a query" methods rather than treating them
the same.

## Where this lives

`../embeddings.py` — a tiny factory that builds the embedding model from config. Everything
else in the pipeline just calls it to turn text (or a query) into vectors.
