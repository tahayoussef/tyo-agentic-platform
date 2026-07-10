# 04 · Ingestion — filling the database

## What ingestion is

**Ingestion** is the offline job that populates the vector database. It's the "static" side of
RAG: you run it whenever your source documents change, and then it just sits there ready to be
searched. It's a straight pipeline of the ideas from the last three files:

```
load documents  →  split into chunks  →  embed each chunk  →  store in Qdrant
   (02)                (02)                   (01)                 (03)
```

Concretely, for our agent: read the Markdown files describing each repo, chunk them, turn each
chunk into a 1024-D vector, and upsert those vectors (plus their text and repo metadata) into
the Qdrant collection.

## "Offline" is the important word

Ingestion is deliberately separate from answering questions. You pay the cost of chunking and
embedding **once**, up front — not on every user question. At query time you only embed the
short question and look up neighbors, which is cheap and fast. This split (index offline, query
online) is what makes RAG practical.

## Getting it right

- **Dimension check** — the collection is created to match the embedding model's output size,
  so vectors actually fit.
- **Re-runnable** — you can wipe and rebuild the index (`--recreate`) so re-indexing after a
  doc change gives a clean result instead of duplicates.
- **Metadata travels with the chunk** — the repo/source tags added during chunking are stored
  in Qdrant's payload, ready for filtering and attribution later.

## How you run it

`rag-agent ingest --recreate` — then open the Qdrant dashboard to see the points appear. That
visual confirmation ("my documents are now vectors I can search") is the payoff of this step.

## Where this lives

`../ingest.py` orchestrates load → chunk → embed → upsert and returns a small report (files,
chunks, collection, dimension). The `ingest` command in `../cli.py` runs it.
