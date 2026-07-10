# 03 · The vector database (Qdrant)

## Why you need one

Once every chunk is a vector, answering a question means: embed the question, then find the
chunks whose vectors are **closest**. With five documents you could compare against all of
them by brute force. With five million, you can't — comparing one-by-one would be far too
slow.

A **vector database** exists to solve exactly this: store huge numbers of vectors and find the
nearest ones **fast**, using clever indexes (approximate-nearest-neighbor search). We use
**Qdrant**, run as a container via `docker compose`.

## The pieces to know

- **Collection** — think of it as a table dedicated to one set of vectors. Ours holds the repo
  chunks.
- **Vector size** — a collection is created with a fixed vector length, and it **must equal
  your embedding model's dimension** (1024 for us). Mismatch this and inserts fail — it's the
  classic first-day RAG bug, which is why the code *probes* the model's dimension instead of
  hardcoding it.
- **Distance metric** — *how* "closeness" is measured. We use **cosine**, which suits these
  retrieval embeddings (it compares direction, i.e. meaning, not magnitude).
- **Payload** — alongside each vector, Qdrant stores the original chunk text and its metadata
  (source, repo). So a search returns not just "the nearest vector" but the actual text and
  where it came from.

## Why hosting it in Docker is a feature, not a chore

Running Qdrant as a real service (rather than an in-memory toy) means:

- the index persists between runs (ingest once, query many times), and
- it has a **web dashboard** (`http://localhost:6333/dashboard`) where you can literally *see*
  your collection, the points, their vectors, and their payloads. That's the moment embeddings
  stop being abstract.

## Where this lives

`../vector_store.py` — creating/verifying the collection (with the right size and distance),
and building the object the rest of the code searches through.
