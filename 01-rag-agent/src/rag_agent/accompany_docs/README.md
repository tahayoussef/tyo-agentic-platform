# RAG concepts — read in order

A junior-friendly tour of **what** we built for the RAG agent and **why** — the ideas, not
the syntax. Each file is one concept; read them in number order.

| # | Concept | The one-line idea |
|---|---------|-------------------|
| [00](00-what-is-rag.md) | What RAG is | Give the model relevant text at answer time so it stops guessing. |
| [01](01-embeddings.md) | Embeddings | Turn text into numbers that capture *meaning*, so you can search by meaning. |
| [02](02-chunking.md) | Chunking | Split documents into small passages so retrieval is precise. |
| [03](03-vector-database.md) | The vector database | A store built to find the nearest vectors fast (Qdrant). |
| [04](04-ingestion.md) | Ingestion | The offline pipeline that fills the database: load → chunk → embed → store. |
| [05](05-retrieval.md) | Retrieval | Turn a question into a vector and pull the closest chunks. |
| [06](06-metadata-filtering.md) | Metadata & filtering | Narrow the search *before* similarity — precision, not just closeness. |
| [07](07-agentic-rag.md) | Agentic RAG | Let the agent *choose* to search, and reconcile the KB with live data. |
| [08](08-evaluating-retrieval.md) | Evaluating retrieval | Measure quality with numbers so "better" isn't a guess. |
| [09](09-reranking.md) | Reranking | Fetch many, then re-score with a smarter cross-encoder. |
| [10](10-hybrid-search.md) | Hybrid search | Combine semantic meaning with exact keyword matching. |

The code these describe lives one directory up (`../embeddings.py`, `../ingest.py`,
`../retriever.py`, `../vector_store.py`, `../tools.py`).
