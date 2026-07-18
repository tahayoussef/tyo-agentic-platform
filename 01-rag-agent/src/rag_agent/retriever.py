"""Retrieval: embed a query and find the most relevant chunks in Qdrant.

Three orthogonal, optional features layer on top of the basic search:
- **repo pre-filtering** — restrict the search to a single repository's chunks;
- **hybrid search** — pass a sparse embedding to fuse dense (semantic) + sparse (keyword);
- **reranking** — pass a cross-encoder to fetch many candidates, then re-score to the top-k.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings
from qdrant_client import QdrantClient, models

from rag_agent.config import Settings
from rag_agent.vector_store import build_vector_store


class Reranker(Protocol):
    """A cross-encoder that reorders documents by their relevance to a query."""

    def compress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]: ...


def _repo_filter(repo: str | None) -> models.Filter | None:
    """Build a Qdrant payload filter restricting results to a single repository."""
    if not repo:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="metadata.repo", match=models.MatchValue(value=repo))]
    )


def _rerank(
    reranker: Reranker,
    query: str,
    pairs: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """Re-score candidate chunks with the cross-encoder and keep the best ``top_k``."""
    docs = [doc for doc, _ in pairs]
    if not docs:
        return []
    reranked = reranker.compress_documents(documents=docs, query=query)
    scored: list[tuple[Document, float]] = [
        (doc, float(doc.metadata.get("relevance_score", 0.0))) for doc in reranked
    ]
    return scored[:top_k]


def search(
    settings: Settings,
    query: str,
    *,
    embeddings: Embeddings,
    client: QdrantClient,
    top_k: int | None = None,
    repo: str | None = None,
    sparse_embedding: SparseEmbeddings | None = None,
    reranker: Reranker | None = None,
) -> list[tuple[Document, float]]:
    """Return the most relevant chunks to ``query`` with their scores.

    When ``reranker`` is set we fetch ``rerank_fetch_k`` candidates first, then let the
    reranker choose the final ``top_k`` — the standard two-stage retrieval pattern. When
    ``sparse_embedding`` is set the underlying store runs hybrid (dense + sparse) search.
    """
    store = build_vector_store(
        client, settings.collection_name, embeddings, sparse_embedding=sparse_embedding
    )
    k = top_k or settings.top_k
    fetch_k = settings.rerank_fetch_k if reranker is not None else k
    query_filter = _repo_filter(repo)

    results: list[tuple[Document, float]] = store.similarity_search_with_score(
        query, k=fetch_k, filter=query_filter
    )
    if not results and query_filter is not None:
        results = store.similarity_search_with_score(query, k=fetch_k)

    if reranker is not None:
        return _rerank(reranker, query, results, k)
    return results[:k]
