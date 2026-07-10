"""Retrieval: embed a query and find the nearest chunks in Qdrant.

Supports optional **pre-filtering by repository**: when a question targets one repo, we
restrict the search to that repo's chunks so a fact buried in its docs isn't crowded out of
the top-k by unrelated repositories. This is the body of the ``search_knowledge_base`` tool
and also powers ``rag-agent search``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from rag_agent.config import Settings
from rag_agent.vector_store import build_vector_store


def _repo_filter(repo: str | None) -> models.Filter | None:
    """Build a Qdrant payload filter restricting results to a single repository."""
    if not repo:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="metadata.repo", match=models.MatchValue(value=repo))]
    )


def search(
    settings: Settings,
    query: str,
    *,
    embeddings: Embeddings,
    client: QdrantClient,
    top_k: int | None = None,
    repo: str | None = None,
) -> list[tuple[Document, float]]:
    """Return the ``top_k`` most similar chunks to ``query`` with their scores.

    When ``repo`` is given, the search is pre-filtered to that repository's chunks. If the
    filter matches nothing (e.g. an unknown repo slug), we fall back to an unfiltered search
    rather than returning nothing.
    """
    store = build_vector_store(client, settings.collection_name, embeddings)
    k = top_k or settings.top_k
    query_filter = _repo_filter(repo)

    results: list[tuple[Document, float]] = store.similarity_search_with_score(
        query, k=k, filter=query_filter
    )
    if not results and query_filter is not None:
        results = store.similarity_search_with_score(query, k=k)
    return results
