"""Retrieval: embed a query and find the nearest chunks in Qdrant.

This is the minimal "static search" primitive. In a later phase it becomes the body of the
``search_knowledge_base`` tool the agent calls; for now it also powers the ``rag-agent
search`` command so retrieval quality is inspectable on its own.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.vector_store import build_vector_store


def search(
    settings: Settings,
    query: str,
    *,
    embeddings: Embeddings,
    client: QdrantClient,
    top_k: int | None = None,
) -> list[tuple[Document, float]]:
    """Return the ``top_k`` most similar chunks to ``query`` with their scores."""
    store = build_vector_store(client, settings.collection_name, embeddings)
    return store.similarity_search_with_score(query, k=top_k or settings.top_k)
