"""Qdrant client, collection lifecycle, and vector-store construction.

We manage the collection explicitly (rather than letting the framework auto-create it) so
the mechanics are visible: a collection has a fixed **vector size** — which must equal the
embedding model's output dimension — and a **distance metric** (cosine for these models).
Mismatching the dimension is the classic first-day RAG bug, so we probe it from the live
embedding model instead of hardcoding it.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from rag_agent.config import Settings
from rag_agent.logging import get_logger

logger = get_logger(__name__)

# Payload fields we index so the retriever can pre-filter efficiently (e.g. by repository).
_INDEXED_PAYLOAD_FIELDS = ("metadata.repo",)


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client from settings."""
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


def probe_embedding_dimension(embeddings: Embeddings) -> int:
    """Return the embedding model's output dimension by embedding a probe string."""
    return len(embeddings.embed_query("dimension probe"))


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    *,
    recreate: bool = False,
    distance: Distance = Distance.COSINE,
) -> None:
    """Ensure a collection exists with the given vector size and distance metric.

    Args:
        client: Qdrant client.
        collection_name: Collection to create/verify.
        dimension: Vector size — must match the embedding model's output dimension.
        recreate: If true, drop an existing collection first (destroys its data).
        distance: Similarity metric (cosine suits normalized retrieval embeddings).
    """
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        logger.info("qdrant.collection.delete", collection=collection_name)
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        logger.info("qdrant.collection.create", collection=collection_name, dim=dimension)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=distance),
        )
        # Index the payload fields we filter on, so pre-filtering is efficient.
        for field in _INDEXED_PAYLOAD_FIELDS:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )


def build_vector_store(
    client: QdrantClient,
    collection_name: str,
    embeddings: Embeddings,
) -> QdrantVectorStore:
    """Build a LangChain vector store bound to an existing Qdrant collection."""
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
