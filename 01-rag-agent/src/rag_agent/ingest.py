"""Ingestion pipeline: load Markdown docs -> chunk -> embed -> upsert into Qdrant.

This is the "static" side of the RAG system — an offline indexing step you run whenever the
knowledge base changes. Each function is small and independently testable; ``run_ingest``
wires them together. Embeddings and the Qdrant client are injected so the whole pipeline can
run in-memory (no NVIDIA, no network) under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.logging import get_logger
from rag_agent.vector_store import (
    build_vector_store,
    ensure_collection,
    probe_embedding_dimension,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """Summary of an ingestion run."""

    files: int
    chunks: int
    collection: str
    dimension: int


def load_documents(knowledge_base_dir: Path) -> list[Document]:
    """Load every ``*.md`` file in a directory into a LangChain Document.

    Each document carries ``source`` (file name) and ``repo`` (file stem) metadata so
    retrieved chunks can be attributed back to a repository.
    """
    files = sorted(knowledge_base_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {knowledge_base_dir!s}")

    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(page_content=text, metadata={"source": path.name, "repo": path.stem})
        )
    logger.debug("ingest.loaded", files=len(documents))
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into overlapping chunks, preserving each chunk's metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.debug("ingest.split", chunks=len(chunks))
    return chunks


def run_ingest(
    settings: Settings,
    *,
    embeddings: Embeddings,
    client: QdrantClient,
    recreate: bool = False,
) -> IngestReport:
    """Run the full ingestion pipeline and return a summary report."""
    documents = load_documents(settings.knowledge_base_dir)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    dimension = probe_embedding_dimension(embeddings)
    ensure_collection(client, settings.collection_name, dimension, recreate=recreate)

    store = build_vector_store(client, settings.collection_name, embeddings)
    store.add_documents(chunks)

    logger.info(
        "ingest.complete",
        files=len(documents),
        chunks=len(chunks),
        collection=settings.collection_name,
        dimension=dimension,
    )
    return IngestReport(
        files=len(documents),
        chunks=len(chunks),
        collection=settings.collection_name,
        dimension=dimension,
    )
