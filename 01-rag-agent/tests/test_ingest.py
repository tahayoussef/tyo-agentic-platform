"""Tests for the ingestion pipeline.

These run fully offline: a deterministic fake embedding model (no NVIDIA) and an in-memory
Qdrant instance (no docker/network). That is possible precisely because ``run_ingest``
takes the embeddings and client as injected dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.ingest import IngestReport, load_documents, run_ingest, split_documents
from rag_agent.vector_store import build_vector_store

EMBED_DIM = 64


def _write_kb(directory: Path) -> None:
    (directory / "alpha.md").write_text("# Alpha\n" + "alpha content. " * 100, encoding="utf-8")
    (directory / "beta.md").write_text("# Beta\n" + "beta content. " * 100, encoding="utf-8")


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "nvidia_api_key": "nvapi-test",
        "knowledge_base_dir": tmp_path,
        "collection_name": "test_repos",
        "chunk_size": 200,
        "chunk_overlap": 20,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type, call-arg]


def test_load_documents_attaches_metadata(tmp_path: Path) -> None:
    _write_kb(tmp_path)

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    assert {d.metadata["repo"] for d in docs} == {"alpha", "beta"}
    assert docs[0].metadata["source"].endswith(".md")


def test_load_documents_errors_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_split_documents_chunks_and_keeps_metadata(tmp_path: Path) -> None:
    _write_kb(tmp_path)
    docs = load_documents(tmp_path)

    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > len(docs)
    assert all(c.metadata.get("repo") in {"alpha", "beta"} for c in chunks)


def test_run_ingest_end_to_end(tmp_path: Path) -> None:
    _write_kb(tmp_path)
    settings = _settings(tmp_path)
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")

    report = run_ingest(settings, embeddings=embeddings, client=client, recreate=True)

    assert isinstance(report, IngestReport)
    assert report.files == 2
    assert report.chunks >= 2
    assert report.dimension == EMBED_DIM
    assert report.collection == "test_repos"

    # The indexed chunks are retrievable.
    store = build_vector_store(client, settings.collection_name, embeddings)
    results = store.similarity_search("alpha content", k=3)
    assert results
    assert any(r.metadata.get("repo") == "alpha" for r in results)


def test_run_ingest_recreate_is_idempotent(tmp_path: Path) -> None:
    _write_kb(tmp_path)
    settings = _settings(tmp_path)
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")

    first = run_ingest(settings, embeddings=embeddings, client=client, recreate=True)
    second = run_ingest(settings, embeddings=embeddings, client=client, recreate=True)

    # Recreating wipes the collection, so counts match rather than doubling.
    count = client.count(settings.collection_name).count
    assert count == first.chunks == second.chunks
