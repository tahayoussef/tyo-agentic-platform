"""Tests for retrieval, including repository pre-filtering.

Runs offline against in-memory Qdrant with deterministic fake embeddings — so these assert
the *filtering* behavior (which is exact), not semantic ranking (which needs real models).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.ingest import run_ingest
from rag_agent.retriever import search

EMBED_DIM = 64


def _ingest(tmp_path: Path) -> tuple[Settings, DeterministicFakeEmbedding, QdrantClient]:
    (tmp_path / "carthage-architecture-center.md").write_text(
        "# carthage\n" + "terraform gcp modules. primary language python. " * 30,
        encoding="utf-8",
    )
    (tmp_path / "gobekli-tepe.md").write_text(
        "# gobekli\n" + "dbt medallion bronze silver gold platform. " * 30, encoding="utf-8"
    )
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        knowledge_base_dir=tmp_path,
        collection_name="test_repos",
        chunk_size=200,
        chunk_overlap=20,
    )
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")
    run_ingest(settings, embeddings=embeddings, client=client, recreate=True)
    return settings, embeddings, client


def test_repo_filter_restricts_results_to_that_repo(tmp_path: Path) -> None:
    settings, embeddings, client = _ingest(tmp_path)

    results = search(
        settings,
        "language",
        embeddings=embeddings,
        client=client,
        top_k=10,
        repo="carthage-architecture-center",
    )

    assert results
    assert all(doc.metadata["repo"] == "carthage-architecture-center" for doc, _ in results)


def test_unfiltered_search_is_not_restricted(tmp_path: Path) -> None:
    settings, embeddings, client = _ingest(tmp_path)

    results = search(settings, "language", embeddings=embeddings, client=client, top_k=10)

    # With no filter, chunks from more than one repo can appear.
    assert {doc.metadata["repo"] for doc, _ in results} >= {"carthage-architecture-center"}


def test_unknown_repo_falls_back_to_unfiltered(tmp_path: Path) -> None:
    settings, embeddings, client = _ingest(tmp_path)

    results = search(
        settings,
        "language",
        embeddings=embeddings,
        client=client,
        top_k=10,
        repo="does-not-exist",
    )

    assert results  # fell back instead of returning nothing
