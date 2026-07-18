"""Tests for hybrid search and reranking.

Run offline with fakes: a deterministic fake sparse embedding (no model download) and a fake
reranker (no NVIDIA call). They validate our *wiring* — collection schema, the two-stage
fetch→rerank flow — not the quality of real models.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import SparseEmbeddings
from langchain_qdrant.sparse_embeddings import SparseVector
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.ingest import run_ingest
from rag_agent.retriever import _rerank, search

EMBED_DIM = 64


class FakeSparse(SparseEmbeddings):
    """A deterministic sparse embedding (hashes words to indices) — no model download."""

    def embed_documents(self, texts: list[str]) -> list[Any]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> Any:
        return self._vec(text)

    def _vec(self, text: str) -> Any:
        indices = sorted({abs(hash(word)) % 500 for word in text.lower().split()}) or [0]
        return SparseVector(indices=indices, values=[1.0] * len(indices))


class FakeReranker:
    """A stand-in cross-encoder: reverses the candidate order and tags relevance scores."""

    def compress_documents(self, documents: Sequence[Document], query: str) -> list[Document]:
        ranked = list(reversed(list(documents)))
        for i, doc in enumerate(ranked):
            doc.metadata["relevance_score"] = round(1.0 - i * 0.01, 4)
        return ranked


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


def _write_kb(tmp_path: Path) -> None:
    (tmp_path / "carthage-architecture-center.md").write_text(
        "# carthage\n" + "terraform gcp infrastructure modules. " * 30, encoding="utf-8"
    )
    (tmp_path / "gobekli-tepe.md").write_text(
        "# gobekli\n" + "dbt medallion bronze silver gold platform. " * 30, encoding="utf-8"
    )


# --- reranking (pure) ---------------------------------------------------------


def test_rerank_reorders_and_scores() -> None:
    doc_a = Document(page_content="a", metadata={"repo": "x"})
    doc_b = Document(page_content="b", metadata={"repo": "y"})
    pairs = [(doc_a, 0.9), (doc_b, 0.1)]

    reranked = _rerank(FakeReranker(), "query", pairs, top_k=2)

    assert [doc.page_content for doc, _ in reranked] == ["b", "a"]  # reversed by fake
    assert reranked[0][0].metadata["relevance_score"] == 1.0


def test_rerank_empty_input() -> None:
    assert _rerank(FakeReranker(), "query", [], top_k=3) == []


def test_search_applies_reranker(tmp_path: Path) -> None:
    _write_kb(tmp_path)
    settings = _settings(tmp_path)
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")
    run_ingest(settings, embeddings=embeddings, client=client, recreate=True)

    results = search(
        settings,
        "terraform",
        embeddings=embeddings,
        client=client,
        top_k=2,
        reranker=FakeReranker(),
    )

    assert 0 < len(results) <= 2
    # The reranker stamps a relevance_score onto every doc it returns.
    assert all("relevance_score" in doc.metadata for doc, _ in results)


# --- hybrid search ------------------------------------------------------------


def test_hybrid_ingest_and_search(tmp_path: Path) -> None:
    _write_kb(tmp_path)
    settings = _settings(tmp_path, retrieval_mode="hybrid")
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    sparse = FakeSparse()
    client = QdrantClient(location=":memory:")

    run_ingest(
        settings, embeddings=embeddings, client=client, sparse_embedding=sparse, recreate=True
    )

    results = search(
        settings,
        "terraform modules",
        embeddings=embeddings,
        client=client,
        sparse_embedding=sparse,
        top_k=3,
    )

    assert results
    assert all(
        doc.metadata.get("repo") in {"carthage-architecture-center", "gobekli-tepe"}
        for doc, _ in results
    )
