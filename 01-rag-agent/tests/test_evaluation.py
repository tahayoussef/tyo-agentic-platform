"""Tests for the retrieval evaluation harness.

The metric functions are pure and deterministic, so they're tested with hand-picked inputs.
The end-to-end runner is exercised against in-memory Qdrant with fake embeddings — that
validates the *plumbing and metric ranges*, not retrieval quality (which needs real models).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.evaluation import (
    EvalCase,
    compute_metrics,
    evaluate,
    load_gold,
)
from rag_agent.ingest import run_ingest

EMBED_DIM = 64


# --- pure metric tests --------------------------------------------------------


def test_perfect_single_repo() -> None:
    m = compute_metrics(["carthage", "gobekli", "machu"], frozenset({"carthage"}), k=3)
    assert m.hit == 1.0
    assert m.recall == 1.0
    assert m.reciprocal_rank == 1.0  # relevant at rank 1
    assert round(m.precision, 4) == round(1 / 3, 4)


def test_relevant_lower_down_hurts_rank() -> None:
    m = compute_metrics(["gobekli", "machu", "carthage"], frozenset({"carthage"}), k=3)
    assert m.hit == 1.0
    assert m.reciprocal_rank == 1 / 3  # first relevant at rank 3


def test_miss_scores_zero() -> None:
    m = compute_metrics(["gobekli", "machu"], frozenset({"carthage"}), k=3)
    assert m.hit == 0.0
    assert m.recall == 0.0
    assert m.reciprocal_rank == 0.0
    assert m.precision == 0.0


def test_partial_multi_repo_recall() -> None:
    m = compute_metrics(["carthage", "unrelated"], frozenset({"carthage", "machu"}), k=2)
    assert m.hit == 1.0
    assert m.recall == 0.5  # found 1 of 2 relevant
    assert m.precision == 0.5


def test_k_truncates_before_scoring() -> None:
    m = compute_metrics(["gobekli", "carthage"], frozenset({"carthage"}), k=1)
    assert m.hit == 0.0  # carthage is at rank 2, outside k=1


# --- gold loading -------------------------------------------------------------


def test_load_gold(tmp_path: Path) -> None:
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        "- query: what language is carthage?\n"
        "  relevant_repos: [carthage-architecture-center]\n"
        "  repo_filter: carthage-architecture-center\n"
        "- query: which use dbt?\n"
        "  relevant_repos: [gobekli-tepe]\n",
        encoding="utf-8",
    )

    cases = load_gold(gold)

    assert len(cases) == 2
    assert cases[0].relevant_repos == frozenset({"carthage-architecture-center"})
    assert cases[0].repo_filter == "carthage-architecture-center"
    assert cases[1].repo_filter is None


# --- end-to-end runner --------------------------------------------------------


def test_evaluate_produces_report_with_valid_ranges(tmp_path: Path) -> None:
    (tmp_path / "carthage-architecture-center.md").write_text(
        "# carthage\n" + "terraform gcp modules. " * 30, encoding="utf-8"
    )
    (tmp_path / "gobekli-tepe.md").write_text(
        "# gobekli\n" + "dbt medallion platform. " * 30, encoding="utf-8"
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

    cases = [
        EvalCase(
            query="terraform modules", relevant_repos=frozenset({"carthage-architecture-center"})
        ),
        EvalCase(
            query="dbt",
            relevant_repos=frozenset({"gobekli-tepe"}),
            repo_filter="gobekli-tepe",
        ),
    ]

    report = evaluate(settings, cases, embeddings=embeddings, client=client, top_k=3)

    assert len(report.results) == 2
    assert report.k == 3
    for value in (report.hit_rate, report.recall, report.mrr, report.precision):
        assert 0.0 <= value <= 1.0
    # The filtered case can only ever retrieve its own repo.
    filtered = report.results[1]
    assert all(repo == "gobekli-tepe" for repo in filtered.retrieved_repos)
