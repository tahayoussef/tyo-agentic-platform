"""Retrieval evaluation harness.

Deliberately **generic**: it runs whatever the current ``search`` does and scores the
repositories that come back against a fixed gold set. It does not know or care *how*
retrieval works — so when we later add reranking, hybrid search, or change chunking, we
re-run the same harness and compare scorecards.

Relevance is judged at the **repository level**, not the exact chunk. Chunk boundaries move
whenever chunk size changes, but "the answer lives in the carthage docs" stays true — so the
same gold set remains valid across those changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import yaml
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.retriever import Reranker, search


@dataclass(frozen=True)
class EvalCase:
    """One evaluation question and the repositories that *should* be retrieved for it."""

    query: str
    relevant_repos: frozenset[str]
    repo_filter: str | None = None  # if set, applies the metadata pre-filter for this query


@dataclass(frozen=True)
class CaseMetrics:
    """Per-question retrieval metrics (all in [0, 1])."""

    hit: float
    recall: float
    reciprocal_rank: float
    precision: float


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    retrieved_repos: tuple[str, ...]
    metrics: CaseMetrics


@dataclass(frozen=True)
class EvalReport:
    """A full run: per-case results plus aggregate (mean) metrics."""

    results: list[CaseResult]
    k: int

    @property
    def hit_rate(self) -> float:
        return _safe_mean([r.metrics.hit for r in self.results])

    @property
    def recall(self) -> float:
        return _safe_mean([r.metrics.recall for r in self.results])

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank — the average of per-case reciprocal ranks."""
        return _safe_mean([r.metrics.reciprocal_rank for r in self.results])

    @property
    def precision(self) -> float:
        return _safe_mean([r.metrics.precision for r in self.results])


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def compute_metrics(
    retrieved_repos: Sequence[str],
    relevant: frozenset[str],
    k: int,
) -> CaseMetrics:
    """Score one ranked list of retrieved repos against the relevant set.

    - hit: did any relevant repo appear in the top-k?
    - recall: fraction of the relevant repos that appeared in the top-k.
    - reciprocal_rank: 1 / (rank of the first relevant repo), else 0.
    - precision: fraction of the top-k that were relevant.
    """
    top = list(retrieved_repos[:k])
    relevant_in_top = [repo for repo in top if repo in relevant]

    hit = 1.0 if relevant_in_top else 0.0
    recall = len(set(relevant_in_top)) / len(relevant) if relevant else 0.0
    precision = len(relevant_in_top) / len(top) if top else 0.0

    reciprocal_rank = 0.0
    for rank, repo in enumerate(top, start=1):
        if repo in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return CaseMetrics(hit=hit, recall=recall, reciprocal_rank=reciprocal_rank, precision=precision)


def load_gold(path: Path) -> list[EvalCase]:
    """Load evaluation cases from a YAML file (a list of {query, relevant_repos, ...})."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Gold file {path} must contain a list of cases.")
    return [
        EvalCase(
            query=str(entry["query"]),
            relevant_repos=frozenset(entry["relevant_repos"]),
            repo_filter=entry.get("repo_filter"),
        )
        for entry in raw
    ]


def evaluate(
    settings: Settings,
    cases: Sequence[EvalCase],
    *,
    embeddings: Embeddings,
    client: QdrantClient,
    sparse_embedding: SparseEmbeddings | None = None,
    reranker: Reranker | None = None,
    top_k: int | None = None,
) -> EvalReport:
    """Run every case through the current retriever (incl. hybrid/rerank) and score it."""
    k = top_k or settings.top_k
    results: list[CaseResult] = []
    for case in cases:
        pairs = search(
            settings,
            case.query,
            embeddings=embeddings,
            client=client,
            top_k=k,
            repo=case.repo_filter,
            sparse_embedding=sparse_embedding,
            reranker=reranker,
        )
        retrieved_repos = tuple(str(doc.metadata.get("repo", "")) for doc, _ in pairs)
        metrics = compute_metrics(retrieved_repos, case.relevant_repos, k)
        results.append(CaseResult(case=case, retrieved_repos=retrieved_repos, metrics=metrics))
    return EvalReport(results=results, k=k)
