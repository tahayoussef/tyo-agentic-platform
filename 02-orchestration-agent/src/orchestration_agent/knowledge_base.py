"""A deliberately simple, lexical search over the local Markdown knowledge base.

Retrieval quality was 01-rag-agent's lesson (embeddings, Qdrant, hybrid, reranking); this
project's lesson is orchestration. So the docs specialist gets the simplest searcher that
works — split files into heading-delimited sections and rank them by keyword overlap — and
the interesting engineering happens a level up, in the router and the supervisor. Swapping
this for 01's vector retrieval would be a drop-in change behind the same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from orchestration_agent.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "this",
        "that",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "does",
        "have",
        "has",
        "from",
        "into",
        "their",
        "there",
        "them",
        "they",
        "its",
        "can",
        "could",
        "should",
        "would",
        "will",
        "been",
        "being",
        "all",
        "any",
        "you",
        "your",
        "our",
        "out",
        "not",
        "use",
        "used",
        "using",
        "about",
    }
)


@dataclass(frozen=True)
class DocSection:
    """One heading-delimited slice of a Markdown document."""

    repo: str  # the file stem, which by convention is the repository name
    heading: str
    text: str


@dataclass(frozen=True)
class DocHit:
    """A section matched by a search, with its lexical relevance score."""

    section: DocSection
    score: float


class DocsSearcher(Protocol):
    """The capability the docs tool depends on (structural interface)."""

    def search(self, query: str, *, top_k: int | None = None) -> list[DocHit]: ...


def _tokenize(text: str) -> frozenset[str]:
    """Lowercased word tokens, minus stopwords and very short words."""
    return frozenset(
        word for word in _WORD_RE.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS
    )


def _split_markdown(repo: str, content: str) -> list[DocSection]:
    """Split one document into sections at Markdown headings; skip empty sections."""
    sections: list[DocSection] = []
    heading = repo
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(DocSection(repo=repo, heading=heading, text=text))
        buffer.clear()

    for line in content.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            heading = line.strip().lstrip("#").strip() or repo
        else:
            buffer.append(line)
    flush()
    return sections


def load_sections(directory: Path) -> list[DocSection]:
    """Split every ``*.md`` file in ``directory`` into heading-delimited sections."""
    sections: list[DocSection] = []
    for path in sorted(directory.glob("*.md")):
        sections.extend(_split_markdown(path.stem, path.read_text(encoding="utf-8")))
    logger.debug("knowledge_base.loaded", directory=str(directory), sections=len(sections))
    return sections


class KeywordSearcher:
    """Rank sections by keyword overlap with the query.

    Scoring: one point per query token found in the section (heading included), plus a
    weighted bonus per query token matching the repository name — so "tell me about
    gobekli tepe" strongly prefers that repo's sections even if the words are common.
    """

    _REPO_NAME_WEIGHT = 2.0

    def __init__(self, sections: list[DocSection], *, top_k: int = 3) -> None:
        self._top_k = top_k
        self._index = [
            (
                section,
                _tokenize(section.text) | _tokenize(section.heading),
                _tokenize(section.repo.replace("-", " ")),
            )
            for section in sections
        ]

    def search(self, query: str, *, top_k: int | None = None) -> list[DocHit]:
        """Return the best-matching sections for ``query``, highest score first."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        hits = []
        for section, text_tokens, repo_tokens in self._index:
            score = float(len(query_tokens & text_tokens))
            score += self._REPO_NAME_WEIGHT * len(query_tokens & repo_tokens)
            if score > 0:
                hits.append(DocHit(section=section, score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: top_k or self._top_k]


def build_searcher(directory: Path, *, top_k: int) -> KeywordSearcher:
    """Load the knowledge base from disk and index it for keyword search."""
    return KeywordSearcher(load_sections(directory), top_k=top_k)
