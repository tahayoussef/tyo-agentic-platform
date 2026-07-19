"""Tests for the lexical knowledge-base searcher (sections, tokenizing, ranking)."""

from __future__ import annotations

from pathlib import Path

from orchestration_agent.knowledge_base import KeywordSearcher, load_sections

_GOBEKLI = """# gobekli-tepe

An analytics platform.

## Architecture

Uses dbt with a bronze, silver, and gold medallion architecture on BigQuery.
"""

_CARTHAGE = """# carthage-architecture-center

## Purpose

A reference library of reusable Terraform modules for Google Cloud.
"""


def _write_kb(tmp_path: Path) -> Path:
    (tmp_path / "gobekli-tepe.md").write_text(_GOBEKLI, encoding="utf-8")
    (tmp_path / "carthage-architecture-center.md").write_text(_CARTHAGE, encoding="utf-8")
    return tmp_path


def test_load_sections_splits_on_headings(tmp_path: Path) -> None:
    sections = load_sections(_write_kb(tmp_path))

    headings = {(s.repo, s.heading) for s in sections}
    assert ("gobekli-tepe", "Architecture") in headings
    assert ("carthage-architecture-center", "Purpose") in headings


def test_sections_carry_repo_from_filename(tmp_path: Path) -> None:
    sections = load_sections(_write_kb(tmp_path))

    assert all(s.repo in {"gobekli-tepe", "carthage-architecture-center"} for s in sections)


def test_empty_sections_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("# only a heading\n\n## and another\n", encoding="utf-8")

    assert load_sections(tmp_path) == []


def test_search_ranks_matching_content_first(tmp_path: Path) -> None:
    searcher = KeywordSearcher(load_sections(_write_kb(tmp_path)), top_k=2)

    hits = searcher.search("which project uses dbt and a medallion architecture?")

    assert hits
    assert hits[0].section.repo == "gobekli-tepe"


def test_repo_name_mention_boosts_that_repo(tmp_path: Path) -> None:
    searcher = KeywordSearcher(load_sections(_write_kb(tmp_path)), top_k=1)

    hits = searcher.search("tell me about carthage")

    assert hits
    assert hits[0].section.repo == "carthage-architecture-center"


def test_search_respects_top_k(tmp_path: Path) -> None:
    searcher = KeywordSearcher(load_sections(_write_kb(tmp_path)), top_k=10)

    assert len(searcher.search("architecture", top_k=1)) == 1


def test_no_token_overlap_means_no_hits(tmp_path: Path) -> None:
    searcher = KeywordSearcher(load_sections(_write_kb(tmp_path)), top_k=3)

    assert searcher.search("zzzz qqqq") == []


def test_stopword_only_query_returns_nothing(tmp_path: Path) -> None:
    searcher = KeywordSearcher(load_sections(_write_kb(tmp_path)), top_k=3)

    assert searcher.search("what does the") == []
