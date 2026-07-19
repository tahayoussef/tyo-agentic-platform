"""Tests for the evaluation harness: gold loading, router accuracy, delegation coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestration_agent.evaluation import (
    DelegationRecorder,
    evaluate_router,
    evaluate_supervisor,
    load_gold,
    specialists_consulted,
)
from orchestration_agent.router import RouteDecision

_GOLD_YAML = """
- question: "stars now?"
  routes: [github]
- question: "hello"
  routes: [general]
- question: "docs vs live?"
  routes: [github, docs]
  needs: [github, docs]
"""


def _write_gold(tmp_path: Path) -> Path:
    path = tmp_path / "gold.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    return path


class FixedClassifier:
    """Returns a pre-programmed route per question."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def classify(self, question: str) -> RouteDecision:
        return RouteDecision(
            route=self.answers[question],  # type: ignore[arg-type]
            confidence=1.0,
            reason="fixed",
        )


def test_load_gold_parses_routes_and_explicit_needs(tmp_path: Path) -> None:
    cases = load_gold(_write_gold(tmp_path))

    assert cases[2].routes == frozenset({"github", "docs"})
    assert cases[2].needs == frozenset({"github", "docs"})


def test_load_gold_defaults_needs_to_routes_minus_general(tmp_path: Path) -> None:
    cases = load_gold(_write_gold(tmp_path))

    assert cases[0].needs == frozenset({"github"})
    assert cases[1].needs == frozenset()  # general-only questions need no delegation


def test_load_gold_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("question: not a list", encoding="utf-8")

    with pytest.raises(ValueError, match="list of cases"):
        load_gold(path)


def test_router_accuracy_counts_acceptable_routes(tmp_path: Path) -> None:
    cases = load_gold(_write_gold(tmp_path))
    classifier = FixedClassifier(
        {"stars now?": "github", "hello": "general", "docs vs live?": "docs"}
    )

    report = evaluate_router(classifier, cases)

    # All three are correct: the cross-source case accepts either github or docs.
    assert report.accuracy == 1.0
    assert report.misroutes == []


def test_router_misroutes_are_reported(tmp_path: Path) -> None:
    cases = load_gold(_write_gold(tmp_path))
    classifier = FixedClassifier(
        {"stars now?": "docs", "hello": "general", "docs vs live?": "github"}
    )

    report = evaluate_router(classifier, cases)

    assert report.accuracy == pytest.approx(2 / 3)
    assert [m.case.question for m in report.misroutes] == ["stars now?"]


def test_supervisor_coverage_requires_all_needed_specialists(tmp_path: Path) -> None:
    cases = load_gold(_write_gold(tmp_path))
    consultations = {
        "stars now?": ("github",),
        "hello": (),
        "docs vs live?": ("docs",),  # missed github → not covered
    }

    report = evaluate_supervisor(lambda q: consultations[q], cases)

    assert report.coverage == pytest.approx(2 / 3)
    assert [r.covered for r in report.results] == [True, True, False]


def test_supervisor_extra_delegations_are_counted(tmp_path: Path) -> None:
    cases = load_gold(_write_gold(tmp_path))
    consultations = {
        "stars now?": ("github", "docs"),  # docs was unnecessary
        "hello": (),
        "docs vs live?": ("github", "docs"),
    }

    report = evaluate_supervisor(lambda q: consultations[q], cases)

    assert report.coverage == 1.0
    assert report.results[0].extra == ("docs",)
    assert report.mean_extra_delegations == pytest.approx(1 / 3)


def test_recorder_plus_mapping_yields_specialist_names() -> None:
    recorder = DelegationRecorder()
    recorder.on_tool_start({"name": "consult_github_specialist"}, "")
    recorder.on_tool_start({"name": "consult_docs_specialist"}, "")
    recorder.on_tool_start({"name": "consult_github_specialist"}, "")  # duplicate
    recorder.on_tool_start({"name": "search_project_docs"}, "")  # not a delegation

    assert specialists_consulted(recorder.tool_calls) == ("github", "docs")
