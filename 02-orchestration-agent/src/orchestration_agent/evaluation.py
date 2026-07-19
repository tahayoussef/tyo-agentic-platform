"""Orchestration evaluation harness: routing accuracy and delegation coverage.

Same philosophy as 01's retrieval harness — a fixed gold set and a scorecard, so a change
to prompts, models, or specialist descriptions is judged by numbers, not by eyeballing one
demo question. Orchestration needs two different measurements:

- **Router accuracy** — a classification metric: did the router pick an acceptable
  specialist for each query?
- **Delegation coverage** — a *trajectory* metric: we don't (here) judge the supervisor's
  final prose; we record WHICH specialists it consulted and check that every source the
  question needs was actually consulted. Wrong trajectories produce confident wrong
  answers, which is why they're worth measuring on their own.

Both evaluators are generic: they take a callable/protocol, not a concrete agent, so tests
run them with fakes and the CLI runs them with the real thing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from langchain_core.callbacks import BaseCallbackHandler

from orchestration_agent.router import RouteClassifier, RouteDecision
from orchestration_agent.specialists import GENERAL
from orchestration_agent.supervisor import specialist_name_from_tool


@dataclass(frozen=True)
class GoldCase:
    """One evaluation question with its expected routing and delegation targets."""

    question: str
    routes: frozenset[str]  # acceptable single destinations for the router
    needs: frozenset[str]  # specialists the supervisor must consult (may be empty)


def load_gold(path: Path) -> list[GoldCase]:
    """Load gold cases from YAML (a list of ``{question, routes, needs?}``).

    ``needs`` defaults to ``routes`` minus ``general`` — a question routable to a data
    specialist should see that specialist consulted, while small talk needs nobody.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Gold file {path} must contain a list of cases.")

    cases: list[GoldCase] = []
    for entry in raw:
        routes = frozenset(str(route) for route in entry["routes"])
        if not routes:
            raise ValueError(f"Gold case {entry['question']!r} lists no routes.")
        needs_raw = entry.get("needs")
        needs = (
            frozenset(str(need) for need in needs_raw)
            if needs_raw is not None
            else routes - {GENERAL}
        )
        cases.append(GoldCase(question=str(entry["question"]), routes=routes, needs=needs))
    return cases


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


# --------------------------------------------------------------------------- #
# Router accuracy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RouterCaseResult:
    case: GoldCase
    decision: RouteDecision

    @property
    def correct(self) -> bool:
        return self.decision.route in self.case.routes


@dataclass(frozen=True)
class RouterReport:
    """A full router run: per-case decisions plus aggregate accuracy."""

    results: list[RouterCaseResult]

    @property
    def accuracy(self) -> float:
        return _safe_mean([1.0 if r.correct else 0.0 for r in self.results])

    @property
    def misroutes(self) -> list[RouterCaseResult]:
        return [r for r in self.results if not r.correct]


def evaluate_router(classifier: RouteClassifier, cases: Sequence[GoldCase]) -> RouterReport:
    """Classify every gold question and score the decisions."""
    results = [
        RouterCaseResult(case=case, decision=classifier.classify(case.question)) for case in cases
    ]
    return RouterReport(results=results)


# --------------------------------------------------------------------------- #
# Supervisor delegation coverage
# --------------------------------------------------------------------------- #


class DelegationRecorder(BaseCallbackHandler):
    """Record the name of every tool the supervisor calls during a run."""

    def __init__(self) -> None:
        self.tool_calls: list[str] = []

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") if serialized else None
        if name:
            self.tool_calls.append(str(name))


def specialists_consulted(tool_calls: Iterable[str]) -> tuple[str, ...]:
    """Map recorded tool names back to specialist names, deduplicated, order preserved."""
    seen: list[str] = []
    for tool_call in tool_calls:
        name = specialist_name_from_tool(tool_call)
        if name is not None and name not in seen:
            seen.append(name)
    return tuple(seen)


@dataclass(frozen=True)
class SupervisorCaseResult:
    case: GoldCase
    consulted: tuple[str, ...]

    @property
    def covered(self) -> bool:
        """Did the supervisor consult every specialist the question needs?"""
        return self.case.needs <= set(self.consulted)

    @property
    def extra(self) -> tuple[str, ...]:
        """Specialists consulted beyond what the question needed (cost/latency signal)."""
        return tuple(name for name in self.consulted if name not in self.case.needs)


@dataclass(frozen=True)
class SupervisorReport:
    """A full supervisor run: per-case trajectories plus aggregate coverage."""

    results: list[SupervisorCaseResult]

    @property
    def coverage(self) -> float:
        return _safe_mean([1.0 if r.covered else 0.0 for r in self.results])

    @property
    def mean_extra_delegations(self) -> float:
        return _safe_mean([float(len(r.extra)) for r in self.results])


# Runs one question through the supervisor and reports which specialists it consulted.
Consult = Callable[[str], Sequence[str]]


def evaluate_supervisor(consult: Consult, cases: Sequence[GoldCase]) -> SupervisorReport:
    """Run every gold question through ``consult`` and score the trajectories."""
    results = [
        SupervisorCaseResult(case=case, consulted=tuple(consult(case.question))) for case in cases
    ]
    return SupervisorReport(results=results)
