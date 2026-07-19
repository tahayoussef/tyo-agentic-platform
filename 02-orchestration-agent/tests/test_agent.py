"""Tests for the OrchestrationAgent facade: mode selection, dispatch, and delegation.

Everything (classifier, specialists, supervisor) is faked — these verify only the facade's
wiring, mirroring 00/01's facade tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from orchestration_agent.agent import OrchestrationAgent
from orchestration_agent.config import Settings
from orchestration_agent.router import RouteDecision
from orchestration_agent.specialists import DOCS, GENERAL, GITHUB, Specialist


def _settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


class FakeRunner:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []
        self.last_callbacks: Sequence[BaseCallbackHandler] | None = None

    def ask(self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None) -> str:
        self.questions.append(question)
        self.last_callbacks = callbacks
        return self.answer

    def stream(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> Iterator[str]:
        yield from self.ask(question, callbacks=callbacks).split(" ")


class FakeClassifier:
    def __init__(self, route: str) -> None:
        self.route = route
        self.questions: list[str] = []

    def classify(self, question: str) -> RouteDecision:
        self.questions.append(question)
        return RouteDecision(route=self.route, confidence=0.9, reason="fake")  # type: ignore[arg-type]


def _team() -> dict[str, Specialist]:
    return {
        name: Specialist(name=name, description=name, runner=FakeRunner(f"{name} answer"))
        for name in (GITHUB, DOCS, GENERAL)
    }


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown orchestration mode"):
        OrchestrationAgent(_settings(), mode="swarm", specialists=_team())


def test_mode_defaults_to_settings() -> None:
    agent = OrchestrationAgent(
        _settings(),  # default mode is router
        classifier=FakeClassifier(DOCS),
        specialists=_team(),
    )

    assert agent.mode == "router"


def test_router_dispatches_to_the_chosen_specialist() -> None:
    team = _team()
    agent = OrchestrationAgent(_settings(), classifier=FakeClassifier(DOCS), specialists=team)

    answer = agent.ask("what do the docs say?")

    assert answer == "docs answer"
    docs_runner = team[DOCS].runner
    assert isinstance(docs_runner, FakeRunner)
    assert docs_runner.questions == ["what do the docs say?"]


def test_router_leaves_other_specialists_untouched() -> None:
    team = _team()
    agent = OrchestrationAgent(_settings(), classifier=FakeClassifier(GITHUB), specialists=team)

    agent.ask("stars?")

    docs_runner = team[DOCS].runner
    assert isinstance(docs_runner, FakeRunner)
    assert docs_runner.questions == []


def test_router_stream_goes_through_the_specialist() -> None:
    agent = OrchestrationAgent(_settings(), classifier=FakeClassifier(GENERAL), specialists=_team())

    assert list(agent.stream("hi")) == ["general", "answer"]


def test_router_unknown_route_raises() -> None:
    team = {GITHUB: _team()[GITHUB]}  # classifier will choose a missing specialist
    agent = OrchestrationAgent(_settings(), classifier=FakeClassifier(DOCS), specialists=team)

    with pytest.raises(ValueError, match="unknown specialist"):
        agent.ask("q")


def test_route_exposes_the_decision_without_dispatching() -> None:
    team = _team()
    agent = OrchestrationAgent(_settings(), classifier=FakeClassifier(DOCS), specialists=team)

    decision = agent.route("what do the docs say?")

    assert decision.route == DOCS
    docs_runner = team[DOCS].runner
    assert isinstance(docs_runner, FakeRunner)
    assert docs_runner.questions == []  # classified, not dispatched


def test_supervisor_mode_uses_the_supervisor_runner() -> None:
    supervisor = FakeRunner("synthesized answer")
    agent = OrchestrationAgent(_settings(), mode="supervisor", supervisor=supervisor)

    assert agent.ask("compare docs and live") == "synthesized answer"
    assert supervisor.questions == ["compare docs and live"]


def test_supervisor_mode_forwards_callbacks() -> None:
    supervisor = FakeRunner("answer")
    agent = OrchestrationAgent(_settings(), mode="supervisor", supervisor=supervisor)
    handler = BaseCallbackHandler()

    agent.ask("q", callbacks=[handler])

    assert supervisor.last_callbacks == [handler]


def test_route_is_unavailable_in_supervisor_mode() -> None:
    agent = OrchestrationAgent(_settings(), mode="supervisor", supervisor=FakeRunner("answer"))

    with pytest.raises(RuntimeError, match="router mode"):
        agent.route("q")
