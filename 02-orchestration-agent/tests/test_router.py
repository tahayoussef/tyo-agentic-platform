"""Tests for the router: schema constraints, prompt rendering, and the LLM classifier.

The LLM is faked, so these verify only our side of the contract: what we send to the
structured-output runnable and how we validate what comes back.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from orchestration_agent.router import (
    LlmRouteClassifier,
    RouteDecision,
    build_router_prompt,
)
from orchestration_agent.specialists import SPECIALIST_DESCRIPTIONS


class FakeStructuredRunnable:
    """Stands in for ``llm.with_structured_output(RouteDecision)``."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.last_input: Any = None

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.last_input = messages
        return self.result


class FakeLlm:
    """Stands in for a chat model; only ``with_structured_output`` is used by the router."""

    def __init__(self, result: Any) -> None:
        self.runnable = FakeStructuredRunnable(result)
        self.requested_schema: Any = None

    def with_structured_output(self, schema: Any) -> FakeStructuredRunnable:
        self.requested_schema = schema
        return self.runnable


def test_route_decision_rejects_unknown_specialist() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(route="nonexistent", confidence=0.9, reason="?")  # type: ignore[arg-type]


def test_route_decision_bounds_confidence() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(route="docs", confidence=1.5, reason="too sure")


def test_router_prompt_lists_every_specialist() -> None:
    prompt = build_router_prompt(SPECIALIST_DESCRIPTIONS)

    for name, description in SPECIALIST_DESCRIPTIONS.items():
        assert f"`{name}`" in prompt
        assert description in prompt


def test_classifier_requests_the_decision_schema() -> None:
    decision = RouteDecision(route="docs", confidence=0.8, reason="docs question")
    llm = FakeLlm(decision)

    LlmRouteClassifier(llm)

    assert llm.requested_schema is RouteDecision


def test_classifier_returns_decision_and_sends_system_plus_question() -> None:
    decision = RouteDecision(route="github", confidence=0.9, reason="live data")
    llm = FakeLlm(decision)
    classifier = LlmRouteClassifier(llm)

    result = classifier.classify("how many stars?")

    assert result is decision
    system, human = llm.runnable.last_input
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert human.content == "how many stars?"


def test_classifier_validates_dict_outputs() -> None:
    llm = FakeLlm({"route": "general", "confidence": 0.5, "reason": "chit chat"})
    classifier = LlmRouteClassifier(llm)

    result = classifier.classify("hello")

    assert isinstance(result, RouteDecision)
    assert result.route == "general"
