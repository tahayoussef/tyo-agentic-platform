"""Tests for the supervisor's delegation layer: tool wrapping and name mapping."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from orchestration_agent.specialists import DOCS, GENERAL, GITHUB, Specialist
from orchestration_agent.supervisor import (
    build_delegation_tools,
    delegation_tool_name,
    specialist_name_from_tool,
)


class FakeRunner:
    """Records the questions it was asked and returns a canned answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []

    def ask(self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None) -> str:
        self.questions.append(question)
        return self.answer

    def stream(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> Iterator[str]:
        yield self.ask(question)


def _team() -> tuple[dict[str, Specialist], dict[str, FakeRunner]]:
    runners = {
        GITHUB: FakeRunner("live answer"),
        DOCS: FakeRunner("docs answer"),
        GENERAL: FakeRunner("general answer"),
    }
    team = {
        name: Specialist(name=name, description=f"{name} stuff", runner=runner)
        for name, runner in runners.items()
    }
    return team, runners


def test_tool_name_round_trip() -> None:
    assert delegation_tool_name(DOCS) == "consult_docs_specialist"
    assert specialist_name_from_tool("consult_docs_specialist") == DOCS


def test_non_delegation_tool_maps_to_none() -> None:
    assert specialist_name_from_tool("search_project_docs") is None
    assert specialist_name_from_tool("list_github_repositories") is None


def test_general_specialist_gets_no_delegation_tool() -> None:
    team, _runners = _team()

    tools = build_delegation_tools(team)

    assert sorted(tool.name for tool in tools) == [
        "consult_docs_specialist",
        "consult_github_specialist",
    ]


def test_delegation_tool_invokes_its_specialist() -> None:
    team, runners = _team()
    tools = {tool.name: tool for tool in build_delegation_tools(team)}

    answer = tools["consult_github_specialist"].invoke({"question": "how many stars?"})

    assert answer == "live answer"
    assert runners[GITHUB].questions == ["how many stars?"]
    assert runners[DOCS].questions == []


def test_delegation_tool_description_carries_specialist_description() -> None:
    team, _runners = _team()
    tools: dict[str, Any] = {tool.name: tool for tool in build_delegation_tools(team)}

    assert "docs stuff" in tools["consult_docs_specialist"].description
