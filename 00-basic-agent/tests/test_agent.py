"""Tests for the BasicAgent facade.

The compiled graph is stubbed so these tests never call the LLM or the network — they
verify only the facade's own wiring (state shape, message extraction, token filtering).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from basic_agent.agent import BasicAgent
from basic_agent.config import Settings


class FakeGraph:
    """A stand-in for a compiled LangGraph agent."""

    def __init__(self) -> None:
        self.last_state: dict[str, Any] | None = None

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.last_state = state
        return {"messages": [AIMessage(content="hello world")]}

    def stream(self, state: dict[str, Any], stream_mode: str = "messages") -> Any:
        self.last_state = state
        # Interleave a tool message to prove it is filtered out of the answer stream.
        yield AIMessageChunk(content="hel"), {}
        yield ToolMessage(content="tool noise", tool_call_id="1"), {}
        yield AIMessageChunk(content="lo"), {}
        yield AIMessageChunk(content=""), {}  # empty chunks are skipped


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        github_username="octocat",
    )


def test_ask_returns_final_message_content() -> None:
    agent = BasicAgent(_settings(), graph=FakeGraph())

    assert agent.ask("hi") == "hello world"


def test_ask_sends_user_message() -> None:
    graph = FakeGraph()
    agent = BasicAgent(_settings(), graph=graph)

    agent.ask("what are my repos?")

    assert graph.last_state is not None
    assert graph.last_state["messages"][0]["content"] == "what are my repos?"


def test_stream_yields_only_assistant_tokens() -> None:
    agent = BasicAgent(_settings(), graph=FakeGraph())

    assert "".join(agent.stream("hi")) == "hello"
