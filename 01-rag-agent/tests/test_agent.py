"""Tests for the RagAgent facade.

The compiled graph is stubbed, so these never touch the LLM, Qdrant, or the network — they
verify only the facade's wiring (state shape, final-message extraction, token filtering).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from rag_agent.agent import RagAgent
from rag_agent.config import Settings
from rag_agent.tracing import ToolTraceCallbackHandler


class FakeGraph:
    """A stand-in for a compiled LangGraph agent."""

    def __init__(self) -> None:
        self.last_state: dict[str, Any] | None = None
        self.last_config: Any = None

    def invoke(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.last_state = state
        self.last_config = kwargs.get("config")
        return {"messages": [AIMessage(content="reconciled answer")]}

    def stream(self, state: dict[str, Any], stream_mode: str = "messages", **kwargs: Any) -> Any:
        self.last_state = state
        self.last_config = kwargs.get("config")
        yield AIMessageChunk(content="rec"), {}
        yield ToolMessage(content="tool noise", tool_call_id="1"), {}
        yield AIMessageChunk(content="onciled"), {}
        yield AIMessageChunk(content=""), {}


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        github_username="octocat",
    )


def test_ask_returns_final_message_content() -> None:
    agent = RagAgent(_settings(), graph=FakeGraph())

    assert agent.ask("what is gobekli?") == "reconciled answer"


def test_ask_sends_user_message() -> None:
    graph = FakeGraph()
    agent = RagAgent(_settings(), graph=graph)

    agent.ask("what is gobekli?")

    assert graph.last_state is not None
    assert graph.last_state["messages"][0]["content"] == "what is gobekli?"


def test_stream_yields_only_assistant_tokens() -> None:
    agent = RagAgent(_settings(), graph=FakeGraph())

    assert "".join(agent.stream("hi")) == "reconciled"


def test_ask_without_callbacks_sends_no_config() -> None:
    graph = FakeGraph()

    RagAgent(_settings(), graph=graph).ask("hi")

    assert graph.last_config is None


def test_ask_forwards_callbacks_as_run_config() -> None:
    graph = FakeGraph()
    handler = ToolTraceCallbackHandler()

    RagAgent(_settings(), graph=graph).ask("hi", callbacks=[handler])

    assert graph.last_config == {"callbacks": [handler]}
