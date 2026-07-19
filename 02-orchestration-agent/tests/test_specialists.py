"""Tests for the specialist building blocks: runners and tools, all with fakes."""

from __future__ import annotations

from collections.abc import Iterator
from types import TracebackType
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from orchestration_agent.config import Settings
from orchestration_agent.knowledge_base import DocHit, DocSection
from orchestration_agent.specialists import (
    GraphRunner,
    LlmRunner,
    build_docs_tool,
    build_github_tool,
)
from orchestration_agent.tracing import ToolTraceCallbackHandler


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"github_username": "octocat", **overrides}
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        **values,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# GraphRunner
# --------------------------------------------------------------------------- #


class FakeGraph:
    """A stand-in for a compiled LangGraph agent."""

    def __init__(self) -> None:
        self.last_state: dict[str, Any] | None = None
        self.last_config: Any = None

    def invoke(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.last_state = state
        self.last_config = kwargs.get("config")
        return {"messages": [AIMessage(content="graph answer")]}

    def stream(self, state: dict[str, Any], stream_mode: str = "messages", **kwargs: Any) -> Any:
        self.last_state = state
        yield AIMessageChunk(content="gra"), {}
        yield ToolMessage(content="tool noise", tool_call_id="1"), {}
        yield AIMessageChunk(content="ph"), {}
        yield AIMessageChunk(content=""), {}


def test_graph_runner_ask_returns_final_message() -> None:
    assert GraphRunner(FakeGraph()).ask("q") == "graph answer"


def test_graph_runner_sends_user_message() -> None:
    graph = FakeGraph()

    GraphRunner(graph).ask("the question")

    assert graph.last_state is not None
    assert graph.last_state["messages"][0]["content"] == "the question"


def test_graph_runner_stream_filters_to_assistant_tokens() -> None:
    assert "".join(GraphRunner(FakeGraph()).stream("q")) == "graph"


def test_graph_runner_forwards_callbacks_as_config() -> None:
    graph = FakeGraph()
    handler = ToolTraceCallbackHandler()

    GraphRunner(graph).ask("q", callbacks=[handler])

    assert graph.last_config == {"callbacks": [handler]}


def test_graph_runner_without_callbacks_sends_no_config() -> None:
    graph = FakeGraph()

    GraphRunner(graph).ask("q")

    assert graph.last_config is None


# --------------------------------------------------------------------------- #
# LlmRunner
# --------------------------------------------------------------------------- #


class FakeLlm:
    """A stand-in chat model for the tool-less specialist."""

    def __init__(self) -> None:
        self.last_messages: Any = None

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        self.last_messages = messages
        return AIMessage(content="llm answer")

    def stream(self, messages: Any, **kwargs: Any) -> Iterator[AIMessageChunk]:
        self.last_messages = messages
        yield AIMessageChunk(content="llm ")
        yield AIMessageChunk(content="")
        yield AIMessageChunk(content="answer")


def test_llm_runner_ask_returns_content_and_sends_system_prompt() -> None:
    llm = FakeLlm()
    runner = LlmRunner(llm, "be general")

    assert runner.ask("hi") == "llm answer"
    system, human = llm.last_messages
    assert system.content == "be general"
    assert human.content == "hi"


def test_llm_runner_stream_yields_nonempty_tokens() -> None:
    assert "".join(LlmRunner(FakeLlm(), "sys").stream("hi")) == "llm answer"


# --------------------------------------------------------------------------- #
# The docs tool
# --------------------------------------------------------------------------- #


class FakeSearcher:
    def __init__(self, hits: list[DocHit]) -> None:
        self.hits = hits
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def search(self, query: str, *, top_k: int | None = None) -> list[DocHit]:
        self.last_query = query
        self.last_top_k = top_k
        return self.hits


def _hit(repo: str, heading: str, text: str) -> DocHit:
    return DocHit(section=DocSection(repo=repo, heading=heading, text=text), score=1.0)


def test_docs_tool_formats_hits_with_attribution() -> None:
    searcher = FakeSearcher([_hit("gobekli-tepe", "Architecture", "Uses dbt.")])
    tool = build_docs_tool(_settings(), searcher=searcher)

    output = tool.invoke({"query": "architecture"})

    assert "[docs · gobekli-tepe · Architecture]" in output
    assert "Uses dbt." in output
    assert searcher.last_query == "architecture"


def test_docs_tool_reports_empty_results() -> None:
    tool = build_docs_tool(_settings(), searcher=FakeSearcher([]))

    assert "No matching sections" in tool.invoke({"query": "nothing"})


def test_docs_tool_passes_top_k_through() -> None:
    searcher = FakeSearcher([])
    tool = build_docs_tool(_settings(), searcher=searcher)

    tool.invoke({"query": "q", "top_k": 7})

    assert searcher.last_top_k == 7


# --------------------------------------------------------------------------- #
# The GitHub tool (behavior around the injected client factory)
# --------------------------------------------------------------------------- #


class FakeClient:
    """A stand-in for GitHubClient, returned by the injected factory."""

    def __init__(self, repos: list[Any]) -> None:
        self.repos = repos
        self.last_username: str | None = None

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def list_public_repositories(
        self, username: str, *, include_forks: bool = False, limit: int = 20
    ) -> list[Any]:
        self.last_username = username
        return self.repos


def test_github_tool_uses_default_username_from_settings() -> None:
    client = FakeClient([])
    tool = build_github_tool(_settings(), client_factory=lambda: client)

    output = tool.invoke({})

    assert client.last_username == "octocat"
    assert "octocat" in output


def test_github_tool_without_any_username_asks_for_one() -> None:
    tool = build_github_tool(_settings(github_username=None), client_factory=lambda: FakeClient([]))

    assert "No GitHub username" in tool.invoke({})
