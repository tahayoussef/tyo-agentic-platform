"""The agent itself: an LLM wired to the GitHub tool via a ReAct loop.

We use LangGraph's prebuilt ``create_react_agent``, the current idiomatic way to build a
tool-calling agent in the LangChain ecosystem. :class:`BasicAgent` is a thin, typed facade
over the compiled graph exposing ``ask`` (blocking) and ``stream`` (token-by-token).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent

from basic_agent.config import Settings
from basic_agent.logging import get_logger
from basic_agent.tools import build_github_tools

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a concise assistant that answers questions about GitHub users' public "
    "repositories. When a question is about someone's projects, languages, popularity, or "
    "recent activity, call the `list_github_repositories` tool to get real data before "
    "answering. Ground every claim in the tool's output — never invent repository names, "
    "star counts, or descriptions. If no repositories are found, say so plainly."
)


def build_llm(settings: Settings) -> ChatNVIDIA:
    """Instantiate the chat model from settings."""
    return ChatNVIDIA(
        model=settings.nvidia_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        seed=settings.llm_seed,
    )


def build_agent(settings: Settings) -> Any:
    """Compile a ReAct agent (LLM + GitHub tool) into an executable graph."""
    llm = build_llm(settings)
    tools = build_github_tools(settings)
    logger.debug("agent.build", model=settings.nvidia_model, tools=[t.name for t in tools])
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


class BasicAgent:
    """A typed facade over the compiled ReAct graph."""

    def __init__(self, settings: Settings, *, graph: Any | None = None) -> None:
        self._settings = settings
        self._graph: Any = graph if graph is not None else build_agent(settings)

    def ask(self, question: str) -> str:
        """Answer ``question`` and return the final assistant message."""
        result = self._graph.invoke(self._initial_state(question))
        return str(result["messages"][-1].content)

    def stream(self, question: str) -> Iterator[str]:
        """Yield the assistant's answer token-by-token as it is generated."""
        for chunk, _metadata in self._graph.stream(
            self._initial_state(question), stream_mode="messages"
        ):
            if not isinstance(chunk, AIMessageChunk):
                continue
            content = chunk.content
            if isinstance(content, str) and content:
                yield content

    @staticmethod
    def _initial_state(question: str) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": question}]}
