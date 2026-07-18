"""The RAG agent: an LLM given two tools (static KB + live GitHub) wired into a ReAct loop.

Same shape as 00-basic-agent's ``BasicAgent`` — a typed facade over a LangGraph
``create_react_agent`` graph — but the system prompt is the interesting part: it tells the
model to consult BOTH sources and to reconcile them, surfacing conflicts where the static
knowledge base has drifted from live GitHub.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent

from rag_agent.config import Settings
from rag_agent.embeddings import build_embeddings, build_reranker, build_sparse_embeddings
from rag_agent.logging import get_logger
from rag_agent.tools import build_tools
from rag_agent.vector_store import build_qdrant_client

logger = get_logger(__name__)

RECONCILIATION_SYSTEM_PROMPT = (
    "You answer questions about a portfolio of GitHub repositories. You have two sources:\n"
    "1. `search_knowledge_base` — a curated knowledge base of detailed repo documentation "
    "(architecture, rationale, history). It is rich but may be OUT OF DATE.\n"
    "2. `list_github_repositories` — the LIVE GitHub API (current description, primary "
    "language, stars, forks, dates). It is current but shallow.\n\n"
    "For any substantive question, consult BOTH: use the knowledge base for background and "
    "narrative, and the live API for current facts. When the two disagree — a different "
    "primary language, different star counts, a repo the knowledge base doesn't mention, or "
    "one it describes that no longer exists — explicitly FLAG the discrepancy, prefer the "
    "live data for present-day facts, and note what the knowledge base says. Ground every "
    "claim in tool output; never invent. Keep the final answer concise."
)


def build_llm(settings: Settings) -> ChatNVIDIA:
    """Instantiate the reasoning chat model from settings."""
    return ChatNVIDIA(
        model=settings.nvidia_chat_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        seed=settings.llm_seed,
    )


def build_agent(settings: Settings) -> Any:
    """Compile a ReAct agent (LLM + KB tool + live GitHub tool) into an executable graph."""
    embeddings = build_embeddings(settings)
    qdrant_client = build_qdrant_client(settings)
    sparse_embedding = build_sparse_embeddings(settings) if settings.is_hybrid else None
    reranker = build_reranker(settings) if settings.use_reranker else None
    tools = build_tools(
        settings,
        embeddings=embeddings,
        qdrant_client=qdrant_client,
        sparse_embedding=sparse_embedding,
        reranker=reranker,
    )
    llm = build_llm(settings)
    logger.debug(
        "agent.build",
        model=settings.nvidia_chat_model,
        retrieval_mode=settings.retrieval_mode,
        reranker=settings.use_reranker,
        tools=[t.name for t in tools],
    )
    return create_react_agent(llm, tools, prompt=RECONCILIATION_SYSTEM_PROMPT)


def _run_kwargs(callbacks: Sequence[BaseCallbackHandler] | None) -> dict[str, Any]:
    """Build graph run kwargs, attaching a callback config only when callbacks are given."""
    return {"config": {"callbacks": callbacks}} if callbacks else {}


class RagAgent:
    """A typed facade over the compiled ReAct graph."""

    def __init__(self, settings: Settings, *, graph: Any | None = None) -> None:
        self._settings = settings
        self._graph: Any = graph if graph is not None else build_agent(settings)

    def ask(
        self,
        question: str,
        *,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> str:
        """Answer ``question`` and return the final assistant message."""
        result = self._graph.invoke(self._initial_state(question), **_run_kwargs(callbacks))
        return str(result["messages"][-1].content)

    def stream(
        self,
        question: str,
        *,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> Iterator[str]:
        """Yield the assistant's answer token-by-token as it is generated."""
        for chunk, _metadata in self._graph.stream(
            self._initial_state(question), stream_mode="messages", **_run_kwargs(callbacks)
        ):
            if not isinstance(chunk, AIMessageChunk):
                continue
            content = chunk.content
            if isinstance(content, str) and content:
                yield content

    @staticmethod
    def _initial_state(question: str) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": question}]}
