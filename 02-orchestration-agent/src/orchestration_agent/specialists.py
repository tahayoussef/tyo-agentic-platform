"""The team of specialist agents that both orchestration styles delegate to.

A *specialist* is a narrow agent: one job, one focused system prompt, and only the tools
that job needs. Three specialists cover this domain:

- ``github``  — answers from the LIVE GitHub API (a ReAct agent with one tool);
- ``docs``    — answers from the local documentation knowledge base (ReAct + one tool);
- ``general`` — small talk / anything else (a plain LLM call — a specialist doesn't have
  to be an agent; with no tools to call, an "agent" is just an LLM with a prompt).

Everything is built through injectable factories (client factory, docs searcher, LLM), so
the whole team is unit-testable without network, disk, or a real model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from orchestration_agent.config import Settings
from orchestration_agent.github_client import GitHubClient, GitHubError, GitHubRepository
from orchestration_agent.knowledge_base import DocHit, DocsSearcher, build_searcher
from orchestration_agent.llm import build_chat_llm
from orchestration_agent.logging import get_logger

logger = get_logger(__name__)

# The canonical specialist names. The router's structured output and the eval gold set both
# refer to these — keep them in sync with ``router.RouteName``.
GITHUB = "github"
DOCS = "docs"
GENERAL = "general"

# One description per specialist, used verbatim by BOTH orchestration styles: the router
# embeds them in its classification prompt, the supervisor in its delegation-tool
# descriptions. A single source of truth keeps the two modes comparable.
SPECIALIST_DESCRIPTIONS: dict[str, str] = {
    GITHUB: (
        "Live GitHub data: current repository lists, descriptions, primary language, "
        "star/fork counts, and recent push activity — anything about how the repositories "
        "are doing RIGHT NOW."
    ),
    DOCS: (
        "The curated project documentation: what each repository is, its architecture, "
        "design rationale, and history. Rich background, but written at a point in time — "
        "it may be out of date."
    ),
    GENERAL: (
        "Greetings, small talk, questions about this assistant itself, and anything that "
        "needs neither repository data nor project documentation."
    ),
}


# --------------------------------------------------------------------------- #
# Runners: a uniform, typed facade over "something that answers a question"
# --------------------------------------------------------------------------- #


class Runner(Protocol):
    """Anything that can answer a question — a compiled graph or a bare LLM."""

    def ask(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> str: ...

    def stream(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> Iterator[str]: ...


def _run_kwargs(callbacks: Sequence[BaseCallbackHandler] | None) -> dict[str, Any]:
    """Build run kwargs, attaching a callback config only when callbacks are given."""
    return {"config": {"callbacks": callbacks}} if callbacks else {}


class GraphRunner:
    """A typed facade over a compiled LangGraph agent (same shape as 00/01's facades)."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def ask(self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None) -> str:
        result = self._graph.invoke(self._initial_state(question), **_run_kwargs(callbacks))
        return str(result["messages"][-1].content)

    def stream(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> Iterator[str]:
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


class LlmRunner:
    """A tool-less specialist: a single LLM call behind the same ``Runner`` interface."""

    def __init__(self, llm: Any, system_prompt: str) -> None:
        self._llm = llm
        self._system_prompt = system_prompt

    def ask(self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None) -> str:
        response = self._llm.invoke(self._messages(question), **_run_kwargs(callbacks))
        return str(response.content)

    def stream(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> Iterator[str]:
        for chunk in self._llm.stream(self._messages(question), **_run_kwargs(callbacks)):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                yield content

    def _messages(self, question: str) -> list[Any]:
        return [SystemMessage(content=self._system_prompt), HumanMessage(content=question)]


@dataclass(frozen=True)
class Specialist:
    """A named team member: who it is, what it covers, and how to run it."""

    name: str
    description: str
    runner: Runner


# --------------------------------------------------------------------------- #
# The docs specialist's tool
# --------------------------------------------------------------------------- #

DOCS_SPECIALIST_PROMPT = (
    "You are the project-documentation specialist for a portfolio of GitHub repositories. "
    "Your ONLY source is the `search_project_docs` tool over the curated documentation. "
    "Search it before answering, and answer strictly from what it returns — attribute "
    "claims to the documentation (e.g. 'the docs describe X as ...'). The documentation "
    "was written at a point in time and may be stale; do not present it as live fact. If "
    "the docs don't cover something, say so plainly."
)


class SearchProjectDocsInput(BaseModel):
    """Arguments for the ``search_project_docs`` tool."""

    query: str = Field(description="A natural-language description of what to look for.")
    top_k: int | None = Field(
        default=None, description="How many sections to return. Defaults to the configured value."
    )


def _format_doc_hits(hits: list[DocHit]) -> str:
    """Render matched sections as text for the LLM, attributed to their repository."""
    if not hits:
        return "No matching sections were found in the project documentation."
    blocks = [
        f"[docs · {hit.section.repo} · {hit.section.heading}]\n{hit.section.text.strip()}"
        for hit in hits
    ]
    return "\n\n---\n\n".join(blocks)


def build_docs_tool(settings: Settings, *, searcher: DocsSearcher) -> BaseTool:
    """Build the ``search_project_docs`` tool bound to a searcher."""

    def search_project_docs(query: str, top_k: int | None = None) -> str:
        return _format_doc_hits(searcher.search(query, top_k=top_k))

    return StructuredTool.from_function(
        func=search_project_docs,
        name="search_project_docs",
        description=(
            "Search the curated project documentation — architecture, design rationale, and "
            "history of each repository. Rich but possibly out of date."
        ),
        args_schema=SearchProjectDocsInput,
    )


# --------------------------------------------------------------------------- #
# The GitHub specialist's tool (ported from 00/01)
# --------------------------------------------------------------------------- #

GITHUB_SPECIALIST_PROMPT = (
    "You are the live-GitHub specialist for a portfolio of repositories. Your ONLY source "
    "is the `list_github_repositories` tool, which returns current data from the GitHub "
    "API. Call it before answering and ground every claim in its output — never invent "
    "repository names, star counts, or descriptions. You know nothing about a project's "
    "internals or history beyond what the API returns; if asked, say the documentation "
    "specialist covers that."
)


class RepositoryLister(Protocol):
    """The client capability the GitHub tool depends on (structural interface)."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def list_public_repositories(
        self,
        username: str,
        *,
        include_forks: bool = ...,
        limit: int = ...,
    ) -> list[GitHubRepository]: ...


ClientFactory = Callable[[], RepositoryLister]


class ListRepositoriesInput(BaseModel):
    """Arguments for the ``list_github_repositories`` tool."""

    username: str | None = Field(
        default=None,
        description="GitHub username. Omit to use the configured default account.",
    )
    include_forks: bool = Field(default=False, description="Include forked repositories.")
    limit: int = Field(default=20, ge=1, le=100, description="Max repositories to return.")


def _format_repositories(username: str, repos: list[GitHubRepository]) -> str:
    """Render live repositories as compact text for the LLM."""
    if not repos:
        return f"{username} has no matching public repositories."

    lines = [f"Live GitHub repositories for {username} (showing {len(repos)}):", ""]
    for repo in repos:
        header = f"- {repo.name}"
        if repo.language:
            header += f" ({repo.language})"
        header += f" — ★{repo.stargazers_count}, forks {repo.forks_count}"
        if repo.archived:
            header += " [archived]"
        lines.append(header)
        if repo.description:
            lines.append(f"    {repo.description}")
        lines.append(f"    {repo.html_url}")
    return "\n".join(lines)


def build_github_tool(
    settings: Settings,
    *,
    client_factory: ClientFactory | None = None,
) -> BaseTool:
    """Build the live-GitHub tool, bound to ``settings``."""

    def _default_factory() -> GitHubClient:
        token = settings.github_token.get_secret_value() if settings.github_token else None
        return GitHubClient(
            base_url=settings.github_api_base_url,
            token=token,
            timeout=settings.request_timeout_seconds,
        )

    factory = client_factory or _default_factory

    def list_github_repositories(
        username: str | None = None,
        include_forks: bool = False,
        limit: int = 20,
    ) -> str:
        resolved = (username or settings.github_username or "").strip()
        if not resolved:
            return (
                "No GitHub username was provided and no default is configured. "
                "Ask the user which GitHub account to inspect."
            )
        try:
            with factory() as client:
                repos = client.list_public_repositories(
                    resolved, include_forks=include_forks, limit=limit
                )
        except GitHubError as exc:
            return f"Could not fetch repositories for {resolved}: {exc}"
        return _format_repositories(resolved, repos)

    return StructuredTool.from_function(
        func=list_github_repositories,
        name="list_github_repositories",
        description=(
            "Fetch a GitHub user's public repositories from the LIVE GitHub API — current "
            "descriptions, primary language, star/fork counts, and recent activity."
        ),
        args_schema=ListRepositoriesInput,
    )


# --------------------------------------------------------------------------- #
# The general specialist (no tools)
# --------------------------------------------------------------------------- #

GENERAL_SPECIALIST_PROMPT = (
    "You are a concise, friendly assistant for an app that answers questions about a "
    "portfolio of GitHub repositories. Handle greetings, small talk, and questions about "
    "what you can do. If the question actually needs live repository data or project "
    "documentation, say so and suggest asking about the repositories directly — do not "
    "guess at facts you don't have."
)


# --------------------------------------------------------------------------- #
# Building the team
# --------------------------------------------------------------------------- #


def build_specialists(
    settings: Settings,
    *,
    llm: Any | None = None,
    github_client_factory: ClientFactory | None = None,
    docs_searcher: DocsSearcher | None = None,
) -> dict[str, Specialist]:
    """Build the full specialist team, keyed by canonical name."""
    llm = llm if llm is not None else build_chat_llm(settings)
    searcher = docs_searcher or build_searcher(
        settings.knowledge_base_dir, top_k=settings.docs_top_k
    )

    github_graph = create_react_agent(
        llm,
        [build_github_tool(settings, client_factory=github_client_factory)],
        prompt=GITHUB_SPECIALIST_PROMPT,
    )
    docs_graph = create_react_agent(
        llm,
        [build_docs_tool(settings, searcher=searcher)],
        prompt=DOCS_SPECIALIST_PROMPT,
    )

    team = {
        GITHUB: Specialist(GITHUB, SPECIALIST_DESCRIPTIONS[GITHUB], GraphRunner(github_graph)),
        DOCS: Specialist(DOCS, SPECIALIST_DESCRIPTIONS[DOCS], GraphRunner(docs_graph)),
        GENERAL: Specialist(
            GENERAL, SPECIALIST_DESCRIPTIONS[GENERAL], LlmRunner(llm, GENERAL_SPECIALIST_PROMPT)
        ),
    }
    logger.debug("specialists.built", team=sorted(team))
    return team
