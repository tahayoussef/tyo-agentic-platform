"""The two tools the RAG agent can call.

1. ``search_knowledge_base`` — the **static** source: semantic search over the Qdrant index
   of curated repo docs. Rich background, but potentially stale.
2. ``list_github_repositories`` — the **live** source: the current GitHub API (ported from
   00-basic-agent). Shallow but always current.

The agent decides when to call each and reconciles their answers. Both tools are built by
factories with their dependencies (embeddings, Qdrant client, GitHub client) injected, so
they stay unit-testable without network or LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.github_client import GitHubClient, GitHubError, GitHubRepository
from rag_agent.retriever import search as kb_search

# --------------------------------------------------------------------------- #
# Tool 1: knowledge-base search (static / Qdrant)
# --------------------------------------------------------------------------- #


class SearchKnowledgeBaseInput(BaseModel):
    """Arguments for the ``search_knowledge_base`` tool."""

    query: str = Field(description="A natural-language description of what to look for.")
    repo: str | None = Field(
        default=None,
        description=(
            "Restrict the search to one repository by its exact name/slug "
            "(e.g. 'carthage-architecture-center'). Set this whenever the question is about a "
            "specific repository, so the search returns only that repo's documentation."
        ),
    )
    top_k: int | None = Field(
        default=None,
        description="How many chunks to return. Defaults to the configured value.",
    )


def _format_kb_results(results: list[tuple[Document, float]]) -> str:
    """Render retrieved chunks as text for the LLM, attributed to their repository."""
    if not results:
        return "No relevant documents were found in the knowledge base."
    blocks = [
        f"[knowledge base · {doc.metadata.get('repo', 'unknown')}]\n{doc.page_content.strip()}"
        for doc, _score in results
    ]
    return "\n\n---\n\n".join(blocks)


def build_knowledge_base_tool(
    settings: Settings,
    *,
    embeddings: Embeddings,
    client: QdrantClient,
) -> BaseTool:
    """Build the ``search_knowledge_base`` tool bound to a Qdrant collection."""

    def search_knowledge_base(query: str, repo: str | None = None, top_k: int | None = None) -> str:
        results = kb_search(
            settings, query, embeddings=embeddings, client=client, top_k=top_k, repo=repo
        )
        return _format_kb_results(results)

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search a curated knowledge base of detailed repository documentation — "
            "architecture, design rationale, history, and context. It is rich but may be "
            "out of date. Use it to answer what a project is, how it works, or why it exists. "
            "When the question targets a specific repository, pass its exact name as `repo` to "
            "focus the search on that repository's docs."
        ),
        args_schema=SearchKnowledgeBaseInput,
    )


# --------------------------------------------------------------------------- #
# Tool 2: live GitHub repositories (ported from 00-basic-agent)
# --------------------------------------------------------------------------- #


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


def build_github_tools(
    settings: Settings,
    *,
    client_factory: ClientFactory | None = None,
) -> list[BaseTool]:
    """Build the live-GitHub tool(s), bound to ``settings``."""

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

    tool = StructuredTool.from_function(
        func=list_github_repositories,
        name="list_github_repositories",
        description=(
            "Fetch a GitHub user's public repositories from the LIVE GitHub API — current "
            "descriptions, primary language, star/fork counts, and recent activity. Use this "
            "for up-to-date facts and to verify anything the knowledge base claims."
        ),
        args_schema=ListRepositoriesInput,
    )
    return [tool]


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #


def build_tools(
    settings: Settings,
    *,
    embeddings: Embeddings,
    qdrant_client: QdrantClient,
    github_client_factory: ClientFactory | None = None,
) -> list[BaseTool]:
    """Build the full tool set the agent is given: static KB search + live GitHub."""
    return [
        build_knowledge_base_tool(settings, embeddings=embeddings, client=qdrant_client),
        *build_github_tools(settings, client_factory=github_client_factory),
    ]
