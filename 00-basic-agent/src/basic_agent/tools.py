"""LangChain tools exposed to the agent.

The single tool wraps :class:`~basic_agent.github_client.GitHubClient`. It is built by a
factory so that configuration (default username, credentials) and the HTTP client can be
injected — which keeps the tool trivially unit-testable without hitting the network.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from basic_agent.config import Settings
from basic_agent.github_client import GitHubClient, GitHubError, GitHubRepository


class RepositoryLister(Protocol):
    """The client capability the GitHub tool depends on.

    Depending on this structural interface (rather than the concrete
    :class:`~basic_agent.github_client.GitHubClient`) lets callers inject any conforming
    implementation — a real client, a fake in tests, or a cached decorator later.
    """

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
        description="GitHub username to inspect. Omit to use the configured default account.",
    )
    include_forks: bool = Field(
        default=False,
        description="Set true to include forked repositories in the results.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of repositories to return.",
    )


def _format_repositories(username: str, repos: list[GitHubRepository]) -> str:
    """Render repositories as compact text for the LLM to reason over."""
    if not repos:
        return f"{username} has no matching public repositories."

    lines = [f"Public repositories for {username} (showing {len(repos)}):", ""]
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
    """Build the GitHub-related tools bound to ``settings``.

    Args:
        settings: Application settings supplying the default username and credentials.
        client_factory: Optional factory returning a :class:`GitHubClient`. Injected in
            tests; defaults to a client configured from ``settings``.
    """

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
            "List a GitHub user's public repositories, most recently pushed first. "
            "Use this to answer questions about someone's projects, languages, popularity "
            "(stars), or recent activity."
        ),
        args_schema=ListRepositoriesInput,
    )
    return [tool]
