"""A small, typed client for the public GitHub REST API.

Only the handful of fields the agent reasons about are modelled; everything else in the
API response is ignored. The client is deliberately narrow and synchronous — it does one
job (list a user's public repositories) and does it well.
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from basic_agent.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT = "tyo-basic-agent"
_GITHUB_API_VERSION = "2022-11-28"


class GitHubError(RuntimeError):
    """Raised when the GitHub API cannot be queried successfully."""


class GitHubRepository(BaseModel):
    """The subset of a GitHub repository the agent cares about."""

    model_config = ConfigDict(extra="ignore")

    name: str
    full_name: str
    html_url: HttpUrl
    description: str | None = None
    language: str | None = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    topics: list[str] = Field(default_factory=list)
    fork: bool = False
    archived: bool = False
    pushed_at: datetime | None = None


class GitHubClient:
    """Fetches public repository data from GitHub.

    Usable as a context manager so the underlying HTTP connection pool is always closed::

        with GitHubClient() as client:
            repos = client.list_public_repositories("octocat")
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.github.com",
        token: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _GITHUB_API_VERSION,
            "User-Agent": _USER_AGENT,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # When an HTTP client is injected (e.g. in tests) we don't own its lifecycle.
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client:
            self._client.close()

    def list_public_repositories(
        self,
        username: str,
        *,
        include_forks: bool = False,
        limit: int = 30,
        max_pages: int = 10,
    ) -> list[GitHubRepository]:
        """List a user's public repositories, most recently pushed first.

        Args:
            username: The GitHub account to inspect.
            include_forks: Whether to include repositories that are forks.
            limit: Maximum number of repositories to return.
            max_pages: Safety cap on the number of API pages fetched.

        Returns:
            Repositories ordered by most recent push.

        Raises:
            GitHubError: If the user does not exist or the API request fails.
        """
        repos: list[GitHubRepository] = []
        for page in range(1, max_pages + 1):
            batch = self._get_repositories_page(username, page)
            if not batch:
                break

            for raw in batch:
                repo = GitHubRepository.model_validate(raw)
                if repo.fork and not include_forks:
                    continue
                repos.append(repo)
                if len(repos) >= limit:
                    logger.debug("repos.limit_reached", username=username, limit=limit)
                    return repos

            if len(batch) < 100:  # last page reached
                break

        logger.debug("repos.fetched", username=username, count=len(repos))
        return repos

    def _get_repositories_page(self, username: str, page: int) -> list[dict[str, Any]]:
        try:
            response = self._client.get(
                f"/users/{username}/repos",
                params={
                    "per_page": 100,
                    "page": page,
                    "sort": "pushed",
                    "direction": "desc",
                    "type": "owner",
                },
            )
        except httpx.HTTPError as exc:  # network / timeout errors
            raise GitHubError(f"Request to GitHub failed: {exc}") from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            raise GitHubError(f"GitHub user '{username}' was not found.")
        if response.status_code != httpx.codes.OK:
            raise GitHubError(f"GitHub API returned {response.status_code}: {response.text[:200]}")

        payload: list[dict[str, Any]] = response.json()
        return payload
