"""Tests for the LangChain tool wrapping the GitHub client."""

from __future__ import annotations

from types import TracebackType
from typing import Self

from basic_agent.config import Settings
from basic_agent.github_client import GitHubError, GitHubRepository
from basic_agent.tools import build_github_tools


class FakeClient:
    """A stand-in for GitHubClient that records calls and returns canned data."""

    def __init__(
        self,
        repos: list[GitHubRepository] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._repos = repos or []
        self._error = error
        self.calls: list[dict[str, object]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def list_public_repositories(
        self,
        username: str,
        *,
        include_forks: bool = False,
        limit: int = 30,
    ) -> list[GitHubRepository]:
        self.calls.append({"username": username, "include_forks": include_forks, "limit": limit})
        if self._error is not None:
            raise self._error
        return self._repos


def _repo(name: str) -> GitHubRepository:
    return GitHubRepository(
        name=name,
        full_name=f"octocat/{name}",
        html_url=f"https://github.com/octocat/{name}",  # type: ignore[arg-type]
        language="Python",
        stargazers_count=5,
    )


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "nvidia_api_key": "nvapi-test",
        "github_username": "octocat",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type, call-arg]


def test_tool_formats_repositories() -> None:
    repos = [_repo("alpha"), _repo("beta")]
    tools = build_github_tools(_settings(), client_factory=lambda: FakeClient(repos=repos))

    output = tools[0].invoke({"username": "octocat"})

    assert "alpha" in output
    assert "beta" in output
    assert "★5" in output


def test_tool_falls_back_to_configured_username() -> None:
    fake = FakeClient(repos=[_repo("alpha")])
    tools = build_github_tools(_settings(), client_factory=lambda: fake)

    output = tools[0].invoke({})

    assert "octocat" in output
    assert fake.calls[0]["username"] == "octocat"


def test_tool_without_any_username() -> None:
    tools = build_github_tools(_settings(github_username=None), client_factory=lambda: FakeClient())

    output = tools[0].invoke({})

    assert "No GitHub username" in output


def test_tool_reports_github_errors_gracefully() -> None:
    tools = build_github_tools(
        _settings(), client_factory=lambda: FakeClient(error=GitHubError("boom"))
    )

    output = tools[0].invoke({"username": "octocat"})

    assert "Could not fetch" in output
    assert "boom" in output


def test_tool_metadata_is_well_formed() -> None:
    tool = build_github_tools(_settings())[0]

    assert tool.name == "list_github_repositories"
    assert tool.description
    assert tool.args_schema is not None
