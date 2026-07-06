"""Tests for the GitHub REST client, with the API mocked via respx."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from basic_agent.github_client import GitHubClient, GitHubError, GitHubRepository

BASE_URL = "https://api.github.com"


def _repo(name: str, *, fork: bool = False, language: str = "Python") -> dict[str, Any]:
    return {
        "name": name,
        "full_name": f"octocat/{name}",
        "description": f"{name} description",
        "html_url": f"https://github.com/octocat/{name}",
        "language": language,
        "stargazers_count": 3,
        "forks_count": 1,
        "open_issues_count": 0,
        "topics": ["demo"],
        "fork": fork,
        "archived": False,
        "pushed_at": "2026-01-01T00:00:00Z",
        # An extra field the model should ignore rather than choke on:
        "watchers_count": 99,
    }


@respx.mock
def test_lists_and_parses_repositories() -> None:
    route = respx.get(f"{BASE_URL}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo("alpha"), _repo("beta")])
    )

    with GitHubClient(base_url=BASE_URL) as client:
        repos = client.list_public_repositories("octocat")

    assert route.called
    assert [r.name for r in repos] == ["alpha", "beta"]
    assert all(isinstance(r, GitHubRepository) for r in repos)
    assert str(repos[0].html_url) == "https://github.com/octocat/alpha"


@respx.mock
def test_forks_are_excluded_by_default() -> None:
    respx.get(f"{BASE_URL}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo("alpha"), _repo("beta", fork=True)])
    )

    with GitHubClient(base_url=BASE_URL) as client:
        repos = client.list_public_repositories("octocat")

    assert [r.name for r in repos] == ["alpha"]


@respx.mock
def test_forks_can_be_included() -> None:
    respx.get(f"{BASE_URL}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo("alpha"), _repo("beta", fork=True)])
    )

    with GitHubClient(base_url=BASE_URL) as client:
        repos = client.list_public_repositories("octocat", include_forks=True)

    assert {r.name for r in repos} == {"alpha", "beta"}


@respx.mock
def test_limit_is_respected() -> None:
    respx.get(f"{BASE_URL}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo(f"repo-{i}") for i in range(10)])
    )

    with GitHubClient(base_url=BASE_URL) as client:
        repos = client.list_public_repositories("octocat", limit=3)

    assert len(repos) == 3


@respx.mock
def test_unknown_user_raises_github_error() -> None:
    respx.get(f"{BASE_URL}/users/ghost/repos").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    with GitHubClient(base_url=BASE_URL) as client, pytest.raises(GitHubError, match="not found"):
        client.list_public_repositories("ghost")


@respx.mock
def test_server_error_raises_github_error() -> None:
    respx.get(f"{BASE_URL}/users/octocat/repos").mock(
        return_value=httpx.Response(503, text="upstream unavailable")
    )

    with GitHubClient(base_url=BASE_URL) as client, pytest.raises(GitHubError, match="503"):
        client.list_public_repositories("octocat")


@respx.mock
def test_network_error_raises_github_error() -> None:
    respx.get(f"{BASE_URL}/users/octocat/repos").mock(side_effect=httpx.ConnectError("boom"))

    with GitHubClient(base_url=BASE_URL) as client, pytest.raises(GitHubError, match="failed"):
        client.list_public_repositories("octocat")
