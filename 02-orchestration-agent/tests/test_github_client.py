"""Tests for the GitHub client, with all HTTP traffic mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from orchestration_agent.github_client import GitHubClient, GitHubError

_BASE = "https://api.github.com"


def _repo(name: str, *, fork: bool = False, stars: int = 0) -> dict[str, object]:
    return {
        "name": name,
        "full_name": f"octocat/{name}",
        "html_url": f"https://github.com/octocat/{name}",
        "description": f"{name} description",
        "language": "Python",
        "stargazers_count": stars,
        "forks_count": 1,
        "fork": fork,
    }


@respx.mock
def test_lists_repositories_and_skips_forks() -> None:
    respx.get(f"{_BASE}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo("keep", stars=5), _repo("forked", fork=True)])
    )

    with GitHubClient() as client:
        repos = client.list_public_repositories("octocat")

    assert [r.name for r in repos] == ["keep"]
    assert repos[0].stargazers_count == 5


@respx.mock
def test_include_forks_keeps_forked_repos() -> None:
    respx.get(f"{_BASE}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo("keep"), _repo("forked", fork=True)])
    )

    with GitHubClient() as client:
        repos = client.list_public_repositories("octocat", include_forks=True)

    assert [r.name for r in repos] == ["keep", "forked"]


@respx.mock
def test_limit_caps_results() -> None:
    respx.get(f"{_BASE}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo(f"repo-{i}") for i in range(5)])
    )

    with GitHubClient() as client:
        repos = client.list_public_repositories("octocat", limit=2)

    assert len(repos) == 2


@respx.mock
def test_unknown_user_raises() -> None:
    respx.get(f"{_BASE}/users/ghost/repos").mock(return_value=httpx.Response(404))

    with GitHubClient() as client, pytest.raises(GitHubError, match="not found"):
        client.list_public_repositories("ghost")


@respx.mock
def test_server_error_raises() -> None:
    respx.get(f"{_BASE}/users/octocat/repos").mock(return_value=httpx.Response(500, text="boom"))

    with GitHubClient() as client, pytest.raises(GitHubError, match="500"):
        client.list_public_repositories("octocat")


@respx.mock
def test_network_error_raises() -> None:
    respx.get(f"{_BASE}/users/octocat/repos").mock(side_effect=httpx.ConnectError("down"))

    with GitHubClient() as client, pytest.raises(GitHubError, match="failed"):
        client.list_public_repositories("octocat")
