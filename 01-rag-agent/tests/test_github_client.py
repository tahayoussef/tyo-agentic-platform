"""Focused tests for the ported GitHub client (mocked with respx).

The client is a direct port of 00-basic-agent's fully-tested client; these cover the two
paths the RAG agent relies on: successful parsing and a clean domain error on 404.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from rag_agent.github_client import GitHubClient, GitHubError, GitHubRepository

BASE_URL = "https://api.github.com"


def _repo(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "full_name": f"octocat/{name}",
        "html_url": f"https://github.com/octocat/{name}",
        "language": "Python",
        "stargazers_count": 3,
        "fork": False,
        "watchers_count": 99,  # unmodeled field — must be ignored
    }


@respx.mock
def test_lists_and_parses_repositories() -> None:
    respx.get(f"{BASE_URL}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[_repo("alpha"), _repo("beta")])
    )

    with GitHubClient(base_url=BASE_URL) as client:
        repos = client.list_public_repositories("octocat")

    assert [r.name for r in repos] == ["alpha", "beta"]
    assert all(isinstance(r, GitHubRepository) for r in repos)


@respx.mock
def test_unknown_user_raises_github_error() -> None:
    respx.get(f"{BASE_URL}/users/ghost/repos").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    with GitHubClient(base_url=BASE_URL) as client, pytest.raises(GitHubError, match="not found"):
        client.list_public_repositories("ghost")
