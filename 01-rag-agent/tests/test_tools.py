"""Tests for the agent's tools: static KB search and live GitHub.

Fully offline: KB search runs against in-memory Qdrant with deterministic fake embeddings;
the GitHub tool uses a fake client. No NVIDIA, no network.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Self

from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag_agent.config import Settings
from rag_agent.github_client import GitHubError, GitHubRepository
from rag_agent.ingest import run_ingest
from rag_agent.tools import build_github_tools, build_knowledge_base_tool

EMBED_DIM = 64


def _settings(tmp_path: Path | None = None, **overrides: object) -> Settings:
    values: dict[str, object] = {"nvidia_api_key": "nvapi-test", "github_username": "octocat"}
    if tmp_path is not None:
        values.update(
            knowledge_base_dir=tmp_path,
            collection_name="test_repos",
            chunk_size=200,
            chunk_overlap=20,
        )
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type, call-arg]


# --- knowledge-base tool ------------------------------------------------------


def test_knowledge_base_tool_returns_attributed_chunks(tmp_path: Path) -> None:
    (tmp_path / "gobekli.md").write_text(
        "# gobekli\n" + "dbt medallion bronze silver gold platform. " * 30, encoding="utf-8"
    )
    (tmp_path / "carthage.md").write_text(
        "# carthage\n" + "terraform gcp infrastructure modules. " * 30, encoding="utf-8"
    )
    settings = _settings(tmp_path)
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")
    run_ingest(settings, embeddings=embeddings, client=client, recreate=True)

    tool = build_knowledge_base_tool(settings, embeddings=embeddings, client=client)
    output = tool.invoke({"query": "dbt medallion architecture", "top_k": 2})

    assert "knowledge base" in output
    assert ("gobekli" in output) or ("carthage" in output)


def test_knowledge_base_tool_filters_by_repo(tmp_path: Path) -> None:
    (tmp_path / "carthage-architecture-center.md").write_text(
        "# carthage\n" + "terraform modules. primary language python. " * 30, encoding="utf-8"
    )
    (tmp_path / "gobekli-tepe.md").write_text(
        "# gobekli\n" + "dbt medallion platform. " * 30, encoding="utf-8"
    )
    settings = _settings(tmp_path)
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")
    run_ingest(settings, embeddings=embeddings, client=client, recreate=True)

    tool = build_knowledge_base_tool(settings, embeddings=embeddings, client=client)
    output = tool.invoke({"query": "language", "repo": "carthage-architecture-center", "top_k": 10})

    assert "carthage-architecture-center" in output
    assert "gobekli-tepe" not in output


def test_knowledge_base_tool_metadata_is_well_formed(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("# one\n" + "content here. " * 30, encoding="utf-8")
    settings = _settings(tmp_path)
    embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    client = QdrantClient(location=":memory:")
    run_ingest(settings, embeddings=embeddings, client=client, recreate=True)

    tool = build_knowledge_base_tool(settings, embeddings=embeddings, client=client)

    assert tool.name == "search_knowledge_base"
    assert tool.description
    assert tool.args_schema is not None


# --- live GitHub tool ---------------------------------------------------------


class FakeClient:
    """A stand-in for GitHubClient (satisfies the RepositoryLister protocol)."""

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
        self.calls.append({"username": username})
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


def test_github_tool_formats_repositories() -> None:
    tools = build_github_tools(
        _settings(), client_factory=lambda: FakeClient(repos=[_repo("alpha")])
    )

    output = tools[0].invoke({"username": "octocat"})

    assert "alpha" in output
    assert "Live GitHub" in output


def test_github_tool_falls_back_to_configured_username() -> None:
    fake = FakeClient(repos=[_repo("alpha")])
    tools = build_github_tools(_settings(), client_factory=lambda: fake)

    tools[0].invoke({})

    assert fake.calls[0]["username"] == "octocat"


def test_github_tool_reports_errors_gracefully() -> None:
    tools = build_github_tools(
        _settings(), client_factory=lambda: FakeClient(error=GitHubError("boom"))
    )

    output = tools[0].invoke({"username": "octocat"})

    assert "Could not fetch" in output
