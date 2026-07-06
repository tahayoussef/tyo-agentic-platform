"""Shared test fixtures and helpers."""

from __future__ import annotations

import pytest

from basic_agent.config import Settings


@pytest.fixture
def settings() -> Settings:
    """A minimal, valid Settings instance that never touches a real ``.env``."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        github_username="octocat",
    )
