"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_agent.config import Settings


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
    monkeypatch.setenv("COLLECTION_NAME", "custom_repos")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.nvidia_api_key.get_secret_value() == "nvapi-from-env"
    assert settings.collection_name == "custom_repos"
    assert settings.embedding_model == "nvidia/nv-embedqa-e5-v5"  # default
    assert isinstance(settings.knowledge_base_dir, Path)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_invalid_chunk_size_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
            chunk_size=0,
        )
