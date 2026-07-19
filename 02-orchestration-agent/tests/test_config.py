"""Tests for typed settings: defaults, env overrides, and secret handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestration_agent.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def test_api_key_is_required() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults() -> None:
    settings = _settings()

    assert settings.orchestration_mode == "router"
    assert settings.docs_top_k == 3
    assert settings.knowledge_base_dir.name == "knowledge_base"


def test_router_model_falls_back_to_chat_model() -> None:
    settings = _settings()

    assert settings.resolved_router_model == settings.nvidia_chat_model


def test_dedicated_router_model_wins() -> None:
    settings = _settings(nvidia_router_model="meta/llama-3.1-8b-instruct")

    assert settings.resolved_router_model == "meta/llama-3.1-8b-instruct"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
    monkeypatch.setenv("ORCHESTRATION_MODE", "supervisor")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.orchestration_mode == "supervisor"


def test_secret_never_leaks_in_repr() -> None:
    settings = _settings()

    assert "nvapi-test" not in repr(settings)
    assert "nvapi-test" not in str(settings)
