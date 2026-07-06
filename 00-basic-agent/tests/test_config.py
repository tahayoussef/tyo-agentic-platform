"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from basic_agent.config import Settings


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
    monkeypatch.setenv("GITHUB_USERNAME", "octocat")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.nvidia_api_key.get_secret_value() == "nvapi-from-env"
    assert settings.github_username == "octocat"
    assert settings.nvidia_model == "z-ai/glm-5.2"  # default applied


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_secret_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-super-secret")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert "nvapi-super-secret" not in repr(settings)
    assert "nvapi-super-secret" not in str(settings)


def test_out_of_range_temperature_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            nvidia_api_key="nvapi-test",  # type: ignore[arg-type]
            llm_temperature=5.0,
        )
