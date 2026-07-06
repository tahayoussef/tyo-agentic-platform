"""Application configuration, validated and loaded from the environment.

All configuration (including secrets) is supplied via environment variables or a local
``.env`` file. Nothing is hardcoded, which keeps the twelve-factor promise and makes the
same image safe to run in any environment.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Field names map to upper-cased environment variables (e.g. ``nvidia_api_key`` reads
    ``NVIDIA_API_KEY``). Values are validated by Pydantic at startup, so misconfiguration
    fails fast and loudly instead of surfacing as a confusing runtime error later.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- NVIDIA / LLM ---
    nvidia_api_key: SecretStr = Field(description="API key for NVIDIA AI endpoints.")
    nvidia_model: str = Field(
        default="z-ai/glm-5.2",
        description="Identifier of the model served by NVIDIA AI endpoints.",
    )
    llm_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=16_384, gt=0)
    llm_seed: int | None = Field(default=42, description="Seed for reproducible sampling.")

    # --- GitHub ---
    github_username: str | None = Field(
        default=None,
        description="Default GitHub account inspected when a question names no user.",
    )
    github_token: SecretStr | None = Field(
        default=None,
        description="Optional Personal Access Token to raise the GitHub rate limit.",
    )
    github_api_base_url: str = Field(default="https://api.github.com")
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Observability ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(
        default=False,
        description="Emit structured JSON logs (recommended in production).",
    )


def get_settings() -> Settings:
    """Build and validate a :class:`Settings` instance.

    Raises :class:`pydantic.ValidationError` if required configuration is missing.
    """
    # The required fields are populated from the environment, which mypy cannot see.
    return Settings()  # type: ignore[call-arg]
