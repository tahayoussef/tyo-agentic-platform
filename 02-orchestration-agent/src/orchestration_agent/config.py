"""Application configuration, validated and loaded from the environment.

Same pattern as 00/01: typed settings via ``pydantic-settings``, secrets as ``SecretStr``,
nothing hardcoded. New here are the orchestration knobs: the default mode (router vs
supervisor) and an optional, cheaper model reserved for the routing decision.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings (field ``x`` reads env var ``X``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- NVIDIA / chat model (specialists + supervisor) ---
    nvidia_api_key: SecretStr = Field(description="API key for NVIDIA AI endpoints.")
    nvidia_chat_model: str = Field(default="z-ai/glm-5.2")
    nvidia_router_model: str | None = Field(
        default=None,
        description=(
            "Optional smaller model used ONLY for the routing decision. Routing is a "
            "classification task, so a cheap model often suffices. Falls back to the chat model."
        ),
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_seed: int | None = Field(default=42)

    # --- Orchestration ---
    orchestration_mode: str = Field(
        default="router",
        description="Default mode for `ask`: 'router' (one specialist) or 'supervisor' (many).",
    )

    # --- Docs specialist (local markdown knowledge base) ---
    knowledge_base_dir: Path = Field(
        default=Path("knowledge_base"),
        description="Directory of Markdown documents the docs specialist searches.",
    )
    docs_top_k: int = Field(default=3, ge=1, le=20, description="Sections per docs search.")

    # --- GitHub specialist (live source) ---
    github_username: str | None = Field(
        default=None, description="Default GitHub account for the live tool."
    )
    github_token: SecretStr | None = Field(
        default=None, description="Optional token to raise the GitHub rate limit."
    )
    github_api_base_url: str = Field(default="https://api.github.com")
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Observability ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    @property
    def resolved_router_model(self) -> str:
        """The model that makes routing decisions (dedicated one, or the chat model)."""
        return self.nvidia_router_model or self.nvidia_chat_model


def get_settings() -> Settings:
    """Build and validate a :class:`Settings` instance.

    Raises :class:`pydantic.ValidationError` if required configuration is missing.
    """
    return Settings()  # type: ignore[call-arg]
