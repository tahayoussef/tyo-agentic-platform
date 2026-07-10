"""Application configuration, validated and loaded from the environment.

Mirrors the 00-basic-agent approach: typed settings via ``pydantic-settings``, secrets as
``SecretStr``, nothing hardcoded. Adds RAG-specific knobs (embedding model, Qdrant
connection, chunking, retrieval) that we will tune in the optimization phase.
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

    # --- NVIDIA / embeddings ---
    nvidia_api_key: SecretStr = Field(description="API key for NVIDIA AI endpoints.")
    embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="NVIDIA embedding model. Determines the vector dimension.",
    )
    embedding_truncate: str = Field(
        default="END",
        description="How to handle over-long inputs: NONE | START | END.",
    )

    # --- NVIDIA / chat model (the agent's reasoning LLM) ---
    nvidia_chat_model: str = Field(default="z-ai/glm-5.2")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_seed: int | None = Field(default=42)

    # --- Qdrant (vector DB) ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: SecretStr | None = Field(
        default=None, description="Only needed for secured / cloud Qdrant."
    )
    collection_name: str = Field(default="github_repos")

    # --- Ingestion / chunking ---
    knowledge_base_dir: Path = Field(
        default=Path("github_repos_readmes"),
        description="Directory of Markdown documents to index.",
    )
    chunk_size: int = Field(default=1000, gt=0, description="Target characters per chunk.")
    chunk_overlap: int = Field(default=150, ge=0, description="Character overlap between chunks.")

    # --- Retrieval ---
    top_k: int = Field(default=4, ge=1, le=50, description="Chunks to return per search.")

    # --- GitHub (live source) ---
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


def get_settings() -> Settings:
    """Build and validate a :class:`Settings` instance.

    Raises :class:`pydantic.ValidationError` if required configuration is missing.
    """
    return Settings()  # type: ignore[call-arg]
