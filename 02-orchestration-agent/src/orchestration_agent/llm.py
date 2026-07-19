"""Chat-model factories, shared by the specialists, the router, and the supervisor.

Two builders instead of one because orchestration has two very different LLM jobs:
- **reasoning** (specialists + supervisor) wants the capable chat model;
- **routing** is a one-shot classification, so it can run on a smaller model, and always at
  temperature 0 — a routing decision should be deterministic, not creative.
"""

from __future__ import annotations

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from orchestration_agent.config import Settings


def build_chat_llm(settings: Settings) -> ChatNVIDIA:
    """Instantiate the reasoning chat model used by specialists and the supervisor."""
    return ChatNVIDIA(
        model=settings.nvidia_chat_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        seed=settings.llm_seed,
    )


def build_router_llm(settings: Settings) -> ChatNVIDIA:
    """Instantiate the routing model: possibly smaller, always deterministic."""
    return ChatNVIDIA(
        model=settings.resolved_router_model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=0.0,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        seed=settings.llm_seed,
    )
