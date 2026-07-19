"""The router: classify a query, dispatch it to exactly ONE specialist.

Routing is not an open-ended reasoning task — it's classification. So instead of letting
the model ramble, we constrain it with **structured output**: the LLM must return a
:class:`RouteDecision` whose ``route`` field is a ``Literal`` over the specialist names.
The schema itself narrows what the model can say — an invalid destination is impossible by
construction, not caught by an if-statement afterwards.

The router is cheap (one small-model call, one hop) but structurally limited: it can pick
only one specialist, so a question that needs several sources gets, at best, a partial
answer. That limitation is the reason the supervisor exists (see ``supervisor.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestration_agent.config import Settings
from orchestration_agent.llm import build_router_llm
from orchestration_agent.logging import get_logger
from orchestration_agent.specialists import SPECIALIST_DESCRIPTIONS

logger = get_logger(__name__)

# Must stay in sync with the canonical names in ``specialists.py``. Keeping this a Literal
# (not a plain str) is the point: it becomes an enum in the JSON schema the LLM must obey.
RouteName = Literal["github", "docs", "general"]


class RouteDecision(BaseModel):
    """The router's verdict: which single specialist should handle the query."""

    route: RouteName = Field(description="The one specialist best suited to answer the query.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="How confident the router is in this choice (0 to 1)."
    )
    reason: str = Field(description="One short sentence justifying the choice.")


class RouteClassifier(Protocol):
    """Anything that can turn a question into a :class:`RouteDecision`."""

    def classify(self, question: str) -> RouteDecision: ...


def build_router_prompt(descriptions: Mapping[str, str]) -> str:
    """Render the classification prompt from the specialist descriptions."""
    catalogue = "\n".join(f"- `{name}`: {text}" for name, text in sorted(descriptions.items()))
    return (
        "You are a router for a team of specialist agents. Read the user's query and pick "
        "the ONE specialist best suited to answer it.\n\n"
        f"The specialists:\n{catalogue}\n\n"
        "Rules: choose exactly one route. If several specialists could contribute, pick "
        "the one most likely to hold the core of the answer and lower your confidence "
        "accordingly. Do not answer the query yourself."
    )


class LlmRouteClassifier:
    """Classify queries with an LLM constrained to the :class:`RouteDecision` schema."""

    def __init__(self, llm: Any, *, descriptions: Mapping[str, str] | None = None) -> None:
        self._structured_llm = llm.with_structured_output(RouteDecision)
        self._system_prompt = build_router_prompt(descriptions or SPECIALIST_DESCRIPTIONS)

    def classify(self, question: str) -> RouteDecision:
        raw = self._structured_llm.invoke(
            [SystemMessage(content=self._system_prompt), HumanMessage(content=question)]
        )
        decision = raw if isinstance(raw, RouteDecision) else RouteDecision.model_validate(raw)
        logger.debug(
            "router.classified",
            route=decision.route,
            confidence=decision.confidence,
            reason=decision.reason,
        )
        return decision


def build_route_classifier(settings: Settings, *, llm: Any | None = None) -> LlmRouteClassifier:
    """Build the production classifier (deterministic, possibly smaller routing model)."""
    return LlmRouteClassifier(llm if llm is not None else build_router_llm(settings))
