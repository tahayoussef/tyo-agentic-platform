"""The orchestration agent: one typed facade, two interchangeable orchestration styles.

``OrchestrationAgent`` exposes the same ``ask``/``stream`` surface as 00/01's facades, but
behind it sit two different architectures selected by ``mode``:

- **router** — classify once, dispatch to exactly one specialist (cheap, single hop);
- **supervisor** — an LLM coordinator that may consult several specialists and synthesize.

Keeping one facade over both is deliberate: callers (the CLI, the eval harness) don't
change when the orchestration style does, which is what makes the two styles comparable.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from langchain_core.callbacks import BaseCallbackHandler

from orchestration_agent.config import Settings
from orchestration_agent.logging import get_logger
from orchestration_agent.router import RouteClassifier, RouteDecision, build_route_classifier
from orchestration_agent.specialists import Runner, Specialist, build_specialists
from orchestration_agent.supervisor import build_supervisor

logger = get_logger(__name__)

ROUTER = "router"
SUPERVISOR = "supervisor"
MODES = (ROUTER, SUPERVISOR)


class OrchestrationAgent:
    """A typed facade that answers questions via the configured orchestration style."""

    def __init__(
        self,
        settings: Settings,
        *,
        mode: str | None = None,
        classifier: RouteClassifier | None = None,
        specialists: Mapping[str, Specialist] | None = None,
        supervisor: Runner | None = None,
    ) -> None:
        resolved = (mode or settings.orchestration_mode).lower()
        if resolved not in MODES:
            raise ValueError(f"Unknown orchestration mode '{resolved}'. Expected one of {MODES}.")
        self._settings = settings
        self._mode = resolved

        self._classifier: RouteClassifier | None = None
        self._specialists: Mapping[str, Specialist] = {}
        self._supervisor: Runner | None = None
        if resolved == ROUTER:
            self._specialists = (
                specialists if specialists is not None else build_specialists(settings)
            )
            self._classifier = (
                classifier if classifier is not None else build_route_classifier(settings)
            )
        else:
            self._supervisor = (
                supervisor
                if supervisor is not None
                else build_supervisor(settings, specialists=specialists)
            )

    @property
    def mode(self) -> str:
        return self._mode

    def ask(self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None) -> str:
        """Answer ``question`` and return the final assistant message."""
        if self._supervisor is not None:
            return self._supervisor.ask(question, callbacks=callbacks)
        _decision, specialist = self._dispatch(question)
        return specialist.runner.ask(question, callbacks=callbacks)

    def stream(
        self, question: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
    ) -> Iterator[str]:
        """Yield the answer token-by-token as it is generated."""
        if self._supervisor is not None:
            yield from self._supervisor.stream(question, callbacks=callbacks)
            return
        _decision, specialist = self._dispatch(question)
        yield from specialist.runner.stream(question, callbacks=callbacks)

    def route(self, question: str) -> RouteDecision:
        """Classify ``question`` without dispatching it (router mode only)."""
        if self._classifier is None:
            raise RuntimeError("route() is only available in router mode.")
        return self._classifier.classify(question)

    def _dispatch(self, question: str) -> tuple[RouteDecision, Specialist]:
        """Classify the question and resolve the specialist it should go to."""
        if self._classifier is None:  # pragma: no cover - guarded by __init__
            raise RuntimeError("Router components are not built in this mode.")
        decision = self._classifier.classify(question)
        specialist = self._specialists.get(decision.route)
        if specialist is None:
            raise ValueError(f"Router chose unknown specialist '{decision.route}'.")
        logger.info(
            "router.dispatch",
            route=decision.route,
            confidence=decision.confidence,
            reason=decision.reason,
        )
        return decision, specialist
