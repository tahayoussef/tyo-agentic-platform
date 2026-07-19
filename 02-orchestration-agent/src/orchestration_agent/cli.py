"""Command-line entry point for the orchestration agent.

Commands:
  - ``orchestration-agent ask``    answer a question via the router or the supervisor
  - ``orchestration-agent route``  show the routing decision only (no specialist runs)
  - ``orchestration-agent eval``   score routing accuracy or delegation coverage
"""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from orchestration_agent.agent import MODES, ROUTER, SUPERVISOR, OrchestrationAgent
from orchestration_agent.config import Settings, get_settings
from orchestration_agent.evaluation import (
    DelegationRecorder,
    RouterReport,
    SupervisorReport,
    evaluate_router,
    evaluate_supervisor,
    load_gold,
    specialists_consulted,
)
from orchestration_agent.logging import configure_logging, get_logger
from orchestration_agent.router import build_route_classifier
from orchestration_agent.tracing import ToolTraceCallbackHandler

app = typer.Typer(
    add_completion=False,
    help="Orchestrate specialist agents: route to one, or supervise several.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Orchestrate specialist agents: route to one, or supervise several."""


def _load_settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        typer.secho(
            "Configuration error: check your environment / .env file (is NVIDIA_API_KEY set?).",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the agent."),
    mode: str = typer.Option(
        None, help=f"Orchestration style: {' | '.join(MODES)} (defaults to ORCHESTRATION_MODE)."
    ),
    stream: bool = typer.Option(
        default=True,
        help="Stream the answer token-by-token (disable with --no-stream).",
    ),
    show_trace: bool = typer.Option(
        default=False,
        help="Print tool calls to stderr: delegations (supervisor) or specialist tools (router).",
    ),
) -> None:
    """Ask the agent; the orchestration style decides who actually answers."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    get_logger(__name__).info("cli.ask", mode=mode, stream=stream, show_trace=show_trace)

    try:
        agent = OrchestrationAgent(settings, mode=mode)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    callbacks = [ToolTraceCallbackHandler()] if show_trace else None

    if stream:
        for token in agent.stream(question, callbacks=callbacks):
            typer.echo(token, nl=False)
        typer.echo()
    else:
        typer.echo(agent.ask(question, callbacks=callbacks))


@app.command()
def route(
    question: str = typer.Argument(..., help="The question to classify."),
) -> None:
    """Show where the router would send a question — without running any specialist."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    decision = build_route_classifier(settings).classify(question)
    typer.secho(f"route:      {decision.route}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"confidence: {decision.confidence:.2f}")
    typer.echo(f"reason:     {decision.reason}")


_DEFAULT_GOLD_PATH = Path("eval/gold.yaml")
_GOLD_PATH_OPTION = typer.Option(_DEFAULT_GOLD_PATH, help="Path to the gold question set (YAML).")


def _print_router_report(report: RouterReport) -> None:
    typer.secho(f"\nRouting eval — {len(report.results)} cases\n", bold=True)
    header = f"{'ok':>3} {'conf':>5}  {'predicted':<10} {'expected':<15} question"
    typer.echo(header)
    typer.echo("-" * len(header))
    for result in report.results:
        question = result.case.question
        if len(question) > 42:
            question = question[:39] + "..."
        expected = "|".join(sorted(result.case.routes))
        typer.echo(
            f"{int(result.correct):>3} {result.decision.confidence:>5.2f}  "
            f"{result.decision.route:<10} {expected:<15} {question}"
        )
    typer.echo("-" * len(header))
    typer.secho(f"accuracy: {report.accuracy:.2f}", fg=typer.colors.GREEN, bold=True)
    for miss in report.misroutes:
        typer.secho(
            f"  misroute: {miss.decision.route!r} for {miss.case.question!r} "
            f"({miss.decision.reason})",
            fg=typer.colors.YELLOW,
        )


def _print_supervisor_report(report: SupervisorReport) -> None:
    typer.secho(f"\nDelegation eval — {len(report.results)} cases\n", bold=True)
    header = f"{'cov':>3} {'extra':>5}  {'consulted':<15} {'needed':<15} question"
    typer.echo(header)
    typer.echo("-" * len(header))
    for result in report.results:
        question = result.case.question
        if len(question) > 38:
            question = question[:35] + "..."
        consulted = "+".join(result.consulted) or "-"
        needed = "+".join(sorted(result.case.needs)) or "-"
        typer.echo(
            f"{int(result.covered):>3} {len(result.extra):>5}  {consulted:<15} {needed:<15} "
            f"{question}"
        )
    typer.echo("-" * len(header))
    typer.secho(
        f"coverage: {report.coverage:.2f}   "
        f"mean extra delegations: {report.mean_extra_delegations:.2f}",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command("eval")
def eval_command(
    gold_path: Path = _GOLD_PATH_OPTION,
    mode: str = typer.Option(
        ROUTER, help="What to evaluate: 'router' (accuracy) or 'supervisor' (coverage)."
    ),
) -> None:
    """Evaluate orchestration against the gold set (routing accuracy / delegation coverage)."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    cases = load_gold(gold_path)

    if mode == ROUTER:
        _print_router_report(evaluate_router(build_route_classifier(settings), cases))
    elif mode == SUPERVISOR:
        agent = OrchestrationAgent(settings, mode=SUPERVISOR)

        def consult(question: str) -> tuple[str, ...]:
            recorder = DelegationRecorder()
            agent.ask(question, callbacks=[recorder])
            return specialists_consulted(recorder.tool_calls)

        _print_supervisor_report(evaluate_supervisor(consult, cases))
    else:
        typer.secho(f"Unknown eval mode '{mode}'. Expected one of {MODES}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
