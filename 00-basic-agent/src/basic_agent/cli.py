"""Command-line entry point.

Exposes ``basic-agent ask "<question>"``. Configuration errors (e.g. a missing API key)
are reported clearly instead of dumping a raw traceback.
"""

from __future__ import annotations

import typer
from pydantic import ValidationError

from basic_agent.agent import BasicAgent
from basic_agent.config import Settings, get_settings
from basic_agent.logging import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    help="Ask questions about a GitHub user's public repositories.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Ask questions about a GitHub user's public repositories.

    Declaring a callback keeps ``ask`` an explicit sub-command (rather than Typer
    collapsing a lone command into the root), leaving room for more commands later.
    """


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
    stream: bool = typer.Option(
        default=True,
        help="Stream the answer token-by-token (disable with --no-stream).",
    ),
) -> None:
    """Ask the agent a question about public GitHub repositories."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    logger = get_logger(__name__)
    logger.info("agent.ask", stream=stream)

    agent = BasicAgent(settings)

    if stream:
        for token in agent.stream(question):
            typer.echo(token, nl=False)
        typer.echo()
    else:
        typer.echo(agent.ask(question))


if __name__ == "__main__":  # pragma: no cover
    app()
