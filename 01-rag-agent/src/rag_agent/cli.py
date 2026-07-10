"""Command-line entry point for the RAG agent.

Commands:
  - ``rag-agent ingest``   build/refresh the Qdrant index from the knowledge base
  - ``rag-agent search``   run a raw similarity search (inspect retrieval quality)
  - ``rag-agent ask``      ask the agent, which fuses the knowledge base with live GitHub

The callback keeps these as explicit sub-commands so more can be added cleanly.
"""

from __future__ import annotations

import typer
from pydantic import ValidationError

from rag_agent.agent import RagAgent
from rag_agent.config import Settings, get_settings
from rag_agent.embeddings import build_embeddings
from rag_agent.ingest import run_ingest
from rag_agent.logging import configure_logging, get_logger
from rag_agent.retriever import search as run_search
from rag_agent.tracing import ToolTraceCallbackHandler
from rag_agent.vector_store import build_qdrant_client

app = typer.Typer(
    add_completion=False,
    help="RAG over a Qdrant knowledge base of GitHub repositories.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """RAG over a Qdrant knowledge base of GitHub repositories."""


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
def ingest(
    recreate: bool = typer.Option(
        default=False,
        help="Drop and recreate the collection before ingesting (destroys existing data).",
    ),
) -> None:
    """Index the knowledge base into Qdrant."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    logger = get_logger(__name__)
    logger.info("cli.ingest", recreate=recreate)

    embeddings = build_embeddings(settings)
    client = build_qdrant_client(settings)
    report = run_ingest(settings, embeddings=embeddings, client=client, recreate=recreate)

    typer.secho(
        f"Ingested {report.chunks} chunks from {report.files} files into "
        f"'{report.collection}' (vector dim = {report.dimension}).",
        fg=typer.colors.GREEN,
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="The text to search the knowledge base for."),
    top_k: int = typer.Option(None, help="Number of chunks to return (defaults to TOP_K)."),
) -> None:
    """Run a raw similarity search against the indexed knowledge base."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    embeddings = build_embeddings(settings)
    client = build_qdrant_client(settings)
    results = run_search(settings, query, embeddings=embeddings, client=client, top_k=top_k)

    if not results:
        typer.echo("No results. Have you run `rag-agent ingest`?")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        repo = doc.metadata.get("repo", "?")
        snippet = " ".join(doc.page_content.split())[:200]
        typer.secho(f"\n#{rank}  score={score:.4f}  repo={repo}", fg=typer.colors.CYAN)
        typer.echo(f"    {snippet}...")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the agent."),
    stream: bool = typer.Option(
        default=True,
        help="Stream the answer token-by-token (disable with --no-stream).",
    ),
    show_trace: bool = typer.Option(
        default=False,
        help="Print each tool call and its result to stderr (diagnose the ReAct loop).",
    ),
) -> None:
    """Ask the agent: it fuses the knowledge base with live GitHub and reconciles them."""
    settings = _load_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    get_logger(__name__).info("cli.ask", stream=stream, show_trace=show_trace)

    agent = RagAgent(settings)
    callbacks = [ToolTraceCallbackHandler()] if show_trace else None

    if stream:
        for token in agent.stream(question, callbacks=callbacks):
            typer.echo(token, nl=False)
        typer.echo()
    else:
        typer.echo(agent.ask(question, callbacks=callbacks))


if __name__ == "__main__":  # pragma: no cover
    app()
