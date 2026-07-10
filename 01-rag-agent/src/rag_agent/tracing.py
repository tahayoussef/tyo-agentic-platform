"""A callback handler that prints the agent's tool calls and results, for diagnostics.

LangChain emits callback events at every step of the ReAct loop (`on_tool_start`,
`on_tool_end`, ...). This handler listens for the tool events and prints them to stderr, so
you can see *which* tools the agent chose, with *what* arguments, and *what each returned* —
the information needed to tell whether the model consulted both sources and reconciled them.

It's the same integration point a real observability backend (LangSmith / Langfuse) plugs
into; this is the minimal, local version.
"""

from __future__ import annotations

from typing import Any

import typer
from langchain_core.callbacks import BaseCallbackHandler


class ToolTraceCallbackHandler(BaseCallbackHandler):
    """Print each tool invocation and its result to stderr."""

    def __init__(self, *, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool") if serialized else "tool"
        inputs = kwargs.get("inputs")
        args = inputs if inputs is not None else input_str
        typer.secho(f"\n→ tool call: {name}  args={args}", fg=typer.colors.YELLOW, err=True)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        typer.secho("← tool result:", fg=typer.colors.BLUE, err=True)
        typer.echo(self._format(output), err=True)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        typer.secho(f"← tool error: {error}", fg=typer.colors.RED, err=True)

    def _format(self, output: Any) -> str:
        content = getattr(output, "content", output)
        text = content if isinstance(content, str) else str(content)
        if len(text) > self.max_chars:
            return f"{text[: self.max_chars]}\n… [+{len(text) - self.max_chars} more chars]"
        return text
