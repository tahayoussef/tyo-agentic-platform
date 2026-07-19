"""A callback handler that prints the agent's tool calls and results, for diagnostics.

Ported from 01-rag-agent. In THIS project the trace is how you *see* orchestration happen:

- in supervisor mode, the tool calls are the delegations (`consult_docs_specialist`, ...) —
  the trace shows which specialists were consulted and what each reported back. The
  specialists' own inner tool calls do NOT appear: they run in isolated contexts, and the
  trace makes that isolation visible.
- in router mode, the chosen specialist runs with these callbacks, so the trace shows its
  internal tool calls (`search_project_docs`, `list_github_repositories`).
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
