"""Tests for the tool-trace callback handler."""

from __future__ import annotations

import pytest

from rag_agent.tracing import ToolTraceCallbackHandler


def test_prints_tool_call_and_result(capsys: pytest.CaptureFixture[str]) -> None:
    handler = ToolTraceCallbackHandler()

    handler.on_tool_start({"name": "search_knowledge_base"}, "query", inputs={"query": "dbt"})
    handler.on_tool_end("some retrieved chunk")

    err = capsys.readouterr().err
    assert "search_knowledge_base" in err
    assert "query" in err
    assert "some retrieved chunk" in err


def test_truncates_long_output(capsys: pytest.CaptureFixture[str]) -> None:
    handler = ToolTraceCallbackHandler(max_chars=10)

    handler.on_tool_start({"name": "t"}, "in")
    handler.on_tool_end("x" * 50)

    err = capsys.readouterr().err
    assert "more chars]" in err
