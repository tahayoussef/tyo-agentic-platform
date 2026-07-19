"""Tests for the tool-trace callback handler (output shape and truncation)."""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from orchestration_agent.tracing import ToolTraceCallbackHandler


def test_on_tool_start_prints_name_and_args(capsys: pytest.CaptureFixture[str]) -> None:
    handler = ToolTraceCallbackHandler()

    handler.on_tool_start({"name": "consult_docs_specialist"}, "what is gobekli?")

    err = capsys.readouterr().err
    assert "consult_docs_specialist" in err
    assert "what is gobekli?" in err


def test_on_tool_end_prints_message_content(capsys: pytest.CaptureFixture[str]) -> None:
    handler = ToolTraceCallbackHandler()

    handler.on_tool_end(ToolMessage(content="the docs say X", tool_call_id="1"))

    assert "the docs say X" in capsys.readouterr().err


def test_long_output_is_truncated(capsys: pytest.CaptureFixture[str]) -> None:
    handler = ToolTraceCallbackHandler(max_chars=10)

    handler.on_tool_end("x" * 50)

    err = capsys.readouterr().err
    assert "xxxxxxxxxx" in err
    assert "+40 more chars" in err


def test_on_tool_error_prints_error(capsys: pytest.CaptureFixture[str]) -> None:
    handler = ToolTraceCallbackHandler()

    handler.on_tool_error(RuntimeError("boom"))

    assert "boom" in capsys.readouterr().err
