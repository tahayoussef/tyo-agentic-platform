"""The supervisor: an LLM coordinator that can consult SEVERAL specialists and synthesize.

This is the "agents as tools" pattern: each specialist is wrapped in a ``consult_*`` tool,
and the supervisor is itself a ReAct agent whose only tools are those delegations. The
ReAct loop it inherits is exactly what routing lacked — the supervisor can consult one
specialist, read the answer, decide it needs another, consult again, and then write one
synthesized reply.

Each delegation passes a *question* and returns a *final answer* — the specialist's inner
conversation (its own tool calls and reasoning) never enters the supervisor's context.
That context isolation is the main argument for this pattern over handoff-style
orchestration, where agents share one message history.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from orchestration_agent.config import Settings
from orchestration_agent.llm import build_chat_llm
from orchestration_agent.logging import get_logger
from orchestration_agent.specialists import (
    GENERAL,
    GraphRunner,
    Specialist,
    build_specialists,
)

logger = get_logger(__name__)

SUPERVISOR_SYSTEM_PROMPT = (
    "You are the supervisor of a team of specialist agents that together answer questions "
    "about a portfolio of GitHub repositories. You do not have direct access to any data — "
    "you delegate by calling the `consult_*` tools, each of which asks one specialist a "
    "question and returns its answer.\n\n"
    "Send each specialist a focused, self-contained question (it cannot see this "
    "conversation). Consult EVERY specialist whose source the question touches — questions "
    "that compare documentation with live reality need both the docs and the github "
    "specialists. Then synthesize one answer: attribute claims to their source ('the docs "
    "say…', 'live GitHub shows…'), and when sources disagree, flag the discrepancy and "
    "prefer live data for present-day facts (the documentation may be stale).\n\n"
    "Only greetings, small talk, or questions about your own capabilities may be answered "
    "directly without consulting anyone. Never invent repository facts yourself."
)

_TOOL_PREFIX = "consult_"
_TOOL_SUFFIX = "_specialist"


def delegation_tool_name(specialist_name: str) -> str:
    """The tool name under which a specialist is exposed to the supervisor."""
    return f"{_TOOL_PREFIX}{specialist_name}{_TOOL_SUFFIX}"


def specialist_name_from_tool(tool_name: str) -> str | None:
    """Invert :func:`delegation_tool_name`; ``None`` for tools that aren't delegations."""
    if tool_name.startswith(_TOOL_PREFIX) and tool_name.endswith(_TOOL_SUFFIX):
        return tool_name[len(_TOOL_PREFIX) : -len(_TOOL_SUFFIX)]
    return None


class ConsultInput(BaseModel):
    """Arguments for a ``consult_*`` delegation tool."""

    question: str = Field(
        description=(
            "A focused, self-contained question for this specialist. Include all context "
            "it needs — it cannot see the conversation."
        )
    )


def _delegation_tool(specialist: Specialist) -> BaseTool:
    """Wrap one specialist as a tool the supervisor can call."""

    def consult(question: str) -> str:
        logger.debug("supervisor.delegate", specialist=specialist.name, question=question)
        return specialist.runner.ask(question)

    return StructuredTool.from_function(
        func=consult,
        name=delegation_tool_name(specialist.name),
        description=f"Ask the {specialist.name} specialist. Covers: {specialist.description}",
        args_schema=ConsultInput,
    )


def build_delegation_tools(specialists: Mapping[str, Specialist]) -> list[BaseTool]:
    """One delegation tool per specialist — except ``general``, whose job (small talk)
    the supervisor handles directly rather than paying an extra hop for."""
    return [
        _delegation_tool(specialist)
        for name, specialist in sorted(specialists.items())
        if name != GENERAL
    ]


def build_supervisor(
    settings: Settings,
    *,
    specialists: Mapping[str, Specialist] | None = None,
    llm: Any | None = None,
) -> GraphRunner:
    """Compile the supervisor agent over the specialist team."""
    llm = llm if llm is not None else build_chat_llm(settings)
    team = specialists if specialists is not None else build_specialists(settings, llm=llm)
    tools = build_delegation_tools(team)
    logger.debug("supervisor.build", tools=[tool.name for tool in tools])
    graph = create_react_agent(llm, tools, prompt=SUPERVISOR_SYSTEM_PROMPT)
    return GraphRunner(graph)
