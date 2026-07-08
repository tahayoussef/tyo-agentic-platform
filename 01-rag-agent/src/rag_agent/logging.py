"""Structured logging setup built on ``structlog`` (identical pattern to 00-basic-agent).

Console-friendly output for local development; single-line JSON for production. Logs go to
stderr so they never interleave with command output on stdout.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(*, level: str = "INFO", json_logs: bool = False) -> None:
    """Configure process-wide structured logging."""
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        renderer,
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound structlog logger, optionally namespaced by ``name``."""
    logger: FilteringBoundLogger = structlog.get_logger(name) if name else structlog.get_logger()
    return logger
