"""
Structured JSON logger for LLM call observability.

Logs prompt metadata, response metadata, latency, and token usage to:
1. Console (human-readable)
2. logs/qa_analysis.jsonl (machine-readable JSONL for monitoring/analytics)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# JSONL file handler
# ──────────────────────────────────────────────────────────────────────────────

_LOG_DIR = Path("logs")
_JSONL_LOG_FILE = _LOG_DIR / "qa_analysis.jsonl"


class JSONLHandler(logging.Handler):
    """Appends log records as newline-delimited JSON to a file."""

    def __init__(self, filepath: Path):
        super().__init__()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Merge any extra structured fields passed to the logger
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "levelname", "levelno", "pathname",
                    "filename", "module", "exc_info", "exc_text", "stack_info",
                    "lineno", "funcName", "created", "msecs", "relativeCreated",
                    "thread", "threadName", "processName", "process", "message",
                    "taskName",
                ):
                    log_entry[key] = value
            self._file.write(json.dumps(log_entry) + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        self._file.close()
        super().close()


# ──────────────────────────────────────────────────────────────────────────────
# Structured logger adapter
# ──────────────────────────────────────────────────────────────────────────────

class StructuredLogger(logging.LoggerAdapter):
    """Wraps a standard logger to support structured key=value fields."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = kwargs.pop("extra", {})
        # Merge any remaining kwargs as structured fields
        extra.update({k: v for k, v in kwargs.items() if k not in ("exc_info", "stack_info")})
        kwargs = {k: v for k, v in kwargs.items() if k in ("exc_info", "stack_info")}
        kwargs["extra"] = extra
        return msg, kwargs

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        msg, kwargs = self.process(msg, kwargs)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        msg, kwargs = self.process(msg, kwargs)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        msg, kwargs = self.process(msg, kwargs)
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        msg, kwargs = self.process(msg, kwargs)
        self.logger.debug(msg, *args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_loggers: dict[str, StructuredLogger] = {}


def configure_logging() -> None:
    """
    Configure root logger. Call once at application startup.
    - Console: human-readable format
    - JSONL file: machine-readable (logs/qa_analysis.jsonl)
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        console.setFormatter(fmt)
        root.addHandler(console)

    # JSONL file handler
    if not any(isinstance(h, JSONLHandler) for h in root.handlers):
        root.addHandler(JSONLHandler(_JSONL_LOG_FILE))


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for the given module name."""
    if name not in _loggers:
        base = logging.getLogger(name)
        _loggers[name] = StructuredLogger(base, {})
    return _loggers[name]
