# -*- coding: utf-8 -*-

import json
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path


DEFAULT_GENERATION_LOG_PATH = "logs/generation.jsonl"
_LOG_CONTEXT = ContextVar("generation_log_context", default={})
_GENERATION_DEADLINE = ContextVar("generation_deadline", default=None)
_WRITE_LOCK = threading.Lock()
_DATA_URL_PATTERN = re.compile(
    r"data:[^;,\s]+(?:;[^,\s]+)*;base64,[A-Za-z0-9+/=_-]+",
    flags=re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", flags=re.IGNORECASE)


def _sanitize_log_value(value, key=""):
    normalized_key = str(key or "").casefold()
    if any(secret in normalized_key for secret in ("api_key", "authorization", "access_token")):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _sanitize_log_value(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, bytes):
        return f"[bytes omitted: {len(value)} bytes]"
    if isinstance(value, str):
        sanitized = _DATA_URL_PATTERN.sub("[base64 image omitted]", value)
        return _BEARER_PATTERN.sub("Bearer [REDACTED]", sanitized)
    return value


def write_log(payload, log_path="logs/llm.jsonl"):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _sanitize_log_value(dict(payload))
    record["created_at"] = datetime.now().isoformat()

    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def generation_log_context(
    log_path=None,
    enabled=None,
    deadline_seconds=None,
    deadline_monotonic=None,
    **fields,
):
    current = dict(_LOG_CONTEXT.get() or {})
    current.update(fields)
    if log_path:
        current["_generation_log_path"] = str(log_path)
        current["_generation_logging_enabled"] = True
    if enabled is not None:
        current["_generation_logging_enabled"] = bool(enabled)
    token = _LOG_CONTEXT.set(current)
    if deadline_seconds is not None:
        current["deadline_seconds"] = deadline_seconds
    deadline_token = None
    if deadline_monotonic is not None:
        deadline_token = _GENERATION_DEADLINE.set(float(deadline_monotonic))
    elif deadline_seconds is not None:
        deadline_token = _GENERATION_DEADLINE.set(time.perf_counter() + float(deadline_seconds))
    try:
        yield
    finally:
        if deadline_token is not None:
            _GENERATION_DEADLINE.reset(deadline_token)
        _LOG_CONTEXT.reset(token)


def get_generation_deadline():
    """Return the monotonic deadline for the current request, if configured."""
    return _GENERATION_DEADLINE.get()


def write_generation_event(stage, status, **details):
    context = dict(_LOG_CONTEXT.get() or {})
    enabled = bool(context.pop("_generation_logging_enabled", False))
    if not enabled:
        return
    log_path = context.pop("_generation_log_path", DEFAULT_GENERATION_LOG_PATH)
    write_log(
        {
            "record_type": "generation_event",
            "stage": str(stage),
            "status": str(status),
            **context,
            **details,
        },
        log_path=log_path,
    )


@contextmanager
def generation_stage(stage, **details):
    started_at = time.perf_counter()
    write_generation_event(stage, "started", **details)
    try:
        yield
    except Exception as exc:
        write_generation_event(
            stage,
            "failed",
            **details,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
        )
        raise
    else:
        write_generation_event(
            stage,
            "completed",
            **details,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
