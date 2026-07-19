"""Structured request logging (BUILD_PLAN T54).

JSON to stdout, one line per request, each carrying a ``request_id`` so a report
of "request abc123 failed" is actually traceable. The id is accepted from an
inbound ``X-Request-ID`` header (so a proxy's id survives) or minted, and echoed
on the response.

Filenames are never logged raw -- an attacker controls them, and a newline in a
filename would otherwise forge log lines.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if rid := request_id_var.get():
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request id and log one structured line per request."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        logger = logging.getLogger("seethru.api")
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # perf_counter is monotonic; safe unlike wall-clock (which the
            # workflow sandbox even forbids). Fine in the live server.
            logger.exception(
                "request failed method=%s path=%s", request.method, request.url.path
            )
            raise
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "method=%s path=%s status=%d dur_ms=%.1f",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        response.headers["X-Request-ID"] = rid
        return response
