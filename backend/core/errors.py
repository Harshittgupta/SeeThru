"""One error envelope for the whole API (BUILD_PLAN T56).

Every error the client sees has the same shape::

    {"error_code": "no_face_detected", "message": "...", "request_id": "..."}

so the frontend can switch on ``error_code`` (T64) instead of parsing prose, and
every response is traceable by ``request_id``.

The error codes are a contract, not an implementation detail. ``no_face_detected``
in particular is a **422, not a guess** -- the model is only meaningful on an
aligned face crop, so "I could not find a face" must never collapse into a
real/fake answer.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

# Raw integers rather than starlette.status.HTTP_* constants: Starlette is mid-
# rename (HTTP_422_UNPROCESSABLE_ENTITY -> ..._CONTENT) and the old names emit a
# DeprecationWarning at CLASS-DEFINITION time, before any pytest filter applies.
# The numbers are stable across every HTTP version; the names are not.
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_410_GONE = 410
HTTP_413_TOO_LARGE = 413
HTTP_415_UNSUPPORTED = 415
HTTP_422_UNPROCESSABLE = 422
HTTP_503_UNAVAILABLE = 503


class ApiError(Exception):
    """Base for every error rendered through the envelope."""

    status_code = HTTP_400_BAD_REQUEST
    error_code = "bad_request"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- 4xx: the client sent something we cannot use ---
class NoFaceDetected(ApiError):
    # 422, deliberately: the request was well-formed, we just cannot analyse it.
    # Collapsing this into a real/fake verdict would be a fabricated result.
    status_code = HTTP_422_UNPROCESSABLE
    error_code = "no_face_detected"


class InsufficientFaces(ApiError):
    status_code = HTTP_422_UNPROCESSABLE
    error_code = "insufficient_faces"


class UnreadableMedia(ApiError):
    status_code = HTTP_422_UNPROCESSABLE
    error_code = "unreadable_media"


class UnsupportedMedia(ApiError):
    status_code = HTTP_415_UNSUPPORTED
    error_code = "unsupported_media"


class PayloadTooLarge(ApiError):
    status_code = HTTP_413_TOO_LARGE
    error_code = "payload_too_large"


class JobNotFound(ApiError):
    status_code = HTTP_404_NOT_FOUND
    error_code = "job_not_found"


class JobExpired(ApiError):
    status_code = HTTP_410_GONE
    error_code = "job_expired"


# --- 5xx / 503: we cannot serve right now ---
class ModelNotReady(ApiError):
    status_code = HTTP_503_UNAVAILABLE
    error_code = "model_not_ready"


class GpuBusy(ApiError):
    status_code = HTTP_503_UNAVAILABLE
    error_code = "gpu_busy"


class QueueFull(ApiError):
    status_code = HTTP_503_UNAVAILABLE
    error_code = "queue_full"


def _envelope(request: Request, code: str, message: str, details: dict) -> dict:
    return {
        "error_code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", None),
        "details": details,
    }


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    headers = {}
    # 503s are transient; tell the client it is worth retrying (T64).
    if exc.status_code == HTTP_503_UNAVAILABLE:
        headers["Retry-After"] = str(exc.details.get("retry_after", 5))
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(request, exc.error_code, exc.message, exc.details),
        headers=headers,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: an unexpected error must still return the envelope, not a
    stack trace, and must not leak internals to the client."""
    import logging

    logging.getLogger("seethru.api").exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=_envelope(
            request, "internal_error", "An unexpected error occurred.", {}
        ),
    )
