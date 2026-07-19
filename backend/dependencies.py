"""FastAPI dependencies (BUILD_PLAN T54)."""

from __future__ import annotations

from fastapi import Request

from backend.core.config import Settings, get_settings
from backend.services.registry import ModelRegistry


def registry(request: Request) -> ModelRegistry:
    """The process-wide registry, created in lifespan and stored on app.state."""
    return request.app.state.registry


def settings() -> Settings:
    return get_settings()


def job_store(request: Request):
    return request.app.state.jobs
