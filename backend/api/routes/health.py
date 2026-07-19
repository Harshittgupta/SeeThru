"""Liveness and readiness (BUILD_PLAN T54).

The distinction is load-bearing for anything that runs this behind an
orchestrator:

* ``/health`` -- **liveness. Touches nothing**: no GPU, no model, no disk. It
  returns 200 the instant the process is up, *including while weights are still
  loading*. If it depended on the model, the orchestrator would see it fail
  during a slow startup and kill the pod mid-load, forever.
* ``/ready`` -- **readiness**. 200 only when the models are loaded and warmed;
  503 otherwise. This is what a load balancer should gate traffic on.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response, status

from backend.dependencies import registry
from backend.schemas.responses import Health, Readiness
from backend.services.registry import ModelRegistry

router = APIRouter(tags=["health"])
_STARTED = time.monotonic()
VERSION = "0.1.0"


@router.get("/health", response_model=Health)
async def health() -> Health:
    # Deliberately does NOT depend on the registry. Liveness must not require the
    # model, or a slow load looks like a crash.
    return Health(status="ok", version=VERSION, uptime_s=time.monotonic() - _STARTED)


@router.get("/ready", response_model=Readiness)
async def ready(response: Response, reg: ModelRegistry = Depends(registry)) -> Readiness:
    if reg.ready:
        return Readiness(ready=True)
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Readiness(
        ready=False,
        reason="Model weights are not loaded. Set SEETHRU_IMAGE_WEIGHTS, or "
               "SEETHRU_ALLOW_UNTRAINED=true for local development.",
    )
