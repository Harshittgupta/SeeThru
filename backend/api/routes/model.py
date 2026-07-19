"""Model info (BUILD_PLAN T54/T58)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.dependencies import registry
from backend.services.registry import ModelRegistry

router = APIRouter(prefix="/v1", tags=["model"])


@router.get("/model/info")
async def model_info(reg: ModelRegistry = Depends(registry)) -> dict:
    """Arch, version, device, calibration, and the non-strippable disclaimer.

    Repeats the disclaimer that rides on every prediction (T58): a client that
    only ever hits this endpoint still learns the model's limitations.
    """
    return reg.model_info()
