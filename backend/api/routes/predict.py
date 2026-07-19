"""Prediction endpoints (BUILD_PLAN T56/T57).

* ``POST /v1/predict/image`` -- synchronous. A single face is ~1-2 s, fine to
  hold on the request.
* ``POST /v1/predict/video`` -- **202 + job_id**. A video is 30 s - 5 min and
  cannot be held on an HTTP request (T57), so it is queued and polled.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from backend.core.config import Settings
from backend.dependencies import job_store, registry, settings
from backend.schemas.responses import ImagePrediction, JobAccepted
from backend.services.inference import InferenceService, build_media_block
from backend.services.registry import ModelRegistry
from backend.services.uploads import (
    IMAGE_TYPES,
    VIDEO_TYPES,
    load_validated_image,
    probe_video,
    save_upload,
)

logger = logging.getLogger("seethru.predict")
router = APIRouter(prefix="/v1", tags=["predict"])


@router.post("/predict/image", response_model=ImagePrediction)
async def predict_image(
    request: Request,
    file: UploadFile,
    reg: ModelRegistry = Depends(registry),
    cfg: Settings = Depends(settings),
) -> ImagePrediction:
    reg.require_image()  # 503 model_not_ready before we bother reading the upload

    saved = await save_upload(
        file,
        max_bytes=cfg.max_image_bytes,
        allowed=IMAGE_TYPES,
        tmp_dir=Path(cfg.upload_tmp_dir),
        declared_length=_content_length(request),
    )
    try:
        raw = saved.path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        # Decoding is CPU-bound; keep it off the event loop.
        image_bgr = await run_in_threadpool(
            load_validated_image, saved.path, cfg.max_image_pixels
        )

        artifact_dir = Path(cfg.artifact_dir) / sha[:16]
        service = InferenceService(reg, cfg)
        result = await service.analyze_image(image_bgr, artifact_dir)
    finally:
        saved.cleanup()  # the temp upload is gone whatever happened

    return _image_response(request, reg, result, saved, sha, artifact_dir.name)


@router.post(
    "/predict/video",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def predict_video(
    request: Request,
    file: UploadFile,
    reg: ModelRegistry = Depends(registry),
    cfg: Settings = Depends(settings),
    jobs=Depends(job_store),
) -> JobAccepted:
    reg.require_video()

    saved = await save_upload(
        file,
        max_bytes=cfg.max_video_bytes,
        allowed=VIDEO_TYPES,
        tmp_dir=Path(cfg.upload_tmp_dir),
        declared_length=_content_length(request),
    )
    # Reject a too-long/malformed video BEFORE queueing (T55). The temp file is
    # kept -- the worker will consume and delete it -- but a bad one dies here.
    try:
        await run_in_threadpool(probe_video, saved.path, cfg.max_video_seconds)
    except Exception:
        saved.cleanup()
        raise

    job = jobs.submit("video")
    # Hand the worker what it needs; it owns the temp file's lifetime now.
    request.app.state.pending_uploads[job.id] = str(saved.path)
    logger.info("queued video job %s", job.id)
    return JobAccepted(job_id=job.id, poll_url=f"/v1/jobs/{job.id}")


# --------------------------------------------------------------------------- #
def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    return int(raw) if raw and raw.isdigit() else None


def _image_response(request, reg, result, saved, sha, artifact_key) -> ImagePrediction:
    from backend.schemas.responses import Disclaimer, ModelBlock

    info = reg.model_info()
    # Rewrite bare artifact names into full, servable URLs (T58).
    for face in result["faces"]:
        face["artifacts"] = {
            name: f"/v1/artifacts/{artifact_key}/{fname}"
            for name, fname in face["artifacts"].items()
        }
    return ImagePrediction(
        request_id=getattr(request.state, "request_id", None),
        model=ModelBlock(
            arch="image",
            version=info["image"]["version"],
            device=info["device"],
            calibrated=info["image"]["calibrated"],
        ),
        media=build_media_block("image", sha, {"filename": saved.display_name, "bytes": saved.size}),
        faces=result["faces"],
        summary=result["summary"],
        warnings=result["warnings"],
        disclaimer=Disclaimer(**info["disclaimer"]),
    )
