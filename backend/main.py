"""SEETHRU API entry point (BUILD_PLAN T54). Exposed as ``backend.main:app``.

The lifespan is where the load-once contract lives: weights load at startup, the
single job worker starts, orphaned jobs are reconciled, and everything is torn
down cleanly on shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.core.errors import ApiError, api_error_handler, unhandled_error_handler
from backend.core.logging import RequestIdMiddleware, setup_logging

logger = logging.getLogger("seethru.api")


def _make_clock():
    """A monotonic-ish wall clock for job timestamps.

    time.time() is fine in the live server. It is wrapped so tests can inject a
    controllable clock into the job store.
    """
    import time

    return time.time


async def _video_job_handler(app: FastAPI, job):
    """Process one queued video job → the result dict (stored by the worker).

    Consumes the temp upload the route stashed, runs extraction + clip inference
    under the GPU lock, writes artifacts, and cleans up the temp file.
    """
    from backend.services.jobs import Job

    assert isinstance(job, Job)
    cfg = get_settings()
    reg = app.state.registry
    jobs = app.state.jobs

    tmp_path = app.state.pending_uploads.pop(job.id, None)
    if not tmp_path or not Path(tmp_path).is_file():
        from backend.core.errors import UnreadableMedia

        raise UnreadableMedia("The uploaded video is no longer available.")

    try:
        jobs.set_progress(job.id, 0.1, "extracting faces")
        result = await _run_video(app, cfg, reg, jobs, job, tmp_path)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _run_video(app, cfg, reg, jobs, job, tmp_path):
    """The heavy path. Kept small; detail lives in the ML layer."""
    import hashlib

    import torch
    from fastapi.concurrency import run_in_threadpool

    from ml.explainability import Explainer
    from ml.preprocessing.video_processor import VideoProcessor

    sha = hashlib.sha256(Path(tmp_path).read_bytes()).hexdigest()
    artifact_dir = Path(cfg.artifact_dir) / sha[:16]

    processor = VideoProcessor(face_detector=reg.face_detector, n_frames=16)
    seq = await run_in_threadpool(processor.build_face_sequence, tmp_path)
    if not seq.usable:
        from backend.core.errors import InsufficientFaces

        raise InsufficientFaces(
            f"Too few frames had a detectable face "
            f"({seq.n_missing}/{len(seq.sample.frames)}).",
            {"frames_with_face": len(seq.sample.frames) - seq.n_missing},
        )

    jobs.set_progress(job.id, 0.6, "analysing")
    model = reg.require_video()

    # Build the (1, T, 3, 224, 224) clip tensor from the aligned crops.
    from backend.services.inference import _to_model_input

    frames = torch.cat(
        [_to_model_input(f, reg.device) for f in seq.faces], dim=0
    ).unsqueeze(0)

    async with reg.gpu_lock:
        try:
            explainer = Explainer(model, reg.video_meta)
            exp, artifacts = explainer.explain_clip(
                frames,
                fps=seq.sample.fps,
                source_indices=seq.sample.source_indices,
                interpolated=seq.interpolated,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            from backend.core.errors import GpuBusy

            raise GpuBusy("GPU out of memory during video analysis.", {"retry_after": 10}) from exc

    jobs.set_progress(job.id, 0.9, "rendering")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    urls = {}
    for name, png in artifacts.images.items():
        (artifact_dir / name).write_bytes(png)
        urls[name] = f"/v1/artifacts/{artifact_dir.name}/{name}"
    jobs._update(job.id, artifact_dir=str(artifact_dir))

    info = reg.model_info()
    payload = exp.to_dict()
    return {
        "request_id": None,
        "model": {"arch": "video", "version": info["video"]["version"],
                  "device": info["device"], "calibrated": info["video"]["calibrated"]},
        "media": {"kind": "video", "sha256": sha, "duration_s": seq.sample.duration_s},
        "verdict": payload["verdict"],
        "scores": {"real": 1 - payload["p_fake"], "fake": payload["p_fake"]},
        "attribution": payload["attribution"],
        "timeline": payload["timeline"],
        "spans": payload["spans"],
        "artifacts": urls,
        "warnings": payload["warnings"],
        "disclaimer": info["disclaimer"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    cfg = get_settings()

    from backend.services.jobs import SqliteJobStore
    from backend.services.registry import ModelRegistry

    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.upload_tmp_dir).mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(cfg)
    registry.load()  # blocks until ready; /ready reports 503 until it returns
    app.state.registry = registry

    jobs = SqliteJobStore(
        db_path=str(Path(cfg.artifact_dir) / "jobs.db"),
        max_queue=cfg.max_queue_size,
        ttl_hours=cfg.job_ttl_hours,
        now=_make_clock(),
    )
    jobs.reconcile_orphans()  # a crash cannot leave a job 'running'
    jobs.sweep_expired()      # the container may have been down past a TTL
    jobs.start_worker(lambda job: _video_job_handler(app, job))
    app.state.jobs = jobs
    app.state.pending_uploads = {}

    logger.info("SEETHRU API ready (device=%s)", registry.device)
    try:
        yield
    finally:
        await jobs.stop_worker()
        jobs.close()


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="SEETHRU Deepfake Detection API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIdMiddleware)
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,  # never '*' -- validated in config
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    from backend.api.routes import health, jobs, model, predict

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(predict.router)
    app.include_router(jobs.router)

    # Prometheus /metrics (T54). Optional -- absence must not stop the app.
    try:
        from prometheus_client import make_asgi_app

        app.mount("/metrics", make_asgi_app())
    except ImportError:
        logger.warning("prometheus_client absent; /metrics disabled")

    return app


app = create_app()
