"""Job status/result/cancel + artifact serving (BUILD_PLAN T57/T58)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import FileResponse

from backend.core.config import Settings
from backend.core.errors import JobExpired, JobNotFound
from backend.dependencies import job_store, settings
from backend.schemas.responses import JobStatus
from backend.services.jobs import EXPIRED, SUCCEEDED

router = APIRouter(prefix="/v1", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, jobs=Depends(job_store)) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFound(f"No job with id {job_id!r}.")
    return JobStatus(**job.to_status())


@router.get("/jobs/{job_id}/result")
async def job_result(job_id: str, response: Response, jobs=Depends(job_store)) -> dict:
    """The result, or a status code that tells the client what to do:

    * 200 -- succeeded, here it is.
    * 202 -- still running; keep polling.
    * 410 -- expired; the result is gone (T57).
    """
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFound(f"No job with id {job_id!r}.")
    if job.state == EXPIRED:
        raise JobExpired(f"Job {job_id} has expired and its result was purged.")
    if job.state == SUCCEEDED:
        return jobs.result(job_id)
    if job.state == "failed":
        return {"state": "failed", "error_code": job.error_code}
    response.status_code = status.HTTP_202_ACCEPTED
    return {"state": job.state, "progress": job.progress}


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(job_id: str, jobs=Depends(job_store)) -> Response:
    """Cancel/purge a job. The frontend needs this or its "Cancel" button is a
    lie that leaves the GPU working (T57/T62)."""
    job = jobs.get(job_id)
    if job is None:
        raise JobNotFound(f"No job with id {job_id!r}.")
    jobs._update(job_id, state="failed", error_code="cancelled")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/artifacts/{artifact_key}/{name}")
async def get_artifact(
    artifact_key: str, name: str, cfg: Settings = Depends(settings)
) -> FileResponse:
    """Serve a rendered PNG (T58).

    Both path parts are constrained so a crafted name cannot escape the artifact
    directory -- classic traversal via ``..%2F``.
    """
    if not _safe(artifact_key) or not _safe(name):
        raise JobNotFound("No such artifact.")
    path = Path(cfg.artifact_dir) / artifact_key / name
    if not path.is_file():
        raise JobNotFound("No such artifact.")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


def _safe(component: str) -> bool:
    """A single path segment with no separators or traversal."""
    return (
        component
        and "/" not in component
        and "\\" not in component
        and ".." not in component
    )
