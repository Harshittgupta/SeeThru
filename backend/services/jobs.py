"""Async job queue for video analysis (BUILD_PLAN T57).

**Decision: in-process asyncio queue + one worker + SQLite. NOT Celery.**

A video takes 30 s - 5 min (frame extraction + RetinaFace + 16-frame inference),
which cannot be held on a synchronous HTTP request. But on a single GPU, Celery
would be a *second* CUDA context plus a second TF/RetinaFace allocation (~2-3 GB
idle VRAM) on the same card, for a queue whose steady-state depth is ~1 --
concurrency must be 1 regardless, because two inferences on one GPU risk OOM.
Redis + a broker buys durability that SQLite gives for free at this scale.

This choice has two hard consequences the deployment MUST honour:

* **``--workers 1``.** A job submitted to worker A is invisible to worker B's
  in-memory queue, so a poll would 404. Enforced by design, documented in
  docker-compose.
* **No ``--reload``.** The reloader kills in-flight jobs.

Everything lives behind the ``JobStore`` protocol so swapping to Celery when a
second GPU appears is one implementation, not a rewrite.

SQLite is the source of truth (survives restart); the asyncio.Queue is just the
in-memory hand-off to the worker. On startup, any job left ``running`` by a crash
is reconciled to ``failed`` -- it cannot still be running, the process that ran it
is gone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("seethru.jobs")

QUEUED, RUNNING, SUCCEEDED, FAILED, EXPIRED = (
    "queued", "running", "succeeded", "failed", "expired",
)
TERMINAL = {SUCCEEDED, FAILED, EXPIRED}


@dataclass
class Job:
    id: str
    kind: str
    state: str
    created_at: float
    progress: float = 0.0
    stage: str | None = None
    error_code: str | None = None
    result_json: str | None = None
    artifact_dir: str | None = None

    def to_status(self) -> dict:
        return {
            "job_id": self.id,
            "state": self.state,
            "progress": self.progress,
            "stage": self.stage,
            "error_code": self.error_code,
            "created_at": self.created_at,
            "poll_url": f"/v1/jobs/{self.id}",
        }


class SqliteJobStore:
    """SQLite-backed job store + a single-worker asyncio queue."""

    def __init__(self, db_path: str, max_queue: int, ttl_hours: int, now: Callable[[], float]) -> None:
        # now is injected: perf_counter/time.time are fine in the live server, but
        # passing it keeps this testable with a controllable clock.
        self._now = now
        self._db_path = db_path
        self._ttl_s = ttl_hours * 3600
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._worker_task: asyncio.Task | None = None
        self._handler: Callable[[Job], Awaitable[dict]] | None = None

    def _init_db(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT, state TEXT, created_at REAL,
                progress REAL, stage TEXT, error_code TEXT,
                result_json TEXT, artifact_dir TEXT
            )"""
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def reconcile_orphans(self) -> None:
        """A job left 'running' by a crash cannot still be running: the process
        that ran it is gone. Mark such rows failed on startup."""
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, error_code=? WHERE state IN (?, ?)",
            (FAILED, "interrupted", RUNNING, QUEUED),
        )
        self._conn.commit()
        if cur.rowcount:
            logger.warning("reconciled %d orphaned job(s) to failed", cur.rowcount)

    def start_worker(self, handler: Callable[[Job], Awaitable[dict]]) -> None:
        """Start the single background worker. Called in lifespan."""
        self._handler = handler
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self) -> None:
        import contextlib

        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    # ------------------------------------------------------------------ #
    def submit(self, kind: str) -> Job:
        """Create a QUEUED job and enqueue it. Raises if the queue is full."""
        from backend.core.errors import QueueFull

        job = Job(id=uuid.uuid4().hex[:16], kind=kind, state=QUEUED, created_at=self._now())
        self._insert(job)
        try:
            self._queue.put_nowait(job.id)
        except asyncio.QueueFull:
            self._update(job.id, state=FAILED, error_code="queue_full")
            raise QueueFull(
                "The analysis queue is full. Try again shortly.",
                {"retry_after": 10},
            ) from None
        return job

    def get(self, job_id: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = Job(**dict(row))
        # Lazily expire on read: a terminal job past its TTL is EXPIRED, and its
        # artifacts are swept separately.
        if job.state in TERMINAL and job.state != FAILED and (self._now() - job.created_at) > self._ttl_s:
            self._update(job_id, state=EXPIRED)
            job.state = EXPIRED
        return job

    def result(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job and job.state == SUCCEEDED and job.result_json:
            return json.loads(job.result_json)
        return None

    def set_progress(self, job_id: str, progress: float, stage: str | None = None) -> None:
        self._update(job_id, progress=progress, stage=stage)

    # ------------------------------------------------------------------ #
    async def _worker_loop(self) -> None:
        """Process one job at a time. Concurrency is deliberately 1 (one GPU)."""
        while True:
            job_id = await self._queue.get()
            job = self.get(job_id)
            if job is None or job.state != QUEUED:
                self._queue.task_done()
                continue

            self._update(job_id, state=RUNNING, progress=0.0)
            try:
                result = await self._handler(self.get(job_id))
                self._update(
                    job_id, state=SUCCEEDED, progress=1.0,
                    result_json=json.dumps(result),
                )
            except Exception as exc:  # noqa: BLE001 - a bad job must not kill the worker
                code = getattr(exc, "error_code", "processing_failed")
                logger.exception("job %s failed: %s", job_id, exc)
                self._update(job_id, state=FAILED, error_code=code)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    def _insert(self, job: Job) -> None:
        self._conn.execute(
            "INSERT INTO jobs (id,kind,state,created_at,progress) VALUES (?,?,?,?,?)",
            (job.id, job.kind, job.state, job.created_at, job.progress),
        )
        self._conn.commit()

    def _update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._conn.execute(
            f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id)
        )
        self._conn.commit()

    def sweep_expired(self) -> int:
        """Delete rows + artifact dirs past the TTL. Called on startup and on a
        timer. Runs on startup too -- the container may have been down past a
        job's TTL, so nothing else would ever clean it."""
        import shutil

        cutoff = self._now() - self._ttl_s
        rows = self._conn.execute(
            "SELECT id, artifact_dir FROM jobs WHERE created_at < ?", (cutoff,)
        ).fetchall()
        for row in rows:
            if row["artifact_dir"]:
                shutil.rmtree(row["artifact_dir"], ignore_errors=True)
        self._conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
        self._conn.commit()
        return len(rows)

    def close(self) -> None:
        self._conn.close()
