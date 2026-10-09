"""Process-isolated job execution for long-running native planners."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_module
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from api.config import settings
from api.errors import ApiError
from api.schemas import ErrorDetail, JobSnapshot, JointPlanRequest, MultiPlanRequest

logger = logging.getLogger(__name__)
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _execute_job_process(
    operation: str,
    payload: dict[str, Any],
    artifact_dir: str,
    result_queue: Any,
) -> None:
    """Child-process entry point. Keep this module-level so ``spawn`` can import it."""
    try:
        if operation == "joint-plan":
            from api.services.planning import run_joint_plan

            result = run_joint_plan(JointPlanRequest.model_validate(payload), artifact_dir)
        elif operation == "multi-plan":
            from api.services.planning import run_multi_plan

            result = run_multi_plan(MultiPlanRequest.model_validate(payload), artifact_dir)
        else:
            raise ApiError(400, "INVALID_OPERATION", f"Unknown job operation: {operation}")
        result_queue.put({"ok": True, "result": result})
    except ApiError as exc:
        result_queue.put(
            {
                "ok": False,
                "error": {"code": exc.code, "message": exc.message},
            }
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("JOB FAILED operation=%s", operation)
        result_queue.put(
            {
                "ok": False,
                "error": {"code": "NATIVE_EXECUTION_FAILED", "message": str(exc)},
                "traceback": traceback.format_exc(),
            }
        )


@dataclass
class Job:
    job_id: str
    operation: str
    payload: dict[str, Any]
    artifact_dir: Path
    status: JobStatus = "queued"
    created_at: datetime = field(default_factory=_utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[ErrorDetail] = None
    process: Any = field(default=None, repr=False)
    result_queue: Any = field(default=None, repr=False)


class JobManager:
    """Bounded local process queue; no database, broker, or shared state is required."""

    def __init__(self, max_workers: int = settings.max_workers) -> None:
        self._max_workers = max_workers
        self._context = mp.get_context("spawn")
        self._jobs: dict[str, Job] = {}
        self._queued: list[str] = []
        self._running: set[str] = set()
        self._lock = threading.RLock()

    def submit(self, operation: str, payload: dict[str, Any]) -> Job:
        with self._lock:
            job_id = str(uuid.uuid4())
            artifact_dir = settings.jobs_dir / job_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            job = Job(job_id=job_id, operation=operation, payload=payload, artifact_dir=artifact_dir)
            self._jobs[job_id] = job
            self._queued.append(job_id)
            logger.info("JOB QUEUED id=%s operation=%s", job_id, operation)
            self._start_available_jobs()
            return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "JOB_NOT_FOUND", "The requested job does not exist")
            return job

    def snapshot(self, job_id: str) -> JobSnapshot:
        with self._lock:
            job = self.get(job_id)
            return self._snapshot(job)

    def result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job.status != "completed":
                if job.status == "failed" and job.error is not None:
                    raise ApiError(409, job.error.code, job.error.message)
                raise ApiError(409, "JOB_NOT_COMPLETED", f"Job is {job.status}; no result is available")
            assert job.result is not None
            return job.result

    def cancel(self, job_id: str) -> JobSnapshot:
        with self._lock:
            job = self.get(job_id)
            if job.status == "queued":
                self._queued.remove(job_id)
                job.status = "cancelled"
                job.completed_at = _utc_now()
                logger.info("JOB CANCELLED id=%s before start", job_id)
                return self._snapshot(job)
            if job.status == "running":
                # Planner work is intentionally in its own process: termination cannot kill uvicorn.
                if job.process is not None and job.process.is_alive():
                    job.process.terminate()
                job.status = "cancelled"
                job.completed_at = _utc_now()
                logger.info("JOB CANCELLED id=%s while running", job_id)
                return self._snapshot(job)
            return self._snapshot(job)

    def _start_available_jobs(self) -> None:
        while len(self._running) < self._max_workers and self._queued:
            job_id = self._queued.pop(0)
            job = self._jobs[job_id]
            if job.status != "queued":
                continue
            result_queue = self._context.Queue()
            process = self._context.Process(
                target=_execute_job_process,
                args=(job.operation, job.payload, str(job.artifact_dir), result_queue),
                daemon=True,
            )
            job.result_queue = result_queue
            job.process = process
            job.status = "running"
            job.started_at = _utc_now()
            self._running.add(job_id)
            try:
                process.start()
            except Exception as exc:
                self._running.remove(job_id)
                job.status = "failed"
                job.completed_at = _utc_now()
                job.error = ErrorDetail(code="JOB_START_FAILED", message=str(exc))
                logger.exception("JOB FAILED TO START id=%s", job_id)
                continue
            logger.info("JOB START id=%s operation=%s pid=%s", job_id, job.operation, process.pid)
            threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()

    def _watch(self, job_id: str) -> None:
        job = self.get(job_id)
        assert job.process is not None
        job.process.join()
        message: Optional[dict[str, Any]] = None
        try:
            message = job.result_queue.get(timeout=1)
        except queue_module.Empty:
            pass
        finally:
            if job.result_queue is not None:
                job.result_queue.close()

        with self._lock:
            current = self._jobs[job_id]
            self._running.discard(job_id)
            if current.status == "cancelled":
                logger.info("JOB CANCELLED id=%s process exited", job_id)
            elif message is not None and message.get("ok"):
                current.status = "completed"
                current.completed_at = _utc_now()
                current.result = message["result"]
                logger.info("JOB SUCCESS id=%s operation=%s", job_id, current.operation)
            elif message is not None:
                current.status = "failed"
                current.completed_at = _utc_now()
                error = message.get("error", {})
                current.error = ErrorDetail(
                    code=error.get("code", "NATIVE_EXECUTION_FAILED"),
                    message=error.get("message", "The native job failed"),
                )
                logger.error("JOB FAILED id=%s code=%s", job_id, current.error.code)
            else:
                current.status = "failed"
                current.completed_at = _utc_now()
                current.error = ErrorDetail(
                    code="NATIVE_PROCESS_CRASH",
                    message="The isolated native worker exited without a result; the API server remains available.",
                )
                logger.error("JOB CRASHED id=%s exit_code=%s", job_id, job.process.exitcode)
            self._start_available_jobs()

    def _snapshot(self, job: Job) -> JobSnapshot:
        artifacts = []
        if job.artifact_dir.is_dir():
            artifacts = sorted(str(path.relative_to(job.artifact_dir)) for path in job.artifact_dir.rglob("*") if path.is_file())
        return JobSnapshot(
            job_id=job.job_id,
            operation=job.operation,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            artifacts=artifacts,
        )


job_manager = JobManager()
