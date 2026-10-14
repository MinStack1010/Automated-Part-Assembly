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
from api.services.artifacts import ensure_npy

logger = logging.getLogger(__name__)
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _joint_plan_no_path_error(result: dict[str, Any]) -> Optional[dict[str, str]]:
    """Fail a joint-plan job whose planner produced no path instead of reporting success."""
    if result.get("path") or (result.get("path_state_count") or 0) > 0:
        return None
    status = str(result.get("status") or "unknown")
    if status == "Start with goal":
        code = "PLANNER_NO_PATH"
        message = (
            "Joint planning produced no artifacts: the initial state already satisfies "
            "the goal, so there is no path to save."
        )
    elif status == "Timeout":
        code = "PLANNER_TIMEOUT"
        message = "Joint planning timed out without finding a path; no artifacts were saved."
    else:
        code = "PLANNER_NO_PATH"
        message = f"Joint planning returned status '{status}' without a path; no artifacts were saved."
    diagnostics = result.get("mesh_diagnostics") or {}
    if diagnostics.get("warning"):
        message = f"{message} {diagnostics['warning']}."
    return {"code": code, "message": message}


def _has_step_artifacts(artifact_dir: str) -> bool:
    """Whether the worker wrote at least one planner path.json for its steps."""
    root = Path(artifact_dir)
    if not root.is_dir():
        return False
    return any(path.is_file() for path in root.rglob("path.json"))


def _multi_plan_no_path_error(
    result: dict[str, Any], payload: dict[str, Any], artifact_dir: str
) -> Optional[dict[str, str]]:
    """Fail a multi-plan job whose sequence produced nothing to animate.

    A finished worker process is not a successful plan: without step artifacts
    the job must not report ``completed`` (the frontend would otherwise show a
    successful run with nothing to play). Partial sequences still fail, but the
    result (and any step artifacts already on disk) is preserved for diagnosis.
    """
    status = str(result.get("status") or "unknown")
    sequence = result.get("sequence") or []
    save_artifacts = bool(payload.get("save_artifacts", False))
    if status == "Success" and (not save_artifacts or _has_step_artifacts(artifact_dir)):
        return None
    if status == "Timeout":
        code = "PLANNER_TIMEOUT"
        message = (
            f"Multi-part planning timed out with {len(sequence)} of the sequence steps "
            "completed; no complete animation is available."
            if sequence
            else "Multi-part planning timed out without completing any sequence step; no artifacts were saved."
        )
    elif not sequence and not _has_step_artifacts(artifact_dir):
        code = "PLANNER_NO_PATH"
        message = f"Multi-part planning returned status '{status}' without any step path; no artifacts were saved."
    else:
        code = "PLANNER_NO_PATH"
        message = (
            f"Multi-part planning returned status '{status}' with an incomplete sequence "
            f"({len(sequence)} steps); no complete animation is available."
        )
    return {"code": code, "message": message}


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
            no_path_error = _joint_plan_no_path_error(result)
            if no_path_error is not None:
                result_queue.put({"ok": False, "error": no_path_error, "result": result})
            else:
                result_queue.put({"ok": True, "result": result})
        elif operation == "multi-plan":
            from api.services.planning import run_multi_plan

            result = run_multi_plan(MultiPlanRequest.model_validate(payload), artifact_dir)
            no_path_error = _multi_plan_no_path_error(result, payload, artifact_dir)
            if no_path_error is not None:
                result_queue.put({"ok": False, "error": no_path_error, "result": result})
            else:
                result_queue.put({"ok": True, "result": result})
        else:
            raise ApiError(400, "INVALID_OPERATION", f"Unknown job operation: {operation}")
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

    def animation(self, job_id: str) -> dict[str, Any]:
        """Build the frontend-ready animation manifest for a completed job.

        Same gating as :meth:`result` (404 unknown, 409 unless completed, the
        job's own error code when it failed) so clients can rely on one
        convention. Only completed jobs have materialized artifacts.
        """
        from api.services.artifacts import list_animation_steps

        with self._lock:
            job = self.get(job_id)
            if job.status != "completed":
                if job.status == "failed" and job.error is not None:
                    raise ApiError(409, job.error.code, job.error.message)
                raise ApiError(409, "JOB_NOT_COMPLETED", f"Job is {job.status}; no animation is available")
            steps = list_animation_steps(job.artifact_dir)
            part_ids: list[str] = []
            try:
                from api.schemas import AssemblyRef
                from api.services.assemblies import get_metadata

                payload_assembly = job.payload.get("assembly")
                if payload_assembly:
                    part_ids = get_metadata(AssemblyRef.model_validate(payload_assembly)).part_ids
            except Exception:
                logger.warning("JOB ANIMATION manifest job=%s assembly unavailable", job_id, exc_info=True)
            base = f"/api/v1/jobs/{job_id}"
            manifest_steps = []
            for step in steps:
                manifest_steps.append(
                    {
                        **step,
                        "json_url": f"{base}/artifacts/{step['artifact']}",
                        "npy_url": f"{base}/artifacts/{step['npy_artifact']}" if step["npy_artifact"] else None,
                        "gif_url": f"{base}/artifacts/{step['gif_artifact']}" if step["gif_artifact"] else None,
                    }
                )
            return {
                "job_id": job.job_id,
                "operation": job.operation,
                "status": job.status,
                "part_ids": part_ids,
                "obj_files": {part_id: f"{base}/assembly/{part_id}.obj" for part_id in part_ids},
                "steps": manifest_steps,
                "artifacts_zip_url": f"{base}/artifacts.zip",
            }

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
        message: Optional[dict[str, Any]] = None
        # Drain the queue while the worker is alive. Joining first deadlocks
        # whenever the pickled result exceeds the OS pipe buffer (~64 KB): the
        # child's queue feeder blocks on write until we read, while we wait
        # for the child to exit before reading at all.
        try:
            while True:
                try:
                    message = job.result_queue.get(timeout=0.5)
                    break
                except queue_module.Empty:
                    if not job.process.is_alive():
                        break
                except EOFError:
                    break
            job.process.join()
            if message is None:
                try:
                    message = job.result_queue.get_nowait()
                except (queue_module.Empty, EOFError):
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
                # Materialize npy motion files before the job is visible as
                # completed, so /result and the job snapshot both list them.
                try:
                    ensure_npy(Path(job.artifact_dir))
                except Exception:
                    logger.warning("JOB NPY PREFETCH FAILED id=%s", job_id, exc_info=True)
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
                if message.get("result") is not None:
                    current.result = message["result"]
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
