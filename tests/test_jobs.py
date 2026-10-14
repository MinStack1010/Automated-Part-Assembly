"""Regression tests for job watcher/result transport.

The parent used to ``process.join()`` before reading the result queue. A result
larger than the OS pipe buffer (~64 KB — easy with an inlined path of up to
10k states) made the child block forever in ``Queue.put()`` while the parent
blocked forever in ``join()``, leaving the job "running" long after ``max_time``.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from api.config import settings
from api.services.jobs import Job, JobManager, _joint_plan_no_path_error, _multi_plan_no_path_error


def _put_large_result(queue, size: int) -> None:
    queue.put({"ok": True, "result": {"blob": "x" * size}})


def _put_success_with_path(queue) -> None:
    queue.put(
        {
            "ok": True,
            "result": {"status": "Success", "path": [[0.0, 0.0, 0.0]], "path_state_count": 1},
        }
    )


def _put_planner_timeout(queue) -> None:
    queue.put(
        {
            "ok": False,
            "error": {
                "code": "PLANNER_TIMEOUT",
                "message": "Joint planning timed out without finding a path; no artifacts were saved.",
            },
            "result": {"status": "Timeout", "path": None, "path_state_count": 0},
        }
    )


class JobWatchTest(unittest.TestCase):
    def test_watch_receives_result_larger_than_pipe_buffer(self) -> None:
        manager = JobManager(max_workers=1)
        job_id = "test-large-result"
        artifact_dir = settings.jobs_dir / job_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, operation="joint-plan", payload={}, artifact_dir=artifact_dir)
        job.status = "running"
        result_queue = manager._context.Queue()
        payload_size = 2_000_000  # well above the ~64 KB OS pipe buffer
        process = manager._context.Process(
            target=_put_large_result, args=(result_queue, payload_size), daemon=True
        )
        job.result_queue = result_queue
        job.process = process
        manager._jobs[job_id] = job
        manager._running.add(job_id)
        process.start()

        finished = threading.Event()

        def _run_watch() -> None:
            manager._watch(job_id)
            finished.set()

        threading.Thread(target=_run_watch, daemon=True).start()
        self.assertTrue(finished.wait(timeout=15), "_watch deadlocked on a large result")
        self.assertEqual(job.status, "completed")
        self.assertEqual(len(job.result["blob"]), payload_size)

    def test_watch_marks_crashed_process(self) -> None:
        manager = JobManager(max_workers=1)
        job_id = "test-crash"
        artifact_dir = settings.jobs_dir / job_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, operation="joint-plan", payload={}, artifact_dir=artifact_dir)
        job.status = "running"
        result_queue = manager._context.Queue()
        process = manager._context.Process(target=_exit_immediately, daemon=True)
        job.result_queue = result_queue
        job.process = process
        manager._jobs[job_id] = job
        manager._running.add(job_id)
        process.start()

        finished = threading.Event()

        def _run_watch() -> None:
            manager._watch(job_id)
            finished.set()

        threading.Thread(target=_run_watch, daemon=True).start()
        self.assertTrue(finished.wait(timeout=15), "_watch did not observe process exit")
        self.assertEqual(job.status, "failed")
        self.assertIsNotNone(job.error)
        self.assertEqual(job.error.code, "NATIVE_PROCESS_CRASH")


def _exit_immediately() -> None:
    return


class NoPathFailureTest(unittest.TestCase):
    """A planner that found no path must fail the job instead of reporting success."""

    def test_timeout_without_path_is_planner_timeout(self) -> None:
        error = _joint_plan_no_path_error({"status": "Timeout", "path": None, "path_state_count": 0})
        self.assertIsNotNone(error)
        self.assertEqual(error["code"], "PLANNER_TIMEOUT")
        self.assertIn("timed out", error["message"])
        self.assertIn("no artifacts", error["message"])

    def test_failure_without_path_is_planner_no_path(self) -> None:
        error = _joint_plan_no_path_error({"status": "Failure", "path": None, "path_state_count": 0})
        self.assertEqual(error["code"], "PLANNER_NO_PATH")
        self.assertIn("Failure", error["message"])

    def test_start_with_goal_without_path_fails(self) -> None:
        error = _joint_plan_no_path_error({"status": "Start with goal", "path": None, "path_state_count": 0})
        self.assertEqual(error["code"], "PLANNER_NO_PATH")
        self.assertIn("already satisfies", error["message"])

    def test_success_with_path_produces_no_error(self) -> None:
        self.assertIsNone(
            _joint_plan_no_path_error({"status": "Success", "path": [[0.0, 0.0, 0.0]], "path_state_count": 1})
        )

    def test_diagnostics_warning_is_appended(self) -> None:
        error = _joint_plan_no_path_error(
            {
                "status": "Timeout",
                "path": None,
                "path_state_count": 0,
                "mesh_diagnostics": {"warning": "raise collision_threshold before planning"},
            }
        )
        self.assertIn("raise collision_threshold before planning", error["message"])

    def test_watch_marks_job_failed_with_planner_error(self) -> None:
        manager = JobManager(max_workers=1)
        job_id = "test-planner-timeout"
        artifact_dir = settings.jobs_dir / job_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, operation="joint-plan", payload={}, artifact_dir=artifact_dir)
        job.status = "running"
        result_queue = manager._context.Queue()
        process = manager._context.Process(target=_put_planner_timeout, args=(result_queue,), daemon=True)
        job.result_queue = result_queue
        job.process = process
        manager._jobs[job_id] = job
        manager._running.add(job_id)
        process.start()

        finished = threading.Event()

        def _run_watch() -> None:
            manager._watch(job_id)
            finished.set()

        threading.Thread(target=_run_watch, daemon=True).start()
        self.assertTrue(finished.wait(timeout=15), "_watch did not observe the planner error")
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error.code, "PLANNER_TIMEOUT")
        self.assertEqual(job.result["status"], "Timeout")

    def test_watch_materializes_npy_when_the_job_completes(self) -> None:
        manager = JobManager(max_workers=1)
        job_id = "test-npy-on-complete"
        artifact_dir = settings.jobs_dir / job_id / "artifacts"
        frames_dir = artifact_dir / "path"
        frames_dir.mkdir(parents=True, exist_ok=True)
        matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        (frames_dir / "path.json").write_text(json.dumps([{"name": "nut", "matrix": matrix}]))
        job = Job(job_id=job_id, operation="joint-plan", payload={}, artifact_dir=artifact_dir)
        job.status = "running"
        result_queue = manager._context.Queue()
        process = manager._context.Process(target=_put_success_with_path, args=(result_queue,), daemon=True)
        job.result_queue = result_queue
        job.process = process
        manager._jobs[job_id] = job
        manager._running.add(job_id)
        process.start()

        finished = threading.Event()

        def _run_watch() -> None:
            manager._watch(job_id)
            finished.set()

        threading.Thread(target=_run_watch, daemon=True).start()
        self.assertTrue(finished.wait(timeout=15), "_watch did not finish")
        self.assertEqual(job.status, "completed")
        self.assertTrue((artifact_dir / "npy" / "path.npy").is_file(), "npy motion file missing after completion")


class MultiPlanNoPathFailureTest(unittest.TestCase):
    """A multi-plan run without step artifacts must fail instead of reporting completed."""

    def _empty_dir(self) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return str(Path(tmp.name) / "artifacts")

    def test_timeout_without_steps_is_planner_timeout(self) -> None:
        error = _multi_plan_no_path_error(
            {"status": "Timeout", "sequence": []}, {"save_artifacts": True}, self._empty_dir()
        )
        self.assertIsNotNone(error)
        self.assertEqual(error["code"], "PLANNER_TIMEOUT")

    def test_failure_without_steps_is_planner_no_path(self) -> None:
        error = _multi_plan_no_path_error(
            {"status": "Failure", "sequence": []}, {"save_artifacts": True}, self._empty_dir()
        )
        self.assertEqual(error["code"], "PLANNER_NO_PATH")

    def test_partial_timeout_reports_step_count(self) -> None:
        error = _multi_plan_no_path_error(
            {"status": "Timeout", "sequence": ["0"]}, {"save_artifacts": True}, self._empty_dir()
        )
        self.assertEqual(error["code"], "PLANNER_TIMEOUT")
        self.assertIn("1", error["message"])

    def test_success_with_step_artifact_produces_no_error(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        step_dir = Path(tmp.name) / "artifacts" / "00013" / "0_0"
        step_dir.mkdir(parents=True)
        (step_dir / "path.json").write_text("[]")
        error = _multi_plan_no_path_error(
            {"status": "Success", "sequence": ["0"]},
            {"save_artifacts": True},
            str(Path(tmp.name) / "artifacts"),
        )
        self.assertIsNone(error)

    def test_success_without_artifact_request_produces_no_error(self) -> None:
        error = _multi_plan_no_path_error(
            {"status": "Success", "sequence": ["0"]}, {"save_artifacts": False}, self._empty_dir()
        )
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
