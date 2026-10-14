"""Animation manifest, single-artifact download, and multi-step npy handling."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.schemas import ErrorDetail
from api.server import app
from api.services.artifacts import ensure_npy, list_animation_steps, npy_target_for
from api.services.jobs import Job, job_manager

_FRAME_MATRIX = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _frames(name: str, count: int = 3) -> list[dict]:
    frames = []
    for i in range(count):
        matrix = [row[:] for row in _FRAME_MATRIX]
        matrix[0][3] = float(i)
        frames.append({"name": name, "matrix": matrix})
    return frames


class NpyCollisionTests(unittest.TestCase):
    """Multi-plan writes one path.json per step; every step needs its own .npy."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self._tmp.name)
        for relative in ("00013/0_0/path.json", "00013/1_2/path.json"):
            target = self.artifact_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(_frames(relative.split("/")[1].split("_")[1])))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_each_step_gets_its_own_npy(self) -> None:
        ensure_npy(self.artifact_dir)
        first = self.artifact_dir / "npy" / "00013_0_0_path.npy"
        second = self.artifact_dir / "npy" / "00013_1_2_path.npy"
        self.assertTrue(first.is_file(), sorted(p.name for p in (self.artifact_dir / "npy").iterdir()))
        self.assertTrue(second.is_file())
        self.assertFalse((self.artifact_dir / "npy" / "path.npy").exists())
        self.assertEqual(np.load(first).shape, (3, 4, 4))

    def test_joint_plan_keeps_the_legacy_npy_name(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            artifact_dir = Path(other)
            target = artifact_dir / "path" / "path.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(_frames("0")))
            self.assertEqual(npy_target_for(artifact_dir, target), artifact_dir / "npy" / "path.npy")
            ensure_npy(artifact_dir)
            self.assertTrue((artifact_dir / "npy" / "path.npy").is_file())

    def test_steps_carry_sequence_order_and_move_ids(self) -> None:
        ensure_npy(self.artifact_dir)
        steps = list_animation_steps(self.artifact_dir)
        self.assertEqual(len(steps), 2)
        self.assertEqual([step["sequence_index"] for step in steps], [0, 1])
        self.assertEqual([step["move_id"] for step in steps], ["0", "2"])
        self.assertEqual(steps[0]["artifact"], "00013/0_0/path.json")
        self.assertEqual(steps[0]["npy_artifact"], "npy/00013_0_0_path.npy")
        self.assertEqual([step["frame_count"] for step in steps], [3, 3])


class AnimationEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.tmp = tempfile.TemporaryDirectory()
        self._registered: list[str] = []
        self.joint_id = self._register_job(
            "joint-plan",
            "completed",
            assembly={"collection": "multi_assembly", "assembly_id": "00013"},
            files={"path/path.json": _frames("0", 4)},
        )
        self.multi_id = self._register_job(
            "multi-plan",
            "completed",
            assembly={"collection": "multi_assembly", "assembly_id": "00013"},
            files={
                "00013/0_0/path.json": _frames("0", 4),
                "00013/1_2/path.json": _frames("2", 5),
            },
        )
        self.running_id = self._register_job("joint-plan", "running", assembly=None, files={})
        self.failed_id = self._register_job(
            "joint-plan",
            "failed",
            assembly=None,
            files={},
            error=ErrorDetail(code="PLANNER_TIMEOUT", message="timed out"),
        )

    def tearDown(self) -> None:
        for job_id in self._registered:
            job_manager._jobs.pop(job_id, None)
            job_manager._running.discard(job_id)
            if job_id in job_manager._queued:
                job_manager._queued.remove(job_id)
        self.tmp.cleanup()

    def _register_job(self, operation, status, assembly, files, error=None) -> str:
        job_id = f"test-animation-{uuid.uuid4()}"
        artifact_dir = Path(self.tmp.name) / job_id / "artifacts"
        artifact_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc)
        job = Job(job_id=job_id, operation=operation, payload={"assembly": assembly} if assembly else {}, artifact_dir=artifact_dir)
        job.status = status  # type: ignore[assignment]
        job.started_at = now
        if status in {"completed", "failed"}:
            job.completed_at = now
        if status == "completed":
            job.result = {"status": "Success"}
        if error is not None:
            job.error = error
        for relative, frames in files.items():
            target = artifact_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(frames))
        if status == "completed":
            ensure_npy(artifact_dir)
        job_manager._jobs[job_id] = job
        self._registered.append(job_id)
        return job_id

    def test_joint_animation_manifest(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.joint_id}/animation")
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertEqual(result["job_id"], self.joint_id)
        self.assertEqual(result["operation"], "joint-plan")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["steps"]), 1)
        step = result["steps"][0]
        self.assertEqual(step["id"], "path")
        self.assertEqual(step["move_id"], "0")
        self.assertIsNone(step["sequence_index"])
        self.assertEqual(step["frame_count"], 4)
        self.assertTrue(step["json_url"].endswith(f"/api/v1/jobs/{self.joint_id}/artifacts/path/path.json"))
        self.assertTrue(step["npy_url"].endswith("artifacts/npy/path.npy"))
        self.assertIn("0", result["part_ids"])
        self.assertTrue(result["obj_files"]["0"].endswith(f"/api/v1/jobs/{self.joint_id}/assembly/0.obj"))
        self.assertTrue(result["artifacts_zip_url"].endswith("artifacts.zip"))

    def test_multi_animation_keeps_per_step_mapping(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.multi_id}/animation")
        self.assertEqual(response.status_code, 200)
        steps = response.json()["result"]["steps"]
        self.assertEqual(len(steps), 2)
        self.assertEqual([(s["sequence_index"], s["move_id"]) for s in steps], [(0, "0"), (1, "2")])
        self.assertNotEqual(steps[0]["npy_url"], steps[1]["npy_url"])
        self.assertEqual([s["frame_count"] for s in steps], [4, 5])
        for step in steps:
            fetched = self.client.get(step["json_url"])
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(len(fetched.json()), step["frame_count"])
            npy_response = self.client.get(step["npy_url"])
            self.assertEqual(npy_response.status_code, 200)
            array = np.load(io.BytesIO(npy_response.content))
            self.assertEqual(array.shape, (step["frame_count"], 4, 4))

    def test_animation_urls_serve_real_bytes(self) -> None:
        manifest = self.client.get(f"/api/v1/jobs/{self.joint_id}/animation").json()["result"]
        obj_url = manifest["obj_files"]["0"]
        obj_response = self.client.get(obj_url)
        self.assertEqual(obj_response.status_code, 200)
        self.assertTrue(obj_response.content.startswith(b"#") or b"v " in obj_response.content[:10_000])

    def test_running_failed_and_unknown_jobs(self) -> None:
        running = self.client.get(f"/api/v1/jobs/{self.running_id}/animation")
        self.assertEqual(running.status_code, 409)
        self.assertEqual(running.json()["error"]["code"], "JOB_NOT_COMPLETED")
        failed = self.client.get(f"/api/v1/jobs/{self.failed_id}/animation")
        self.assertEqual(failed.status_code, 409)
        self.assertEqual(failed.json()["error"]["code"], "PLANNER_TIMEOUT")
        unknown = self.client.get("/api/v1/jobs/test-animation-does-not-exist/animation")
        self.assertEqual(unknown.status_code, 404)

    def test_single_artifact_download_and_errors(self) -> None:
        ok_response = self.client.get(f"/api/v1/jobs/{self.joint_id}/artifacts/path/path.json")
        self.assertEqual(ok_response.status_code, 200)
        self.assertTrue(ok_response.headers["content-type"].startswith("application/json"))
        missing = self.client.get(f"/api/v1/jobs/{self.joint_id}/artifacts/does/not-exist.json")
        self.assertEqual(missing.status_code, 404)
        traversal = self.client.get(f"/api/v1/jobs/{self.joint_id}/artifacts/..%2F..%2Fserver.py")
        self.assertIn(traversal.status_code, (400, 404))
        unknown = self.client.get("/api/v1/jobs/test-animation-does-not-exist/artifacts/path/path.json")
        self.assertEqual(unknown.status_code, 404)

    def test_assembly_file_download(self) -> None:
        ok_response = self.client.get(f"/api/v1/jobs/{self.joint_id}/assembly/0.obj")
        self.assertEqual(ok_response.status_code, 200)
        missing = self.client.get(f"/api/v1/jobs/{self.joint_id}/assembly/9.obj")
        self.assertEqual(missing.status_code, 404)

    def test_zip_still_contains_both_multi_steps(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.multi_id}/artifacts.zip")
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = set(archive.namelist())
        self.assertIn("artifact/npy/00013_0_0_path.npy", names)
        self.assertIn("artifact/npy/00013_1_2_path.npy", names)


if __name__ == "__main__":
    unittest.main()
