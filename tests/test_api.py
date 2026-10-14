"""Smoke tests for the HTTP wrapper and its real native integration."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from api.server import app
from api.schemas import ErrorDetail
from api.services.jobs import Job, job_manager

DELETED_ENDPOINTS = [
    ("GET", "/api/v1/assemblies"),
    ("GET", "/api/v1/assemblies/multi_assembly/00003"),
    ("POST", "/api/v1/assemblies/preprocess"),
    ("POST", "/api/v1/mesh-distance"),
    ("POST", "/api/v1/simulations/demo"),
]

_FRAME_MATRIX = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class ApiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_service_discovery_endpoints(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertIn(health.json()["status"], {"ok", "degraded"})

        info = self.client.get("/info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["name"], "assemble-them-all-api")

        capabilities = self.client.get("/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        names = {operation["name"] for operation in capabilities.json()["operations"]}
        self.assertTrue({"joint.plan", "multi.plan", "mesh.gap", "assembly.upload"}.issubset(names))
        self.assertTrue(
            {"assembly.inspect", "assembly.preprocess", "mesh.distance", "simulation.demo"}.isdisjoint(names)
        )

    def test_deleted_endpoints_are_not_registered(self) -> None:
        for method, path in DELETED_ENDPOINTS:
            with self.subTest(method=method, path=path):
                response = (
                    self.client.post(path, json={}) if method == "POST" else self.client.get(path)
                )
                self.assertEqual(response.status_code, 404, f"{method} {path} is still registered")
                body = response.json()
                self.assertNotIn("result", body)

    def test_openapi_document_hides_deleted_endpoints(self) -> None:
        paths = self.client.get("/openapi.json").json()["paths"]
        for _, path in DELETED_ENDPOINTS:
            self.assertNotIn(path, paths)
        self.assertIn("/api/v1/assemblies/upload", paths)
        self.assertIn("/api/v1/jobs/joint-plan", paths)

    def test_validation_errors_use_the_standard_shape(self) -> None:
        response = self.client.post("/api/v1/mesh-gap", json={"assembly": {"collection": "multi_assembly"}})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.json()["success"])
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")


class JobResultEndpointTests(unittest.TestCase):
    """GET /result answers with the job snapshot JSON; artifacts.zip stays a zip."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.tmp = tempfile.TemporaryDirectory()
        self._registered: list[str] = []
        self.joint_id = self._register_job(
            "joint-plan",
            "completed",
            files=["path/path.json", "npy/path.npy"],
        )
        self.multi_id = self._register_job(
            "multi-plan",
            "completed",
            files=["0_nut/path.json"],
        )
        self.running_id = self._register_job("joint-plan", "running")
        self.failed_id = self._register_job(
            "joint-plan",
            "failed",
            error=ErrorDetail(code="PLANNER_TIMEOUT", message="Joint planning timed out without finding a path."),
        )

    def tearDown(self) -> None:
        for job_id in self._registered:
            job_manager._jobs.pop(job_id, None)
            job_manager._running.discard(job_id)
            if job_id in job_manager._queued:
                job_manager._queued.remove(job_id)
        self.tmp.cleanup()

    def _register_job(
        self,
        operation: str,
        status: str,
        files: tuple[str, ...] | list[str] = (),
        error: ErrorDetail | None = None,
    ) -> str:
        job_id = f"test-result-{uuid.uuid4()}"
        artifact_dir = Path(self.tmp.name) / job_id / "artifacts"
        artifact_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc)
        job = Job(job_id=job_id, operation=operation, payload={}, artifact_dir=artifact_dir)
        job.status = status  # type: ignore[assignment]
        job.started_at = now
        if status in {"completed", "failed"}:
            job.completed_at = now
        if status == "completed":
            job.result = {"status": "Success", "path_state_count": 1}
        if error is not None:
            job.error = error
        for relative in files:
            target = artifact_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".json":
                target.write_text(json.dumps([{"name": "nut", "matrix": _FRAME_MATRIX}]))
            else:
                target.write_bytes(b"\x93NUMPY-fake")
        job_manager._jobs[job_id] = job
        self._registered.append(job_id)
        return job_id

    def test_completed_joint_plan_result_is_json_snapshot(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.joint_id}/result")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertNotIn("content-disposition", response.headers)
        body = response.json()
        self.assertTrue(body["success"])
        result = body["result"]
        self.assertEqual(result["job_id"], self.joint_id)
        self.assertEqual(result["operation"], "joint-plan")
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["created_at"])
        self.assertIsNotNone(result["started_at"])
        self.assertIsNotNone(result["completed_at"])
        self.assertEqual(
            result["artifacts"],
            ["npy/path.npy", "path/path.json"],
        )
        self.assertTrue(all(isinstance(entry, str) for entry in result["artifacts"]))

    def test_completed_multi_plan_result_is_json_snapshot(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.multi_id}/result")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        result = response.json()["result"]
        self.assertEqual(result["operation"], "multi-plan")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["artifacts"], ["0_nut/path.json"])

    def test_running_job_keeps_the_409_convention(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.running_id}/result")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "JOB_NOT_COMPLETED")

    def test_failed_job_answers_with_its_own_error_code(self) -> None:
        response = self.client.get(f"/api/v1/jobs/{self.failed_id}/result")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "PLANNER_TIMEOUT")

    def test_unknown_job_is_404(self) -> None:
        response = self.client.get("/api/v1/jobs/test-result-does-not-exist/result")
        self.assertEqual(response.status_code, 404)

    def test_artifacts_zip_still_downloads_a_zip(self) -> None:
        for job_id, expected_npy in (
            (self.joint_id, "artifact/npy/path.npy"),
            # multi-plan steps keep their own npy: 0_nut/path.json must not
            # collapse onto the joint-plan npy/path.npy name.
            (self.multi_id, "artifact/npy/0_nut_path.npy"),
        ):
            with self.subTest(job_id=job_id):
                response = self.client.get(f"/api/v1/jobs/{job_id}/artifacts.zip")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "application/zip")
                self.assertIn("attachment", response.headers["content-disposition"])
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = set(archive.namelist())
                self.assertIn("artifact/json/path.json", names)
                self.assertIn(expected_npy, names)

    def test_result_and_zip_serve_the_same_completed_job(self) -> None:
        json_response = self.client.get(f"/api/v1/jobs/{self.multi_id}/result")
        zip_response = self.client.get(f"/api/v1/jobs/{self.multi_id}/artifacts.zip")
        self.assertEqual(json_response.status_code, 200)
        self.assertEqual(zip_response.status_code, 200)
        listed = json_response.json()["result"]["artifacts"]
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
            zipped = set(archive.namelist())
        self.assertIn("0_nut/path.json", listed)
        self.assertIn("artifact/json/path.json", zipped)


if __name__ == "__main__":
    unittest.main()
