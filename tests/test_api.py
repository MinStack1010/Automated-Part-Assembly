"""Smoke tests for the HTTP wrapper and its real native integration."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from api.server import app
from api.services.native import native_status


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
        self.assertTrue({"mesh.distance", "simulation.demo", "joint.plan", "multi.plan"}.issubset(names))

    def test_installed_assembly_inspection(self) -> None:
        response = self.client.get("/api/v1/assemblies/multi_assembly/00003")
        self.assertEqual(response.status_code, 200)
        metadata = response.json()["result"]
        self.assertGreaterEqual(metadata["part_count"], 2)
        self.assertEqual(len(metadata["part_ids"]), metadata["part_count"])

    def test_validation_errors_use_the_standard_shape(self) -> None:
        response = self.client.post("/api/v1/mesh-distance", json={"assembly": {"collection": "multi_assembly"}})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.json()["success"])
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    @unittest.skipUnless(native_status()[0], "redmax_py is not built for this Python runtime")
    def test_native_demo_crosses_python_pybind_cpp_boundary(self) -> None:
        response = self.client.post(
            "/api/v1/simulations/demo",
            json={"environment": "SinglePendulum-Test", "integrator": "BDF2", "num_steps": 1},
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertEqual(result["environment"], "SinglePendulum-Test")
        self.assertEqual(result["num_steps"], 1)
        self.assertIsInstance(result["q"], list)
