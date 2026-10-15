"""Single-file multi-mesh upload: one OBJ holding several parts.

Covers the upload → split → plan → artifacts workflow plus the error and
backward-compatibility cases. Planning tests call the services directly
(same pattern as ThreadPlanningIntegrationTests) with the geometric engine,
which completed in seconds during manual verification.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from fastapi.testclient import TestClient

from api.config import settings
from api.schemas import AssemblyRef, JointPlanRequest, MultiPlanRequest
from api.server import app
from api.services.planning import run_joint_plan, run_multi_plan


def _combined_obj(specs: list[tuple[str, tuple[float, float, float]]]) -> bytes:
    # One scene exported as a single OBJ keeps global vertex indices valid.
    scene = trimesh.Scene()
    for name, offset in specs:
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.apply_translation(np.asarray(offset, dtype=float))
        scene.add_geometry(mesh, geom_name=name)
    exported = scene.export(file_type="obj")
    return exported.encode("utf-8") if isinstance(exported, str) else bytes(exported)


class SingleFileUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls._uploads: list[Path] = []

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls._uploads:
            shutil.rmtree(path, ignore_errors=True)

    def _upload_files(self, payload: list[tuple[str, bytes, str]], expected_status: int = 200):
        response = self.client.post(
            "/api/v1/assemblies/upload",
            files=[("files", (name, data, mime)) for name, data, mime in payload],
        )
        self.assertEqual(response.status_code, expected_status, response.text)
        return response

    def _upload_single(self, data: bytes, filename: str = "assembly.obj") -> str:
        response = self._upload_files([(filename, data, "application/octet-stream")])
        result = response.json()["result"]
        self._uploads.append(settings.uploads_dir / result["source"]["upload_id"])
        return result["source"]["upload_id"]

    def test_single_file_splits_into_named_parts(self) -> None:
        upload_id = self._upload_single(
            _combined_obj([("screw", (0.0, 0.0, 0.0)), ("nut", (5.0, 0.0, 0.0)), ("washer", (0.0, 5.0, 0.0))])
        )
        upload_dir = settings.uploads_dir / upload_id
        # The container file is replaced by one file per part.
        self.assertEqual(
            sorted(p.name for p in upload_dir.glob("*.obj")),
            ["nut.obj", "screw.obj", "washer.obj"],
        )
        result = self.client.post(
            "/api/v1/mesh-gap",
            json={"assembly": {"upload_id": upload_id}, "move_id": "screw", "still_ids": ["nut"]},
        ).json()["result"]
        self.assertEqual(result["part_ids"], ["nut", "screw", "washer"])
        # Geometry (including world placement) is preserved through the split.
        self.assertAlmostEqual(result["gap"], 4.0, places=3)

    def test_single_file_with_translation(self) -> None:
        data = _combined_obj([("screw", (0.0, 0.0, 0.0)), ("nut", (5.0, 0.0, 0.0))])
        translation = json.dumps({"screw": [1.0, 0.0, 0.0], "nut": [0.0, 0.0, 0.0]}).encode()
        response = self._upload_files(
            [("assembly.obj", data, "application/octet-stream"), ("translation.json", translation, "application/json")]
        )
        result = response.json()["result"]
        self._uploads.append(settings.uploads_dir / result["source"]["upload_id"])
        self.assertTrue(result["has_translation"])
        self.assertEqual(result["part_ids"], ["nut", "screw"])

    def test_single_file_translation_missing_part_is_rejected(self) -> None:
        data = _combined_obj([("screw", (0.0, 0.0, 0.0)), ("nut", (5.0, 0.0, 0.0))])
        translation = json.dumps({"screw": [0.0, 0.0, 0.0]}).encode()
        response = self._upload_files(
            [("assembly.obj", data, "application/octet-stream"), ("translation.json", translation, "application/json")],
            expected_status=400,
        )
        self.assertIn("nut", response.json()["error"]["message"])

    def test_duplicate_and_unsafe_object_names_are_sanitized(self) -> None:
        upload_id = self._upload_single(
            _combined_obj([("my part", (0.0, 0.0, 0.0)), ("my part", (5.0, 0.0, 0.0)), ("my/part", (0.0, 5.0, 0.0))])
        )
        upload_dir = settings.uploads_dir / upload_id
        names = sorted(p.stem for p in upload_dir.glob("*.obj"))
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        for stem in names:
            self.assertRegex(stem, r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def test_single_mesh_file_is_rejected(self) -> None:
        response = self._upload_files(
            [("one.obj", _combined_obj([("lonely", (0.0, 0.0, 0.0))]), "application/octet-stream")],
            expected_status=400,
        )
        self.assertEqual(response.json()["error"]["code"], "INVALID_ASSEMBLY")

    def test_unparsable_file_is_rejected(self) -> None:
        response = self._upload_files(
            [("bad.obj", b"this is not mesh data {{{", "application/octet-stream")], expected_status=400
        )
        self.assertFalse(response.json()["success"])

    def test_real_pin_bushing_fixture_splits_by_object_name(self) -> None:
        # Regression test for the reported frontend mismatch: the shipped
        # assets/my_parts/assembly_model.obj must yield stable part ids
        # "bushing" and "pin" (not the filename), matching planner move_ids.
        fixture = Path(__file__).resolve().parent.parent / "assets" / "my_parts" / "assembly_model.obj"
        self.assertTrue(fixture.is_file(), f"missing fixture: {fixture}")
        response = self._upload_files(
            [("assembly_model.obj", fixture.read_bytes(), "application/octet-stream")]
        )
        result = response.json()["result"]
        self.assertEqual(result["part_ids"], ["bushing", "pin"])
        self.assertEqual(result["part_count"], 2)
        self._uploads.append(settings.uploads_dir / result["source"]["upload_id"])
        upload_id = result["source"]["upload_id"]
        # Split parts keep their world placement: bushing is the larger ring.
        bounds = {}
        import trimesh as _trimesh

        for part_id in ("bushing", "pin"):
            mesh = _trimesh.load(str(settings.uploads_dir / upload_id / f"{part_id}.obj"), process=False)
            bounds[part_id] = mesh.bounds
        self.assertGreater(bounds["bushing"][1][0] - bounds["bushing"][0][0], 2.0)
        gap = self.client.post(
            "/api/v1/mesh-gap",
            json={"assembly": {"upload_id": upload_id}, "move_id": "pin", "still_ids": ["bushing"]},
        ).json()["result"]
        self.assertEqual(gap["move_id"], "pin")

    def test_multi_file_upload_still_works(self) -> None:
        # Backward compatibility: separate files keep working, one part each.
        box_a = trimesh.creation.box().export(file_type="obj")
        box_b = trimesh.creation.box().export(file_type="obj")
        if isinstance(box_a, str):
            box_a = box_a.encode()
            box_b = box_b.encode()  # type: ignore[union-attr]
        response = self._upload_files(
            [("0.obj", box_a, "application/octet-stream"), ("1.obj", box_b, "application/octet-stream")]
        )
        result = response.json()["result"]
        self._uploads.append(settings.uploads_dir / result["source"]["upload_id"])
        self.assertEqual(result["part_ids"], ["0", "1"])


class SingleFilePlanningTests(unittest.TestCase):
    """joint-plan and multi-plan run unchanged on a split single-file upload."""

    # Real interlocked parts: widely separated boxes are already disassembled
    # ('Start with goal'), which exports no path.json by design.
    _ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "multi_assembly" / "00013"

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls._uploads: list[Path] = []
        scene = trimesh.Scene()
        for file_id, part_id in (("0", "screw"), ("1", "nut"), ("2", "washer")):
            mesh = trimesh.load(str(cls._ASSET_DIR / f"{file_id}.obj"), process=False)
            scene.add_geometry(mesh, geom_name=part_id)
        exported = scene.export(file_type="obj")
        data = exported.encode("utf-8") if isinstance(exported, str) else bytes(exported)
        response = cls.client.post(
            "/api/v1/assemblies/upload", files=[("files", ("assembly.obj", data, "application/octet-stream"))]
        )
        assert response.status_code == 200, response.text
        cls.upload_id = response.json()["result"]["source"]["upload_id"]
        cls._uploads.append(settings.uploads_dir / cls.upload_id)
        cls._tmp = tempfile.TemporaryDirectory()
        # run_*_plan expect an existing artifact dir (the job manager mkdirs it in production).
        (Path(cls._tmp.name) / "joint").mkdir(parents=True, exist_ok=True)
        (Path(cls._tmp.name) / "multi").mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls._uploads:
            shutil.rmtree(path, ignore_errors=True)
        cls._tmp.cleanup()

    def test_joint_plan_on_split_upload(self) -> None:
        result = run_joint_plan(
            JointPlanRequest(
                assembly=AssemblyRef(upload_id=self.upload_id),
                move_id="screw",
                still_ids=["nut", "washer"],
                engine="geometric",
                planner="rrt",
                max_time=120,
                mesh_diagnostics=False,
                seed=1,
            ),
            str(Path(self._tmp.name) / "joint"),
        )
        self.assertEqual(result["status"], "Success")
        self.assertGreater(result["path_state_count"], 0)

    def test_multi_plan_on_split_upload(self) -> None:
        result = run_multi_plan(
            MultiPlanRequest(
                assembly=AssemblyRef(upload_id=self.upload_id),
                engine="geometric",
                sequence_planner="queue",
                path_planner="rrt",
                sequence_max_time=300,
                path_max_time=60,
                save_artifacts=True,
                seed=1,
            ),
            str(Path(self._tmp.name) / "multi"),
        )
        self.assertEqual(result["status"], "Success")
        step_dirs = list((Path(self._tmp.name) / "multi").rglob("path.json"))
        self.assertGreaterEqual(len(step_dirs), 2, "each disassembled part must export a path.json")


if __name__ == "__main__":
    unittest.main()
