"""Automatic screw-thread detection and simplification for planning inputs."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from fastapi.testclient import TestClient

from api.config import settings
from api.schemas import AssemblyRef, JointPlanRequest
from api.server import app
from api.services.meshes import _straight_pull
from api.services.planning import run_joint_plan
from api.services.threads import _load_parts, planning_assembly, process_parts

# The original helical-threaded pair used to live in assets/my_parts; since
# Task 2 simplified that demo assembly, the threaded reference meshes are kept
# here so thread-removal coverage does not depend on the demo geometry.
MY_PARTS = Path(__file__).resolve().parent / "fixtures" / "threaded_nut_screw"
# assets/my_parts now holds user-supplied threaded meshes; the simplified
# thread-free demo pair is preserved under joint_assembly/screw_nut_demo.
SIMPLE_PARTS = Path(__file__).resolve().parent.parent / "assets" / "joint_assembly" / "screw_nut_demo"
PART_NAMES = ("nut.obj", "screw.obj")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ThreadSimplificationTests(unittest.TestCase):
    """The threaded nut/screw pair in assets/my_parts is the reference case."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parts = _load_parts(MY_PARTS)
        cls.processed, cls.report = process_parts(cls.parts)

    def test_internal_and_external_threads_are_detected_and_simplified(self) -> None:
        nut = self.report["parts"]["nut"]
        screw = self.report["parts"]["screw"]
        self.assertTrue(nut["detected"], nut)
        self.assertTrue(nut["applied"], nut)
        self.assertEqual(nut["kind"], "internal")
        self.assertTrue(screw["detected"], screw)
        self.assertTrue(screw["applied"], screw)
        self.assertEqual(screw["kind"], "external")
        for entry in (nut, screw):
            self.assertEqual(abs(entry["axis"][2]), 1.0, entry)
            self.assertGreater(entry["vertices_moved"], 0, entry)
            self.assertEqual(entry["residual_candidates"], 0, entry)
            self.assertLess(abs(entry["volume_delta_pct"]), 30.0, entry)
            self.assertLess(entry["ripple_ratio"], 0.4, entry)
        self.assertEqual(sorted(self.report["simplified_parts"]), ["nut", "screw"])
        self.assertEqual(self.report["interference"]["regressions"], [])

    def test_simplified_parts_keep_topology_bounds_and_watertightness(self) -> None:
        for part_id, original in self.parts.items():
            simplified = self.processed[part_id]
            with self.subTest(part=part_id):
                self.assertEqual(len(simplified.vertices), len(original.vertices))
                self.assertEqual(len(simplified.faces), len(original.faces))
                self.assertTrue(original.is_watertight)
                self.assertTrue(simplified.is_watertight)
                self.assertTrue(np.allclose(simplified.bounds, original.bounds, atol=1e-9))

    def test_interlocked_nut_becomes_straight_pullable(self) -> None:
        original = _straight_pull(self.parts, "nut", steps=48)
        simplified = _straight_pull(self.processed, "nut", steps=48)
        if not original["available"] or not simplified["available"]:
            self.skipTest("python-fcl is not installed")
        self.assertFalse(original["free"], original)
        self.assertTrue(simplified["free"], simplified)
        self.assertEqual(simplified["direction"], [0.0, 0.0, -1.0])

    def test_source_files_are_never_modified(self) -> None:
        digests = {name: _digest(MY_PARTS / name) for name in PART_NAMES}
        with planning_assembly(MY_PARTS, enabled=True) as (planning_dir, report):
            self.assertNotEqual(planning_dir, MY_PARTS)
            self.assertTrue((planning_dir / "nut.obj").is_file())
            self.assertEqual(sorted(report["simplified_parts"]), ["nut", "screw"])
            scratch = planning_dir
        self.assertFalse(scratch.exists())
        for name, digest in digests.items():
            self.assertEqual(_digest(MY_PARTS / name), digest, name)
        self.assertFalse(report["source_modified"])

    def test_scratch_copy_keeps_translation_json_and_skips_sdf_files(self) -> None:
        translation = '{"nut": [0, 0, 0], "screw": [0, 0, 0]}'
        with tempfile.TemporaryDirectory() as tmp:
            assembly = Path(tmp) / "assembly"
            assembly.mkdir()
            for name in PART_NAMES:
                shutil.copy2(MY_PARTS / name, assembly / name)
            (assembly / "translation.json").write_text(translation)
            (assembly / "screw.sdf").write_text("<sdf/>")
            with planning_assembly(assembly, enabled=True) as (planning_dir, _report):
                self.assertEqual((planning_dir / "translation.json").read_text(), translation)
                self.assertFalse((planning_dir / "screw.sdf").exists())

    def test_disabled_planner_keeps_the_original_meshes(self) -> None:
        with planning_assembly(MY_PARTS, enabled=False, skip_reason="planner_requires_threads") as (
            planning_dir,
            report,
        ):
            self.assertEqual(planning_dir, MY_PARTS)
            self.assertFalse(report["enabled"])
            self.assertEqual(report["reason"], "planner_requires_threads")
            self.assertEqual(report["simplified_parts"], [])

    def test_unthreaded_parts_are_reported_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assembly = Path(tmp) / "assembly"
            assembly.mkdir()
            box = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
            other = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
            other.apply_translation([10.0, 0.0, 0.0])
            box.export(str(assembly / "a.obj"))
            other.export(str(assembly / "b.obj"))
            parts = _load_parts(assembly)
            processed, report = process_parts(parts)
            self.assertEqual(report["simplified_parts"], [])
            for part_id, entry in report["parts"].items():
                with self.subTest(part=part_id):
                    self.assertFalse(entry["detected"])
                    self.assertEqual(entry["reason"], "no_thread_detected")
            self.assertIs(processed["a"], parts["a"])
            self.assertIs(processed["b"], parts["b"])

    def test_non_watertight_thread_is_reported_not_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assembly = Path(tmp) / "assembly"
            assembly.mkdir()
            nut = _load_parts(MY_PARTS)["nut"].copy()
            nut.update_faces(np.arange(len(nut.faces) - 1))
            self.assertFalse(nut.is_watertight)
            nut.export(str(assembly / "nut.obj"))
            parts = _load_parts(assembly)
            self.assertFalse(parts["nut"].is_watertight)
            processed, report = process_parts(parts)
            entry = report["parts"]["nut"]
            self.assertTrue(entry["detected"], entry)
            self.assertFalse(entry["applied"], entry)
            self.assertEqual(entry["reason"], "not_watertight")
            self.assertIs(processed["nut"], parts["nut"])


class SimpleDemoPairTests(unittest.TestCase):
    """The simplified assets/my_parts demo pair needs no thread preprocessing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parts = _load_parts(SIMPLE_PARTS)
        cls.processed, cls.report = process_parts(cls.parts)

    def test_pair_is_watertight_and_thread_free(self) -> None:
        for part_id, mesh in self.parts.items():
            with self.subTest(part=part_id):
                self.assertTrue(mesh.is_watertight)
        self.assertEqual(self.report["simplified_parts"], [])
        for entry in self.report["parts"].values():
            self.assertFalse(entry["detected"], entry)

    def test_pair_has_clearance_and_a_straight_exit(self) -> None:
        from api.services.meshes import mesh_gap

        gap = mesh_gap(SIMPLE_PARTS, move_id="screw", still_ids=["nut"])
        self.assertFalse(gap["overlapping"])
        self.assertGreater(gap["gap"], 0.0)
        pull = _straight_pull(self.parts, "nut", steps=48)
        self.assertTrue(pull["available"])
        self.assertTrue(pull["free"], pull)


class ThreadPlanningIntegrationTests(unittest.TestCase):
    """joint-plan must run on the simplified scratch copy and report it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls._uploads: list[Path] = []

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls._uploads:
            shutil.rmtree(path, ignore_errors=True)

    def _upload(self) -> str:
        files = [
            ("files", (name, (MY_PARTS / name).read_bytes(), "application/octet-stream"))
            for name in PART_NAMES
        ]
        response = self.client.post("/api/v1/assemblies/upload", files=files)
        self.assertEqual(response.status_code, 200, response.text)
        upload_id = response.json()["result"]["source"]["upload_id"]
        self._uploads.append(settings.uploads_dir / upload_id)
        return upload_id

    def test_geometric_joint_plan_runs_on_simplified_meshes(self) -> None:
        upload_id = self._upload()
        result = run_joint_plan(
            JointPlanRequest(
                assembly=AssemblyRef(upload_id=upload_id),
                move_id="nut",
                still_ids=["screw"],
                engine="geometric",
                planner="rrt",
                max_time=1.0,
                mesh_diagnostics=False,
                seed=1,
            )
        )
        removal = result["thread_removal"]
        self.assertTrue(removal["enabled"])
        self.assertEqual(sorted(removal["simplified_parts"]), ["nut", "screw"])
        self.assertFalse(removal["source_modified"])
        # One second is deliberately far too short for this planner to succeed;
        # what matters is that it planned against the simplified meshes.
        self.assertEqual(result["status"], "Timeout")
        self.assertEqual(result["path_state_count"], 0)

    def test_helical_planner_keeps_the_original_threads(self) -> None:
        upload_id = self._upload()
        result = run_joint_plan(
            JointPlanRequest(
                assembly=AssemblyRef(upload_id=upload_id),
                move_id="nut",
                still_ids=["screw"],
                engine="physics",
                planner="helical",
                max_time=1.0,
                mesh_diagnostics=False,
                seed=1,
            )
        )
        removal = result["thread_removal"]
        self.assertFalse(removal["enabled"])
        self.assertEqual(removal["reason"], "planner_requires_threads")
        self.assertEqual(removal["simplified_parts"], [])
        self.assertIn(result["status"], {"Timeout", "Failure", "Success"})


if __name__ == "__main__":
    unittest.main()
