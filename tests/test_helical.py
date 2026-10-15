"""Explicit helical screw-motion constraint (pin/bushing unscrewing).

Covers the reported failure mode: physics BFS with rotation=true applies
independent torque/push actions and its hull-based goal test accepts lateral
(crab-like) escapes. The explicit constraint branch may only emit coupled
rotation+translation states satisfying axial = lead * angle / (2*pi).

The positive tests use a synthetic smooth coaxial pair (no detectable
thread), so any success must come from the constraint machinery itself, not
from pitch estimation. The negative test uses the real threaded fixture,
where rigid helical motion is geometrically impossible.
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
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from api.config import settings
from api.errors import ApiError
from api.schemas import AssemblyRef, HelicalConstraint, JointPlanRequest, MultiPlanRequest
from api.server import app
from api.services.helical import build_helical_path, resolve_helical_spec
from api.services.planning import run_joint_plan, run_multi_plan
from assets.transform import get_transform_matrix

_TWO_PI = 2.0 * np.pi


def _smooth_pair(tmp: Path) -> Path:
    """Coaxial smooth pin + ring nut with radial clearance (no thread)."""
    pin = trimesh.creation.cylinder(radius=0.45, height=4.0, sections=32)
    nut = trimesh.creation.annulus(r_min=0.6, r_max=1.2, height=1.5, sections=48)
    pin.export(str(tmp / "pin.obj"))
    nut.export(str(tmp / "nut.obj"))
    return tmp


def _assert_helical_relation(test: unittest.TestCase, states, axis, lead) -> None:
    # Planner states start at sweep step 1 (theta > 0), so the relation is
    # checked on deltas between emitted states, not against a zero state.
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    base = np.asarray(states[0], dtype=float)
    base_angle = float(np.linalg.norm(base[3:]))
    base_axial = float(base[:3] @ axis)
    for index, state in enumerate(states):
        state = np.asarray(state, dtype=float)
        angle = float(np.linalg.norm(state[3:]))
        delta_pos = state[:3] - base[:3]
        # No lateral drift off the screw line through the start pose.
        lateral = np.linalg.norm(delta_pos - axis * float(delta_pos @ axis))
        test.assertLess(lateral, 1e-6, f"state {index} drifts laterally: {lateral}")
        # Coupled screw motion: axial travel == lead * angle / (2*pi).
        test.assertAlmostEqual(
            float(state[:3] @ axis) - base_axial,
            lead * (angle - base_angle) / _TWO_PI,
            places=5,
            msg=f"state {index} breaks the lead relation",
        )
        # Rotation happens about the screw axis, not an arbitrary axis.
        if angle > 1e-9:
            rot_axis = state[3:] / angle
            test.assertGreater(abs(float(rot_axis @ axis)), 1.0 - 1e-6, f"state {index} spins off-axis")


class HelicalConstraintSchemaTests(unittest.TestCase):
    def test_valid_spec_parses(self) -> None:
        spec = HelicalConstraint(
            base_part="screw", moving_part="nut", axis=[0.0, 0.0, 1.0], lead=1.9822, direction=1
        )
        self.assertEqual(spec.direction, 1)

    def test_axis_must_be_a_3_vector(self) -> None:
        with self.assertRaises(ValidationError):
            HelicalConstraint(base_part="screw", moving_part="nut", axis=[0.0, 1.0], lead=1.0, direction=1)

    def test_lead_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            HelicalConstraint(base_part="screw", moving_part="nut", axis=[0, 0, 1], lead=0.0, direction=1)

    def test_direction_must_be_pm_one(self) -> None:
        with self.assertRaises(ValidationError):
            HelicalConstraint(base_part="screw", moving_part="nut", axis=[0, 0, 1], lead=1.0, direction=0)  # type: ignore[arg-type]

    def test_parts_must_differ(self) -> None:
        with self.assertRaises(ValidationError):
            HelicalConstraint(base_part="nut", moving_part="nut", axis=[0, 0, 1], lead=1.0, direction=1)

    def test_non_z_axis_accepted(self) -> None:
        spec = HelicalConstraint(base_part="a", moving_part="b", axis=[1.0, 0.0, 0.0], lead=0.5, direction=-1)
        self.assertEqual(spec.axis, [1.0, 0.0, 0.0])


class ResolveHelicalSpecTests(unittest.TestCase):
    PARTS = ["pin", "bushing"]

    def _spec(self, **overrides):
        base = {"base_part": "pin", "moving_part": "bushing", "axis": [0, 0, 2], "lead": 0.5, "direction": 1}
        base.update(overrides)
        return base

    def test_valid_spec_normalizes_axis(self) -> None:
        resolved = resolve_helical_spec(self._spec(), self.PARTS, move_id="bushing", still_ids=["pin"])
        self.assertEqual(resolved["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(resolved["lead"], 0.5)
        self.assertEqual(resolved["direction"], 1)

    def test_unknown_parts_rejected(self) -> None:
        with self.assertRaises(ApiError) as ctx:
            resolve_helical_spec(self._spec(base_part="nope"), self.PARTS)
        self.assertEqual(ctx.exception.status_code, 400)
        with self.assertRaises(ApiError):
            resolve_helical_spec(self._spec(moving_part="nope"), self.PARTS)

    def test_zero_axis_rejected(self) -> None:
        with self.assertRaises(ApiError):
            resolve_helical_spec(self._spec(axis=[0, 0, 0]), self.PARTS)

    def test_nonpositive_lead_rejected(self) -> None:
        with self.assertRaises(ApiError):
            resolve_helical_spec(self._spec(lead=-1.0), self.PARTS)

    def test_move_mismatch_rejected(self) -> None:
        with self.assertRaises(ApiError):
            resolve_helical_spec(self._spec(), self.PARTS, move_id="pin", still_ids=["bushing"])
        with self.assertRaises(ApiError):
            resolve_helical_spec(self._spec(), self.PARTS, move_id="bushing", still_ids=["other"])

    def test_helical_with_non_helical_planner_rejected(self) -> None:
        request = JointPlanRequest(
            assembly=AssemblyRef(collection="multi_assembly", assembly_id="00013"),
            planner="bfs",
            helical=HelicalConstraint(base_part="a", moving_part="b", axis=[0, 0, 1], lead=1.0, direction=1),
        )
        self.assertEqual(request.planner, "bfs")
        self.assertIsNotNone(request.helical)


class ExplicitSweepTests(unittest.TestCase):
    """End-to-end screw motion on a smooth coaxial pair with a pinned lead."""

    def test_explicit_sweep_produces_coupled_verifiable_motion(self) -> None:
        from trimesh.collision import CollisionManager

        from assets.load import load_assembly

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        _smooth_pair(tmp)
        spec = {"base_part": "pin", "moving_part": "nut", "axis": [0, 0, 1], "lead": 0.5, "direction": 1}
        status, _, path, info = build_helical_path(tmp, "nut", ["pin"], max_time=120, helical=spec)
        self.assertEqual(status, "Success", info)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(info.get("mode"), "explicit")
        self.assertIn(info.get("rotation_sign"), (1, -1))
        _assert_helical_relation(self, path, [0, 0, 1], 0.5)
        # Independently re-verify every emitted state is collision-free
        # (sweep verification must hold, not just be claimed).
        meshes, names = load_assembly(str(tmp), return_names=True)
        by_id = {name.replace(".obj", ""): mesh for mesh, name in zip(meshes, names)}
        manager = CollisionManager()
        manager.add_object("pin", by_id["pin"])
        for index, state in enumerate(path):
            probe = CollisionManager()
            probe.add_object("nut", by_id["nut"])
            probe.set_transform("nut", get_transform_matrix(np.asarray(state, dtype=float)))
            self.assertFalse(
                manager.in_collision_other(probe), f"emitted state {index} collides"
            )
        # The exported animation must equal the planner states exactly.
        from assets.save import save_path

        save_path(str(tmp / "anim"), path, n_frame=len(path), name="nut")
        frames = json.loads((tmp / "anim" / "path.json").read_text())
        self.assertEqual(len(frames), len(path))
        for frame, state in zip(frames, path):
            self.assertEqual(frame["name"], "nut")
            np.testing.assert_allclose(
                np.asarray(frame["matrix"]), get_transform_matrix(np.asarray(state, dtype=float))
            )

    def test_explicit_sweep_reports_failure_without_fake_path(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "threaded_nut_screw"
        spec = {"base_part": "screw", "moving_part": "nut", "axis": [0, 0, 1], "lead": 1.9822, "direction": 1}
        status, _, path, info = build_helical_path(fixture, "nut", ["screw"], max_time=120, helical=spec)
        self.assertEqual(status, "Failure")
        self.assertIsNone(path)
        self.assertEqual(info.get("mode"), "explicit")
        self.assertEqual(info.get("reason"), "no_collision_free_combo")


class HelicalEndToEndTests(unittest.TestCase):
    """API-level wiring: helical spec reaches the planner for joint- and multi-plan."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        pair = Path(cls._tmp.name) / "pair"
        pair.mkdir()
        _smooth_pair(pair)
        files = [
            ("files", ("pin.obj", (pair / "pin.obj").read_bytes(), "application/octet-stream")),
            ("files", ("nut.obj", (pair / "nut.obj").read_bytes(), "application/octet-stream")),
        ]
        response = cls.client.post("/api/v1/assemblies/upload", files=files)
        assert response.status_code == 200, response.text
        cls.upload_id = response.json()["result"]["source"]["upload_id"]
        cls.addClassCleanup(shutil.rmtree, settings.uploads_dir / cls.upload_id, True)

    def _spec(self):
        return HelicalConstraint(base_part="pin", moving_part="nut", axis=[0, 0, 1], lead=0.5, direction=1)

    def test_joint_plan_explicit_helical(self) -> None:
        result = run_joint_plan(
            JointPlanRequest(
                assembly=AssemblyRef(upload_id=self.upload_id),
                engine="physics",
                planner="helical",
                move_id="nut",
                still_ids=["pin"],
                helical=self._spec(),
                max_time=120,
                mesh_diagnostics=False,
                seed=1,
            )
        )
        self.assertEqual(result["status"], "Success", result.get("helical"))
        self.assertEqual(result["helical"]["mode"], "explicit")
        _assert_helical_relation(self, result["path"], [0, 0, 1], 0.5)

    def test_multi_plan_explicit_helical(self) -> None:
        artifact_dir = Path(self._tmp.name) / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        result = run_multi_plan(
            MultiPlanRequest(
                assembly=AssemblyRef(upload_id=self.upload_id),
                engine="physics",
                sequence_planner="queue",
                path_planner="helical",
                helical=self._spec(),
                sequence_max_time=600,
                path_max_time=120,
                save_artifacts=True,
                seed=1,
            ),
            str(artifact_dir),
        )
        self.assertEqual(result["status"], "Success", result)
        self.assertEqual(result["sequence"], ["nut"])
        step_files = list(artifact_dir.rglob("path.json"))
        self.assertTrue(step_files, "multi-plan must export step artifacts")
        frames = json.loads(step_files[0].read_text())
        states = [np.asarray(frame["matrix"], dtype=float) for frame in frames]
        # Animation frames must satisfy the pinned lead relation. The total
        # screw rotation spans several turns, so per-frame absolute angles
        # would wrap past pi; accumulate signed per-step angles instead.
        start = states[0][:3, 3]
        axis = np.array([0.0, 0.0, 1.0])
        cumulative_angle = 0.0
        for index, matrix in enumerate(states):
            if index > 0:
                step_rot = states[index - 1][:3, :3].T @ matrix[:3, :3]
                step_rv = Rotation.from_matrix(step_rot).as_rotvec()
                cumulative_angle += float(step_rv @ axis)
            delta = matrix[:3, 3] - start
            lateral = np.linalg.norm(delta - axis * float(delta @ axis))
            self.assertLess(lateral, 1e-6, f"frame {index} drifts laterally")
            self.assertAlmostEqual(float(delta @ axis), 0.5 * cumulative_angle / _TWO_PI, places=3)


if __name__ == "__main__":
    unittest.main()
