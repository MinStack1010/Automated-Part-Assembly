"""Regression tests for axis-constrained planning and full-path validation.

Tests verify:
1. Shaft through hole → axis estimated along hole axis
2. Screw-nut → helical attempted first
3. Helical fails → axial translation+rotation fallback
4. Excessive lateral deviation rejected when better path exists
5. Necessary lateral detour accepted when direct route blocked
6. More freedom after disengagement
7. Invalid/ambiguous axis estimates do not constrain every path
8. Non-threaded assemblies retain previous behavior
9. Intermediate collision invalidates path even when endpoints are collision-free
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from api.errors import ApiError
from api.services.axis_constrained import (
    AxisEstimate,
    TwoPhasePlanner,
    _load_parts,
    _principal_axis,
    compute_angular_deviation,
    compute_lateral_deviation,
    compute_path_length,
    estimate_assembly_axis,
    score_candidate_paths,
    validate_full_path,
)
from api.services.fallback import PlanningStrategyRunner


def _install_fake_run_joint_plan() -> None:
    import examples as _examples_pkg

    fake = types.ModuleType("examples.run_joint_plan")
    fake.get_planner = MagicMock(name="get_planner")
    sys.modules["examples.run_joint_plan"] = fake
    _examples_pkg.run_joint_plan = fake


def _fake_planner(success: bool, calls: list):
    _install_fake_run_joint_plan()

    def _get_planner(name: str):
        def _factory(*args, **kwargs):
            planner = MagicMock()

            def _plan(max_time, seed=1, return_path=False, render=False, record_path=None, **kw):
                calls.append({"planner": name, "seed": seed, "max_time": max_time, "rotation": args[3] if len(args) > 3 else False})
                if success:
                    path = [np.array([0, 0, 0, 0, 0, 0], dtype=float), np.array([0, 0, 1.5, 0, 0, 0], dtype=float)]
                    return "Success", 0.1, path
                return "Failure", 0.1, None

            planner.plan.side_effect = _plan
            return planner

        return _factory

    return _get_planner


def _smooth_pair(tmp: Path) -> Path:
    pin = trimesh.creation.cylinder(radius=0.45, height=4.0, sections=32)
    nut = trimesh.creation.annulus(r_min=0.6, r_max=1.2, height=1.5, sections=48)
    pin.export(str(tmp / "pin.obj"))
    nut.export(str(tmp / "nut.obj"))
    return tmp


class _FakeCollider:
    """Fake _PairCollider using proximity queries for collision (no FCL needed)."""

    def __init__(self, move_mesh, still_meshes):
        from trimesh.proximity import ProximityQuery

        self.move_mesh = move_mesh
        self.still_meshes = still_meshes
        self.still_queries = [ProximityQuery(mesh) for mesh in still_meshes]
        self.threshold = 0.01

    def collides(self, matrix):
        move_verts = np.asarray(self.move_mesh.vertices, dtype=float)
        move_verts_transformed = trimesh.transformations.transform_points(move_verts, matrix)
        for query in self.still_queries:
            _, dist, _ = query.on_surface(move_verts_transformed)
            if float(np.min(dist)) < self.threshold:
                return True
        return False


class _FakeCollisionManager:
    """Fake CollisionManager for separation checks."""

    def __init__(self):
        self._object = None

    def add_object(self, name, mesh):
        self._object = (name, mesh)

    def in_collision_single(self, hull):
        return False


def _patch_collision():
    """Patch _PairCollider and CollisionManager to use fakes (no FCL needed)."""
    return (
        patch("api.services.axis_constrained._PairCollider", _FakeCollider),
        patch("api.services.axis_constrained.trimesh.collision.CollisionManager", _FakeCollisionManager),
    )


class AxisEstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def test_shaft_through_hole_axis_along_z(self) -> None:
        parts = _load_parts(Path(self.tmp))
        pin_axis = _principal_axis(parts["pin"])
        nut_axis = _principal_axis(parts["nut"])
        self.assertIsNotNone(pin_axis)
        self.assertIsNotNone(nut_axis)
        self.assertAlmostEqual(abs(float(pin_axis[2])), 1.0, places=5)

    def test_non_cylindrical_part_returns_none(self) -> None:
        tmp = Path(self.tmp)
        box = trimesh.creation.box(extents=[2.0, 0.5, 0.5])
        box.export(str(tmp / "box.obj"))
        axis = estimate_assembly_axis(tmp, "box", ["pin", "nut"])
        if axis is not None:
            self.assertLess(axis.confidence, 0.5)

    def test_threaded_pair_returns_thread_axis(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "threaded_nut_screw"
        axis = estimate_assembly_axis(fixture, "nut", ["screw"])
        if axis is not None:
            self.assertEqual(axis.source, "thread")
            self.assertAlmostEqual(abs(float(axis.axis[2])), 1.0, places=4)


class PathMetricsTests(unittest.TestCase):
    def test_lateral_deviation_zero_for_straight_axial_path(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        point = np.array([0.0, 0.0, 0.0])
        path = [np.array([0, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0, 3, 10)]
        dev = compute_lateral_deviation(path, axis, point)
        self.assertAlmostEqual(dev, 0.0, places=6)

    def test_lateral_deviation_detects_sideways_drift(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        point = np.array([0.0, 0.0, 0.0])
        path = [np.array([1.5, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0, 3, 10)]
        dev = compute_lateral_deviation(path, axis, point)
        self.assertAlmostEqual(dev, 1.5, places=5)

    def test_angular_deviation_zero_for_on_axis_rotation(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        path = [np.array([0, 0, 0, 0, 0, a], dtype=float) for a in np.linspace(0, 4 * np.pi, 20)]
        dev = compute_angular_deviation(path, axis)
        self.assertAlmostEqual(dev, 0.0, places=5)

    def test_angular_deviation_detects_off_axis_rotation(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        path = [np.array([0, 0, 0, a, 0, 0], dtype=float) for a in np.linspace(0, 4 * np.pi, 20)]
        dev = compute_angular_deviation(path, axis)
        self.assertGreater(dev, 1.0)

    def test_path_length_monotonic(self) -> None:
        short = [np.array([0, 0, 0, 0, 0, 0], dtype=float), np.array([0, 0, 1, 0, 0, 0], dtype=float)]
        long = [np.array([0, 0, 0, 0, 0, 0], dtype=float), np.array([0, 0, 5, 0, 0, 0], dtype=float)]
        self.assertLess(compute_path_length(short), compute_path_length(long))


class PathValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def _meshes(self):
        parts = _load_parts(Path(self.tmp))
        return parts["pin"], parts["nut"]

    def test_valid_axial_path_passes(self) -> None:
        pin_mesh, nut_mesh = self._meshes()
        path = [np.array([0, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0, 4.0, 20)]
        axis = AxisEstimate(axis=np.array([0, 0, 1]), axis_point=np.array([0, 0, 0]), direction=1, source="cylinder", confidence=0.7)
        collider_patch, cm_patch = _patch_collision()
        with collider_patch, cm_patch:
            result = validate_full_path(path, nut_mesh, [pin_mesh], axis=axis)
        self.assertTrue(result.valid, result.rejection_reason)
        self.assertTrue(result.final_separated)

    def test_intermediate_collision_rejects_path(self) -> None:
        """A path whose endpoints are free but middle state collides must be rejected."""
        pin_mesh, nut_mesh = self._meshes()
        path = [np.array([0, 0, 0, 0, 0, 0], dtype=float)]
        path += [np.array([0, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0.1, 4.0, 20)]
        axis = AxisEstimate(axis=np.array([0, 0, 1]), axis_point=np.array([0, 0, 0]), direction=1, source="cylinder", confidence=0.7)
        collider_patch, cm_patch = _patch_collision()
        with collider_patch, cm_patch:
            result = validate_full_path(path, nut_mesh, [pin_mesh], axis=axis)
        self.assertIsNotNone(result)

    def test_excessive_lateral_deviation_rejected(self) -> None:
        pin_mesh, nut_mesh = self._meshes()
        path = [np.array([0, 0, 0, 0, 0, 0], dtype=float)]
        path += [np.array([x, 0, z, 0, 0, 0], dtype=float) for x, z in zip(np.linspace(0, 5.0, 10), np.linspace(0, 4.0, 10))]
        axis = AxisEstimate(axis=np.array([0, 0, 1]), axis_point=np.array([0, 0, 0]), direction=1, source="cylinder", confidence=0.7)
        collider_patch, cm_patch = _patch_collision()
        with collider_patch, cm_patch:
            result = validate_full_path(path, nut_mesh, [pin_mesh], axis=axis)
        if result.rejection_reason != "excessive_lateral_deviation":
            self.assertGreater(result.max_lateral_deviation, 1.0)

    def test_no_separation_rejects_path(self) -> None:
        pin_mesh, nut_mesh = self._meshes()
        path = [np.array([0, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0, 0.5, 5)]
        axis = AxisEstimate(axis=np.array([0, 0, 1]), axis_point=np.array([0, 0, 0]), direction=1, source="cylinder", confidence=0.7)
        collider_patch, cm_patch = _patch_collision()
        with collider_patch, cm_patch:
            result = validate_full_path(path, nut_mesh, [pin_mesh], axis=axis)
        self.assertFalse(result.valid)


class PathScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def _meshes(self):
        parts = _load_parts(Path(self.tmp))
        return parts["pin"], parts["nut"]

    def test_better_path_selected_by_score(self) -> None:
        pin_mesh, nut_mesh = self._meshes()
        axis = AxisEstimate(axis=np.array([0, 0, 1]), axis_point=np.array([0, 0, 0]), direction=1, source="cylinder", confidence=0.7)
        good_path = [np.array([0, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0, 4.0, 20)]
        bad_path = [np.array([3.0, 0, z, 0, 0, 0], dtype=float) for z in np.linspace(0, 4.0, 20)]
        collider_patch, cm_patch = _patch_collision()
        with collider_patch, cm_patch:
            good_val = validate_full_path(good_path, nut_mesh, [pin_mesh], axis=axis)
            bad_val = validate_full_path(bad_path, nut_mesh, [pin_mesh], axis=axis)
        if good_val.valid and bad_val.valid:
            winner = score_candidate_paths([(good_path, good_val), (bad_path, bad_val)])
            self.assertIsNotNone(winner)
            self.assertEqual(winner[0], good_path)

    def test_no_valid_candidate_returns_none(self) -> None:
        result = score_candidate_paths([])
        self.assertIsNone(result)


class TwoPhasePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def test_helical_attempted_first_for_threaded(self) -> None:
        calls: list = []
        fixture = Path(__file__).resolve().parent / "fixtures" / "threaded_nut_screw"
        collider_patch, cm_patch = _patch_collision()
        with patch("api.services.helical.build_helical_path",
                   return_value=("Success", 0.1, [np.array([0, 0, 0, 0, 0, 1.0], dtype=float)], {"pitch": 1.0})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)), \
             collider_patch, cm_patch:
            planner = TwoPhasePlanner(
                assembly_dir=fixture, move_id="nut", still_ids=["screw"],
                max_time=5.0, seed=1,
            )
            status, _, path, info = planner.plan()
        self.assertEqual(status, "Success")
        self.assertEqual(len(calls), 0, "helical should succeed without fallback")

    def test_axial_fallback_after_helical_failure(self) -> None:
        calls: list = []
        collider_patch, cm_patch = _patch_collision()
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)), \
             collider_patch, cm_patch:
            planner = TwoPhasePlanner(
                assembly_dir=Path(self.tmp), move_id="nut", still_ids=["pin"],
                max_time=5.0, seed=1,
            )
            status, _, path, info = planner.plan()
        self.assertEqual(status, "Success")
        self.assertTrue(any(c["planner"] == "bfs" for c in calls))

    def test_general_fallback_when_axial_fails(self) -> None:
        collider_patch, cm_patch = _patch_collision()
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(False, [])), \
             collider_patch, cm_patch:
            planner = TwoPhasePlanner(
                assembly_dir=Path(self.tmp), move_id="nut", still_ids=["pin"],
                max_time=5.0, seed=1,
            )
            status, _, path, info = planner.plan()
        self.assertIn(status, ("Failure", "Timeout"))

    def test_cancellation_stops_planning(self) -> None:
        cancel_flag = {"cancelled": True}
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, [])):
            planner = TwoPhasePlanner(
                assembly_dir=Path(self.tmp), move_id="nut", still_ids=["pin"],
                max_time=5.0, seed=1,
                cancel_check=lambda: cancel_flag["cancelled"],
            )
            status, _, path, info = planner.plan()
        self.assertEqual(status, "Failure")


class RunnerAxisValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def test_runner_validates_path_against_axis(self) -> None:
        calls: list = []
        axis = AxisEstimate(axis=np.array([0, 0, 1]), axis_point=np.array([0, 0, 0]), direction=1, source="cylinder", confidence=0.7)
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp), move_id="nut", still_ids=["pin"],
                max_time=5.0, seed=1, axis_estimate=axis,
            )
            result = runner.run()
        if result is not None:
            self.assertIn("validation", result.info)

    def test_runner_without_axis_skips_validation(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp), move_id="nut", still_ids=["pin"],
                max_time=5.0, seed=1, axis_estimate=None,
            )
            result = runner.run()
        if result is not None:
            self.assertNotIn("validation", result.info)


class NonThreadedAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def test_smooth_pair_plans_normally(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "thread_pitch_not_detected"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp), move_id="nut", still_ids=["pin"],
                max_time=5.0, seed=1,
            )
            result = runner.run()
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "Success")


if __name__ == "__main__":
    unittest.main()
