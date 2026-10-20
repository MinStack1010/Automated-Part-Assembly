"""Regression tests for hierarchical planning with robust motion fallbacks.

Covers the restored behavior: when helical (screw) planning fails or times out,
the system falls back to combined translation+rotation motion (bfs/bk-rrt with
rotation=True), then to alternative seeds, before declaring failure.

Tests mock the native planner and helical synthesizer so they run deterministically
without requiring the redmax_py native backend or long real planning runs.
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
from api.schemas import AssemblyRef, JointPlanRequest
from api.services.fallback import FallbackConfig, PlanningStrategyRunner
from api.services.planning import run_joint_plan


def _install_fake_run_joint_plan() -> None:
    """Pre-inject a fake examples.run_joint_plan module so patching get_planner
    never triggers a real redmax_py import."""
    import examples as _examples_pkg

    fake = types.ModuleType("examples.run_joint_plan")
    fake.get_planner = MagicMock(name="get_planner")
    sys.modules["examples.run_joint_plan"] = fake
    _examples_pkg.run_joint_plan = fake


def _smooth_pair(tmp: Path) -> Path:
    """Coaxial smooth pin + ring nut (no thread)."""
    pin = trimesh.creation.cylinder(radius=0.45, height=4.0, sections=32)
    nut = trimesh.creation.annulus(r_min=0.6, r_max=1.2, height=1.5, sections=48)
    pin.export(str(tmp / "pin.obj"))
    nut.export(str(tmp / "nut.obj"))
    return tmp


def _fake_planner(success: bool, calls: list, delay: float = 0.0):
    """Build a fake get_planner factory that records calls and returns canned status."""
    _install_fake_run_joint_plan()

    def _get_planner(name: str):
        def _factory(*args, **kwargs):
            planner = MagicMock()

            def _plan(max_time, seed=1, return_path=False, render=False, record_path=None, **kw):
                import time as _time
                calls.append({"planner": name, "seed": seed, "max_time": max_time, "rotation": args[3] if len(args) > 3 else False})
                if delay:
                    _time.sleep(delay)
                if success:
                    path = [np.array([0, 0, 0, 0, 0, 0], dtype=float), np.array([0, 0, 1.5, 0, 0, 0], dtype=float)]
                    return "Success", 0.1, path
                return "Failure", 0.1, None

            planner.plan.side_effect = _plan
            return planner

        return _factory

    return _get_planner


class HelicalFallbackStrategyTests(unittest.TestCase):
    """Unit tests for PlanningStrategyRunner strategy selection."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))
        self.assembly_dir = Path(self.tmp)

    def _runner(self, config=None):
        return PlanningStrategyRunner(
            assembly_dir=self.assembly_dir,
            move_id="nut",
            still_ids=["pin"],
            max_time=10.0,
            seed=1,
            config=config,
        )

    def test_helical_success_skips_fallback(self) -> None:
        calls: list = []
        path = [np.array([0, 0, 0, 0, 0, 0], dtype=float)]
        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("api.services.fallback")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.addCleanup(logger.removeHandler, handler)

        with patch("api.services.helical.build_helical_path", return_value=("Success", 0.1, path, {"pitch": 0.5})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            runner = self._runner()
            result = runner.run()
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy, "helical")
        self.assertEqual(result.status, "Success")
        self.assertEqual(len(calls), 0, "fallback must not run after helical success")
        self.assertIn("HELICAL_PATH_FOUND", stream.getvalue())

    def test_helical_failure_triggers_combined_motion_fallback(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.1, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            runner = self._runner()
            result = runner.run()
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy, "combined")
        self.assertEqual(result.status, "Success")
        self.assertTrue(any(c["planner"] == "bfs" for c in calls))

    def test_helical_timeout_triggers_fallback(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Timeout", 0.1, None, {"reason": "timeout"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            runner = self._runner()
            result = runner.run()
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(calls), 1)

    def test_all_strategies_exhausted_returns_failure(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(False, calls)):
            runner = self._runner()
            result = runner.run()
        self.assertIsNone(result)
        self.assertGreaterEqual(len(calls), 2)

    def test_cancellation_stops_retries(self) -> None:
        calls: list = []
        cancel_flag = {"cancelled": False}

        def _cancel_check():
            return cancel_flag["cancelled"]

        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(False, calls)):
            runner = self._runner()
            runner.cancel_check = _cancel_check
            cancel_flag["cancelled"] = True
            result = runner.run()
        self.assertIsNone(result)
        self.assertTrue(runner.cancelled)
        self.assertEqual(len(calls), 0, "no fallback should run after cancellation")

    def test_alternative_planner_retried_after_combined_fails(self) -> None:
        calls: list = []

        def _fake_get_planner_sequence(name: str):
            def _factory(*args, **kwargs):
                planner = MagicMock()

                def _plan(max_time, seed=1, return_path=False, render=False, record_path=None, **kw):
                    is_alt = seed in (7, 13)
                    calls.append({"planner": name, "seed": seed})
                    if is_alt:
                        path = [np.array([0, 0, 0, 0, 0, 0], dtype=float)]
                        return "Success", 0.1, path
                    return "Failure", 0.1, None

                planner.plan.side_effect = _plan
                return planner

            return _factory

        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_get_planner_sequence):
            runner = self._runner()
            result = runner.run()
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy, "alternative")
        self.assertTrue(any(c["seed"] in (7, 13) for c in calls))

    def _capture_logs(self, runner):
        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("api.services.fallback")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.addCleanup(logger.removeHandler, handler)
        return ""


class HelicalPlannerFallbackTests(unittest.TestCase):
    """HelicalPlanner (multi-plan adapter) must also fall back on timeout/collision."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def _make_planner(self):
        from api.services.helical import HelicalPlanner

        return HelicalPlanner(
            asset_folder="assets",
            assembly_dir=str(self.tmp),
            move_id="nut",
            still_ids=["pin"],
            rotation=False,
            body_type="sdf",
        )

    def test_helical_planner_falls_back_on_timeout(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Timeout", 0.05, None, {"reason": "timeout"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            planner = self._make_planner()
            status, _, _ = planner.plan(max_time=5.0, return_path=True)
        self.assertEqual(status, "Success")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["rotation"], "fallback must enable rotation")
        self.assertEqual(planner.info.get("fallback"), "bfs")

    def test_helical_planner_falls_back_on_no_collision_free_combo(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            planner = self._make_planner()
            status, _, _ = planner.plan(max_time=5.0, return_path=True)
        self.assertEqual(status, "Success")
        self.assertEqual(len(calls), 1)

    def test_helical_planner_no_fallback_on_success(self) -> None:
        calls: list = []
        path = [np.array([0, 0, 0, 0, 0, 0], dtype=float)]
        with patch("api.services.helical.build_helical_path", return_value=("Success", 0.05, path, {})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            planner = self._make_planner()
            status, _, result_path = planner.plan(max_time=5.0, return_path=True)
        self.assertEqual(status, "Success")
        self.assertEqual(len(calls), 0)
        self.assertEqual(result_path, path)


class JointPlanIntegrationTests(unittest.TestCase):
    """run_joint_plan must route helical requests through the strategy runner."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))
        self.assembly_dir = Path(self.tmp)

    def _request(self, **overrides):
        base = {
            "assembly": AssemblyRef(upload_id="test-upload"),
            "engine": "physics",
            "planner": "helical",
            "move_id": "nut",
            "still_ids": ["pin"],
            "max_time": 5.0,
            "mesh_diagnostics": False,
            "seed": 1,
        }
        base.update(overrides)
        return JointPlanRequest(**base)

    @patch("api.services.planning.require_native", return_value=None)
    @patch("api.services.planning.get_metadata")
    @patch("api.services.planning.resolve_assembly")
    @patch("api.services.threads.planning_assembly")
    def test_joint_plan_helical_success(self, mock_assembly, mock_resolve, mock_metadata, mock_native) -> None:
        from contextlib import contextmanager

        @contextmanager
        def _ctx(*a, **k):
            yield self.assembly_dir, {"enabled": False}

        mock_assembly.side_effect = _ctx
        mock_resolve.return_value = self.assembly_dir
        mock_metadata.return_value = MagicMock(part_ids=["pin", "nut"], model_dump=lambda **_: {})

        path = [np.array([0, 0, 0, 0, 0, 0], dtype=float)]
        with patch("api.services.helical.build_helical_path", return_value=("Success", 0.1, path, {"pitch": 0.5})):
            result = run_joint_plan(self._request())
        self.assertEqual(result["status"], "Success")
        self.assertEqual(result["helical"]["strategy"], "helical")

    @patch("api.services.planning.require_native", return_value=None)
    @patch("api.services.planning.get_metadata")
    @patch("api.services.planning.resolve_assembly")
    @patch("api.services.threads.planning_assembly")
    def test_joint_plan_helical_fallback_to_combined(self, mock_assembly, mock_resolve, mock_metadata, mock_native) -> None:
        from contextlib import contextmanager

        @contextmanager
        def _ctx(*a, **k):
            yield self.assembly_dir, {"enabled": False}

        mock_assembly.side_effect = _ctx
        mock_resolve.return_value = self.assembly_dir
        mock_metadata.return_value = MagicMock(part_ids=["pin", "nut"], model_dump=lambda **_: {})

        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)):
            result = run_joint_plan(self._request())
        self.assertEqual(result["status"], "Success")
        self.assertEqual(result["helical"]["strategy"], "combined")
        self.assertEqual(result["helical"]["planner_used"], "bfs")
        self.assertGreaterEqual(len(calls), 1)

    @patch("api.services.planning.require_native", return_value=None)
    @patch("api.services.planning.get_metadata")
    @patch("api.services.planning.resolve_assembly")
    @patch("api.services.threads.planning_assembly")
    def test_joint_plan_all_strategies_exhausted(self, mock_assembly, mock_resolve, mock_metadata, mock_native) -> None:
        from contextlib import contextmanager

        @contextmanager
        def _ctx(*a, **k):
            yield self.assembly_dir, {"enabled": False}

        mock_assembly.side_effect = _ctx
        mock_resolve.return_value = self.assembly_dir
        mock_metadata.return_value = MagicMock(part_ids=["pin", "nut"], model_dump=lambda **_: {})

        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(False, [])):
            result = run_joint_plan(self._request())
        self.assertEqual(result["status"], "Failure")
        self.assertEqual(result["helical"]["reason"], "all_strategies_exhausted")

    @patch("api.services.planning.require_native", return_value=None)
    @patch("api.services.planning.get_metadata")
    @patch("api.services.planning.resolve_assembly")
    @patch("api.services.threads.planning_assembly")
    def test_non_threaded_assembly_uses_bfs_directly(self, mock_assembly, mock_resolve, mock_metadata, mock_native) -> None:
        """planner=bfs must not go through the helical strategy runner."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx(*a, **k):
            yield self.assembly_dir, {"enabled": True, "simplified_parts": []}

        mock_assembly.side_effect = _ctx
        mock_resolve.return_value = self.assembly_dir
        mock_metadata.return_value = MagicMock(part_ids=["pin", "nut"], model_dump=lambda **_: {})

        calls: list = []
        with patch("examples.run_joint_plan.get_planner", _fake_planner(True, calls)), \
             patch("api.services.helical.build_helical_path") as mock_helical:
            result = run_joint_plan(self._request(planner="bfs"))
        self.assertEqual(result["status"], "Success")
        mock_helical.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["planner"], "bfs")

    def test_invalid_planner_rejected_before_planning(self) -> None:
        with patch("api.services.planning.require_native", return_value=None), \
             self.assertRaises(ApiError) as ctx:
            run_joint_plan(self._request(planner="unsupported-planner"))
        self.assertEqual(ctx.exception.status_code, 400)


class TimeoutBudgetTests(unittest.TestCase):
    """Strategy budgets must respect the overall deadline."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def test_budgets_sum_to_max_time(self) -> None:
        config = FallbackConfig.from_max_time(10.0)
        self.assertAlmostEqual(
            config.helical_fraction + config.combined_fraction + config.alternative_fraction,
            10.0,
            places=5,
        )

    def test_deadline_exceeded_stops_further_strategies(self) -> None:
        calls: list = []
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(False, calls)):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp),
                move_id="nut",
                still_ids=["pin"],
                max_time=0.01,
                seed=1,
            )
            result = runner.run()
        self.assertIsNone(result)


class LoggingEventTests(unittest.TestCase):
    """Strategy transitions must emit structured log events."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _smooth_pair(Path(self.tmp))

    def _capture_logs(self):
        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("api.services.fallback")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.addCleanup(logger.removeHandler, handler)
        return stream

    def test_helical_attempt_logged(self) -> None:
        stream = self._capture_logs()
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(False, [])):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp),
                move_id="nut",
                still_ids=["pin"],
                max_time=5.0,
                seed=1,
            )
            runner.run()
        output = stream.getvalue()
        self.assertIn("HELICAL_ATTEMPT_STARTED", output)
        self.assertIn("HELICAL_PATH_NOT_FOUND", output)
        self.assertIn("COMBINED_MOTION_ATTEMPT_STARTED", output)
        self.assertIn("ALL_STRATEGIES_EXHAUSTED", output)

    def test_combined_motion_success_logged(self) -> None:
        stream = self._capture_logs()
        with patch("api.services.helical.build_helical_path",
                   return_value=("Failure", 0.05, None, {"reason": "no_collision_free_combo"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, [])):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp),
                move_id="nut",
                still_ids=["pin"],
                max_time=5.0,
                seed=1,
            )
            runner.run()
        output = stream.getvalue()
        self.assertIn("COMBINED_MOTION_PATH_FOUND", output)

    def test_helical_timeout_fallback_logged(self) -> None:
        stream = self._capture_logs()
        with patch("api.services.helical.build_helical_path",
                   return_value=("Timeout", 0.05, None, {"reason": "timeout"})), \
             patch("examples.run_joint_plan.get_planner", _fake_planner(True, [])):
            runner = PlanningStrategyRunner(
                assembly_dir=Path(self.tmp),
                move_id="nut",
                still_ids=["pin"],
                max_time=5.0,
                seed=1,
            )
            runner.run()
        output = stream.getvalue()
        self.assertIn("HELICAL_TIMEOUT_FALLBACK", output)


if __name__ == "__main__":
    unittest.main()
