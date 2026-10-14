"""movement_direction was removed from the joint-plan API.

The field, its validator, and the API plumbing are gone: ``JointPlanRequest``
neither defines nor forwards ``movement_direction``. The planner-level
half-space helpers below remain part of the library code (they still default
to ``None`` = unconstrained) and are covered here for regression safety.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.schemas import AssemblyRef, JointPlanRequest
from api.server import app
from api.services.threads import planning_assembly
from baselines.run_joint_plan import PyPlanner
from examples.run_joint_plan import allowed_actions

# assets/my_parts now holds user-supplied threaded meshes; use the preserved
# simplified pair so these half-space assertions keep their original geometry.
MY_PARTS = Path(__file__).resolve().parent.parent / "assets" / "joint_assembly" / "screw_nut_demo"
DOWN = [0.0, 0.0, -1.0]


def _request(**overrides) -> JointPlanRequest:
    payload = {
        "assembly": AssemblyRef(collection="multi_assembly", assembly_id="00003"),
        "move_id": "0",
        "still_ids": ["1"],
    }
    payload.update(overrides)
    return JointPlanRequest(**payload)


class MovementDirectionRemovedTests(unittest.TestCase):
    def test_field_is_absent_from_the_schema(self) -> None:
        self.assertNotIn("movement_direction", JointPlanRequest.model_fields)
        self.assertFalse(hasattr(_request(), "movement_direction"))

    def test_extra_payload_field_is_ignored_not_forwarded(self) -> None:
        request = _request(movement_direction=[0.0, 0.0, -1.0])
        self.assertNotIn("movement_direction", request.model_dump(mode="json"))

    def test_openapi_schema_has_no_movement_direction(self) -> None:
        client = TestClient(app)
        schema = client.get("/openapi.json").json()["components"]["schemas"]["JointPlanRequest"]
        self.assertNotIn("movement_direction", schema.get("properties", {}))

    def test_direction_normalization_validator_is_gone(self) -> None:
        # A zero vector used to be rejected with a validation error; the field
        # no longer exists, so the payload must validate and drop the key.
        request = _request(movement_direction=[0.0, 0.0, 0.0])
        self.assertNotIn("movement_direction", request.model_dump(mode="json"))


class PhysicsActionFilterTests(unittest.TestCase):
    def test_translation_pushes_against_the_direction_are_dropped(self) -> None:
        actions = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
        kept = allowed_actions(actions, DOWN)
        self.assertEqual(kept.tolist(), [[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    def test_rotational_actions_are_never_restricted(self) -> None:
        actions = np.array(
            [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, -1], [0, 0, 0, 1, 0, 0], [0, 0, 0, -1, 0, 0]],
            dtype=float,
        )
        kept = allowed_actions(actions, DOWN)
        self.assertEqual(len(kept), 4)
        # a pure translation inside a rotation action set is still restricted
        mixed = np.array([[0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]], dtype=float)
        self.assertEqual(allowed_actions(mixed, DOWN).tolist(), [[0, 0, -1, 0, 0, 0]])

    def test_without_direction_the_action_set_is_unchanged(self) -> None:
        actions = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
        self.assertEqual(allowed_actions(actions, None).tolist(), actions.tolist())


class GeometricHalfSpaceTests(unittest.TestCase):
    def test_states_behind_the_start_plane_are_collisions(self) -> None:
        with planning_assembly(MY_PARTS, enabled=True) as (planning_dir, _report):
            planner = PyPlanner(
                str(planning_dir), "nut", ["screw"], False, 0.01, False, "bvh", 0.05,
                movement_direction=[0.0, 0.0, 1.0],
            )
            collision_fn = planner.get_collision_fn()
            self.assertFalse(collision_fn(np.zeros(3)))
            self.assertFalse(collision_fn(np.array([0.0, 0.0, 2.0])))
            self.assertTrue(collision_fn(np.array([0.0, 0.0, -0.2])))
            self.assertTrue(collision_fn(np.array([0.4, 0.0, -0.05])))

    def test_direction_none_leaves_the_start_state_unconstrained(self) -> None:
        with planning_assembly(MY_PARTS, enabled=True) as (planning_dir, _report):
            planner = PyPlanner(str(planning_dir), "nut", ["screw"], False, 0.01, False, "bvh", 0.05)
            collision_fn = planner.get_collision_fn()
            self.assertFalse(collision_fn(np.array([0.0, 0.0, -0.2])))


if __name__ == "__main__":
    unittest.main()
