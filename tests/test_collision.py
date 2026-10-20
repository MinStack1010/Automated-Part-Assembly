"""Collision-detection regression tests.

Reproduces the mesh-crossing bug: redmax BVHMesh.min_distance only samples
query *vertices* and returns a +10 sentinel for any outside point, so
vertex-sampling collision checks miss face-face interpenetration between
parallel same-size parts (and miss it at every interpolation step, enabling
tunneling). The fix switches the geometric planner and physics path
validation to triangle-level FCL over the real meshes.

Fixtures:
  tests/fixtures/collision_cubes/  - three unit cubes (0 still, 1 moving, 2 above)
  tests/fixtures/assembly_parts/   - the 5 connected components of
                                     assets/my_parts/assembly.obj (never modified)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import trimesh

from assets.load import load_assembly
from assets.transform import get_transform_matrix

CUBES = os.path.join(os.path.dirname(__file__), 'fixtures', 'collision_cubes')
PARTS = os.path.join(os.path.dirname(__file__), 'fixtures', 'assembly_parts')


def _make_planner(rotation=False):
    from baselines.run_joint_plan import PyPlanner
    return PyPlanner(CUBES, '1', ['0', '2'], rotation=rotation,
                     max_collision=1e-2, adaptive_collision=False,
                     body_type='bvh', sdf_dx=0.05)


class PenetrationDetectionTests(unittest.TestCase):
    """The FCL oracle must detect face-face interpenetration the old
    vertex-sampling oracle missed (d == 0.0 inside the overlap zone)."""

    def setUp(self):
        self.planner = _make_planner()
        self.collision_fn = self.planner.get_collision_fn()

    def test_rest_is_free(self):
        self.assertFalse(self.collision_fn(np.zeros(3)))

    def test_overlap_zone_detected(self):
        # Cubes A [-0.5,0.5] and B [1,2] along x; q.x in (-2.5,-0.5) overlaps.
        for qv in (-0.6, -1.0, -1.5, -2.0, -2.4):
            self.assertTrue(self.collision_fn(np.array([qv, 0.0, 0.0])),
                            f'penetration at q.x={qv} must be detected')

    def test_clear_zones_free(self):
        for qv in (0.0, -2.6, -3.0, 1.0):
            self.assertFalse(self.collision_fn(np.array([qv, 0.0, 0.0])),
                             f'clear state at q.x={qv} must stay free')

    def test_two_meshes_head_on(self):
        # Move B directly toward A along -x; an intermediate overlapped state
        # must be flagged when sampled by extend_fn.
        dist_fn = self.planner.get_distance_fn()
        extend_fn = self.planner.get_extend_fn(dist_fn, step_size=0.1)
        q0, q1 = np.zeros(3), np.array([-3.0, 0.0, 0.0])
        hits = [self.collision_fn(q) for q in extend_fn(q0, q1)]
        self.assertTrue(any(hits), 'head-on path must hit the overlap zone')


class TunnelingTests(unittest.TestCase):
    """A single large step must not leap past an obstacle."""

    def setUp(self):
        self.planner = _make_planner()
        self.collision_fn = self.planner.get_collision_fn()

    def test_large_step_blocked(self):
        dist_fn = self.planner.get_distance_fn()
        extend_fn = self.planner.get_extend_fn(dist_fn, step_size=100.0)
        q0, q1 = np.zeros(3), np.array([-3.0, 0.0, 0.0])
        steps = list(extend_fn(q0, q1))
        self.assertGreater(len(steps), 1, 'step must be clamped by max_safe_step')
        hits = [self.collision_fn(q) for q in steps]
        self.assertTrue(any(hits), 'clamped stepping must detect the obstacle')

    def test_step_clamped_to_mesh_feature(self):
        self.assertLessEqual(self.planner.max_safe_step,
                             0.5 * min(m.extents.min() for m in
                                       [self.planner.viz_mesh_move,
                                        *self.planner.viz_meshes_still]) + 1e-9)

    def test_extend_lands_on_target(self):
        dist_fn = self.planner.get_distance_fn()
        extend_fn = self.planner.get_extend_fn(dist_fn, step_size=0.3)
        q0, q1 = np.zeros(3), np.array([-0.3, 0.0, 0.0])  # clear zone, 0.3 away
        steps = list(extend_fn(q0, q1))
        np.testing.assert_allclose(steps[-1], q1, atol=1e-12)
        # no overshoot beyond q1
        for s in steps:
            self.assertLessEqual(s[0], q1[0] + 1e-12)


class RotationTranslationTests(unittest.TestCase):
    """Combined translation + rotation states must be checked with the full
    6-DOF transform, not just translation."""

    def setUp(self):
        self.planner = _make_planner(rotation=True)
        self.collision_fn = self.planner.get_collision_fn()

    def test_rotation_into_obstacle_detected(self):
        # q = (translation, rotvec). Translate B to overlap A while rotating.
        q_overlap = np.array([-1.5, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.assertTrue(self.collision_fn(q_overlap))

    def test_rotation_away_stays_free(self):
        q_clear = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertFalse(self.collision_fn(q_clear))


class AssemblyPartsFCLTests(unittest.TestCase):
    """Real assembly geometry (the 5 components of assets/my_parts/assembly.obj):
    FCL must agree with the geometric oracle on rest and overlap states."""

    def setUp(self):
        from baselines.run_joint_plan import PyPlanner
        # part 1 is a ring around part 0 (nested); disassembly must move it up.
        self.planner = PyPlanner(PARTS, '1', ['0'], rotation=False,
                                 max_collision=1e-2, adaptive_collision=False,
                                 body_type='bvh', sdf_dx=0.05)
        self.collision_fn = self.planner.get_collision_fn()

    def test_rest_not_deeply_penetrating(self):
        # Rest state: parts touch/nest but must not read as deep penetration.
        # (FCL contact at shared surfaces is allowed; deep overlap is not.)
        d = self.planner._still_fcl.min_distance_other(self.planner._move_fcl)
        self.assertGreater(d, -0.5,
                           'rest assembly must not be deeply interpenetrating')

    def test_lateral_shove_flagged(self):
        # Shoving part 1 sideways into part 0 must be flagged as collision.
        self.assertTrue(self.collision_fn(np.array([0.5, 0.0, 0.0])))

    def test_axial_pull_away_eventually_clear(self):
        # Pulling part 1 up along +z clears part 0 eventually.
        for z in (0.0, 1.0, 2.0, 4.0, 8.0):
            hit = self.collision_fn(np.array([0.0, 0.0, z]))
        self.assertFalse(hit, 'far pull must be clear')


class MultiPlanOrderTests(unittest.TestCase):
    """Multi-plan: each removal step must be planned against the remaining
    parts and verified; a path crossing a still-installed part is rejected."""

    def test_each_part_checked_against_rest(self):
        from baselines.run_joint_plan import PyPlanner
        pids = ['0', '1', '2', '3', '4']
        for move in pids:
            still = [p for p in pids if p != move]
            planner = PyPlanner(PARTS, move, still, rotation=False,
                                max_collision=1e-2, adaptive_collision=False,
                                body_type='bvh', sdf_dx=0.05)
            collision_fn = planner.get_collision_fn()
            # rest state must be free (assembled parts touch but the margin
            # check uses FCL contact which is allowed at rest via max_collision)
            # and a big sideways shove into the nest must be flagged.
            self.assertFalse(collision_fn(np.zeros(3)),
                             f'rest state of part {move} must be planned free')
            # a state deep inside the assembly bounds must be flagged for at
            # least the nested parts (0/1); top-level parts may already be free.
            if move in ('0', '1'):
                self.assertTrue(collision_fn(np.array([0.3, 0.0, 0.0])),
                                f'overlapped state of part {move} must be rejected')


if __name__ == '__main__':
    unittest.main()
