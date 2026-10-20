"""Axis-constrained planning with two-phase extraction and path validation.

Estimates the expected mating/extraction axis from assembly geometry, uses it
to guide planning (helical → axial → general fallback), validates the full path
at intermediate states, and scores candidates to prefer axis-consistent motion.

Phase A (engaged): prefer translation along the mating axis with controlled
rotation; helical motion for threaded engagement when supported.
Phase B (disengaged): allow greater translational/rotational freedom once
geometric clearance confirms separation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import trimesh

from api.config import settings
from api.errors import ApiError

logger = logging.getLogger(__name__)

# Validation: max translation/rotation jump between consecutive states
DEFAULT_JUMP_TRANS_THRESHOLD = 2.0
DEFAULT_JUMP_ROT_THRESHOLD = np.pi / 2.0
# Scoring weights (lower score = better path)
WEIGHT_LATERAL = 1.0
WEIGHT_LENGTH = 0.01
WEIGHT_ANGULAR = 0.5


@dataclass
class AxisEstimate:
    """Estimated assembly/extraction axis with metadata."""
    axis: np.ndarray  # unit direction vector
    axis_point: np.ndarray  # a point on the axis (world coordinates)
    direction: int  # +1 or -1: extraction travel direction along axis
    source: str  # "thread", "cylinder", "pca", "user"
    confidence: float  # 0.0–1.0

    def __post_init__(self) -> None:
        self.axis = np.asarray(self.axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.axis_point = np.asarray(self.axis_point, dtype=float)
        self.direction = int(self.direction) if self.direction != 0 else 1


@dataclass
class ValidationResult:
    """Full-path validation outcome."""
    valid: bool
    max_lateral_deviation: float = 0.0
    max_angular_deviation: float = 0.0
    path_length: float = 0.0
    final_separated: bool = False
    collision_states: int = 0
    max_jump_translation: float = 0.0
    max_jump_rotation: float = 0.0
    score: float = float("inf")
    rejection_reason: Optional[str] = None


def estimate_assembly_axis(
    assembly_dir: Path,
    move_id: str,
    still_ids: list[str],
    thread_info: Optional[dict[str, Any]] = None,
) -> Optional[AxisEstimate]:
    """Estimate the mating/extraction axis from assembly geometry.

    Priority: thread axis > cylindrical common axis > PCA principal axis.
    Returns None when the axis cannot be identified reliably.
    """
    from api.services.helical import estimate_thread, vertices_center

    parts = _load_parts(assembly_dir)
    if move_id not in parts:
        raise ApiError(400, "INVALID_INPUT", "move_id is not a part in the assembly")
    move_mesh = parts[move_id]

    # 1. Thread axis (most reliable for threaded connections)
    if thread_info is None:
        thread_info = estimate_thread(move_mesh)
    if thread_info is not None:
        axis = np.asarray(thread_info["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        point = np.asarray(thread_info.get("axis_point") or vertices_center(move_mesh), dtype=float)
        return AxisEstimate(
            axis=axis,
            axis_point=point,
            direction=1,
            source="thread",
            confidence=float(thread_info.get("confidence", 0.8)),
        )

    # 2. Cylindrical common axis: check if moving part is roughly cylindrical
    pca_axis = _principal_axis(move_mesh)
    if pca_axis is not None and _is_cylindrical(move_mesh, pca_axis):
        still_axes = []
        for sid in still_ids:
            if sid in parts:
                sa = _principal_axis(parts[sid])
                if sa is not None and abs(float(np.dot(sa, pca_axis))) > 0.95:
                    still_axes.append(sa)
        if still_axes:
            # Still parts share the axis — likely a shaft-through-hole pair
            return AxisEstimate(
                axis=pca_axis,
                axis_point=vertices_center(move_mesh),
                direction=1,
                source="cylinder",
                confidence=0.7,
            )

    # 3. PCA principal axis as weak fallback (only for elongated parts)
    if pca_axis is not None and _is_elongated(move_mesh, pca_axis):
        return AxisEstimate(
            axis=pca_axis,
            axis_point=vertices_center(move_mesh),
            direction=1,
            source="pca",
            confidence=0.3,
        )

    logger.info("AXIS_ESTIMATION_FAILED move=%s reason=no_reliable_axis", move_id)
    return None


def _load_parts(assembly_dir: Path) -> dict[str, trimesh.Trimesh]:
    from assets.load import load_assembly

    meshes, names = load_assembly(str(assembly_dir), return_names=True)
    return {name.replace(".obj", ""): mesh for mesh, name in zip(meshes, names)}


def _principal_axis(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """Principal axis from PCA of vertices (direction of greatest variance)."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 3:
        return None
    centered = vertices - vertices.mean(axis=0)
    try:
        _, vecs = np.linalg.eigh(np.cov(centered.T))
        return vecs[:, -1]
    except np.linalg.LinAlgError:
        return None


def _is_cylindrical(mesh: trimesh.Trimesh, axis: np.ndarray, tol: float = 0.15) -> bool:
    """Check if the mesh is roughly a surface of revolution about the axis."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = vertices.mean(axis=0)
    offset = vertices - center
    along = offset @ axis
    perp = offset - np.outer(along, axis)
    radii = np.linalg.norm(perp, axis=1)
    if radii.mean() < 1e-9:
        return False
    radial_std = float(np.std(radii))
    radial_mean = float(np.mean(radii))
    return radial_std / radial_mean < tol


def _is_elongated(mesh: trimesh.Trimesh, axis: np.ndarray, ratio: float = 3.0) -> bool:
    """Check if the mesh is elongated along the axis."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = vertices.mean(axis=0)
    offset = vertices - center
    along_extent = float((offset @ axis).max() - (offset @ axis).min())
    perp = offset - np.outer(offset @ axis, axis)
    perp_extent = float(np.linalg.norm(perp, axis=1).max()) * 2.0
    if perp_extent < 1e-9:
        return False
    return along_extent / perp_extent >= ratio


def compute_lateral_deviation(
    path: list[np.ndarray], axis: np.ndarray, axis_point: np.ndarray
) -> float:
    """Maximum perpendicular distance of path positions from the axis line."""
    if not path:
        return 0.0
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    point = np.asarray(axis_point, dtype=float)
    max_dev = 0.0
    for state in path:
        state = np.asarray(state, dtype=float)
        pos = state[:3]
        delta = pos - point
        along = float(delta @ axis)
        lateral = delta - axis * along
        max_dev = max(max_dev, float(np.linalg.norm(lateral)))
    return max_dev


def compute_angular_deviation(path: list[np.ndarray], axis: np.ndarray) -> float:
    """Maximum angular deviation of the rotation axis from the expected axis.

    Measures the angle between each state's rotvec direction and the expected
    axis. States with near-zero rotation are skipped.
    """
    if not path:
        return 0.0
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    max_dev = 0.0
    for state in path:
        state = np.asarray(state, dtype=float)
        rotvec = state[3:]
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-9:
            continue
        rot_axis = rotvec / angle
        dot = abs(float(rot_axis @ axis))
        dev = float(np.arccos(np.clip(dot, 0.0, 1.0)))
        max_dev = max(max_dev, dev)
    return max_dev


def compute_path_length(path: list[np.ndarray]) -> float:
    """Total path length in state space (translation + rotation components)."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        prev = np.asarray(path[i - 1], dtype=float)
        curr = np.asarray(path[i], dtype=float)
        total += float(np.linalg.norm(curr[:3] - prev[:3])) + float(np.linalg.norm(curr[3:] - prev[3:]))
    return total


def validate_full_path(
    path: list[np.ndarray],
    move_mesh: trimesh.Trimesh,
    still_meshes: list[trimesh.Trimesh],
    axis: Optional[AxisEstimate] = None,
    lateral_tolerance: Optional[float] = None,
    collision_manager: Any = None,
    still_bounds: Optional[np.ndarray] = None,
    hull_move: Optional[trimesh.Trimesh] = None,
) -> ValidationResult:
    """Validate the full path: collision at intermediate states, final separation,
    jump detection, and optional lateral-deviation constraint.

    Returns a ValidationResult with valid=True only when the path passes all
    hard requirements (collision-free + separated).
    """
    from assets.transform import get_transform_matrix

    if not path:
        return ValidationResult(valid=False, rejection_reason="empty_path")

    # Build collider (reuse pattern from helical.py _PairCollider)
    collider = _PairCollider(move_mesh, still_meshes)
    manager = collision_manager
    hull = hull_move
    bounds = still_bounds

    if manager is None or hull is None or bounds is None:
        # Build them from scratch
        hull = trimesh.convex.convex_hull(np.asarray(move_mesh.vertices, dtype=float))
        still_vertices = np.vstack([np.asarray(m.vertices, dtype=float) for m in still_meshes])
        hull_still = trimesh.convex.convex_hull(still_vertices)
        manager = trimesh.collision.CollisionManager()
        manager.add_object("still", hull_still)
        bounds = hull_still.bounds.copy()

    # Compute lateral tolerance from geometry if not given
    if lateral_tolerance is None and axis is not None:
        lateral_tolerance = _compute_lateral_tolerance(move_mesh, still_meshes, axis)

    collision_states = 0
    max_jump_trans = 0.0
    max_jump_rot = 0.0

    for i, state in enumerate(path):
        state = np.asarray(state, dtype=float)
        # Collision check at every state
        if collider.collides(get_transform_matrix(state)):
            collision_states += 1
            if collision_states > max(1, len(path) // 10):
                # Too many collisions — reject early
                return ValidationResult(
                    valid=False,
                    collision_states=collision_states,
                    rejection_reason="collision_at_intermediate_states",
                )

        # Jump detection
        if i > 0:
            prev = np.asarray(path[i - 1], dtype=float)
            jump_trans = float(np.linalg.norm(state[:3] - prev[:3]))
            jump_rot = float(np.linalg.norm(state[3:] - prev[3:]))
            max_jump_trans = max(max_jump_trans, jump_trans)
            max_jump_rot = max(max_jump_rot, jump_rot)
            if jump_trans > DEFAULT_JUMP_TRANS_THRESHOLD or jump_rot > DEFAULT_JUMP_ROT_THRESHOLD:
                return ValidationResult(
                    valid=False,
                    max_jump_translation=max_jump_trans,
                    max_jump_rotation=max_jump_rot,
                    rejection_reason="abrupt_state_jump",
                )

    # Lateral deviation check
    max_lateral = 0.0
    if axis is not None:
        max_lateral = compute_lateral_deviation(path, axis.axis, axis.axis_point)
        if lateral_tolerance is not None and max_lateral > lateral_tolerance:
            return ValidationResult(
                valid=False,
                max_lateral_deviation=max_lateral,
                rejection_reason="excessive_lateral_deviation",
            )

    # Angular deviation check
    max_angular = 0.0
    if axis is not None:
        max_angular = compute_angular_deviation(path, axis.axis)

    # Final separation check
    final_state = np.asarray(path[-1], dtype=float)
    separated = _check_separation(collider, manager, hull, bounds, get_transform_matrix(final_state))
    if not separated:
        return ValidationResult(
            valid=False,
            final_separated=False,
            rejection_reason="final_state_not_separated",
        )

    path_length = compute_path_length(path)
    score = (
        WEIGHT_LATERAL * max_lateral
        + WEIGHT_LENGTH * path_length
        + WEIGHT_ANGULAR * max_angular
    )

    return ValidationResult(
        valid=True,
        max_lateral_deviation=max_lateral,
        max_angular_deviation=max_angular,
        path_length=path_length,
        final_separated=True,
        collision_states=collision_states,
        max_jump_translation=max_jump_trans,
        max_jump_rotation=max_jump_rot,
        score=score,
    )


def _check_separation(
    collider: Any,
    manager: Any,
    hull: trimesh.Trimesh,
    bounds: np.ndarray,
    matrix: np.ndarray,
) -> bool:
    """Check if the moving part is separated from still parts at the given pose."""
    if manager.in_collision_single(hull):
        return False
    hull_moved = hull.copy()
    hull_moved.apply_transform(matrix)
    move_min, move_max = hull_moved.bounds
    still_min, still_max = bounds
    move_contains_still = (move_min <= still_min).all() and (move_max >= still_max).all()
    still_contains_move = (still_min <= move_min).all() and (still_max >= move_max).all()
    return not (move_contains_still or still_contains_move)


def _compute_lateral_tolerance(
    move_mesh: trimesh.Trimesh, still_meshes: list[trimesh.Trimesh], axis: AxisEstimate
) -> float:
    """Compute lateral tolerance from local geometry and clearances.

    Uses the radial clearance between moving and still parts plus a fraction
    of the moving part extent — scales with assembly size.
    """
    move_vertices = np.asarray(move_mesh.vertices, dtype=float)
    move_radii = np.linalg.norm(
        move_vertices - np.outer(move_vertices @ axis.axis, axis.axis) - axis.axis_point, axis=1
    )
    move_extent = float(move_vertices.ptp(axis=0).max()) if len(move_vertices) > 0 else 1.0
    still_vertices = np.vstack([np.asarray(m.vertices, dtype=float) for m in still_meshes])
    still_radii = np.linalg.norm(
        still_vertices - np.outer(still_vertices @ axis.axis, axis.axis) - axis.axis_point, axis=1
    )
    if len(still_radii) > 0 and len(move_radii) > 0:
        radial_clearance = abs(float(still_radii.min()) - float(move_radii.min()))
    else:
        radial_clearance = 0.0
    return max(0.5 * radial_clearance, 0.02 * move_extent, 1e-6)


class _PairCollider:
    """Triangle-level collision between the moving part and still parts."""

    def __init__(self, move_mesh: trimesh.Trimesh, still_meshes: list[trimesh.Trimesh]):
        from trimesh.collision import CollisionManager

        self.move_manager = CollisionManager()
        self.move_manager.add_object("move", move_mesh)
        self.still_manager = CollisionManager()
        for index, mesh in enumerate(still_meshes):
            self.still_manager.add_object(f"still{index}", mesh)

    def collides(self, matrix: np.ndarray) -> bool:
        self.move_manager.set_transform("move", matrix)
        return self.still_manager.in_collision_other(self.move_manager)


def score_candidate_paths(
    candidates: list[tuple[list[np.ndarray], ValidationResult]],
) -> Optional[tuple[list[np.ndarray], ValidationResult]]:
    """Select the best valid candidate by score (lower = better).

    Only valid candidates (collision-free + separated) are considered.
    """
    valid = [(p, v) for p, v in candidates if v.valid]
    if not valid:
        return None
    return min(valid, key=lambda pv: pv[1].score)


class TwoPhasePlanner:
    """Two-phase extraction planner.

    Phase A (engaged): helical/axial motion with axis constraint and small
    lateral tolerance. Phase B (disengaged): free collision-aware planning.
    """

    def __init__(
        self,
        assembly_dir: Path,
        move_id: str,
        still_ids: list[str],
        max_time: float = 120.0,
        seed: int = 1,
        rotation: bool = False,
        body_type: str = "sdf",
        sdf_dx: float = 0.05,
        collision_threshold: float = 0.01,
        force_magnitude: float = 100.0,
        frame_skip: int = 100,
        helical_spec: Optional[dict[str, Any]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.assembly_dir = assembly_dir
        self.move_id = move_id
        self.still_ids = still_ids
        self.max_time = max_time
        self.seed = seed
        self.rotation = rotation
        self.body_type = body_type
        self.sdf_dx = sdf_dx
        self.collision_threshold = collision_threshold
        self.force_magnitude = force_magnitude
        self.frame_skip = frame_skip
        self.helical_spec = helical_spec
        self.cancel_check = cancel_check
        self.deadline = time.monotonic() + max_time
        self.axis_estimate: Optional[AxisEstimate] = None
        self.validation_result: Optional[ValidationResult] = None

    def _remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def _check_cancelled(self) -> bool:
        if self.cancel_check is not None and self.cancel_check():
            return True
        return False

    def _estimate_axis(self) -> Optional[AxisEstimate]:
        from api.services.threads import detect_thread

        parts = _load_parts(self.assembly_dir)
        if self.move_id not in parts:
            return None
        thread = detect_thread(parts[self.move_id])
        try:
            self.axis_estimate = estimate_assembly_axis(
                self.assembly_dir, self.move_id, self.still_ids, thread_info=thread
            )
        except Exception:
            logger.warning("AXIS_ESTIMATION_FAILED move=%s", self.move_id, exc_info=True)
            self.axis_estimate = None
        return self.axis_estimate

    def _validate_and_score(self, path: list[np.ndarray]) -> Optional[ValidationResult]:
        parts = _load_parts(self.assembly_dir)
        move_mesh = parts[self.move_id]
        still_meshes = [parts[sid] for sid in self.still_ids if sid in parts]
        result = validate_full_path(
            path, move_mesh, still_meshes, axis=self.axis_estimate
        )
        return result if result.valid else None

    def plan(self) -> tuple[str, float, Optional[list[np.ndarray]], dict[str, Any]]:
        """Run two-phase extraction. Returns (status, elapsed, path, info)."""
        started_at = time.perf_counter()
        self._estimate_axis()

        if self.axis_estimate is not None:
            logger.info(
                "AXIS_ESTIMATED move=%s source=%s axis=%s direction=%d confidence=%.2f",
                self.move_id, self.axis_estimate.source,
                np.round(self.axis_estimate.axis, 4).tolist(),
                self.axis_estimate.direction, self.axis_estimate.confidence,
            )

        # Phase A: helical (if threaded) or axial-constrained motion
        phase_a_status, phase_a_elapsed, phase_a_path, phase_a_info = self._plan_phase_a()
        if phase_a_status == "Success" and phase_a_path:
            validation = self._validate_and_score(phase_a_path)
            if validation is not None:
                logger.info(
                    "VALID_SEPARATION_PATH_FOUND move=%s phase=helical lateral=%.4f score=%.3f",
                    self.move_id, validation.max_lateral_deviation, validation.score,
                )
                return (
                    "Success",
                    time.perf_counter() - started_at,
                    phase_a_path,
                    {**phase_a_info, "validation": validation.__dict__},
                )
            logger.warning(
                "PATH_VALIDATION_FAILED move=%s phase=helical reason=%s",
                self.move_id, validation.rejection_reason if validation else "unknown",
            )

        # Phase B: free collision-aware planning
        if self._check_cancelled() or self._remaining() <= 0.5:
            return "Failure", time.perf_counter() - started_at, None, {"reason": "cancelled_or_timeout"}

        logger.info("GENERAL_MOTION_FALLBACK_STARTED move=%s", self.move_id)
        phase_b_status, phase_b_elapsed, phase_b_path, phase_b_info = self._plan_phase_b()
        if phase_b_status == "Success" and phase_b_path:
            validation = self._validate_and_score(phase_b_path)
            if validation is not None:
                logger.info(
                    "VALID_SEPARATION_PATH_FOUND move=%s phase=general lateral=%.4f score=%.3f",
                    self.move_id, validation.max_lateral_deviation, validation.score,
                )
                return (
                    "Success",
                    time.perf_counter() - started_at,
                    phase_b_path,
                    {**phase_b_info, "validation": validation.__dict__},
                )
            logger.warning(
                "PATH_VALIDATION_FAILED move=%s phase=general reason=%s",
                self.move_id, validation.rejection_reason if validation else "unknown",
            )

        logger.info(
            "PLANNER_TIMEOUT move=%s phase_a=%s phase_b=%s axis=%s",
            self.move_id, phase_a_status, phase_b_status,
            self.axis_estimate.source if self.axis_estimate else "none",
        )
        return "Timeout", time.perf_counter() - started_at, None, {}

    def _plan_phase_a(self) -> tuple[str, float, Optional[list[np.ndarray]], dict[str, Any]]:
        """Phase A: helical motion for threaded engagement, or axial-constrained
        combined motion for cylindrical connections."""
        budget = self._remaining() * 0.6
        if budget < 1.0:
            return "Failure", 0.0, None, {"reason": "insufficient_budget"}

        if self.helical_spec is not None or self.axis_estimate is not None and self.axis_estimate.source == "thread":
            # Try helical first
            from api.services.helical import build_helical_path

            status, elapsed, path, info = build_helical_path(
                self.assembly_dir, self.move_id, self.still_ids,
                max_time=budget, helical=self.helical_spec,
            )
            if status == "Success":
                return status, elapsed, path, {**info, "phase": "helical"}
            logger.warning(
                "HELICAL_PATH_FAILED move=%s status=%s reason=%s",
                self.move_id, status, info.get("reason"),
            )

        # Fallback: axial translation + rotation with axis constraint
        if self._check_cancelled() or self._remaining() <= 0.5:
            return "Failure", 0.0, None, {"reason": "cancelled_or_timeout"}

        logger.info("AXIAL_FALLBACK_STARTED move=%s", self.move_id)
        from examples.run_joint_plan import get_planner

        planner = get_planner("bfs")(
            str(settings.assets_dir),
            str(self.assembly_dir),
            self.move_id,
            self.still_ids,
            True,
            self.body_type,
            self.sdf_dx,
            self.collision_threshold,
            self.force_magnitude,
            self.frame_skip,
            False,
        )
        status, elapsed, path = planner.plan(
            self._remaining() * 0.4,
            seed=self.seed,
            return_path=True,
            render=False,
            record_path=None,
        )
        return status, elapsed, path, {"phase": "axial_rotation"}

    def _plan_phase_b(self) -> tuple[str, float, Optional[list[np.ndarray]], dict[str, Any]]:
        """Phase B: free collision-aware planning (no axis constraint)."""
        if self._check_cancelled() or self._remaining() <= 0.5:
            return "Failure", 0.0, None, {"reason": "cancelled_or_timeout"}

        from examples.run_joint_plan import get_planner

        # Try bk-rrt (more exploratory) then bfs
        for planner_name in ("bk-rrt", "bfs"):
            if self._check_cancelled() or self._remaining() <= 0.5:
                return "Failure", 0.0, None, {"reason": "cancelled_or_timeout"}
            planner = get_planner(planner_name)(
                str(settings.assets_dir),
                str(self.assembly_dir),
                self.move_id,
                self.still_ids,
                True,
                self.body_type,
                self.sdf_dx,
                self.collision_threshold,
                self.force_magnitude,
                self.frame_skip,
                False,
            )
            status, elapsed, path = planner.plan(
                self._remaining(),
                seed=self.seed,
                return_path=True,
                render=False,
                record_path=None,
            )
            if status == "Success":
                return status, elapsed, path, {"phase": "general", "planner": planner_name}
        return "Failure", 0.0, None, {"reason": "no_valid_path"}
