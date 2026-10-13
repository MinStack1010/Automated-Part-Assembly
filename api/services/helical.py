"""Helical screw-out path synthesis: estimate the thread pitch, sweep rotation
plus axial travel together, and verify every step against the real meshes."""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from api.errors import ApiError

logger = logging.getLogger(__name__)

N_THETA_BINS = 72
N_Z_BINS = 48
STEPS_PER_TURN = 16
PITCH_MIN_FRACTION = 0.02
PITCH_MAX_FRACTION = 0.5
ROW_RADIUS_FRACTION = 0.7
ROW_VARIANCE_FRACTION = 0.4
ALPHA_TOLERANCE_BINS = 2.0
MIN_INLIER_FRACTION = 0.5
MIN_INLIER_PAIRS = 4
AXIS_CONSENSUS_DOT = 0.9
MAX_MISMATCHES = 8
MAX_AXIS_CANDIDATES = 4
MAX_STEPS = 6000
_2PI = 2.0 * np.pi


def _load_parts(assembly_dir: Path) -> dict[str, trimesh.Trimesh]:
    from assets.load import load_assembly

    meshes, names = load_assembly(str(assembly_dir), return_names=True)
    return {name.replace(".obj", ""): mesh for mesh, name in zip(meshes, names)}


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _principal_axes(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    values, vectors = np.linalg.eigh(np.cov(vertices.T))
    order = values.argsort()[::-1]
    return [vectors[:, index] for index in order]


def _axis_profiles(
    mesh: trimesh.Trimesh, axis: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bucket vertex radii into (z band x angle) cells; cells without vertices
    take the row median so partially covered rows keep a usable profile."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = vertices.mean(axis=0)
    offset = vertices - center
    along = offset @ axis
    first, second = _orthonormal_basis(axis)
    radius = np.linalg.norm(offset - np.outer(along, axis), axis=1)
    angle = np.arctan2(offset @ first, offset @ second)

    edges = np.linspace(along.min(), along.max(), N_Z_BINS + 1)
    z_index = np.clip(np.digitize(along, edges) - 1, 0, N_Z_BINS - 1)
    theta_index = np.clip(((angle + np.pi) / _2PI * N_THETA_BINS).astype(int), 0, N_THETA_BINS - 1)

    start = -np.inf if kind == "max" else np.inf
    profile = np.full((N_Z_BINS, N_THETA_BINS), start, dtype=float)
    reducer = np.maximum.at if kind == "max" else np.minimum.at
    reducer(profile, (z_index, theta_index), radius)

    filled = np.isfinite(profile)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_median = np.where(filled.any(axis=1), np.nanmedian(np.where(filled, profile, np.nan), axis=1), np.nan)
    profile = np.where(filled, profile, row_median[:, None])
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    row_filled = np.isfinite(row_median)
    z_centers = (edges[:-1] + edges[1:]) / 2.0
    extent = float(along.max() - along.min())
    return z_centers, profile, row_filled, extent


def _estimate_for_axis(
    mesh: trimesh.Trimesh, axis: np.ndarray, kind: str
) -> Optional[dict[str, Any]]:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    z_centers, profile, row_filled, extent = _axis_profiles(mesh, axis, kind)
    if extent <= 0:
        return None

    row_max = profile.max(axis=1)
    row_min = profile.min(axis=1)
    filled_rows = row_max[row_filled]
    if len(filled_rows) == 0:
        return None
    # Thresholds follow the median row so a large constant feature (a screw head,
    # a hex prism) cannot drown out the smaller periodic thread rows.
    radius_floor = ROW_RADIUS_FRACTION * float(np.median(filled_rows))
    spread_floor = ROW_VARIANCE_FRACTION * float(np.median(filled_rows - row_min[row_filled]))
    active = row_filled & (row_max >= radius_floor) & ((row_max - row_min) >= spread_floor)
    active_rows = np.flatnonzero(active)
    if len(active_rows) < 4:
        return None

    pitch_min = PITCH_MIN_FRACTION * extent
    pitch_max = PITCH_MAX_FRACTION * extent

    pairs: list[tuple[int, int]] = list(zip(active_rows[:-1], active_rows[1:]))
    measured: list[tuple[float, float, float]] = []  # (dz, alpha, score)
    tolerance = ALPHA_TOLERANCE_BINS * _2PI / N_THETA_BINS
    for first_row, second_row in pairs:
        delta_z = float(z_centers[second_row] - z_centers[first_row])
        if delta_z <= 0:
            continue
        left = profile[first_row] - profile[first_row].mean()
        right = profile[second_row] - profile[second_row].mean()
        if left.std() < 1e-12 or right.std() < 1e-12:
            continue
        correlation = np.array([float(np.dot(left, np.roll(right, -shift))) for shift in range(N_THETA_BINS)])
        shift = int(np.argmax(correlation))
        alpha = _2PI * shift / N_THETA_BINS
        if alpha > np.pi:
            alpha -= _2PI
        score = float(correlation[shift] / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
        measured.append((delta_z, alpha, score))
    if len(measured) < MIN_INLIER_PAIRS:
        return None

    hypotheses: dict[float, tuple[float, float]] = {}
    for delta_z, alpha, _ in measured:
        for mismatch in range(-MAX_MISMATCHES, MAX_MISMATCHES + 1):
            alpha_true = alpha + _2PI * mismatch
            if abs(alpha_true) < 1e-9:
                continue
            pitch = _2PI * delta_z / abs(alpha_true)
            if pitch_min <= pitch <= pitch_max:
                hypotheses.setdefault(round(pitch, 6), (pitch, 0.0))

    best_pitch: Optional[float] = None
    best_inliers: list[tuple[float, float, float]] = []
    for pitch in hypotheses:
        inliers = [
            entry
            for entry in measured
            if abs(np.angle(np.exp(1j * (_2PI * entry[0] / pitch - entry[1])))) <= tolerance
        ]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_pitch = pitch
    if best_pitch is None or len(best_inliers) < MIN_INLIER_PAIRS:
        return None
    confidence = len(best_inliers) / len(measured)
    if confidence < MIN_INLIER_FRACTION:
        return None

    alphas = np.array([entry[1] for entry in best_inliers])
    if float(np.median(np.abs(alphas))) <= 1.5 * _2PI / N_THETA_BINS:
        # constant profiles (prisms, cylinders) correlate at zero shift for every
        # row pair, which fits many pitches equally well; threads do not.
        return None

    scores = [entry[2] for entry in best_inliers]
    handedness = int(np.sign(np.median(alphas))) if np.any(alphas != 0) else 0
    return {
        "pitch": float(best_pitch),
        "axis": axis.tolist(),
        "axis_point": vertices_center(mesh),
        "confidence": round(confidence, 4),
        "inliers": len(best_inliers),
        "pairs": len(measured),
        "profile": kind,
        "handedness": handedness,
        "mean_correlation": round(float(np.mean(scores)), 4),
    }


def vertices_center(mesh: trimesh.Trimesh) -> list[float]:
    return np.asarray(mesh.vertices, dtype=float).mean(axis=0).tolist()


def estimate_thread_candidates(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    """Return every viable thread hypothesis (axis x profile), best first."""
    results: list[dict[str, Any]] = []
    for axis in _principal_axes(mesh):
        for kind in ("max", "min"):
            try:
                result = _estimate_for_axis(mesh, axis, kind)
            except Exception:
                logger.warning("thread estimation failed for profile=%s", kind, exc_info=True)
                continue
            if result is not None:
                results.append(result)
    results.sort(key=lambda entry: (entry["inliers"], entry["confidence"]), reverse=True)
    return results


def estimate_thread(mesh: trimesh.Trimesh) -> Optional[dict[str, Any]]:
    """Best thread hypothesis for a standalone mesh, or None when unthreaded."""
    candidates = estimate_thread_candidates(mesh)
    return candidates[0] if candidates else None


def _axis_alignment(first: np.ndarray, second: np.ndarray) -> float:
    return abs(float(np.dot(first, second)))


def _estimation_key(
    candidate: dict[str, Any], other_candidates: list[dict[str, Any]]
) -> tuple[int, float, int, float]:
    axis = np.asarray(candidate["axis"], dtype=float)
    aligned = [
        other
        for other in other_candidates
        if _axis_alignment(axis, np.asarray(other["axis"], dtype=float)) >= AXIS_CONSENSUS_DOT
    ]
    if not aligned:
        return (0, 0.0, candidate["inliers"], candidate["confidence"])
    pitch_error = min(
        abs(candidate["pitch"] - other["pitch"]) / max(candidate["pitch"], other["pitch"])
        for other in aligned
    )
    return (1, -pitch_error, candidate["inliers"], candidate["confidence"])


def rank_estimations(
    moving_candidates: list[dict[str, Any]], other_candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Sort moving-part hypotheses: an axis shared with another part (and a
    matching pitch) ranks above a stronger standalone fit along a wrong axis,
    such as slicing a hex nut across its flats."""
    return sorted(
        moving_candidates,
        key=lambda candidate: _estimation_key(candidate, other_candidates),
        reverse=True,
    )


def _pitch_variants(
    candidate: dict[str, Any], other_candidates: list[dict[str, Any]]
) -> list[tuple[float, str]]:
    """The candidate's own pitch first, then distinct pitches measured on parts
    that share its axis; one of them is usually the mating pitch even when this
    part's own estimate is aliased."""
    variants = [(float(candidate["pitch"]), "own")]
    axis = np.asarray(candidate["axis"], dtype=float)
    for other in other_candidates:
        if _axis_alignment(axis, np.asarray(other["axis"], dtype=float)) < AXIS_CONSENSUS_DOT:
            continue
        pitch = float(other["pitch"])
        if all(abs(pitch - known) / max(pitch, known) > 0.2 for known, _ in variants):
            variants.append((pitch, "aligned_part"))
    return variants


def _disassembled(hull_moved: trimesh.Trimesh, manager: Any, still_bounds: np.ndarray) -> bool:
    if manager.in_collision_single(hull_moved):
        return False
    move_min, move_max = hull_moved.bounds
    still_min, still_max = still_bounds
    move_contains_still = (move_min <= still_min).all() and (move_max >= still_max).all()
    still_contains_move = (still_min <= move_min).all() and (still_max >= move_max).all()
    return not (move_contains_still or still_contains_move)


class _PairCollider:
    """Triangle-level collision between the moving part and the still parts.

    Built once per plan: both BVHs stay resident and only the moving pose is
    updated per step. fcl is the same oracle ``_straight_pull`` uses, so a path
    verified here agrees with the preprocess report."""

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


def _sweep(
    *,
    center: np.ndarray,
    axis: np.ndarray,
    pitch: float,
    rotation_sign: int,
    out_sign: int,
    collider: _PairCollider,
    hull_move: trimesh.Trimesh,
    collision_manager: Any,
    still_bounds: np.ndarray,
    max_steps: int,
    started_at: float,
    max_time: Optional[float],
) -> tuple[str, Optional[list[np.ndarray]], dict[str, Any]]:
    from assets.transform import get_transform_matrix

    theta_step = _2PI / STEPS_PER_TURN
    theta = 0.0
    states: list[np.ndarray] = []
    for step in range(1, max_steps + 1):
        if max_time is not None and time.time() - started_at > max_time:
            return "Timeout", None, {"steps": step - 1}
        theta += theta_step
        rotation = (rotation_sign * theta) * axis
        matrix = Rotation.from_rotvec(rotation).as_matrix()
        translation = center - matrix @ center + (out_sign * axis) * (pitch * theta / _2PI)
        state = np.concatenate([translation, rotation])
        if collider.collides(get_transform_matrix(state)):
            return "collided", None, {"failed_step": step, "theta": round(theta, 4)}
        hull = hull_move.copy()
        hull.apply_transform(get_transform_matrix(state))
        states.append(state)
        if _disassembled(hull, collision_manager, still_bounds):
            return "Success", states, {"theta_final": round(theta, 4), "steps": step}
    return "exhausted", None, {"steps": max_steps}


def build_helical_path(
    assembly_dir: Path,
    move_id: str,
    still_ids: list[str],
    max_time: Optional[float] = None,
) -> tuple[str, Optional[float], Optional[list[np.ndarray]], dict[str, Any]]:
    """Synthesize a verified screw-out path; returns (status, elapsed, path, info).

    Every returned state is collision-free according to redmax BVH distance
    against the real part meshes, and the path ends as soon as the convex
    hulls of the moving and still parts separate."""
    started_at = time.time()
    parts = _load_parts(assembly_dir)
    if move_id not in parts:
        raise ApiError(400, "INVALID_INPUT", "move_id is not a part in the assembly")
    still_meshes = []
    for still_id in still_ids:
        if still_id not in parts:
            raise ApiError(400, "INVALID_INPUT", "still_ids must name other parts in the assembly")
        still_meshes.append(parts[still_id])

    info: dict[str, Any] = {"move_id": move_id, "still_ids": list(still_ids)}
    move_mesh = parts[move_id]

    moving_candidates = estimate_thread_candidates(move_mesh)
    other_candidates = [
        candidate
        for part_id, mesh in parts.items()
        if part_id != move_id
        for candidate in estimate_thread_candidates(mesh)
    ]
    if not moving_candidates:
        info.update(
            reason="thread_pitch_not_detected",
            advice=(
                "the moving part does not look threaded; use planner=bfs, or if it is "
                "threaded but the profiles are too noisy, preprocess the assembly first"
            ),
        )
        return "Failure", time.time() - started_at, None, info
    ranked_candidates = rank_estimations(moving_candidates, other_candidates)
    info["candidates"] = [
        {
            "axis": [round(float(value), 6) for value in candidate["axis"]],
            "pitch": round(float(candidate["pitch"]), 6),
            "profile": candidate["profile"],
            "inliers": candidate["inliers"],
            "pairs": candidate["pairs"],
            "confidence": candidate["confidence"],
        }
        for candidate in ranked_candidates
    ]

    try:
        from trimesh.collision import CollisionManager
    except ImportError as exc:
        raise ApiError(
            503,
            "MISSING_DEPENDENCY",
            "python-fcl is not installed; add python-fcl to requirements.txt and rebuild the image",
        ) from exc

    collider = _PairCollider(move_mesh, still_meshes)
    if collider.collides(np.eye(4)):
        info.update(
            reason="initial_state_collides",
            advice=(
                "the parts already collide at the initial state; run "
                "POST /api/v1/assemblies/preprocess with shrink_parts for the moving part"
            ),
        )
        return "Failure", time.time() - started_at, None, info

    hull_move = trimesh.convex.convex_hull(move_mesh.vertices)
    hull_still = trimesh.convex.convex_hull(np.vstack([mesh.vertices for mesh in still_meshes]))
    collision_manager = CollisionManager()
    collision_manager.add_object("still", hull_still)
    still_bounds = hull_still.bounds.copy()

    if _disassembled(hull_move.copy(), collision_manager, still_bounds):
        info.update(reason="start_with_goal")
        return "Start with goal", time.time() - started_at, None, info

    attempts: list[dict[str, Any]] = []
    theta_step = _2PI / STEPS_PER_TURN
    for axis_index, candidate in enumerate(ranked_candidates[:MAX_AXIS_CANDIDATES]):
        axis = np.asarray(candidate["axis"], dtype=float)
        center = np.asarray(candidate["axis_point"], dtype=float)
        along_move = move_mesh.vertices @ axis
        along_still = hull_still.vertices @ axis
        overlap = min(float(along_move.max()), float(along_still.max())) - max(
            float(along_move.min()), float(along_still.min())
        )
        for pitch, pitch_source in _pitch_variants(candidate, other_candidates):
            travel = max(overlap, 0.0) + 3.0 * pitch
            max_steps = int(
                min(MAX_STEPS, np.ceil((_2PI * travel / pitch) / theta_step) + 2)
            )
            for rotation_sign in (1, -1):
                for out_sign in (1, -1):
                    status, path, detail = _sweep(
                        center=center,
                        axis=axis,
                        pitch=pitch,
                        rotation_sign=rotation_sign,
                        out_sign=out_sign,
                        collider=collider,
                        hull_move=hull_move,
                        collision_manager=collision_manager,
                        still_bounds=still_bounds,
                        max_steps=max_steps,
                        started_at=started_at,
                        max_time=max_time,
                    )
                    attempt: dict[str, Any] = {
                        "axis_index": axis_index,
                        "pitch": round(pitch, 6),
                        "pitch_source": pitch_source,
                        "rotation_sign": rotation_sign,
                        "out_sign": out_sign,
                        "status": status,
                        **detail,
                    }
                    attempts.append(attempt)
                    if status == "Success":
                        assert path is not None
                        info.update(
                            pitch=round(pitch, 6),
                            pitch_source=pitch_source,
                            axis=[round(float(value), 6) for value in axis],
                            axis_point=[round(float(value), 6) for value in center],
                            axis_index=axis_index,
                            profile=candidate["profile"],
                            confidence=candidate["confidence"],
                            inliers=candidate["inliers"],
                            pairs=candidate["pairs"],
                            handedness=candidate["handedness"],
                            rotation_sign=rotation_sign,
                            out_sign=out_sign,
                            turns=round(detail["theta_final"] / _2PI, 3),
                            steps=detail["steps"],
                            attempts=attempts,
                        )
                        logger.info(
                            "helical path move=%s pitch=%.4f axis_index=%d steps=%d attempts=%d",
                            move_id,
                            pitch,
                            axis_index,
                            detail["steps"],
                            len(attempts),
                        )
                        return "Success", time.time() - started_at, path, info
                    if status == "Timeout":
                        info.update(attempts=attempts, reason="timeout")
                        return "Timeout", time.time() - started_at, None, info

    info.update(
        attempts=attempts,
        reason="no_collision_free_combo",
        advice=(
            "no rotation/translation combination keeps the parts apart; the meshes "
            "likely interpenetrate in the threaded region - run POST /api/v1/assemblies/preprocess "
            "with shrink_parts on the moving part and plan again"
        ),
    )
    return "Failure", time.time() - started_at, None, info


class HelicalPlanner:
    """Adapter exposing the examples planner interface so multi-plan can use the
    helical synthesizer as a path planner. Falls back to physics BFS when the
    moving part is not threaded."""

    def __init__(
        self,
        asset_folder: str,
        assembly_dir: str,
        move_id: str,
        still_ids: list[str],
        rotation: bool = False,
        body_type: str = "bvh",
        sdf_dx: float = 0.05,
        collision_th: float = 0.01,
        force_mag: float = 1e3,
        frame_skip: int = 100,
        save_sdf: bool = False,
    ):
        self.asset_folder = asset_folder
        self.assembly_dir = Path(assembly_dir)
        self.move_id = move_id
        self.still_ids = list(still_ids)
        self.rotation = rotation
        self.body_type = body_type
        self.sdf_dx = sdf_dx
        self.collision_th = collision_th
        self.force_mag = force_mag
        self.frame_skip = frame_skip
        self.save_sdf = save_sdf
        self.info: dict[str, Any] = {}

    def plan(self, max_time, seed=1, return_path=False, render=False, record_path=None):
        status, elapsed, path, info = build_helical_path(
            self.assembly_dir, self.move_id, self.still_ids, max_time=max_time
        )
        if status == "Failure" and info.get("reason") == "thread_pitch_not_detected":
            from examples.run_joint_plan import get_planner

            fallback = get_planner("bfs")(
                self.asset_folder,
                str(self.assembly_dir),
                self.move_id,
                self.still_ids,
                self.rotation,
                self.body_type,
                self.sdf_dx,
                self.collision_th,
                self.force_mag,
                self.frame_skip,
                self.save_sdf,
            )
            status, elapsed, path = fallback.plan(
                max_time, seed=seed, return_path=True, render=False, record_path=None
            )
            info = {**info, "fallback": "bfs"}
        self.info = info
        logger.info("helical plan move=%s status=%s info=%s", self.move_id, status, info)
        return (status, elapsed, path) if return_path else (status, elapsed)

    def save_path(self, path, save_dir, n_save_state):
        from assets.save import save_path

        save_path(save_dir, path, n_frame=n_save_state, name=self.move_id)

    def get_contact_bodies(self, part_id):
        return []
