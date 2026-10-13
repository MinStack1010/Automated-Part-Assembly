"""Clearance diagnostics for planner inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh

from api.errors import ApiError

logger = logging.getLogger(__name__)

MAX_ALL_PAIRS_PARTS = 10
TOUCHING_GAP = 0.05
DEFAULT_COLLISION_THRESHOLD = 0.01
TOUCHING_COLLISION_THRESHOLD = 0.1


def _load_parts(assembly_dir: Path) -> dict[str, trimesh.Trimesh]:
    parts: dict[str, trimesh.Trimesh] = {}
    for path in sorted(assembly_dir.glob("*.obj")):
        if path.stem == "assembly":
            continue
        mesh = trimesh.load_mesh(str(path), process=False, maintain_order=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        parts[path.stem] = mesh
    if len(parts) < 2:
        raise ApiError(400, "INVALID_ASSEMBLY", "an assembly must contain at least two OBJ part files")
    return parts


def _overlap_pairs(parts: dict[str, trimesh.Trimesh]) -> tuple[set[frozenset[str]], bool]:
    try:
        from trimesh.collision import CollisionManager
    except ImportError:
        logger.warning("python-fcl is unavailable; overlap detection is disabled")
        return set(), False
    manager = CollisionManager()
    for part_id, mesh in parts.items():
        manager.add_object(part_id, mesh)
    _, names = manager.in_collision_internal(return_names=True)
    return {frozenset(name) for name in names}, True


def _gap_report(
    parts: dict[str, trimesh.Trimesh], pairs: list[tuple[str, str]], mode: str
) -> dict[str, Any]:
    from trimesh.proximity import ProximityQuery

    queries = {part_id: ProximityQuery(mesh) for part_id, mesh in parts.items()}
    overlap_pairs, overlap_available = _overlap_pairs(parts)

    results = []
    for first, second in pairs:
        first_to_second = float(np.min(queries[first].on_surface(parts[second].vertices)[1]))
        second_to_first = float(np.min(queries[second].on_surface(parts[first].vertices)[1]))
        overlap = frozenset((first, second)) in overlap_pairs
        results.append(
            {
                "a": first,
                "b": second,
                "gap": round(min(first_to_second, second_to_first), 6),
                "overlap": overlap,
            }
        )

    gap = min(result["gap"] for result in results)
    overlapping = any(result["overlap"] for result in results)
    touching = overlapping or gap <= TOUCHING_GAP
    suggested = TOUCHING_COLLISION_THRESHOLD if touching else DEFAULT_COLLISION_THRESHOLD
    if overlapping:
        advice = f"Parts interpenetrate at the initial state; use collision_threshold={suggested}."
    elif touching:
        advice = f"Parts touch (gap {gap} <= {TOUCHING_GAP}); use collision_threshold={suggested}."
    else:
        advice = f"Parts are separated by {gap}; collision_threshold={DEFAULT_COLLISION_THRESHOLD} is appropriate."
    return {
        "mode": mode,
        "pairs": results,
        "gap": gap,
        "overlapping": overlapping,
        "overlap_detection": overlap_available,
        "touching": touching,
        "suggested_collision_threshold": suggested,
        "advice": advice,
    }


def _select_pairs(
    parts: dict[str, trimesh.Trimesh], move_id: Optional[str], still_ids: Optional[list[str]]
) -> list[tuple[str, str]]:
    part_ids = list(parts)
    if move_id is None:
        if len(part_ids) > MAX_ALL_PAIRS_PARTS:
            raise ApiError(
                400,
                "INVALID_INPUT",
                f"comparing every pair supports at most {MAX_ALL_PAIRS_PARTS} parts; supply move_id instead",
            )
        return [(first, second) for index, first in enumerate(part_ids) for second in part_ids[index + 1 :]]
    if move_id not in parts:
        raise ApiError(400, "INVALID_INPUT", "move_id is not a part in the assembly")
    targets = list(still_ids) if still_ids else [part_id for part_id in part_ids if part_id != move_id]
    if len(set(targets)) != len(targets) or move_id in targets or not set(targets).issubset(parts):
        raise ApiError(400, "INVALID_INPUT", "still_ids must name other parts in the assembly")
    return [(move_id, target) for target in targets]


def mesh_gap(
    assembly_dir: Path,
    move_id: Optional[str] = None,
    still_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Measure initial-state clearance between parts and suggest a collision threshold."""
    parts = _load_parts(assembly_dir)
    mode = "all_pairs" if move_id is None else "move"
    report = _gap_report(parts, _select_pairs(parts, move_id, still_ids), mode)
    report["part_ids"] = list(parts)
    if move_id is not None:
        report["move_id"] = move_id
    return report


def _straight_pull(
    parts: dict[str, trimesh.Trimesh], move_id: str, steps: int = 240
) -> dict[str, Any]:
    """Sweep a part along the six axis directions and report the first collision-free exit."""
    try:
        from trimesh.collision import CollisionManager
    except ImportError:
        return {"available": False, "free": False}
    manager = CollisionManager()
    for part_id, mesh in parts.items():
        if part_id != move_id:
            manager.add_object(part_id, mesh)
    moving = parts[move_id]
    span = 2.0 * float(np.linalg.norm(moving.extents))
    step = span / steps
    for direction in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)):
        for index in range(1, steps + 1):
            transform = np.eye(4)
            transform[:3, 3] = np.asarray(direction) * (index * step)
            if manager.in_collision_single(moving, transform=transform):
                break
        else:
            return {"available": True, "free": True, "direction": list(direction), "distance": round(span, 4)}
    return {"available": True, "free": False, "direction": None, "distance": None}
