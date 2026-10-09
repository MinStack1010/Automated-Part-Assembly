"""Mesh repair, preprocessing, and clearance diagnostics for planner inputs."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh

from api.config import Settings, settings
from api.errors import ApiError
from api.schemas import AssemblyMetadata, AssemblyRef
from api.services.assemblies import get_metadata, resolve_assembly

logger = logging.getLogger(__name__)

BBOX_SIZE = 10.0
MAX_ALL_PAIRS_PARTS = 10
TOUCHING_GAP = 0.05
DEFAULT_COLLISION_THRESHOLD = 0.01
TOUCHING_COLLISION_THRESHOLD = 0.1

_GAP_ADVICE_REPAIR = "Run POST /api/v1/assemblies/preprocess on this assembly, then plan again."


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


def repair_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool, Optional[str]]:
    """Close holes with pymeshfix; the input mesh is returned when repair fails."""
    try:
        import pymeshfix
    except ImportError as exc:
        raise ApiError(
            503,
            "MISSING_DEPENDENCY",
            "pymeshfix is not installed; add pymeshfix to requirements.txt and rebuild the image",
        ) from exc
    try:
        vertices, faces = pymeshfix.clean_from_arrays(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
        )
    except Exception as exc:
        logger.warning("pymeshfix failed: %s", exc)
        return mesh, False, str(exc)
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(repaired.vertices) == 0 or len(repaired.faces) == 0:
        return mesh, False, "pymeshfix produced an empty mesh"
    return repaired, True, None


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
        advice = f"Parts interpenetrate at the initial state; use collision_threshold={suggested}. {_GAP_ADVICE_REPAIR}"
    elif touching:
        advice = f"Parts touch (gap {gap} <= {TOUCHING_GAP}); use collision_threshold={suggested}. {_GAP_ADVICE_REPAIR}"
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


def _optional_gap_report(parts: dict[str, trimesh.Trimesh]) -> Optional[dict[str, Any]]:
    if len(parts) > MAX_ALL_PAIRS_PARTS:
        return None
    try:
        return _gap_report(
            parts,
            [(first, second) for index, first in enumerate(parts) for second in list(parts)[index + 1 :]],
            "all_pairs",
        )
    except Exception:
        logger.warning("gap measurement failed", exc_info=True)
        return None


def _normalize(meshes: list[trimesh.Trimesh], bbox_size: float = BBOX_SIZE) -> np.ndarray:
    """Center and scale the assembly to a bbox_size bounding box, as assets/process_mesh.py does."""
    stacked = np.vstack([mesh.vertices for mesh in meshes])
    lower, upper = stacked.min(axis=0), stacked.max(axis=0)
    center = (lower + upper) / 2
    scale_factor = bbox_size / float(np.max(upper - lower))
    transform = np.eye(4)
    transform[:3, :3] *= scale_factor
    transform[:3, 3] = -scale_factor * center
    for mesh in meshes:
        mesh.apply_transform(transform)
    return transform


def _shrink_radial(mesh: trimesh.Trimesh, scale: float) -> None:
    """Scale a part about its own x/y axis so mating threads stop interlocking."""
    center = mesh.bounds.mean(axis=0)
    transform = np.eye(4)
    transform[0, 0] = scale
    transform[1, 1] = scale
    transform[:3, 3] = center - transform[:3, :3].dot(center)
    mesh.apply_transform(transform)


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


def _load_subdivide():
    assets_path = str(settings.project_root / "assets")
    if assets_path not in sys.path:
        sys.path.insert(0, assets_path)
    from assets.subdivide import subdivide_to_size

    return subdivide_to_size


def _write_assembly(
    parts: dict[str, trimesh.Trimesh],
    source_dir: Path,
    transform: Optional[np.ndarray],
    config: Settings,
) -> tuple[AssemblyRef, AssemblyMetadata]:
    upload_id = str(uuid.uuid4())
    target_dir = config.uploads_dir / upload_id
    try:
        target_dir.mkdir(parents=True, exist_ok=False)
        for part_id, mesh in sorted(parts.items()):
            mesh.export(str(target_dir / f"{part_id}.obj"), file_type="obj", header=None, include_color=False)
        translation_source = source_dir / "translation.json"
        if translation_source.is_file():
            coms = json.loads(translation_source.read_text())
            if transform is not None:
                coms = {
                    part_id: trimesh.transform_points(np.asarray(value, dtype=float).reshape(1, 3), transform)[0].tolist()
                    for part_id, value in coms.items()
                }
            (target_dir / "translation.json").write_text(json.dumps(coms, indent=2))
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    output_ref = AssemblyRef(upload_id=upload_id)
    return output_ref, get_metadata(output_ref, config)


def _validate_shrink(
    parts: dict[str, trimesh.Trimesh],
    shrink_parts: Optional[dict[str, float]],
    verify_part_id: Optional[str],
) -> None:
    for part_id, scale in (shrink_parts or {}).items():
        if part_id not in parts:
            raise ApiError(400, "INVALID_INPUT", f"shrink_parts names an unknown part: {part_id}")
        if not 0.0 < float(scale) <= 1.0:
            raise ApiError(400, "INVALID_INPUT", f"shrink_parts[{part_id}] must be greater than 0 and at most 1")
    if verify_part_id is not None and verify_part_id not in parts:
        raise ApiError(400, "INVALID_INPUT", f"verify_part_id is not a part in the assembly: {verify_part_id}")


def preprocess_assembly(
    ref: AssemblyRef,
    *,
    repair: bool = True,
    normalize: bool = True,
    subdivide: bool = False,
    max_edge: float = 0.5,
    shrink_parts: Optional[dict[str, float]] = None,
    verify_part_id: Optional[str] = None,
    config: Settings = settings,
) -> dict[str, Any]:
    """Repair, normalize, and optionally subdivide an assembly into a new upload."""
    get_metadata(ref, config)
    assembly_dir = resolve_assembly(ref, config)
    parts = _load_parts(assembly_dir)
    _validate_shrink(parts, shrink_parts, verify_part_id)
    gap_before = _optional_gap_report(parts)

    report: list[dict[str, Any]] = []
    for part_id in sorted(parts):
        mesh = parts[part_id]
        entry: dict[str, Any] = {"part_id": part_id, "watertight_before": bool(mesh.is_watertight), "repaired": False}
        if repair and not entry["watertight_before"]:
            mesh, repaired, error = repair_mesh(mesh)
            entry["repaired"] = bool(repaired)
            if error is not None:
                entry["repair_error"] = error
        entry["watertight_after"] = bool(mesh.is_watertight)
        parts[part_id] = mesh
        report.append(entry)

    transform = _normalize(list(parts.values())) if normalize else None

    for part_id, scale in sorted((shrink_parts or {}).items()):
        _shrink_radial(parts[part_id], float(scale))
        for entry in report:
            if entry["part_id"] == part_id:
                entry["radial_scale"] = float(scale)

    verify_ids = sorted({*(shrink_parts or {}), *({verify_part_id} if verify_part_id else set())})
    straight_pull = {part_id: _straight_pull(parts, part_id) for part_id in verify_ids}

    if subdivide:
        for entry in report:
            if not entry["watertight_after"]:
                raise ApiError(
                    400,
                    "INVALID_ASSEMBLY",
                    f"part {entry['part_id']} is not watertight; subdivide requires a repaired mesh",
                )
        subdivide_to_size = _load_subdivide()
        for part_id in sorted(parts):
            parts[part_id] = subdivide_to_size(parts[part_id], max_edge=max_edge)

    output_ref, output_metadata = _write_assembly(parts, assembly_dir, transform, config)

    for entry in report:
        mesh = parts[entry["part_id"]]
        entry["vertices"] = int(len(mesh.vertices))
        entry["faces"] = int(len(mesh.faces))

    all_watertight = all(entry["watertight_after"] for entry in report)
    gap_after = _optional_gap_report(parts)
    result: dict[str, Any] = {
        "source": ref.model_dump(mode="json"),
        "output": output_metadata.model_dump(mode="json"),
        "actions": {
            "repair": repair,
            "normalize": normalize,
            "subdivide": subdivide,
            "max_edge": max_edge,
            "shrink_parts": shrink_parts or None,
        },
        "part_count": len(parts),
        "parts": report,
        "all_watertight": all_watertight,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "suggested_collision_threshold": (gap_after or {}).get(
            "suggested_collision_threshold", DEFAULT_COLLISION_THRESHOLD
        ),
    }
    if straight_pull:
        result["straight_pull"] = straight_pull
    if not all_watertight:
        result["warning"] = "some parts are still not watertight; planning may time out on this assembly"
    return result
