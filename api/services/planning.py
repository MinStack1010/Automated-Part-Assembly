"""Adapters for the repository's existing physics and geometric planners."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from api.config import settings
from api.errors import ApiError
from api.schemas import JointPlanRequest, MultiPlanRequest
from api.services.assemblies import get_metadata, resolve_assembly
from api.services.native import require_native

logger = logging.getLogger(__name__)

_PHYSICS_JOINT_PLANNERS = {"bfs", "bk-rrt"}
_GEOMETRIC_JOINT_PLANNERS = {"rrt", "rrt-connect", "birrt", "trrt", "matevec-trrt"}
_PHYSICS_SEQUENCE_PLANNERS = {"random", "queue", "prog-queue"}
_GEOMETRIC_SEQUENCE_PLANNERS = {"random", "queue"}


def _validate_part_selection(metadata_part_ids: list[str], move_id: str, still_ids: list[str]) -> None:
    part_ids = set(metadata_part_ids)
    if move_id not in part_ids:
        raise ApiError(400, "INVALID_INPUT", "move_id is not a part in the assembly")
    if len(set(still_ids)) != len(still_ids):
        raise ApiError(400, "INVALID_INPUT", "still_ids must not contain duplicates")
    if move_id in still_ids or not set(still_ids).issubset(part_ids):
        raise ApiError(400, "INVALID_INPUT", "still_ids must name other parts in the assembly")


def _serialize_path(path: Any, artifact_dir: Optional[Path], n_save_states: int) -> dict[str, Any]:
    if path is None:
        return {"path": None, "path_state_count": 0}
    state_count = len(path)
    result: dict[str, Any] = {"path_state_count": state_count}
    if state_count <= settings.max_result_states:
        result["path"] = [np.asarray(state, dtype=float).tolist() for state in path]
    else:
        result["path"] = None
        result["path_omitted"] = True
        result["path_omission_reason"] = "The path is available as an artifact when save_artifacts=true."
    if artifact_dir is not None:
        from assets.save import save_path

        path_dir = artifact_dir / "path"
        save_path(str(path_dir), path, n_frame=n_save_states)
        result["artifact_directory"] = "path"
    return result


def _consolidate_motion_artifacts(output_dir: Path) -> dict[str, Any]:
    """Merge the per-attempt motion folders written by the sequence planners into ``path/``.

    The multi-plan engine saves ``{assembly_id}/{attempt}_{move_id}/{frame}.npy`` for every
    successful attempt, which mixes the moving part of each step with the attempt numbering.
    Clients animate a single continuous trajectory, so the frames are renumbered into one
    ``path/{frame}.npy`` sequence ordered by attempt, and the per-attempt folders are dropped
    from the artifact listing.
    """
    path_dir = output_dir / "path"
    steps: list[tuple[int, str, Path]] = []
    if output_dir.is_dir():
        for step in output_dir.rglob("*"):
            if not step.is_dir() or step == path_dir or path_dir in step.parents:
                continue
            attempt_text, separator, move_id = step.name.partition("_")
            if not separator or not attempt_text.isdigit() or not move_id:
                continue
            if any(frame.is_file() and frame.stem.isdigit() for frame in step.glob("*.npy")):
                steps.append((int(attempt_text), move_id, step))
    if not steps:
        return {}
    steps.sort(key=lambda step: step[0])

    path_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    path_parts: list[dict[str, Any]] = []
    step_parents: set[Path] = set()
    for attempt, move_id, step in steps:
        frames = sorted(
            (frame for frame in step.glob("*.npy") if frame.stem.isdigit()),
            key=lambda frame: int(frame.stem),
        )
        frame_start = frame_index
        for frame in frames:
            shutil.copyfile(frame, path_dir / f"{frame_index}.npy")
            frame_index += 1
        path_parts.append(
            {
                "part_id": move_id,
                "attempt": attempt,
                "frame_start": frame_start,
                "frame_end": frame_index - 1,
            }
        )
        step_parents.add(step.parent)

    for parent in step_parents:
        if parent == output_dir:
            for _, _, step in steps:
                if step.parent == output_dir:
                    shutil.rmtree(step, ignore_errors=True)
        else:
            shutil.rmtree(parent, ignore_errors=True)

    return {
        "artifact_directory": "path",
        "path_state_count": frame_index,
        "path_parts": path_parts,
    }


def run_joint_plan(request: JointPlanRequest, artifact_dir: Optional[str] = None) -> dict[str, Any]:
    """Run an original single-part planner; no planning algorithm is duplicated here."""
    require_native()
    metadata = get_metadata(request.assembly)
    _validate_part_selection(metadata.part_ids, request.move_id, request.still_ids)
    assembly_dir = resolve_assembly(request.assembly)
    output_dir = Path(artifact_dir) if request.save_artifacts and artifact_dir else None

    logger.info(
        "NATIVE CALL joint-plan engine=%s planner=%s assembly=%s",
        request.engine,
        request.planner,
        assembly_dir.name,
    )
    if request.engine == "physics":
        if request.planner not in _PHYSICS_JOINT_PLANNERS:
            raise ApiError(400, "INVALID_INPUT", "physics planner must be bfs or bk-rrt")
        from examples.run_joint_plan import get_planner

        planner = get_planner(request.planner)(
            str(settings.assets_dir), str(assembly_dir), request.move_id, request.still_ids,
            request.rotation, request.body_type, request.sdf_dx, request.collision_threshold,
            request.force_magnitude, request.frame_skip, False,
        )
        status, elapsed_seconds, path = planner.plan(
            request.max_time, seed=request.seed, return_path=True, render=False, record_path=None
        )
    else:
        if request.planner not in _GEOMETRIC_JOINT_PLANNERS:
            raise ApiError(400, "INVALID_INPUT", "geometric planner name is not supported")
        from baselines.run_joint_plan import PyPlanner

        planner = PyPlanner(
            str(assembly_dir), request.move_id, request.still_ids, request.rotation,
            request.max_collision, request.adaptive_collision, request.body_type, request.sdf_dx,
        )
        status, elapsed_seconds, path = planner.plan(
            request.planner, request.step_size, request.max_time, seed=request.seed,
            return_path=True, simplify=request.simplify, render=False,
        )

    result = {
        "engine": request.engine,
        "planner": request.planner,
        "assembly": metadata.model_dump(mode="json"),
        "move_id": request.move_id,
        "still_ids": request.still_ids,
        "status": status,
        "elapsed_seconds": float(elapsed_seconds),
    }
    result.update(_serialize_path(path, output_dir, request.n_save_states))
    return result


def run_multi_plan(request: MultiPlanRequest, artifact_dir: Optional[str] = None) -> dict[str, Any]:
    """Run an original sequence planner for a multi-part assembly."""
    require_native()
    metadata = get_metadata(request.assembly)
    assembly_dir = resolve_assembly(request.assembly)
    output_dir = str(Path(artifact_dir)) if request.save_artifacts and artifact_dir else None

    logger.info(
        "NATIVE CALL multi-plan engine=%s sequence=%s path=%s assembly=%s",
        request.engine,
        request.sequence_planner,
        request.path_planner,
        assembly_dir.name,
    )
    if request.engine == "physics":
        if request.sequence_planner not in _PHYSICS_SEQUENCE_PLANNERS:
            raise ApiError(400, "INVALID_INPUT", "unsupported physics sequence planner")
        if request.path_planner not in _PHYSICS_JOINT_PLANNERS:
            raise ApiError(400, "INVALID_INPUT", "physics path planner must be bfs or bk-rrt")
        from examples.run_multi_plan import get_seq_planner
        from assets.save import clear_saved_sdfs

        clear_saved_sdfs(str(assembly_dir))
        try:
            planner = get_seq_planner(request.sequence_planner)(str(settings.assets_dir), str(assembly_dir))
            status, sequence, attempts, elapsed_seconds = planner.plan_sequence(
                request.path_planner, request.rotation, request.body_type, request.sdf_dx,
                request.collision_threshold, request.force_magnitude, request.frame_skip,
                request.sequence_max_time, request.path_max_time, request.seed, False, None,
                output_dir, request.n_save_states, verbose=False,
            )
        finally:
            clear_saved_sdfs(str(assembly_dir))
    else:
        if request.sequence_planner not in _GEOMETRIC_SEQUENCE_PLANNERS:
            raise ApiError(400, "INVALID_INPUT", "geometric sequence planner must be random or queue")
        if request.path_planner not in _GEOMETRIC_JOINT_PLANNERS:
            raise ApiError(400, "INVALID_INPUT", "unsupported geometric path planner")
        from baselines.run_multi_plan import get_seq_planner
        from assets.save import clear_saved_sdfs

        clear_saved_sdfs(str(assembly_dir))
        try:
            planner = get_seq_planner(request.sequence_planner)(str(assembly_dir))
            status, sequence, attempts, elapsed_seconds = planner.plan_sequence(
                request.path_planner, request.rotation, request.body_type, request.sdf_dx,
                request.max_collision, request.adaptive_collision, request.step_size,
                request.sequence_max_time, request.path_max_time, request.seed, False,
                output_dir, request.n_save_states, verbose=False,
            )
        finally:
            clear_saved_sdfs(str(assembly_dir))

    motion: dict[str, Any] = {}
    if output_dir is not None:
        motion = _consolidate_motion_artifacts(Path(output_dir))
        if motion:
            logger.info(
                "MULTI-PLAN ARTIFACTS dir=%s frames=%s parts=%s",
                motion["artifact_directory"],
                motion["path_state_count"],
                [(part["part_id"], part["frame_start"], part["frame_end"]) for part in motion["path_parts"]],
            )

    return {
        "engine": request.engine,
        "sequence_planner": request.sequence_planner,
        "path_planner": request.path_planner,
        "assembly": metadata.model_dump(mode="json"),
        "status": status,
        "sequence": list(sequence),
        "attempts": int(attempts),
        "elapsed_seconds": float(elapsed_seconds),
        **motion,
    }
