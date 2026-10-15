"""Adapters for the repository's existing physics and geometric planners."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from api.config import settings
from api.errors import ApiError
from api.schemas import JointPlanRequest, MultiPlanRequest
from api.services.assemblies import get_metadata, resolve_assembly
from api.services.native import require_native

logger = logging.getLogger(__name__)

_PHYSICS_JOINT_PLANNERS = {"bfs", "bk-rrt", "helical"}
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


def _serialize_path(
    path: Any, artifact_dir: Optional[Path], n_save_states: int, name: str = "0"
) -> dict[str, Any]:
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
        save_path(str(path_dir), path, n_frame=n_save_states, name=name)
        result["artifact_directory"] = "path"
    return result


def _mesh_diagnostics(request: JointPlanRequest, assembly_dir: Path) -> Optional[dict[str, Any]]:
    """Report the initial-state clearance that decides whether the planner can search deeply."""
    if not request.mesh_diagnostics:
        return None
    try:
        from api.services.meshes import DEFAULT_COLLISION_THRESHOLD, mesh_gap

        diagnostics = mesh_gap(assembly_dir, move_id=request.move_id, still_ids=request.still_ids)
    except Exception:
        logger.warning("mesh diagnostics failed for %s", assembly_dir.name, exc_info=True)
        return {"available": False}
    threshold_name = "collision_threshold" if request.engine == "physics" else "max_collision"
    threshold = request.collision_threshold if request.engine == "physics" else request.max_collision
    diagnostics[threshold_name] = threshold
    suggested = diagnostics.get("suggested_collision_threshold", DEFAULT_COLLISION_THRESHOLD)
    if diagnostics.get("touching") and threshold < suggested:
        diagnostics["warning"] = (
            f"parts touch at the initial state but {threshold_name}={threshold} is below {suggested}; "
            "raise it before planning"
        )
    return diagnostics


def run_joint_plan(request: JointPlanRequest, artifact_dir: Optional[str] = None) -> dict[str, Any]:
    """Run an original single-part planner; no planning algorithm is duplicated here."""
    t_job = time.perf_counter()
    require_native()
    metadata = get_metadata(request.assembly)
    move_id = request.move_id or metadata.part_ids[0]
    still_ids = (
        request.still_ids
        if request.still_ids is not None
        else [part_id for part_id in metadata.part_ids if part_id != move_id]
    )
    request = request.model_copy(update={"move_id": move_id, "still_ids": still_ids})
    _validate_part_selection(metadata.part_ids, move_id, still_ids)
    assembly_dir = resolve_assembly(request.assembly)
    output_dir = Path(artifact_dir) if request.save_artifacts and artifact_dir else None
    # Default collision representation per engine: physics contact must stay fast
    # on dense meshes (bvh forward() measures ~100x slower per step), while
    # geometric distance checks are exact and cheap with bvh and avoid grid builds.
    body_type = request.body_type or ("sdf" if request.engine == "physics" else "bvh")
    # The helical synthesizer measures the real thread, so it always sees the
    # original meshes; every other planner works on threads removed beforehand.
    simplify_threads = request.planner != "helical"
    skip_reason = "planner_requires_threads" if not simplify_threads else None

    from api.services.threads import planning_assembly

    with planning_assembly(assembly_dir, enabled=simplify_threads, skip_reason=skip_reason) as (
        planning_dir,
        thread_report,
    ):
        diagnostics = _mesh_diagnostics(request, planning_dir)

        logger.info(
            "NATIVE CALL joint-plan engine=%s planner=%s assembly=%s",
            request.engine,
            request.planner,
            assembly_dir.name,
        )
        helical_info: Optional[dict[str, Any]] = None
        helical_spec = request.helical.model_dump(mode="json") if request.helical is not None else None
        if helical_spec is not None and request.planner != "helical":
            raise ApiError(
                400, "INVALID_INPUT", "helical constraint requires planner='helical' (it is ignored by bfs/bk-rrt)"
            )
        if request.engine == "physics":
            if request.planner == "helical":
                from api.services.helical import build_helical_path, resolve_helical_spec

                if helical_spec is not None:
                    # Fail fast on a misconfigured constraint before any planning.
                    resolve_helical_spec(
                        helical_spec, metadata.part_ids,
                        move_id=request.move_id, still_ids=request.still_ids,
                    )
                status, elapsed_seconds, path, helical_info = build_helical_path(
                    planning_dir, request.move_id, request.still_ids, max_time=request.max_time,
                    helical=helical_spec,
                )
            else:
                if request.planner not in _PHYSICS_JOINT_PLANNERS:
                    raise ApiError(400, "INVALID_INPUT", "physics planner must be bfs or bk-rrt")
                from examples.run_joint_plan import get_planner

                planner = get_planner(request.planner)(
                    str(settings.assets_dir), str(planning_dir), request.move_id, request.still_ids,
                    request.rotation, body_type, request.sdf_dx, request.collision_threshold,
                    request.force_magnitude, request.frame_skip, False,
                )
                status, elapsed_seconds, path = planner.plan(
                    request.max_time, seed=request.seed, return_path=True, render=False, record_path=None,
                )
                helical_info = None
        elif request.planner == "helical":
            from api.services.helical import build_helical_path, resolve_helical_spec

            if helical_spec is not None:
                resolve_helical_spec(
                    helical_spec, metadata.part_ids,
                    move_id=request.move_id, still_ids=request.still_ids,
                )
            status, elapsed_seconds, path, helical_info = build_helical_path(
                planning_dir, request.move_id, request.still_ids, max_time=request.max_time,
                helical=helical_spec,
            )
        else:
            if request.planner not in _GEOMETRIC_JOINT_PLANNERS:
                raise ApiError(400, "INVALID_INPUT", "geometric planner name is not supported")
            from baselines.run_joint_plan import PyPlanner

            planner = PyPlanner(
                str(planning_dir), request.move_id, request.still_ids, request.rotation,
                request.max_collision, request.adaptive_collision, body_type, request.sdf_dx,
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
        "thread_removal": thread_report,
    }
    if diagnostics is not None:
        result["mesh_diagnostics"] = diagnostics
    if helical_info is not None:
        result["helical"] = helical_info
    result.update(_serialize_path(path, output_dir, request.n_save_states, name=request.move_id))
    logger.info(
        "JOB_PHASES joint-plan status=%s wall_seconds=%.3f search_seconds=%.3f",
        status, time.perf_counter() - t_job, float(elapsed_seconds),
    )
    return result


def run_multi_plan(request: MultiPlanRequest, artifact_dir: Optional[str] = None) -> dict[str, Any]:
    """Run an original sequence planner for a multi-part assembly."""
    require_native()
    metadata = get_metadata(request.assembly)
    assembly_dir = resolve_assembly(request.assembly)
    output_dir = str(Path(artifact_dir)) if request.save_artifacts and artifact_dir else None
    helical_spec = request.helical.model_dump(mode="json") if request.helical is not None else None
    if helical_spec is not None:
        if request.path_planner != "helical":
            raise ApiError(
                400, "INVALID_INPUT", "helical constraint requires path_planner='helical' (it is ignored by bfs/bk-rrt)"
            )
        from api.services.helical import resolve_helical_spec

        # Fail fast on unknown parts; per-move applicability is decided inside
        # HelicalPlanner so other moves keep auto estimation.
        resolve_helical_spec(helical_spec, metadata.part_ids)
    simplify_threads = request.path_planner != "helical"
    skip_reason = "planner_requires_threads" if not simplify_threads else None

    from api.services.threads import planning_assembly

    logger.info(
        "NATIVE CALL multi-plan engine=%s sequence=%s path=%s assembly=%s",
        request.engine,
        request.sequence_planner,
        request.path_planner,
        assembly_dir.name,
    )
    with planning_assembly(assembly_dir, enabled=simplify_threads, skip_reason=skip_reason) as (
        planning_dir,
        thread_report,
    ):
        if request.engine == "physics":
            if request.sequence_planner not in _PHYSICS_SEQUENCE_PLANNERS:
                raise ApiError(400, "INVALID_INPUT", "unsupported physics sequence planner")
            if request.path_planner not in _PHYSICS_JOINT_PLANNERS:
                raise ApiError(400, "INVALID_INPUT", "physics path planner must be bfs, bk-rrt, or helical")
            import examples.run_multi_plan as run_multi_plan_module
            from examples.run_multi_plan import get_seq_planner
            from assets.save import clear_saved_sdfs

            clear_saved_sdfs(str(planning_dir))
            original_get_path_planner = run_multi_plan_module.get_path_planner
            if request.path_planner == "helical":
                from api.services.helical import HelicalPlanner

                def _get_path_planner(name, _original=original_get_path_planner, _helical=helical_spec):
                    if name != "helical":
                        return _original(name)

                    def _with_helical(*args, **kwargs):
                        kwargs.setdefault("helical", _helical)
                        return HelicalPlanner(*args, **kwargs)

                    return _with_helical

                run_multi_plan_module.get_path_planner = _get_path_planner
            try:
                planner = get_seq_planner(request.sequence_planner)(str(settings.assets_dir), str(planning_dir))
                status, sequence, attempts, elapsed_seconds = planner.plan_sequence(
                    request.path_planner, request.rotation, request.body_type, request.sdf_dx,
                    request.collision_threshold, request.force_magnitude, request.frame_skip,
                    request.sequence_max_time, request.path_max_time, request.seed, False, None,
                    output_dir, request.n_save_states, verbose=False,
                )
            finally:
                run_multi_plan_module.get_path_planner = original_get_path_planner
                clear_saved_sdfs(str(planning_dir))
        else:
            if request.sequence_planner not in _GEOMETRIC_SEQUENCE_PLANNERS:
                raise ApiError(400, "INVALID_INPUT", "geometric sequence planner must be random or queue")
            if request.path_planner not in _GEOMETRIC_JOINT_PLANNERS:
                raise ApiError(400, "INVALID_INPUT", "unsupported geometric path planner")
            from baselines.run_multi_plan import get_seq_planner
            from assets.save import clear_saved_sdfs

            clear_saved_sdfs(str(planning_dir))
            try:
                planner = get_seq_planner(request.sequence_planner)(str(planning_dir))
                status, sequence, attempts, elapsed_seconds = planner.plan_sequence(
                    request.path_planner, request.rotation, request.body_type, request.sdf_dx,
                    request.max_collision, request.adaptive_collision, request.step_size,
                    request.sequence_max_time, request.path_max_time, request.seed, False,
                    output_dir, request.n_save_states, verbose=False,
                )
            finally:
                clear_saved_sdfs(str(planning_dir))

    return {
        "engine": request.engine,
        "sequence_planner": request.sequence_planner,
        "path_planner": request.path_planner,
        "assembly": metadata.model_dump(mode="json"),
        "status": status,
        "sequence": list(sequence),
        "attempts": int(attempts),
        "elapsed_seconds": float(elapsed_seconds),
        "thread_removal": thread_report,
    }
