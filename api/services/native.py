"""Small, safe adapters around the pybind11 ``redmax_py`` extension."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional, Tuple

import numpy as np

from api.errors import ApiError, NativeBackendUnavailable
from api.schemas import MeshDistanceRequest, NativeDemoRequest
from api.services.assemblies import get_metadata, resolve_assembly

logger = logging.getLogger(__name__)


def native_status() -> Tuple[bool, Optional[str]]:
    try:
        importlib.import_module("redmax_py")
        return True, None
    except Exception as exc:  # ImportError is common; broken shared libraries also arrive here.
        return False, str(exc)


def require_native() -> Any:
    try:
        return importlib.import_module("redmax_py")
    except Exception as exc:
        raise NativeBackendUnavailable(
            "redmax_py could not be loaded. Build it with `pip install ./simulation` and check native runtime libraries."
        ) from exc


def run_native_demo(request: NativeDemoRequest) -> dict[str, Any]:
    """Run one of the native demos explicitly exposed by ``make_sim``."""
    native = require_native()
    logger.info("NATIVE CALL demo=%s steps=%s", request.environment, request.num_steps)
    simulation = native.make_sim(request.environment, request.integrator)
    if simulation is None:
        raise ApiError(400, "INVALID_INPUT", "unsupported native demo environment")
    simulation.reset()
    simulation.forward(request.num_steps)
    return {
        "environment": request.environment,
        "integrator": request.integrator,
        "num_steps": request.num_steps,
        "ndof": int(simulation.ndof_r),
        "q": np.asarray(simulation.q).tolist(),
        "qdot": np.asarray(simulation.qdot).tolist(),
        "converged": bool(simulation.is_converged()),
    }


def compute_mesh_distance(request: MeshDistanceRequest) -> dict[str, Any]:
    """Call the repository's mesh-distance helper and native mesh implementation."""
    from assets.load import load_assembly
    from assets.mesh_distance import compute_all_mesh_distance

    native = require_native()
    metadata = get_metadata(request.assembly)
    if len(request.states) != metadata.part_count:
        raise ApiError(
            400,
            "INVALID_INPUT",
            f"states must contain exactly one state for each part ({metadata.part_count})",
        )

    assembly_dir = resolve_assembly(request.assembly)
    logger.info("NATIVE CALL mesh-distance assembly=%s body_type=%s", assembly_dir.name, request.body_type)
    meshes = load_assembly(str(assembly_dir))
    native_meshes = []
    for mesh in meshes:
        if request.body_type == "bvh":
            native_meshes.append(native.BVHMesh(mesh.vertices.T, mesh.faces.T))
        else:
            native_meshes.append(native.SDFMesh(mesh.vertices.T, mesh.faces.T, request.sdf_dx))
    distance = compute_all_mesh_distance(native_meshes, [np.asarray(state) for state in request.states])
    return {
        "assembly": metadata.model_dump(mode="json"),
        "body_type": request.body_type,
        "minimum_distance": float(distance),
    }
