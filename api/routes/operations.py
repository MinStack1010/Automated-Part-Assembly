"""Fast synchronous operations backed by the existing native extension."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import MeshDistanceRequest, NativeDemoRequest, SuccessResponse
from api.services.native import compute_mesh_distance, run_native_demo

router = APIRouter(prefix="/api/v1", tags=["operations"])


@router.post(
    "/mesh-distance",
    response_model=SuccessResponse,
    summary="Compute minimum mesh distance",
    description="Builds the repository's BVHMesh or SDFMesh backend and calls assets.mesh_distance.compute_all_mesh_distance.",
)
def mesh_distance(request: MeshDistanceRequest) -> SuccessResponse:
    return SuccessResponse(result=compute_mesh_distance(request))


@router.post(
    "/simulations/demo",
    response_model=SuccessResponse,
    summary="Run an exposed native simulation demo",
    description="Calls redmax_py.make_sim, Simulation.reset, and Simulation.forward directly.",
)
def native_demo(request: NativeDemoRequest) -> SuccessResponse:
    return SuccessResponse(result=run_native_demo(request))
