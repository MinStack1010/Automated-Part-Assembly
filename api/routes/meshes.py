"""Pure-Python clearance diagnostics between assembly parts."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import MeshGapRequest, SuccessResponse
from api.services import meshes as mesh_service
from api.services.assemblies import resolve_assembly

router = APIRouter(prefix="/api/v1", tags=["meshes"])


@router.post(
    "/mesh-gap",
    response_model=SuccessResponse,
    summary="Measure clearance between assembly parts",
    description=(
        "Measures the initial-state gap between parts with trimesh proximity queries, detects overlap "
        "through python-fcl, and returns a collision threshold that avoids BFS time-outs on touching meshes."
    ),
)
def mesh_gap(request: MeshGapRequest) -> SuccessResponse:
    assembly_dir = resolve_assembly(request.assembly)
    return SuccessResponse(
        result=mesh_service.mesh_gap(assembly_dir, move_id=request.move_id, still_ids=request.still_ids)
    )
