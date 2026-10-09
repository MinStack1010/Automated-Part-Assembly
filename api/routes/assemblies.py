"""Assembly discovery and multipart upload routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from api.schemas import (
    AssemblyListResponse,
    AssemblyRef,
    AssemblyUploadResponse,
    PreprocessRequest,
    SuccessResponse,
)
from api.services import meshes as mesh_service
from api.services.assemblies import get_metadata, list_assemblies, save_upload

router = APIRouter(prefix="/api/v1", tags=["assemblies"])


@router.get("/assemblies", response_model=AssemblyListResponse, summary="List assemblies in an installed collection")
def assemblies(collection: str = Query(default="multi_assembly", description="Asset collection to enumerate.")) -> AssemblyListResponse:
    return AssemblyListResponse(result=list_assemblies(collection))


@router.get("/assemblies/{collection}/{assembly_id}", response_model=SuccessResponse, summary="Inspect one assembly")
def assembly(collection: str, assembly_id: str) -> SuccessResponse:
    metadata = get_metadata(AssemblyRef(collection=collection, assembly_id=assembly_id))
    return SuccessResponse(result=metadata.model_dump(mode="json"))


@router.post(
    "/assemblies/upload",
    response_model=AssemblyUploadResponse,
    summary="Upload a multipart OBJ assembly",
    description="Accepts two or more .obj files and optional translation.json. The returned upload_id can be used in any AssemblyRef.",
)
async def upload_assembly(files: list[UploadFile] = File(description="Assembly OBJ files and optional translation.json.")) -> AssemblyUploadResponse:
    return AssemblyUploadResponse(result=await save_upload(files))


@router.post(
    "/assemblies/preprocess",
    response_model=SuccessResponse,
    summary="Repair, normalize, and subdivide an assembly",
    description=(
        "Closes holes in non-watertight parts with pymeshfix, centers and scales the assembly to a 10-unit "
        "bounding box like assets/process_mesh.py, optionally subdivides, and optionally shrinks parts radially "
        "so mating threads stop interlocking. A straight-pull collision sweep reports whether the processed "
        "assembly can be pulled apart. The processed assembly is stored as a new upload; the source assembly "
        "is never modified."
    ),
)
def preprocess_assembly(request: PreprocessRequest) -> SuccessResponse:
    return SuccessResponse(
        result=mesh_service.preprocess_assembly(
            request.assembly,
            repair=request.repair,
            normalize=request.normalize,
            subdivide=request.subdivide,
            max_edge=request.max_edge,
            shrink_parts=request.shrink_parts,
            verify_part_id=request.verify_part_id,
        )
    )
