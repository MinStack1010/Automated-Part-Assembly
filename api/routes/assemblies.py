"""Assembly discovery and multipart upload routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from api.schemas import AssemblyListResponse, AssemblyRef, AssemblyUploadResponse, SuccessResponse
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
