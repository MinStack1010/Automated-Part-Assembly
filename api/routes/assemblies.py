"""Multipart upload routes."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from api.schemas import AssemblyUploadResponse
from api.services.assemblies import save_upload

router = APIRouter(prefix="/api/v1", tags=["assemblies"])


@router.post(
    "/assemblies/upload",
    response_model=AssemblyUploadResponse,
    summary="Upload a multipart OBJ assembly",
    description="Accepts two or more .obj files and optional translation.json. The returned upload_id can be used in any AssemblyRef.",
)
async def upload_assembly(files: list[UploadFile] = File(description="Assembly OBJ files and optional translation.json.")) -> AssemblyUploadResponse:
    return AssemblyUploadResponse(result=await save_upload(files))
