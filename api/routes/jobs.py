"""Long-running planner job routes and result-artifact delivery."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import Response

from api.errors import ApiError
from api.schemas import AssemblyRef, JobAccepted, JobResponse, JobResultResponse, JointPlanRequest, MultiPlanRequest
from api.services.artifacts import build_artifact_zip
from api.services.assemblies import resolve_assembly
from api.services.jobs import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("/joint-plan", response_model=JobAccepted, status_code=202, summary="Queue a single-part disassembly plan")
def create_joint_plan_job(request: JointPlanRequest) -> JobAccepted:
    job = job_manager.submit("joint-plan", request.model_dump(mode="json"))
    return JobAccepted(result={"job_id": job.job_id, "status": job.status})


@router.post("/multi-plan", response_model=JobAccepted, status_code=202, summary="Queue a multi-part sequence plan")
def create_multi_plan_job(request: MultiPlanRequest) -> JobAccepted:
    job = job_manager.submit("multi-plan", request.model_dump(mode="json"))
    return JobAccepted(result={"job_id": job.job_id, "status": job.status})


@router.get("/{job_id}", response_model=JobResponse, summary="Get job state")
def get_job(job_id: str) -> JobResponse:
    return JobResponse(result=job_manager.snapshot(job_id))


@router.get("/{job_id}/result", response_model=JobResultResponse, summary="Get a completed job result")
def get_job_result(job_id: str) -> JobResultResponse:
    return JobResultResponse(result=job_manager.result(job_id))


@router.get(
    "/{job_id}/artifacts.zip",
    summary="Download every job artifact as one zip",
    description=(
        "Returns artifact.zip containing artifact/npy (path.json frames converted to "
        "matrices), artifact/json (the original files), and artifact/gif (rendered from "
        "path.json on first download, cached afterwards). Empty folders are included "
        "when a group has no files."
    ),
    responses={200: {"content": {"application/zip": {}}}},
    response_class=Response,
)
def download_artifacts(job_id: str) -> Response:
    job = job_manager.get(job_id)
    assembly_dir = None
    payload_assembly = job.payload.get("assembly")
    if payload_assembly:
        try:
            assembly_dir = resolve_assembly(AssemblyRef.model_validate(payload_assembly))
        except ApiError:
            logger.warning("artifacts.zip job=%s assembly unavailable; gif skipped", job_id)
    payload = build_artifact_zip(job.artifact_dir, assembly_dir)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"content-disposition": 'attachment; filename="artifact.zip"'},
    )


@router.delete("/{job_id}", response_model=JobResponse, summary="Cancel a queued or running job")
def cancel_job(job_id: str) -> JobResponse:
    return JobResponse(result=job_manager.cancel(job_id))
