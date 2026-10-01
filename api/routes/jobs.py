"""Long-running planner job routes and result-artifact delivery."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from api.schemas import JobAccepted, JobResponse, JobResultResponse, JointPlanRequest, MultiPlanRequest
from api.services.jobs import job_manager

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


@router.get("/{job_id}/artifacts/{artifact_path:path}", summary="Download a saved job artifact")
def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
    artifact = job_manager.resolve_artifact(job_id, artifact_path)
    return FileResponse(artifact, filename=artifact.name)


@router.delete("/{job_id}", response_model=JobResponse, summary="Cancel a queued or running job")
def cancel_job(job_id: str) -> JobResponse:
    return JobResponse(result=job_manager.cancel(job_id))
