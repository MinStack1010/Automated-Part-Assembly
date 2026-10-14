"""Long-running planner job routes and result-artifact delivery."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import Response

from api.errors import ApiError
from api.schemas import (
    AnimationResponse,
    AssemblyRef,
    JobAccepted,
    JobResponse,
    JointPlanRequest,
    MultiPlanRequest,
)
from api.services.artifacts import build_artifact_zip, resolve_artifact
from api.services.assemblies import resolve_assembly
from api.services.jobs import job_manager

_ARTIFACT_MEDIA_TYPES = {
    ".json": "application/json",
    ".npy": "application/octet-stream",
    ".gif": "image/gif",
}

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


@router.get(
    "/{job_id}/result",
    response_model=JobResponse,
    summary="Get a completed job result",
    description=(
        "Returns the job snapshot as JSON: job_id, operation, status, timestamps, "
        "error, and artifacts — the relative paths of every file the job wrote, "
        "including the .npy motion arrays once the job completed (joint-plan: "
        "path/path.json + npy/path.npy; multi-plan: <assembly>/<seq>_<move>/path.json "
        "+ npy/<assembly>_<seq>_<move>_path.npy). "
        "For playback use GET /api/v1/jobs/{job_id}/animation (per-step frame counts "
        "and working file URLs) or download bytes via "
        "GET /api/v1/jobs/{job_id}/artifacts/{path} and the full "
        "GET /api/v1/jobs/{job_id}/artifacts.zip. "
        "Errors keep the usual JSON error body: 404 for unknown jobs, 409 while "
        "the job is queued/running (and with the job's own error code when it "
        "failed)."
    ),
)
def get_job_result(job_id: str) -> JobResponse:
    # Existing conventions stay in place: 404 for unknown jobs, 409 unless the
    # job completed (a failed job answers with its own PLANNER_* error code).
    job_manager.result(job_id)
    return JobResponse(result=job_manager.snapshot(job_id))


@router.get(
    "/{job_id}/animation",
    response_model=AnimationResponse,
    summary="Get a playable animation manifest for a completed job",
    description=(
        "Returns one step per path.json the planners wrote (joint-plan: a single "
        "'path' step; multi-plan: one step per '<assembly>/<seq>_<move>' with "
        "sequence_index and move_id). Each step carries frame_count plus working "
        "json_url/npy_url/gif_url downloads, and obj_files maps every part id to "
        "its source OBJ download, so a frontend can fetch meshes once and drive "
        "each part's 4x4 matrix track. Same 404/409 conventions as .../result."
    ),
)
def get_job_animation(job_id: str) -> AnimationResponse:
    return AnimationResponse(result=job_manager.animation(job_id))


@router.get(
    "/{job_id}/artifacts/{artifact_path:path}",
    summary="Download a single job artifact file",
    description=(
        "Returns the raw bytes of one artifact listed by .../result (path.json, "
        ".npy motion array, or rendered .gif). Paths cannot escape the job folder."
    ),
    responses={200: {"content": {"application/octet-stream": {}}}},
    response_class=Response,
)
def download_artifact(job_id: str, artifact_path: str) -> Response:
    job = job_manager.get(job_id)
    try:
        target = resolve_artifact(job.artifact_dir, artifact_path)
    except ValueError as exc:
        raise ApiError(400, "INVALID_INPUT", str(exc)) from exc
    if not target.is_file():
        raise ApiError(404, "ARTIFACT_NOT_FOUND", "The requested artifact does not exist")
    media_type = _ARTIFACT_MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return Response(
        content=target.read_bytes(),
        media_type=media_type,
        headers={"content-disposition": f'attachment; filename="{target.name}"'},
    )


@router.get(
    "/{job_id}/assembly/{filename}",
    summary="Download one source mesh of the job's assembly",
    description=(
        "Returns the assembled part OBJ (or translation.json) that the planner "
        "loaded, so animation clients need no separate mesh source."
    ),
    responses={200: {"content": {"application/octet-stream": {}}}},
    response_class=Response,
)
def download_assembly_file(job_id: str, filename: str) -> Response:
    job = job_manager.get(job_id)
    payload_assembly = job.payload.get("assembly")
    if not payload_assembly:
        raise ApiError(404, "ASSEMBLY_NOT_FOUND", "This job has no assembly reference")
    assembly_dir = resolve_assembly(AssemblyRef.model_validate(payload_assembly))
    if "/" in filename or filename not in {f"{stem}.obj" for stem in _assembly_part_stems(assembly_dir)} | {"translation.json"}:
        raise ApiError(404, "ASSEMBLY_NOT_FOUND", "The requested assembly file does not exist")
    target = (assembly_dir / filename).resolve()
    if not target.is_file():
        raise ApiError(404, "ASSEMBLY_NOT_FOUND", "The requested assembly file does not exist")
    media_type = "application/json" if target.suffix.lower() == ".json" else "application/octet-stream"
    return Response(
        content=target.read_bytes(),
        media_type=media_type,
        headers={"content-disposition": f'attachment; filename="{target.name}"'},
    )


def _assembly_part_stems(assembly_dir) -> set[str]:
    from assets.load import load_part_ids

    try:
        return set(load_part_ids(str(assembly_dir)))
    except OSError:
        return set()


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
