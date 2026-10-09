"""Health, service metadata, and capability discovery endpoints."""

from __future__ import annotations

import platform

from fastapi import APIRouter

from api.schemas import CapabilitiesResponse, Capability, HealthResponse, InfoResponse
from api.services.native import native_status

router = APIRouter(tags=["service"])

_CAPABILITIES = [
    Capability(name="assembly.inspect", endpoint="/api/v1/assemblies", execution="synchronous", native_backend_required=False, description="List installed OBJ assemblies and inspect their part IDs."),
    Capability(name="assembly.upload", endpoint="/api/v1/assemblies/upload", execution="synchronous", native_backend_required=False, description="Upload an OBJ assembly for later planning."),
    Capability(name="assembly.preprocess", endpoint="/api/v1/assemblies/preprocess", execution="synchronous", native_backend_required=False, description="Repair non-watertight parts, normalize, subdivide, and radially shrink parts into a new upload, with a straight-pull feasibility check."),
    Capability(name="mesh.gap", endpoint="/api/v1/mesh-gap", execution="synchronous", native_backend_required=False, description="Measure initial-state part clearance and suggest a collision threshold."),
    Capability(name="mesh.distance", endpoint="/api/v1/mesh-distance", execution="synchronous", native_backend_required=True, description="Calculate minimum assembly-part distance through BVHMesh or SDFMesh."),
    Capability(name="simulation.demo", endpoint="/api/v1/simulations/demo", execution="synchronous", native_backend_required=True, description="Run an exposed redmax_py built-in simulation."),
    Capability(name="joint.plan", endpoint="/api/v1/jobs/joint-plan", execution="job", native_backend_required=True, description="Run the existing physics or geometric single-part disassembly planner."),
    Capability(name="multi.plan", endpoint="/api/v1/jobs/multi-plan", execution="job", native_backend_required=True, description="Run the existing physics or geometric multi-part sequence planner."),
    Capability(name="job.artifacts.zip", endpoint="/api/v1/jobs/{job_id}/artifacts.zip", execution="synchronous", native_backend_required=False, description="Download all job artifacts as one zip: artifact/npy (path.json converted), artifact/json, artifact/gif (rendered on first download)."),
]


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    available, _ = native_status()
    return HealthResponse(status="ok" if available else "degraded", native_backend="ready" if available else "unavailable")


@router.get("/info", response_model=InfoResponse, summary="Service information")
def info() -> InfoResponse:
    available, _ = native_status()
    return InfoResponse(
        version="0.1.0",
        python_version=platform.python_version(),
        native_backend="ready" if available else "unavailable",
        status="ready" if available else "degraded",
    )


@router.get("/capabilities", response_model=CapabilitiesResponse, summary="Discover real API operations")
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(operations=_CAPABILITIES)
