"""Request and response contracts exposed in OpenAPI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ErrorDetail(BaseModel):
    code: str = Field(examples=["INVALID_INPUT"])
    message: str = Field(examples=["The requested assembly does not exist."])


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail


class SuccessResponse(BaseModel):
    success: Literal[True] = True
    result: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    native_backend: Literal["ready", "unavailable"]


class InfoResponse(BaseModel):
    name: str = "assemble-them-all-api"
    version: str
    python_version: str
    native_backend: Literal["ready", "unavailable"]
    status: Literal["ready", "degraded"]


class Capability(BaseModel):
    name: str
    endpoint: str
    execution: Literal["synchronous", "job"]
    native_backend_required: bool
    description: str


class CapabilitiesResponse(BaseModel):
    operations: list[Capability]


class AssemblyRef(BaseModel):
    """An installed assembly or one previously uploaded through this API."""

    collection: Optional[str] = Field(
        default=None,
        description="Directory below ASSEMBLY_ASSETS_DIR, for example multi_assembly.",
        examples=["multi_assembly"],
    )
    assembly_id: Optional[str] = Field(default=None, examples=["00003"])
    upload_id: Optional[str] = Field(
        default=None,
        description="ID returned by POST /api/v1/assemblies/upload.",
    )

    @model_validator(mode="after")
    def select_exactly_one_source(self) -> "AssemblyRef":
        installed = self.collection is not None or self.assembly_id is not None
        if installed and (self.collection is None or self.assembly_id is None):
            raise ValueError("collection and assembly_id must be supplied together")
        if installed == (self.upload_id is not None):
            raise ValueError("supply either collection + assembly_id or upload_id")
        return self


class AssemblyMetadata(BaseModel):
    source: AssemblyRef
    part_ids: list[str]
    part_count: int
    has_translation: bool


class AssemblyUploadResponse(BaseModel):
    success: Literal[True] = True
    result: AssemblyMetadata


class MeshGapRequest(BaseModel):
    assembly: AssemblyRef
    move_id: Optional[str] = Field(
        default=None,
        description="Measure clearance only against this part. Omit to compare every pair (up to 10 parts).",
        examples=["0"],
    )
    still_ids: Optional[list[str]] = Field(
        default=None,
        description="Restrict move_id comparisons to these parts; defaults to every other part.",
        examples=[["1"]],
    )


class JointPlanRequest(BaseModel):
    assembly: AssemblyRef
    engine: Literal["physics", "geometric"] = Field(
        default="physics",
        description="physics uses examples/run_joint_plan.py; geometric uses baselines/run_joint_plan.py.",
    )
    planner: str = Field(
        default="bfs",
        description=(
            "physics: bfs, bk-rrt, or helical (synthesizes a verified screw-out path: "
            "thread pitch is estimated from the mesh, then rotation and axial travel "
            "are verified against the real meshes; helical ignores rotation/body_type). "
            "geometric: rrt, rrt-connect, birrt, trrt, or matevec-trrt."
        ),
    )
    move_id: str = Field(default="0")
    still_ids: list[str] = Field(default_factory=lambda: ["1"], min_length=1)
    rotation: bool = False
    body_type: Optional[Literal["bvh", "sdf"]] = Field(
        default=None,
        description=(
            "Collision representation. Omit to use the default: sdf for engine=physics "
            "(contact stays fast on dense meshes and grids are cached across jobs; bvh "
            "contact measures ~100x slower per simulation step on thread-sized meshes), "
            "bvh for engine=geometric (exact triangle distance with no grid build)."
        ),
    )
    sdf_dx: float = Field(default=0.05, gt=0)
    max_time: float = Field(default=120, gt=0, le=86400)
    seed: int = 1
    save_artifacts: bool = Field(default=False, description="Store path.json frames ({name, 4x4 matrix}) as downloadable job artifacts.")
    n_save_states: int = Field(default=100, ge=1, le=100000)
    mesh_diagnostics: bool = Field(
        default=True,
        description="Measure the initial-state part gap and report a suggested collision threshold in the result.",
    )
    # Physics planner options
    collision_threshold: float = Field(default=0.01, ge=0)
    force_magnitude: float = Field(default=100, gt=0)
    frame_skip: int = Field(default=100, ge=1)
    # Geometric planner options
    max_collision: float = Field(default=0.01, ge=0)
    adaptive_collision: bool = False
    step_size: float = Field(default=0.01, gt=0)
    simplify: bool = False


class MultiPlanRequest(BaseModel):
    assembly: AssemblyRef
    engine: Literal["physics", "geometric"] = Field(
        default="physics",
        description="physics uses examples/run_multi_plan.py; geometric uses baselines/run_multi_plan.py.",
    )
    sequence_planner: str = Field(
        default="queue",
        description="physics: random, queue, prog-queue. geometric: random or queue.",
    )
    path_planner: str = Field(
        default="bfs",
        description=(
            "physics: bfs, bk-rrt, or helical (verified screw-out path per move; parts "
            "without a detectable thread fall back to bfs). "
            "geometric: rrt, rrt-connect, birrt, trrt, or matevec-trrt."
        ),
    )
    rotation: bool = False
    body_type: Literal["bvh", "sdf"] = Field(
        default="bvh",
        description=(
            "BVH contact is exact but scales poorly on dense meshes; use sdf for assemblies "
            "with interlocking threads or large part meshes when the planner times out."
        ),
    )
    sdf_dx: float = Field(default=0.05, gt=0)
    sequence_max_time: float = Field(default=3600, gt=0, le=86400)
    path_max_time: float = Field(default=120, gt=0, le=86400)
    seed: int = 1
    save_artifacts: bool = False
    n_save_states: int = Field(default=100, ge=1, le=100000)
    # Physics planner options
    collision_threshold: float = Field(default=0.01, ge=0)
    force_magnitude: float = Field(default=100, gt=0)
    frame_skip: int = Field(default=100, ge=1)
    # Geometric planner options
    max_collision: float = Field(default=0.01, ge=0)
    adaptive_collision: bool = False
    step_size: float = Field(default=0.01, gt=0)


class JobAccepted(BaseModel):
    success: Literal[True] = True
    result: dict[str, str]


class JobSnapshot(BaseModel):
    job_id: str
    operation: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[ErrorDetail] = None
    artifacts: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    success: Literal[True] = True
    result: JobSnapshot
