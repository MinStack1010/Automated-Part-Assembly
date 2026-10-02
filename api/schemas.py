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


class AssemblyListResponse(BaseModel):
    success: Literal[True] = True
    result: list[AssemblyMetadata]


class AssemblyUploadResponse(BaseModel):
    success: Literal[True] = True
    result: AssemblyMetadata


class MeshDistanceRequest(BaseModel):
    assembly: AssemblyRef
    states: list[list[float]] = Field(
        description="One 3D translation or 6D translation+rotation-vector state per assembly part.",
        examples=[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    )
    body_type: Literal["bvh", "sdf"] = "bvh"
    sdf_dx: float = Field(default=0.05, gt=0, description="SDF grid resolution when body_type=sdf.")

    @model_validator(mode="after")
    def validate_state_shapes(self) -> "MeshDistanceRequest":
        if len(self.states) < 2:
            raise ValueError("at least two states are required")
        if any(len(state) not in (3, 6) for state in self.states):
            raise ValueError("each state must contain either 3 or 6 values")
        return self


class NativeDemoRequest(BaseModel):
    environment: Literal[
        "SinglePendulum-Test",
        "Prismatic-Test",
        "Free2D-Test",
        "GroundContact-Test",
        "BoxContact-Test",
        "TorqueFinger-Demo",
        "TorqueFingerFlick-Demo",
    ] = "SinglePendulum-Test"
    integrator: Literal["BDF1", "BDF2"] = "BDF2"
    num_steps: int = Field(default=1, ge=1, le=10000)


class JointPlanRequest(BaseModel):
    assembly: AssemblyRef
    engine: Literal["physics", "geometric"] = Field(
        default="physics",
        description="physics uses examples/run_joint_plan.py; geometric uses baselines/run_joint_plan.py.",
    )
    planner: str = Field(
        default="bfs",
        description="physics: bfs or bk-rrt. geometric: rrt, rrt-connect, birrt, trrt, or matevec-trrt.",
    )
    move_id: str = Field(default="0")
    still_ids: list[str] = Field(default_factory=lambda: ["1"], min_length=1)
    rotation: bool = False
    body_type: Literal["bvh", "sdf"] = "bvh"
    sdf_dx: float = Field(default=0.05, gt=0)
    max_time: float = Field(default=120, gt=0, le=86400)
    seed: int = 1
    save_artifacts: bool = Field(default=False, description="Store .npy path transforms as downloadable job artifacts.")
    n_save_states: int = Field(default=100, ge=1, le=100000)
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
        description="physics: bfs or bk-rrt. geometric: rrt, rrt-connect, birrt, trrt, or matevec-trrt.",
    )
    rotation: bool = False
    body_type: Literal["bvh", "sdf"] = "bvh"
    sdf_dx: float = Field(default=0.05, gt=0)
    sequence_max_time: float = Field(default=3600, gt=0, le=86400)
    path_max_time: float = Field(default=120, gt=0, le=86400)
    seed: int = 1
    save_artifacts: bool = Field(
        default=False,
        description=(
            "Store the moving parts' trajectory as path/<frame>.npy job artifacts; "
            "each frame is a 4x4 transform of the part moving during that segment."
        ),
    )
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


class JobResultResponse(BaseModel):
    success: Literal[True] = True
    result: dict[str, Any]
