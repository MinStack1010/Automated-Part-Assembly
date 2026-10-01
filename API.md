# Assemble-Them-All HTTP API

Base URL: `http://<host>:8000`. FastAPI publishes the complete request and
response schemas at `/docs`, `/redoc`, and `/openapi.json`.

All operation success bodies use `{"success": true, "result": ...}`. Errors
use the following stable shape:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "..."
  }
}
```

`400` is used for invalid resource/operation input, `404` for unknown
assemblies/jobs/artifacts, `422` for schema validation, `500` for unexpected
server failures, and `503` when `redmax_py` or a native runtime library cannot
load.

## Endpoint list

| Method | Endpoint | Purpose | Input | Output |
| --- | --- | --- | --- | --- |
| GET | `/health` | Liveness plus native module status. | — | `status`, `native_backend` |
| GET | `/info` | API/runtime metadata. | — | name, version, Python/native status |
| GET | `/capabilities` | Discover operations found in this repository. | — | operation names, endpoints, execution mode |
| GET | `/api/v1/assemblies?collection=multi_assembly` | List installed OBJ assemblies. | collection query | part metadata |
| GET | `/api/v1/assemblies/{collection}/{assembly_id}` | Inspect installed assembly IDs and translation metadata. | path parameters | part metadata |
| POST | `/api/v1/assemblies/upload` | Persist external OBJ parts for later operations. | multipart `files` | `upload_id`, part metadata |
| POST | `/api/v1/mesh-distance` | Compute mesh minimum distance. | JSON `MeshDistanceRequest` | native distance result |
| POST | `/api/v1/simulations/demo` | Execute an exposed `redmax_py.make_sim` demo. | JSON `NativeDemoRequest` | final native state |
| POST | `/api/v1/jobs/joint-plan` | Queue a physics or geometric single-part planner. | JSON `JointPlanRequest` | job ID |
| POST | `/api/v1/jobs/multi-plan` | Queue a physics or geometric sequence planner. | JSON `MultiPlanRequest` | job ID |
| GET | `/api/v1/jobs/{job_id}` | Read job state. | job ID | queued/running/completed/failed/cancelled state |
| GET | `/api/v1/jobs/{job_id}/result` | Read completed planner output. | job ID | planner result |
| GET | `/api/v1/jobs/{job_id}/artifacts/{artifact_path}` | Download an optional saved transform artifact. | job and relative artifact path | binary `.npy` file |
| DELETE | `/api/v1/jobs/{job_id}` | Cancel queued/running planner process. | job ID | final job state |

## Assembly references

Use exactly one installed or uploaded reference:

```json
{"collection": "multi_assembly", "assembly_id": "00013"}
```

```json
{"upload_id": "9be62d7f-c2a6-455a-bf5e-06b315a6d497"}
```

`POST /api/v1/assemblies/upload` accepts two or more files with multipart key
`files`. Only `.obj` and optional `translation.json` are allowed, file names
cannot contain paths, and the total size is constrained by `API_MAX_UPLOAD_MB`.

## Native synchronous operations

### Mesh distance

This maps directly to `assets.load.load_assembly`, native `redmax_py.BVHMesh`
or `SDFMesh`, and `assets.mesh_distance.compute_all_mesh_distance`.

```json
POST /api/v1/mesh-distance
{
  "assembly": {"collection": "multi_assembly", "assembly_id": "00013"},
  "body_type": "bvh",
  "states": [[0, 0, 0], [0, 0, 0]]
}
```

Supply one 3-value translation or 6-value translation/rotation-vector state
for every part in the referenced assembly; use the assembly inspect endpoint
to get its part count.

### Native demo simulation

This is a small, deterministic proof that the HTTP layer crosses
Python → pybind11 → C++:

```json
POST /api/v1/simulations/demo
{
  "environment": "SinglePendulum-Test",
  "integrator": "BDF2",
  "num_steps": 1
}
```

The supported environments are exactly the names accepted by the public
`redmax_py.make_sim` binding: `SinglePendulum-Test`, `Prismatic-Test`,
`Free2D-Test`, `GroundContact-Test`, `BoxContact-Test`, `TorqueFinger-Demo`,
and `TorqueFingerFlick-Demo`.

## Planner jobs

Planner requests run in a bounded local process queue because they can take
seconds or hours. The process boundary means an unexpected native worker crash
is marked `NATIVE_PROCESS_CRASH`; it does not terminate the HTTP server.
`DELETE` terminates a running worker process or removes a queued job.

### Single-part planner

`JointPlanRequest.engine` selects the original implementation:

| Engine | Existing module | Valid `planner` values |
| --- | --- | --- |
| `physics` | `examples/run_joint_plan.py` | `bfs`, `bk-rrt` |
| `geometric` | `baselines/run_joint_plan.py` | `rrt`, `rrt-connect`, `birrt`, `trrt`, `matevec-trrt` |

```bash
curl -X POST http://localhost:8000/api/v1/jobs/joint-plan \
  -H 'content-type: application/json' \
  -d '{
    "assembly":{"collection":"multi_assembly","assembly_id":"00013"},
    "engine":"physics",
    "planner":"bfs",
    "move_id":"0",
    "still_ids":["1"],
    "body_type":"bvh",
    "max_time":120,
    "save_artifacts":true
  }'
```

The response returns a job ID. Poll it, then fetch its result:

```bash
curl http://localhost:8000/api/v1/jobs/<job_id>
curl http://localhost:8000/api/v1/jobs/<job_id>/result
```

Joint plan results include the upstream status (`Success`, `Timeout`, etc.),
elapsed seconds, and the existing planner's state path. Physics settings map
to its existing `collision_threshold`, `force_magnitude`, and `frame_skip`;
geometric settings map to `max_collision`, `adaptive_collision`, `step_size`,
and `simplify`.

### Multi-part sequence planner

`MultiPlanRequest.engine` selects:

| Engine | Existing module | Valid `sequence_planner` | Valid `path_planner` |
| --- | --- | --- | --- |
| `physics` | `examples/run_multi_plan.py` | `random`, `queue`, `prog-queue` | `bfs`, `bk-rrt` |
| `geometric` | `baselines/run_multi_plan.py` | `random`, `queue` | `rrt`, `rrt-connect`, `birrt`, `trrt`, `matevec-trrt` |

The result contains the existing sequence planner's status, order of removed
parts, number of attempts, and total elapsed time. Set `save_artifacts=true`
to preserve successful path transforms created by the original `save_path`
helper.

## Intentionally not exposed as raw endpoints

The pybind extension also exposes mutable low-level `Simulation` state,
arbitrary XML constructors, renderer replay, backward differentiation, and
direct body/joint mutators. They have no existing stateless request contract,
and arbitrary XML file paths would give a remote caller server filesystem
access. They are therefore not exposed as unsafe mock endpoints. The API
exposes the repository's public assembly-level planning flows, safe OBJ upload,
native distance query, and whitelisted built-in native simulation instead.
