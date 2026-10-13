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
| POST | `/api/v1/assemblies/upload` | Persist external OBJ parts for later operations. | multipart `files` | `upload_id`, part metadata |
| POST | `/api/v1/mesh-gap` | Measure initial-state clearance between parts. | JSON `MeshGapRequest` | gap, overlap, suggested collision threshold |
| POST | `/api/v1/jobs/joint-plan` | Queue a physics or geometric single-part planner. | JSON `JointPlanRequest` | job ID |
| POST | `/api/v1/jobs/multi-plan` | Queue a physics or geometric sequence planner. | JSON `MultiPlanRequest` | job ID |
| GET | `/api/v1/jobs/{job_id}` | Read job state. | job ID | queued/running/completed/failed/cancelled state |
| GET | `/api/v1/jobs/{job_id}/result` | Read a completed job's status and artifact list. | job ID | JSON job snapshot (`status`, timestamps, `error`, `artifacts`) |
| GET | `/api/v1/jobs/{job_id}/artifacts.zip` | Download every job artifact as one zip. | job ID | `application/zip` (`artifact/npy`, `artifact/json`, `artifact/gif`) |
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

## Clearance diagnostics

Meshes whose parts touch or interpenetrate at the initial state keep the
physics BFS from searching deeply, which surfaces as `Timeout`. One
synchronous, pure-Python endpoint makes that measurable before a job runs.

### Measure clearance

```json
POST /api/v1/mesh-gap
{
  "assembly": {"collection": "multi_assembly", "assembly_id": "00013"},
  "move_id": "0",
  "still_ids": ["1"]
}
```

`move_id`/`still_ids` are optional; without them every pair is compared, which
is limited to assemblies of at most 10 parts. The result contains each pair's
`gap`, `overlapping` (python-fcl), the minimum `gap`, `touching`, and
`suggested_collision_threshold`: `0.01` for separated parts and `0.1` when
parts touch or interpenetrate, the value that keeps BFS from timing out.
Without `move_id` the same request can be repeated for another part.

Joint plan results include `mesh_diagnostics` with this measurement for the
requested `move_id`/`still_ids` (disable it with `mesh_diagnostics: false`);
when parts touch and `collision_threshold` or `max_collision` is below the
suggested value, a `warning` field explains what to change.

## Automatic screw-thread removal

Interlocking threads (a nut and screw that are meshed together) make every
straight pull collision-free path impossible, so BFS/RRT time out no matter
how long they run. Planning jobs therefore remove the helical thread shape
from the meshes before the planner sees them — no separate preprocessing
request is needed.

For every part in the assembly (`joint-plan` and `multi-plan` alike) the API:

1. estimates a thread hypothesis per part with the helical pitch estimator
   (axis profile cross-correlation) and keeps only hypotheses whose radial
   ripple looks like a real thread — a small, periodic envelope variation on a
   surface of revolution; hex prisms and screw heads are rejected;
2. collapses the threaded axial region onto its *inner* radial envelope: an
   external thread is cut back to its root cylinder, an internal thread's
   grooves are filled up to its crest cylinder. Only the helical shape is
   removed; vertex counts, topology, watertightness, part placement
   (`translation.json` is copied verbatim), and the source OBJ files are never
   modified;
3. validates the result (bounds may not grow, volume change is capped, and
   python-fcl must not find new part-part interference) and falls back to the
   untouched mesh when any check fails, reporting why.

The simplified meshes live in a scratch copy that is deleted when the job
ends. Each planner run records the report:

```json
"thread_removal": {
  "enabled": true,
  "source_modified": false,
  "simplified_parts": ["nut", "screw"],
  "interference": {"available": true, "before": [], "after": [], "regressions": []},
  "parts": {
    "nut": {
      "detected": true, "kind": "internal", "axis": [0.0, 0.0, 1.0],
      "pitch": 0.179047, "applied": true, "vertices_moved": 5248,
      "target_radius": [3.248283, 3.250859], "volume_delta_pct": 7.598,
      "residual_candidates": 0, "watertight": true
    },
    "screw": {
      "detected": true, "kind": "external", "axis": [0.0, 0.0, 1.0],
      "pitch": 1.982236, "applied": true, "vertices_moved": 21862,
      "target_radius": [3.183405, 3.199], "volume_delta_pct": -10.78,
      "residual_candidates": 0, "watertight": true
    }
  },
  "warnings": []
}
```

Notes and limits:

- `planner: "helical"` (or `path_planner: "helical"` for multi-plan) needs the
  real thread, so those requests report
  `"reason": "planner_requires_threads"` and plan on the original meshes.
- Parts whose thread candidate sits on a non-watertight mesh are reported
  (`"reason": "not_watertight"`) but never repaired or modified.
- Only the dominant thread per part is simplified; the measured `pitch` may be
  an alias of the true pitch (the helical planner resolves the real pitch from
  mating parts).
- With `assets/my_parts` (nut + screw meshed together) the nut becomes
  pullable straight out along `[0, 0, -1]` after removal, while the original
  pair is interlocked in every axis direction.

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
    "max_time":120,
    "save_artifacts":true
  }'
```

The response returns a job ID. Poll it, then fetch its result:

```bash
curl http://localhost:8000/api/v1/jobs/<job_id>
curl http://localhost:8000/api/v1/jobs/<job_id>/result
```

Both answer with the job snapshot: `status`, the timestamps, `error`, and the
`artifacts` list (relative paths of every file the job wrote — after completion
that includes the `.npy` motion arrays). Download the file contents through
`/artifacts.zip`. Physics settings map to the planner's existing
`collision_threshold`, `force_magnitude`, and `frame_skip`; geometric settings
map to `max_collision`, `adaptive_collision`, `step_size`, and `simplify`.

### Collision representation and failure semantics

`body_type` selects the collision representation. Omit it to use the default:
`sdf` for `engine=physics` (contact stays fast on dense meshes; the grids are
cached across jobs) and `bvh` for `engine=geometric` (exact triangle distance
with no grid build). Explicit values are honored for both engines.

A joint-plan job only completes with a valid result. If the planner finds no
path (for example `Timeout`), the job finishes as `failed` with a clear error
(`PLANNER_TIMEOUT` or `PLANNER_NO_PATH`), no artifacts are written, and
`GET .../result` answers `409` with that error. With `save_artifacts=true`,
`path/path.json` (plus the derived `.npy`/`.gif` inside `artifacts.zip`) is
written only when a path exists.

### Multi-part sequence planner

`MultiPlanRequest.engine` selects:

| Engine | Existing module | Valid `sequence_planner` | Valid `path_planner` |
| --- | --- | --- | --- |
| `physics` | `examples/run_multi_plan.py` | `random`, `queue`, `prog-queue` | `bfs`, `bk-rrt` |
| `geometric` | `baselines/run_multi_plan.py` | `random`, `queue` | `rrt`, `rrt-connect`, `birrt`, `trrt`, `matevec-trrt` |

`save_artifacts=true` preserves each successful part's path transforms through
the original `save_path` helper (one `path.json` per sequence step, converted
to `.npy` when the job completes).

### Read the completed result as JSON

```bash
curl http://localhost:8000/api/v1/jobs/<job_id>/result
```

Returns `application/json` — the same snapshot shape as `GET /api/v1/jobs/{job_id}`:

```json
{
  "success": true,
  "result": {
    "job_id": "15d39ebd-5ac1-41c5-9116-a2140c361227",
    "operation": "multi-plan",
    "status": "completed",
    "created_at": "2026-10-02T08:46:21.275489Z",
    "started_at": "2026-10-02T08:46:21.278613Z",
    "completed_at": "2026-10-02T08:46:21.983597Z",
    "error": null,
    "artifacts": ["0_nut/path.json", "npy/path.npy"]
  }
}
```

- Only completed jobs return `200`; queued/running jobs answer `409`
  (`JOB_NOT_COMPLETED`), failed jobs answer `409` with their own error code
  (e.g. `PLANNER_TIMEOUT`), unknown job IDs answer `404`.
- `artifacts` lists relative paths inside the job's artifact folder. The `.npy`
  motion arrays (`npy/path.npy`, shape `(N,4,4)`) are written when the job
  completes, so the frontend can read them right away; download file contents
  through `/artifacts.zip`.

### Download every artifact as a zip

```bash
curl -OJ http://localhost:8000/api/v1/jobs/<job_id>/artifacts.zip
```

Returns `artifact.zip`:

```
artifact/
├── npy/    path.npy   # path.json frames converted to an (N,4,4) array
├── json/   path.json  # the original frames [{"name", "matrix"}]
└── gif/    screw.gif  # rendered from path.json on first download
```

- The GIF is rendered with `utils/render_path_gif.py` on first download
  (about 10-70 s depending on mesh size) and cached inside the job's
  artifacts folder; later downloads are instant.
- All three folders are always present in the zip even when empty
  (e.g. `path: null` after a `Timeout` yields empty npy/gif folders).
- Jobs live in memory: after an API restart only jobs queued again can be
  downloaded.

## Intentionally not exposed as raw endpoints

The pybind extension also exposes mutable low-level `Simulation` state,
arbitrary XML constructors, renderer replay, backward differentiation, and
direct body/joint mutators. They have no existing stateless request contract,
and arbitrary XML file paths would give a remote caller server filesystem
access. They are therefore not exposed as unsafe mock endpoints. The API
exposes the repository's public assembly-level planning flows and safe OBJ
upload instead.
