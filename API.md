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
| POST | `/api/v1/assemblies/preprocess` | Repair, normalize, and subdivide an assembly into a new upload. | JSON `PreprocessRequest` | part repair report, gap before/after |
| POST | `/api/v1/mesh-gap` | Measure initial-state clearance between parts. | JSON `MeshGapRequest` | gap, overlap, suggested collision threshold |
| POST | `/api/v1/mesh-distance` | Compute mesh minimum distance. | JSON `MeshDistanceRequest` | native distance result |
| POST | `/api/v1/simulations/demo` | Execute an exposed `redmax_py.make_sim` demo. | JSON `NativeDemoRequest` | final native state |
| POST | `/api/v1/jobs/joint-plan` | Queue a physics or geometric single-part planner. | JSON `JointPlanRequest` | job ID |
| POST | `/api/v1/jobs/multi-plan` | Queue a physics or geometric sequence planner. | JSON `MultiPlanRequest` | job ID |
| GET | `/api/v1/jobs/{job_id}` | Read job state. | job ID | queued/running/completed/failed/cancelled state |
| GET | `/api/v1/jobs/{job_id}/result` | Read completed planner output. | job ID | planner result |
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

## Mesh repair and clearance diagnostics

Meshes whose parts touch or interpenetrate at the initial state keep the
physics BFS from searching deeply, which surfaces as `Timeout`. Two
synchronous, pure-Python endpoints make that measurable before a job runs.

### Preprocess an assembly

```json
POST /api/v1/assemblies/preprocess
{
  "assembly": {"collection": "multi_assembly", "assembly_id": "00013"},
  "repair": true,
  "normalize": true,
  "subdivide": false,
  "max_edge": 0.5,
  "shrink_parts": {"screw": 0.87},
  "verify_part_id": "screw"
}
```

Parts that are not watertight are closed with `pymeshfix`, then the assembly
is centered and scaled to a 10-unit bounding box exactly like
`assets/process_mesh.py` (`normalize`), and optionally subdivided until every
edge is shorter than `max_edge` (`subdivide` requires watertight parts). The
result is written to a new upload; the source assembly is never modified.

`shrink_parts` scales the named parts radially (about their own x/y axis,
`0 < scale <= 1`). It is the fix for threaded male/female pairs such as
screw + nut whose threads interlock: no straight or helical motion is
collision-free, so every planner times out. Shrinking the male part until the
threads no longer engage makes a straight pull possible; the sweep reports
whether it actually is:

```json
"straight_pull": {"screw": {"available": true, "free": true, "direction": [0, 0, 1], "distance": 74.4}}
```

`free: false` means the processed mesh still cannot be pulled apart along any
axis direction. `verify_part_id` runs the same sweep for a part that was not
shrunk; without it the sweep runs only for the parts in `shrink_parts`.

The response reports `parts` (`watertight_before`, `repaired`,
`watertight_after`, `radial_scale` when shrunk, vertex/face counts),
`all_watertight`, `gap_before`, `gap_after`, `straight_pull`, and
`suggested_collision_threshold`. Use `output.source` as the
`assembly` reference of the following planning job:

```bash
curl -X POST http://localhost:8000/api/v1/jobs/joint-plan \
  -H 'content-type: application/json' \
  -d '{"assembly": {"upload_id": "<upload_id from output.source>"},
       "engine": "physics", "planner": "bfs",
       "move_id": "0", "still_ids": ["1"],
       "collision_threshold": 0.1}'
```

For threaded assemblies (screw + nut) use `"body_type": "sdf"` as well:
BVH contact spends 30-90 s per simulation step on the ~20k-face thread
meshes and times out even when the path is collision-free; the same
`bfs` plan on the preprocessed upload finished in ~2 s with SDF
(`{"body_type": "sdf", "sdf_dx": 0.05}`). Shrinking alone is not enough —
the unprocessed assembly still times out with SDF because the threads are
genuinely interlocked.

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
exposes the repository's public assembly-level planning flows, safe OBJ upload,
native distance query, and whitelisted built-in native simulation instead.
