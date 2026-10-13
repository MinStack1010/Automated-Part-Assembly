# Assemble-Them-All — Architecture & Execution Flow

This document describes how the project runs from Docker startup to a completed request,
based on the actual execution flow in the code (not guessed from file names).

## 1. Overview

The project is an **HTTP wrapper (FastAPI) around the existing planners** plus a **C++ module `redmax_py`**
(pybind11, under `simulation/`). There is no database, no Redis, no central cache,
and no external API calls. Job state is kept **in RAM** of the FastAPI process;
files are written to **disk** (`API_STORAGE_DIR`).

| Folder | Role |
| --- | --- |
| `server.py` | CLI entry point: runs uvicorn with `api.server:app`. |
| `api/` | HTTP layer: `config.py` (env → Settings), `errors.py` (ApiError), `schemas.py` (Pydantic contract), `server.py` (app factory, middleware, exception handlers, router mount), `routes/` (endpoints), `services/` (business logic: assemblies, meshes, jobs, planning, native). |
| `examples/` | **Physics** planners (drive the `redmax_py` simulator): `run_joint_plan.py`, `run_multi_plan.py`. |
| `baselines/` | **Geometric** planners (pure-Python RRT-family + trimesh): `run_joint_plan.py`, `run_multi_plan.py`. |
| `assets/` | Assembly loading/storage: `load.py` (reads `.obj`, `translation.json`), `save.py` (writes `path.json`), `transform.py`, `mesh_distance.py`. |
| `simulation/` | C++ `redmax` sources + `setup.py` that builds `redmax_py` (physics, contact, rendering). |
| `utils/`, `images/`, `examples/test_*` | Renderer/CLI/auxiliary assets, not part of the API flow. |
| `tests/` | HTTP smoke tests + service tests: mesh, thread removal, no-path failure semantics (`unittest` + `pytest`). |
| `docker/`, `Dockerfile`, `docker-compose.yml` | 2-stage image build, Xvfb entrypoint, services. |

Relationship: **Client → FastAPI (`api/`) → service layer → planner (`examples/` or `baselines/`)
→ `redmax_py` (C++) or trimesh → artifact files on disk → JSON response.**

## 2. Entry point & startup

### 2.1 Docker

`docker compose up -d` reads `docker-compose.yml`:

- **`api`** (main service, image `assemble-them-all:latest`, port `8000:8000`,
  `restart: unless-stopped`, volume `api-data → /var/lib/assemble-them-all`):
  - `ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]` (`docker/entrypoint.sh`) →
    starts **Xvfb** (virtual X server `:99`, for headless rendering) then `exec "$@"`.
  - `CMD ["python", "server.py"]`.
- **`app`**: same image, mounts the repo `.:/app`, `command: sleep infinity` (empty container for manual runs).
- **`dev`**: builder image (`assemble-them-all:dev`), `sleep infinity`.
- **`gui`** (profile `gui`): mounts the real X server, for the interactive viewer.

Image build (`Dockerfile`): stage `builder` → `pip install ./simulation` (compiles the C++
`redmax_py` with CMake/pybind11, ~12 minutes) + `pip install -r requirements.txt`
+ verifies `import redmax_py`; stage `runtime` → copies `/opt/venv` + `COPY . /app`.
Important env: `API_STORAGE_DIR=/var/lib/assemble-them-all`, `ASSEMBLY_ASSETS_DIR=/app/assets`,
`API_MAX_WORKERS=1`.

### 2.2 Python process

`python server.py`
→ imports `api.server`
→ `api/config.py:get_settings()` runs **once at import** (`settings = get_settings()`, api/config.py:63):
reads env, `mkdir` `uploads/` and `jobs/` under `storage_dir`
→ `api/server.py`: `logging.basicConfig`, creates `FastAPI(...)` (api/server.py:21),
adds CORS (if `CORS_ORIGINS`), registers the request-logging middleware (server.py:40),
registers 4 exception handlers (server.py:54-91), then
`app.include_router(health, assemblies, meshes, jobs)` (server.py:94-97)
→ importing `api.services.jobs` initializes the singleton **`job_manager = JobManager()`** (jobs.py:244):
`mp.get_context("spawn")`, `_jobs = {}`, `_queued = []`, `_running = set()`
→ `uvicorn.run(...)` (server.py:10) opens port `:8000`.

Framework initialization order: **Settings → FastAPI (handlers/middleware) → router → JobManager → uvicorn**.

## 3. Main execution flows

### 3.1 Health / capabilities (no native required)

`GET /health` → `api/routes/health.py:health()` → `api/services/native.py:native_status()`
(`importlib.import_module("redmax_py")`, catches every exception) →
`{"status": "ok"|"degraded", "native_backend": "ready"|"unavailable"}`.
`GET /info` does the same plus `platform.python_version()`. `GET /capabilities` returns the static
`_CAPABILITIES` list (health.py:14) — clients use it to discover the 5 operations
(`assembly.upload`, `mesh.gap`, `joint.plan`, `multi.plan`, `job.artifacts.zip`).

### 3.2 Assembly: upload (file upload)

An assembly installed under `assets/{collection}/{assembly_id}` is referenced with
`{"collection", "assembly_id"}`; there is no list/inspect endpoint — only upload:

`POST /api/v1/assemblies/upload` (multipart `files`)
→ `api/routes/assemblies.py:upload_assembly()` (async)
→ `api/services/assemblies.py:save_upload()`:
  validates each file name (`_SAFE_SEGMENT`, only `.obj` + `translation.json`, no duplicates,
  path traversal blocked, size limited by `API_MAX_UPLOAD_MB`)
  → `mkdir uploads/{uuid4}` → streams each file
  → `get_metadata()` → `assets/load.py:load_part_ids()` (sorted list of `.obj` stems)
  → returns `AssemblyMetadata{source.upload_id, part_ids, part_count, has_translation}`.
On error → `shutil.rmtree(target_dir)` then `raise ApiError` → the handler returns JSON `{success:false, error:{code,message}}`.
**Output:** `upload_id` (UUID) pointing at the on-disk directory; used as `assembly.upload_id` for every later call.

### 3.2b Gap measurement (synchronous, pure Python, no native required)

Parts that touch or interpenetrate in the input state (gap ≈ 0) leave the physical BFS unable
to make progress → the job returns `Timeout`. The endpoint below measures this before running a job:

`POST /api/v1/mesh-gap` (JSON `MeshGapRequest`)
→ `api/routes/meshes.py:mesh_gap()` → `api/services/meshes.py:mesh_gap()`:
  `_select_pairs()` (`move_id`+`still_ids`, or every pair when ≤ 10 parts)
  → `_gap_report()`: `trimesh.proximity.ProximityQuery.on_surface()` both ways
    per pair (distance ≥ 0) + `trimesh.collision.CollisionManager.in_collision_internal`
    to catch intersecting pairs
  → `touching` = intersecting or `gap ≤ 0.05`
  → `suggested_collision_threshold` = `0.1` when `touching` (empirically avoids
    `Timeout`), otherwise `0.01` (the `JointPlanRequest` default).

### 3.2c Automatic thread detection & flattening (before every planner)

Interlocking threads (screw + nut) make **every** straight/helical collision-free
trajectory nonexistent → BFS/RRT times out indefinitely, regardless of `max_time`. Instead of
asking the user to preprocess, each planner job flattens the threads on a temporary
copy (module `api/services/threads.py`, context manager `planning_assembly()`):

1. **Detection** — for each part, `api/services/helical.py:estimate_thread_candidates()`
   (cross-correlates the axial radius profile, 3 principal axes × `min`/`max` profiles)
   → `_classify_candidate()` keeps only candidates whose ripple really looks like a thread:
   `ripple_ratio = median(spread)/median(row_max) ≤ 0.4` **and** the pitch inferred
   between adjacent rows is stable (CV ≤ 0.35, ≥ 4 pairs). Hex faces or screw heads
   (spread ≈ radius, not periodic in z) are rejected.
2. **Flattening** — `_simplify()` re-orients the axis to the nearest world axis, then for
   each axial row takes the minimum radius of the **inner envelope**
   (`min` profile, smoothed ±1 row, clamped to the ripple band): external threads are cut back to
   the root cylinder (crest → root), internal threads fill the grooves up to the crest cylinder. Only
   crests inside the ripple band are moved; intact heads/cylindrical parts outside the band
   are left unchanged, and the `head`/hex region outside the band is skipped.
3. **Validation** — `_validate()`: vertex/face counts unchanged, watertight preserved,
   bounds not expanded, every displacement ≤ the measured ripple, volume change ≤ 30%;
   `_overlap_pairs()` (fcl) before/after: new collisions → revert everything
   (`interference_regressed`). Non-watertight meshes are **not** auto-repaired
   (`reason: "not_watertight"`), only reported. Remaining threads are re-measured
   (`residual_candidates`).
4. **Handoff** — if any part was flattened: the assembly is copied to
   `tempfile.mkdtemp()` (all files except `.sdf`, `translation.json` copied verbatim),
   only the modified parts are overwritten, and the planners read the temp directory; `finally` deletes it.
   **The source assembly is never modified.** Every job result always includes
   `thread_removal` (enabled, simplified_parts, per-part details, interference).
   `planner/path_planner = "helical"` needs the real threads → flattening is disabled
   (`reason: "planner_requires_threads"`).

Experiment on `assets/my_parts` (interlocking nut + screw): both parts are detected
(nut `internal`, screw `external`, z axis), flattened to the inner cylinder, fcl reports
no new collisions, and `meshes._straight_pull()` flips from `free=false` (every
direction collides) to `free=true` along `[0,0,-1]` — a straight pull-out trajectory now exists.

### 3.3 Planner job creation (joint-plan / multi-plan) — main flow

**Step 1 — enqueue (returns immediately):**

`POST /api/v1/jobs/joint-plan` or `POST /api/v1/jobs/multi-plan`
→ `api/routes/jobs.py:create_joint_plan_job()` / `create_multi_plan_job()`
→ Pydantic has already validated the body into `JointPlanRequest`/`MultiPlanRequest` (schemas.py;
joint-plan's optional `body_type` is `None` by default — resolved per engine inside `run_joint_plan`)
→ `request.model_dump(mode="json")`
→ `services/jobs.py:JobManager.submit(operation, payload)`:
  `uuid4()` → `artifact_dir = settings.jobs_dir / job_id / "artifacts"` (mkdir)
  → `Job(...)` stored in `_jobs`, pushed to `_queued`
  → `_start_available_jobs()`: if a slot is free (`API_MAX_WORKERS`), spawns
     `mp.Process(target=_execute_job_process, args=(operation, payload, artifact_dir, result_queue), daemon=True)`
     → sets `status="running"`, `started_at=now` → starts thread `self._watch(job_id)`
→ response **202** `{"success": true, "result": {"job_id": "...", "status": "queued"}}`.

**Step 2 — worker process (child, `spawn`):**

`_execute_job_process()` (jobs.py:28) — module-level so the spawned process can import it:
- `"joint-plan"` → `services/planning.py:run_joint_plan(payload, artifact_dir)`; when the
  result has no path (`path_state_count == 0`), puts
  `{ok:False, error:{code:PLANNER_TIMEOUT|PLANNER_NO_PATH, message}, result}`
  instead of `ok:True` — a job without a path is `failed`, never `completed`
- `"multi-plan"` → `services/planning.py:run_multi_plan(payload, artifact_dir)`
- catches `ApiError` → `result_queue.put({ok:False, error:{code,message}})`
- catches any other exception → `NATIVE_EXECUTION_FAILED` + traceback
- success → `result_queue.put({ok:True, result: {...}})`

**Step 3a — `run_joint_plan()` (single part):**

`require_native()` → `get_metadata()` (reads part_ids) → `_validate_part_selection()`
(move_id ∈ part_ids, valid non-duplicate still_ids) → `resolve_assembly()` → `output_dir`
(only when `save_artifacts=true`) → resolves `body_type = request.body_type or
("sdf" if engine == "physics" else "bvh")` (per-engine default matches the CLI;
`geometric` keeps `bvh` because building SDF grids takes minutes) →
**`threads.planning_assembly(assembly_dir, enabled=...)`**
(context manager §3.2c: auto-flattens threads on a temp copy, unless
`planner="helical"` — then `enabled=false`, `reason="planner_requires_threads"`) →
inside the context: `_mesh_diagnostics()` (when `mesh_diagnostics=true`, measured on the **temp
copy** so the gap reflects what the planner actually sees):
`services/meshes.py:mesh_gap()` measures gap/interpenetration in state 0 and attaches
`result.mesh_diagnostics` (including `suggested_collision_threshold`, and a `warning` when
parts touch but `collision_threshold`/`max_collision` is too small) → engine branch
(every planner receives the temp `planning_dir`):

- `engine=physics` → `examples/run_joint_plan.py:get_planner(name)` (`bfs`→`BFSPlanner`, `bk-rrt`→`BK_RRT`)
  → constructor `PhysicsPlanner.__init__` (run_joint_plan.py:192):
    `load_assembly()` → builds `redmax.BVHMesh/SDFMesh` → `compute_move_mesh_distance()` for the automatic collision threshold
    → `get_xml_string()` generates the model XML (joint `translational`/`free3d-exp` for the moving part, `fixed` for the rest, contact pairs)
    → **`redmax.Simulation(model_string, asset_folder)`** (pybind11 → C++)
  → `planner.plan(max_time, seed, return_path=True, render=False)`: resets the sim, opens the `Tree`, loops
    selecting state/action → `sim.set_body_external_force` → `sim.forward()` × `frame_skip`
    → `is_disassembled()` (ConvexHull overlap via `trimesh.collision.CollisionManager`)
    → `Success` returns `path` = the list of `q` for the moving part, or `Timeout`/`Failure`
    → returns `(status, t_plan, path)`.
- `engine=geometric` → `baselines/run_joint_plan.py:PyPlanner` → builds
  `distance_fn/collision_fn/sample_fn/extend_fn/goal_test` (builds `redmax.BVHMesh/SDFMesh`
  then `assets/mesh_distance.compute_move_mesh_distance` for collision/distance)
  → `rrt | rrt-connect | birrt | trrt | matevec-trrt` (files under `baselines/pyplanners/`)
  → optional `smooth_path()` → `(status, t_plan, path)`.
- `engine=physics|geometric` with `planner="helical"` → `helical.build_helical_path(...)`;
  `rotation_sign` tries both rotation directions.

Afterwards `run_joint_plan` assembles the result (always including `thread_removal`)
and calls `_serialize_path(path, output_dir, n_save_states)`
(planning.py): inlines `path` into the JSON when ≤ `API_MAX_RESULT_STATES`;
when `save_artifacts` is set, `assets/save.py:save_path()` writes `path.json`
(an array of `{"name": move_id, "matrix": 4×4}` per frame) to `artifacts/path/`
and sets `artifact_directory: "path"`. The context manager deletes the temp copy on exit.

**Step 3b — `run_multi_plan()` (multi-part sequence):**

Also wrapped in `threads.planning_assembly()` (unless `path_planner="helical"`) →
`clear_saved_sdfs(planning_dir)` (drops old `.sdf` caches on the temp copy) → try:
- `engine=physics` → `examples/run_multi_plan.py:get_seq_planner(name)`
  (`random`/`queue`/`prog-queue`) → `SequencePlanner.plan_sequence(...)`.
  Loop: pick `move_id` (shuffled queue / random), `still_ids` = the rest
  → `plan_path()` → builds the physics path planner like 3a (a new `redmax.Simulation` each time)
  → on `status='Success'` `graph.remove_node(move_id)`, `sequence.append(move_id)`
  → **writes artifact** `save_dir/{assembly_id}/{seq_count}_{move_id}/path.json`
    (`assets/save.py:save_path`, ≤ `n_save_states` frames/step)
  → stops when one node remains (`Success`), time runs out (`Timeout`), or `sequence_max_time` is exceeded.
- `engine=geometric` → `baselines/run_multi_plan.py` + `PyPlanner`, same pattern.
→ `finally: clear_saved_sdfs(planning_dir)`.

`run_multi_plan` returns `{engine, sequence_planner, path_planner, assembly, status, sequence,
attempts, elapsed_seconds, thread_removal}` (no inline `path` key — the motion lives
entirely in the artifacts).

**Step 4 — worker finishes, `_watch()` (jobs.py:186):**

`process.join()` → take `result_queue` (timeout 1s, tolerates `queue.Empty`) → under lock:
`ok` → `status="completed"`, `completed_at`, `job.result = result`;
`error` present → `status="failed"` + `ErrorDetail` (the planner `result` is stored too when sent);
`ok:False` with a `PLANNER_*` error (joint-plan found no path) → same failed branch;
empty queue (native crash/segfault) → `status="failed"` with `NATIVE_PROCESS_CRASH`
→ finally `_start_available_jobs()` to pick up the next job.

**Step 5 — client polls and fetches results:**

- `GET /api/v1/jobs/{job_id}` → `JobManager.snapshot()` → `_snapshot()` (jobs.py:228):
  `artifacts = sorted(rel path of every file rglob'd under artifact_dir)` — computed live, so
  a running job still shows files already written → `JobSnapshot{status, created/started/completed_at, error, artifacts}`.
- `GET /api/v1/jobs/{job_id}/result` → `JobManager.result()` gates first (409 unless
  `completed`, with the error code when the job `failed`, 404 for unknown IDs), then
  returns `JobResponse(result=JobManager.snapshot())` — the same job snapshot as
  `GET /{job_id}`: `status`, timestamps, `error`, and the relative `artifacts` paths.
  On completion `_watch` runs `artifacts.ensure_npy()` first, so `npy/path.npy`
  already appears in the listing when `status` flips to `completed`.
- `GET /api/v1/jobs/{job_id}/artifacts.zip` → `routes/jobs.py:download_artifacts()`
  → `services/artifacts.py:build_artifact_zip()` (serialized by lock):
  `_ensure_npy()` converts every `path.json` (list of `{"name","matrix"}`) →
  `artifacts/npy/path.npy` shape `(N,4,4)` (cached); `_ensure_gif()` renders
  `artifacts/gif/<move>.gif` via `utils/render_path_gif.render_gif()` when
  `assembly_dir` resolves (cached; failures are ignored); then zips
  `artifact/{npy,json,gif}/` → response `application/zip`,
  `content-disposition: attachment; filename="artifact.zip"` — the 3 folders are always
  present (empty when the job has no files).
- `DELETE /api/v1/jobs/{job_id}` → `cancel()`: queued → removed from `_queued`;
  running → `job.process.terminate()` (kills only the child; uvicorn keeps running).

**Final output of the flow:** a JSON `202 {job_id}` response immediately; when done,
`GET .../result` (and `GET .../jobs/{id}`) return the job snapshot with `status`,
timestamps, `error` and `artifacts` as the list of files on disk
(e.g. multi-plan: `04799/0_0/path.json…`, joint-plan: `path/path.json` + `npy/path.npy`
after completion), and the client fetches
`GET .../artifacts.zip` for `artifact/{npy,json,gif}/` with the rendered animation.

### 3.4 Error handling

- Client input errors: Pydantic → `RequestValidationError` → 422 handler `VALIDATION_ERROR`
  (server.py:62); `ApiError` (invalid ref/assembly, backend unavailable) → 4xx/503 handler
  (server.py:54); unexpected → 500 handler `INTERNAL_ERROR` (server.py:85).
- Errors in the child process never kill the server: `ApiError`/`Exception` are wrapped into
  `result_queue` (jobs.py:47-62), and `_watch` assigns them to `job.error`.
- Native crash (segfault): empty queue → `NATIVE_PROCESS_CRASH`; the container has
  `restart: unless-stopped`, so the API does not stay dead.

## Execution Flow Summary

```
User/Client
→ Docker (docker compose up -d: entrypoint.sh → Xvfb → python server.py)
→ FastAPI app (api/server.py: config → middleware → handlers → routers)
→ Route (api/routes/jobs.py | meshes.py | assemblies.py | health.py)
→ Service (api/services/jobs.py::submit | planning.py::run_* | threads.py::planning_assembly | meshes.py | assemblies.py)
→ [sync] Business Logic + Native (examples|baselines planner → redmax_py C++ / trimesh)
→ [async] spawn child process → _execute_job_process → planner → artifact files on disk
→ JobManager._watch updates job.result/status (in-RAM, no DB)
→ Job snapshot JSON (GET /jobs/{id}/result and GET /jobs/{id}: status + artifacts)
→ artifact.zip (GET /jobs/{id}/artifacts.zip: artifact/npy + json + gif)
→ JSON response to Client
```
