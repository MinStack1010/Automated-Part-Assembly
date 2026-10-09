# Assemble-Them-All — Kiến trúc & Execution Flow

Tài liệu mô tả cách project chạy từ lúc khởi động Docker cho đến khi một request
hoàn thành, dựa trên execution flow thực tế trong code (không mô tả theo tên file đoán chức năng).

## 1. Kiến trúc tổng quan

Project là **HTTP wrapper (FastAPI) quanh các planner có sẵn** + **module C++ `redmax_py`**
(pybind11, nằm trong `simulation/`). Không có database, không có Redis, không có cache
trung tâm và không gọi API bên ngoài. Trạng thái job được giữ **trong RAM** của process
FastAPI; file được ghi ra **disk** (`API_STORAGE_DIR`).

| Folder | Vai trò |
| --- | --- |
| `server.py` | Entry point CLI: chạy uvicorn với `api.server:app`. |
| `api/` | Lớp HTTP: `config.py` (env → Settings), `errors.py` (ApiError), `schemas.py` (Pydantic contract), `server.py` (app factory, middleware, exception handlers, mount router), `routes/` (endpoint), `services/` (business logic: assemblies, meshes, jobs, planning, native). |
| `examples/` | Planner **physics** (gọi simulator `redmax_py`): `run_joint_plan.py`, `run_multi_plan.py`. |
| `baselines/` | Planner **geometric** (RRT-family thuần Python + trimesh): `run_joint_plan.py`, `run_multi_plan.py`. |
| `assets/` | Nạp/giữ dữ liệu assembly: `load.py` (đọc `.obj`, `translation.json`), `save.py` (ghi `path.json`), `transform.py`, `mesh_distance.py`. |
| `simulation/` | Nguồn C++ `redmax` + `setup.py` build ra `redmax_py` (simulator vật lý, contact, render). |
| `utils/`, `images/`, `examples/test_*` | Renderer/CLI/asset phụ, không nằm trong luồng API. |
| `tests/` | Smoke test HTTP + service test mesh (`unittest`). |
| `docker/`, `Dockerfile`, `docker-compose.yml` | Build image 2 stage, entrypoint Xvfb, các service. |

Quan hệ: **Client → FastAPI (`api/`) → service layer → planner (`examples/` hoặc `baselines/`)
→ `redmax_py` (C++) hoặc trimesh → file artifact trên disk → response JSON.**

## 2. Entry point & khởi động

### 2.1 Docker

`docker compose up -d` đọc `docker-compose.yml`:

- **`api`** (service chính, image `assemble-them-all:latest`, port `8000:8000`,
  `restart: unless-stopped`, volume `api-data → /var/lib/assemble-them-all`):
  - `ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]` (`docker/entrypoint.sh`) →
    start **Xvfb** (X server ảo `:99`, để render headless) rồi `exec "$@"`.
  - `CMD ["python", "server.py"]`.
- **`app`**: cùng image, mount repo `.:/app`, `command: sleep infinity` (container rỗng để chạy tay).
- **`dev`**: image builder (`assemble-them-all:dev`), `sleep infinity`.
- **`gui`** (profile `gui`): mount X server thật, dành cho viewer tương tác.

Build image (`Dockerfile`): stage `builder` → `pip install ./simulation` (compile C++
`redmax_py` bằng CMake/pybind11, ~12 phút) + `pip install -r requirements.txt`
+ kiểm tra `import redmax_py`; stage `runtime` → copy `/opt/venv` + `COPY . /app`.
Env quan trọng: `API_STORAGE_DIR=/var/lib/assemble-them-all`, `ASSEMBLY_ASSETS_DIR=/app/assets`,
`API_MAX_WORKERS=1`.

### 2.2 Python process

`python server.py`
→ import `api.server`
→ `api/config.py:get_settings()` chạy **1 lần khi import** (`settings = get_settings()`, api/config.py:63):
đọc env, `mkdir` `uploads/` và `jobs/` dưới `storage_dir`
→ `api/server.py`: `logging.basicConfig`, tạo `FastAPI(...)` (api/server.py:21),
add CORS (nếu `CORS_ORIGINS`), đăng ký middleware log request (server.py:40),
đăng ký 4 exception handler (server.py:54-91), rồi
`app.include_router(health, assemblies, operations, jobs)` (server.py:94-97)
→ import `api.services.jobs` khởi tạo singleton **`job_manager = JobManager()`** (jobs.py:244):
`mp.get_context("spawn")`, `_jobs = {}`, `_queued = []`, `_running = set()`
→ `uvicorn.run(...)` (server.py:10) mở cổng `:8000`.

Framework khởi tạo theo thứ tự: **Settings → FastAPI (handlers/middleware) → router → JobManager → uvicorn**.

## 3. Các execution flow chính

### 3.1 Health / capabilities (không cần native)

`GET /health` → `api/routes/health.py:health()` → `api/services/native.py:native_status()`
(`importlib.import_module("redmax_py")`, catch mọi exception) →
`{"status": "ok"|"degraded", "native_backend": "ready"|"unavailable"}`.
`GET /info` làm tương tự + `platform.python_version()`. `GET /capabilities` trả list tĩnh
`_CAPABILITIES` (health.py:14) — dùng để client discover 6 operation.

### 3.2 Assembly: đọc & upload (file upload)

Đọc (đồng bộ, không cần native):

- `GET /api/v1/assemblies?collection=multi_assembly`
  → `routes/assemblies.py:assemblies()` → `services/assemblies.py:list_assemblies(collection)`
  → `_validate_segment()` (chặn path traversal) → duyệt từng thư mục con của
  `assets_dir/{collection}` → `get_metadata()` mỗi cái (bỏ qua thư mục lỗi) →
  `{"result": [AssemblyMetadata, ...]}`.
- `GET /api/v1/assemblies/{collection}/{assembly_id}`
  → `routes/assemblies.py:assembly()` → `get_metadata(AssemblyRef(...))`
  → `resolve_assembly()` (validate ref, chặn thoát `assets_dir`, 404 nếu không có)
  → `load_part_ids()` → `{"result": {source, part_ids, part_count, has_translation}}`.
  Không có file nào bị ghi/đọc mesh ở 2 endpoint này — chỉ liệt kê file `.obj`.

Upload:

`POST /api/v1/assemblies/upload` (multipart `files`)
→ `api/routes/assemblies.py:upload_assembly()` (async)
→ `api/services/assemblies.py:save_upload()`:
  validate từng tên file (`_SAFE_SEGMENT`, chỉ `.obj` + `translation.json`, cấm duplicate,
  chặn path traversal, giới hạn `API_MAX_UPLOAD_MB`)
  → `mkdir uploads/{uuid4}` → ghi stream từng file
  → `get_metadata()` → `assets/load.py:load_part_ids()` (danh sách stem `.obj`, sort)
  → trả `AssemblyMetadata{source.upload_id, part_ids, part_count, has_translation}`.
Lỗi → `shutil.rmtree(target_dir)` rồi `raise ApiError` → handler trả JSON `{success:false, error:{code,message}}`.
**Output:** `upload_id` (UUID) trỏ tới thư mục trên disk; dùng làm `assembly.upload_id` cho mọi lệnh sau.

### 3.2b Sửa mesh & đo khe hở (đồng bộ, thuần Python, không cần native)

Part chạm hoặc xuyên nhau ở trạng thái đầu vào (gap ≈ 0) làm BFS vật lý không
đi sâu được → job trả `Timeout`. Hai endpoint dưới đây đo và sửa trước khi chạy job:

`POST /api/v1/assemblies/preprocess` (JSON `PreprocessRequest`)
→ `api/routes/assemblies.py:preprocess_assembly()`
→ `api/services/meshes.py:preprocess_assembly()`:
  `get_metadata()`/`resolve_assembly()` (validate ref như các endpoint khác)
  → `_load_parts()` (trimesh nạp từng `.obj` với `process=False`)
  → check `mesh.is_watertight`; part chưa kín → `repair_mesh()` gọi
    `pymeshfix.clean_from_arrays()` (thất bại thì giữ mesh cũ, ghi `repair_error`)
  → `_normalize()` (lặp lại logic `assets/process_mesh.py`: tâm + scale bbox 10)
  → (tùy chọn) `_shrink_radial()` cho từng part trong `shrink_parts`: scale x/y
    quanh trục riêng của part (`0 < scale ≤ 1`) — dành cho cặp vít/đai có ren
    ăn nhau, khiến mọi quỹ đạo thẳng/vít đều chạm → không bao giờ plan được
  → (tùy chọn) `_straight_pull()` cho part trong `shrink_parts`/`verify_part_id`:
    quét fcl 6 chiều (`±x/±y/±z`) từng bước, dừng ở lần chạm đầu →
    `straight_pull.{part}.free` = còn kéo thẳng ra được không (chống Timeout)
  → (tùy chọn) `assets/subdivide.py:subdivide_to_size()` tới `max_edge`, chỉ chạy
    khi mọi part watertight — ngược lại `400 INVALID_ASSEMBLY`
  → `_write_assembly()` ghi `uploads/{uuid4}` (OBJ mới + `translation.json` đã áp
    đúng affine) — **không bao giờ sửa assembly nguồn**
  → trả `{source, output(AssemblyMetadata), parts[], all_watertight, gap_before,
    gap_after, straight_pull, suggested_collision_threshold}`.

Kết quả thực nghiệm với cặp ren vít/đaiốc (assembly `ca793cac…`, `move_id=screw`):
không shrink → **mọi** quỹ đạo (thẳng + z, xoay, vít pitch 1.25) đều chạm → mọi
planner Timeout; shrink `screw=0.85` → `straight_pull.free=true`, gap 0.109;
nhưng planner physics `body_type=bvh` vẫn Timeout vì broad/narrow-phase BVH
trên ~20k mặt ren tốn 30–90 s/bước mô phỏng — cùng upload với
`body_type="sdf"` plan **Success trong ~2 s**. Assembly gốc + SDF vẫn Timeout
(n ren thật sự ăn nhau) → cần **cả** `shrink_parts` **và** `body_type=sdf`.

`POST /api/v1/mesh-gap` (JSON `MeshGapRequest`)
→ `api/routes/meshes.py:mesh_gap()` → `api/services/meshes.py:mesh_gap()`:
  `_select_pairs()` (`move_id`+`still_ids`, hoặc mọi cặp khi ≤ 10 part)
  → `_gap_report()`: `trimesh.proximity.ProximityQuery.on_surface()` hai chiều
    từng cặp (khoảng cách ≥ 0) + `trimesh.collision.CollisionManager.in_collision_internal`
    để bắt pair đang xuyên nhau
  → `touching` = xuyên nhau hoặc `gap ≤ 0.05`
  → `suggested_collision_threshold` = `0.1` khi `touching` (giá trị thực nghiệm
    không Timeout), ngược lại `0.01` (default của `JointPlanRequest`).

### 3.3 Mesh distance (đồng bộ, gọi native)

`POST /api/v1/mesh-distance`
→ `routes/operations.py:mesh_distance()`
→ `services/native.py:compute_mesh_distance()`:
  `require_native()` (503 `NATIVE_BACKEND_UNAVAILABLE` nếu import lỗi)
  → `get_metadata()`/`resolve_assembly()` (validate ref, chặn thoát root assets)
  → `assets/load.py:load_assembly()` (trimesh nạp `.obj`, áp `translation.json`)
  → tạo `native.BVHMesh` hoặc `native.SDFMesh` cho từng part
  → `assets/mesh_distance.py:compute_all_mesh_distance()` (C++ tính khoảng cách tối thiểu)
  → `{assembly, body_type, minimum_distance}`.

`POST /api/v1/simulations/demo` → `native.py:run_native_demo()` →
`redmax_py.make_sim(env, integrator)` → `sim.reset()` → `sim.forward(n)` →
trả `{q, qdot, ndof, converged}`.

### 3.4 Tạo job planner (joint-plan / multi-plan) — flow chính

**Bước 1 — enqueue (trả về ngay):**

`POST /api/v1/jobs/joint-plan` hoặc `POST /api/v1/jobs/multi-plan`
→ `api/routes/jobs.py:create_joint_plan_job()` / `create_multi_plan_job()`
→ Pydantic đã validate body thành `JointPlanRequest`/`MultiPlanRequest` (schemas.py)
→ `request.model_dump(mode="json")`
→ `services/jobs.py:JobManager.submit(operation, payload)`:
  `uuid4()` → `artifact_dir = settings.jobs_dir / job_id / "artifacts"` (mkdir)
  → `Job(...)` lưu vào `_jobs`, đẩy vào `_queued`
  → `_start_available_jobs()`: nếu còn slot (`API_MAX_WORKERS`), spawn
     `mp.Process(target=_execute_job_process, args=(operation, payload, artifact_dir, result_queue), daemon=True)`
     → đặt `status="running"`, `started_at=now` → start thread `self._watch(job_id)`
→ response **202** `{"success": true, "result": {"job_id": "...", "status": "queued"}}`.

**Bước 2 — worker process (child, `spawn`):**

`_execute_job_process()` (jobs.py:28) — module-level để process spawn import được:
- `"joint-plan"` → `services/planning.py:run_joint_plan(payload, artifact_dir)`
- `"multi-plan"` → `services/planning.py:run_multi_plan(payload, artifact_dir)`
- catch `ApiError` → `result_queue.put({ok:False, error:{code,message}})`
- catch exception khác → `NATIVE_EXECUTION_FAILED` + traceback
- thành công → `result_queue.put({ok:True, result: {...}})`

**Bước 3a — `run_joint_plan()` (single part):**

`require_native()` → `get_metadata()` (đọc part_ids) → `_validate_part_selection()`
(move_id ∈ part_ids, still_ids hợp lệ, không trùng) → `resolve_assembly()` → `output_dir`
(chỉ khi `save_artifacts=true`) → `_mesh_diagnostics()` (nếu `mesh_diagnostics=true`):
`services/meshes.py:mesh_gap()` đo gap/xuyên nhau ở trạng thái 0 rồi gắn
`result.mesh_diagnostics` (gồm `suggested_collision_threshold`, và `warning` khi
part chạm nhưng `collision_threshold`/`max_collision` chưa đủ) → chi nhánh engine:

- `engine=physics` → `examples/run_joint_plan.py:get_planner(name)` (`bfs`→`BFSPlanner`, `bk-rrt`→`BK_RRT`)
  → constructor `PhysicsPlanner.__init__` (run_joint_plan.py:192):
    `load_assembly()` → dựng `redmax.BVHMesh/SDFMesh` → `compute_move_mesh_distance()` để auto collision threshold
    → `get_xml_string()` sinh XML mô hình (joint `translational`/`free3d-exp` cho part chuyển, `fixed` cho phần còn lại, contact pairs)
    → **`redmax.Simulation(model_string, asset_folder)`** (pybind11 → C++)
  → `planner.plan(max_time, seed, return_path=True, render=False)`: reset sim, mở `Tree`, lặp
    chọn state/action → `sim.set_body_external_force` → `sim.forward()` × `frame_skip`
    → `is_disassembled()` (trùng lặp ConvexHull qua `trimesh.collision.CollisionManager`)
    → `Success` lấy `path` = danh sách `q` của part chuyển, hoặc `Timeout`/`Failure`
    → trả `(status, t_plan, path)`.
- `engine=geometric` → `baselines/run_joint_plan.py:PyPlanner` → dựng
  `distance_fn/collision_fn/sample_fn/extend_fn/goal_test` (dựng `redmax.BVHMesh/SDFMesh`
  rồi `assets/mesh_distance.compute_move_mesh_distance` để đo va chạm/khoảng cách)
  → `rrt | rrt-connect | birrt | trrt | matevec-trrt` (file trong `baselines/pyplanners/`)
  → optional `smooth_path()` → `(status, t_plan, path)`.

Sau đó `run_joint_plan` gom result và gọi `_serialize_path(path, output_dir, n_save_states)`
(planning.py:35): inline `path` vào JSON nếu ≤ `API_MAX_RESULT_STATES`,
nếu `save_artifacts` thì `assets/save.py:save_path()` → ghi `path.json`
(mảng `{"name": move_id, "matrix": 4×4}` theo frame) vào `artifacts/path/`
và set `artifact_directory: "path"`.

**Bước 3b — `run_multi_plan()` (sequence nhiều part):**

`clear_saved_sdfs(assembly_dir)` (dọn cache `.sdf` cũ) → try:
- `engine=physics` → `examples/run_multi_plan.py:get_seq_planner(name)`
  (`random`/`queue`/`prog-queue`) → `SequencePlanner.plan_sequence(...)`.
  Vòng lặp: chọn `move_id` (queue xáo trộn / random), `still_ids` = phần còn lại
  → `plan_path()` → dựng path planner physics như 3a (mỗi lần 1 `redmax.Simulation` mới)
  → `status='Success'` thì `graph.remove_node(move_id)`, `sequence.append(move_id)`
  → **ghi artifact** `save_dir/{assembly_id}/{seq_count}_{move_id}/path.json`
    (`assets/save.py:save_path`, ≤ `n_save_states` frame/bước)
  → dừng khi chỉ còn 1 node (`Success`), hết thời gian (`Timeout`) hoặc vượt `sequence_max_time`.
- `engine=geometric` → `baselines/run_multi_plan.py` + `PyPlanner`, cùng pattern.
→ `finally: clear_saved_sdfs()`.

`run_multi_plan` trả `{engine, sequence_planner, path_planner, assembly, status, sequence,
attempts, elapsed_seconds}` (không có key `path` inline — motion nằm hoàn toàn trong artifact).

**Bước 4 — worker hoàn tất, `_watch()` (jobs.py:186):**

`process.join()` → lấy `result_queue` (timeout 1s, chịu `queue.Empty`) → under lock:
`ok` → `status="completed"`, `completed_at`, `job.result = result`;
có `error` → `status="failed"` + `ErrorDetail`;
queue rỗng (crash/segfault native) → `status="failed"` với `NATIVE_PROCESS_CRASH`
→ finally `_start_available_jobs()` để nhặt job kế tiếp.

**Bước 5 — client poll & lấy kết quả:**

- `GET /api/v1/jobs/{job_id}` → `JobManager.snapshot()` → `_snapshot()` (jobs.py:228):
  `artifacts = sorted(rel path của mọi file rglob dưới artifact_dir)` — tính realtime nên
  job đang chạy vẫn thấy file đã ghi → `JobSnapshot{status, created/started/completed_at, error, artifacts}`.
- `GET /api/v1/jobs/{job_id}/result` → `JobManager.result()`: 409 nếu chưa `completed`
  (kèm code lỗi nếu job `failed`), trả `job.result` verbatim.
- `GET /api/v1/jobs/{job_id}/artifacts.zip` → `routes/jobs.py:download_artifacts()`
  → `services/artifacts.py:build_artifact_zip()` (serialize bằng lock):
  `_ensure_npy()` chuyển mọi `path.json` (list `{"name","matrix"}`) →
  `artifacts/npy/path.npy` shape `(N,4,4)` (cache); `_ensure_gif()` render
  `artifacts/gif/<move>.gif` qua `utils/render_path_gif.render_gif()` nếu
  `assembly_dir` resolve được (cache, fail thì bỏ qua); sau đó zip
  `artifact/{npy,json,gif}/` → response `application/zip`,
  `content-disposition: attachment; filename="artifact.zip"` — 3 folder luôn
  có mặt (rỗng nếu job không có file).
- `DELETE /api/v1/jobs/{job_id}` → `cancel()`: queued → bỏ khỏi `_queued`;
  running → `job.process.terminate()` (chỉ kill child, uvicorn sống tiếp).

**Kết quả cuối cùng của flow:** JSON response `202 {job_id}` ngay; sau khi xong,
`GET .../result` trả JSON kết quả planner (status/sequence/elapsed/path…),
`GET .../jobs/{id}` trả `artifacts` là danh sách file trên disk
(ví dụ multi-plan: `04799/0_0/path.json…`, joint-plan: `path/path.json`), client tải
`GET .../artifacts.zip` để có `artifact/{npy,json,gif}/` render animation.

### 3.5 Xử lý lỗi

- Lỗi input từ client: Pydantic → `RequestValidationError` → handler 422 `VALIDATION_ERROR`
  (server.py:62); `ApiError` (validate ref/assembly, backend unavailable) → handler 4xx/503
  (server.py:54); không lường trước → handler 500 `INTERNAL_ERROR` (server.py:85).
- Lỗi trong child process không giết server: `ApiError`/`Exception` được đóng gói vào
  `result_queue` (jobs.py:47-62), `_watch` gán vào `job.error`.
- Crash native (segfault): queue rỗng → `NATIVE_PROCESS_CRASH`, container có
  `restart: unless-stopped` nên API không chết hẳn.

## Execution Flow Summary

```
User/Client
→ Docker (docker compose up -d: entrypoint.sh → Xvfb → python server.py)
→ FastAPI app (api/server.py: config → middleware → handlers → routers)
→ Route (api/routes/jobs.py | operations.py | assemblies.py | health.py)
→ Service (api/services/jobs.py::submit | planning.py::run_* | native.py | assemblies.py)
→ [sync] Business Logic + Native (examples|baselines planner → redmax_py C++ / trimesh)
→ [async] spawn child process → _execute_job_process → planner → artifact files trên disk
→ JobManager._watch cập nhật job.result/status (in-RAM, không DB)
→ Result JSON (GET /jobs/{id}/result) + artifacts list (GET /jobs/{id})
→ artifact.zip (GET /jobs/{id}/artifacts.zip: artifact/npy + json + gif)
→ Response JSON cho Client
```
