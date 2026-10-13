# Assemble-Them-All

Assemble-Them-All provides disassembly planning for multi-part 3D assemblies.
Its HTTP service is a thin, documented wrapper around the existing planners
and the `redmax_py` pybind11 extension; it does not reimplement physics or
planning algorithms in Python.

## Project architecture

```text
HTTP / JSON client
        |
        v
FastAPI routes (api/)
        |
        +-- services: upload/asset validation, clearance diagnostics, automatic screw-thread removal, job isolation, response mapping
        |
        v
existing Python planners and asset helpers
  examples/run_joint_plan.py, examples/run_multi_plan.py
  baselines/run_joint_plan.py, baselines/run_multi_plan.py
        |
        v
redmax_py (pybind11 extension)
        |
        v
RedMax C++ simulation, BVH/SDF mesh distance, collision and dynamics
```

The native build produces both the `redmax` C++ test executable and the
`redmax_py` Python extension. `simulation/setup.py` drives CMake and
`simulation/redmax/python_interface.cpp` is the only Python/native binding.
There is no `ctypes`, `cffi`, or SWIG layer.

## API server

Requirements: Python 3.9+, CMake, a C++ compiler, and the existing native
OpenGL/X11 runtime dependencies. Docker supplies these dependencies on Linux.

```bash
python -m pip install ./simulation
python -m pip install -r requirements.txt
python server.py
```

The server listens on `http://0.0.0.0:8000` by default. Interactive API docs
are available at `/docs`, `/redoc`, and `/openapi.json`.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
curl http://localhost:8000/capabilities
```

`GET /health` returns `degraded` when the HTTP server is running but
`redmax_py` cannot load. Native endpoints then return a structured `503`
instead of crashing the service.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_HOST` | `0.0.0.0` | Interface used by `python server.py`. |
| `API_PORT` | `8000` | API listen port. |
| `LOG_LEVEL` | `INFO` | Application log level. |
| `ASSEMBLY_ASSETS_DIR` | `<repo>/assets` | Installed assembly collections. |
| `API_STORAGE_DIR` | `<repo>/.api-data` | Uploaded assemblies, job state, and downloaded artifacts. |
| `API_MAX_WORKERS` | `1` | Concurrent isolated planner processes. Keep low unless RAM allows more simulations. |
| `API_MAX_UPLOAD_MB` | `250` | Total multipart upload limit. |
| `API_MAX_RESULT_STATES` | `10000` | Maximum path states embedded in JSON job results. |
| `CORS_ORIGINS` | empty (disabled) | Comma-separated allowed browser origins, such as `https://3d.example.com`. |

CORS is deliberately disabled by default. Set explicit origins in production;
use `CORS_ORIGINS=*` only when that access policy is intentional.

### Inputs and outputs

An assembly is a directory containing at least two OBJ files named by part ID,
such as `0.obj` and `1.obj`. `translation.json` is optional and follows the
existing project format: an object mapping each part ID to its three-value
translation. Installed assets are referenced by `collection` and
`assembly_id`; an external 3D project can upload OBJ files first and use the
returned `upload_id`.

Job results are JSON: `GET /api/v1/jobs/<job_id>/result` returns the job
snapshot with `status`, timestamps, `error`, and the `artifacts` list (relative
paths, including the `.npy` motion arrays once the job completes). Set
`save_artifacts=true` to have the job save the `path.json` transform frames (an
array of `{"name": "<obj_id>", "matrix": [[...4x4...]]}` objects); download the
file contents from the job's `artifacts.zip`. Every planner job first flattens
detected screw threads
(nut/screw thread pairs) on a temporary copy of the assembly and reports what
it changed under `thread_removal`. See [API.md](API.md) for details.

### Real native planning example

Submitting a planner job is the supported way to exercise the C++ backend: the
job worker constructs a `redmax_py.Simulation` through pybind11 and runs the
existing physics or geometric planner inside an isolated process.

```bash
curl -X POST http://localhost:8000/api/v1/jobs/joint-plan \
  -H 'content-type: application/json' \
  -d '{"assembly":{"collection":"multi_assembly","assembly_id":"00013"},"engine":"physics","planner":"bfs","move_id":"0","still_ids":["1"],"max_time":120}'
```

### External 3D AI integration

```python
import httpx

base_url = "http://assemble-them-all:8000"

# Verify the service (and native backend) before submitting expensive planner jobs.
response = httpx.get(f"{base_url}/health", timeout=30)
response.raise_for_status()
print(response.json())  # {"status": "ok", "native_backend": "ready"}

# Upload generated mesh parts, then use the returned upload_id in a job.
with open("0.obj", "rb") as part0, open("1.obj", "rb") as part1:
    upload = httpx.post(
        f"{base_url}/api/v1/assemblies/upload",
        files=[("files", ("0.obj", part0, "text/plain")), ("files", ("1.obj", part1, "text/plain"))],
        timeout=120,
    )
upload.raise_for_status()
assembly = upload.json()["result"]["source"]

job = httpx.post(
    f"{base_url}/api/v1/jobs/joint-plan",
    json={
        "assembly": assembly,
        "engine": "physics",
        "planner": "bfs",
        "move_id": "0",
        "still_ids": ["1"],
        "max_time": 120,
        "save_artifacts": True,
    },
    timeout=30,
)
job.raise_for_status()
print(job.json())  # includes "thread_removal"
```

See [API.md](API.md) for every endpoint and request contract.

## Docker

The Dockerfile builds `redmax_py` with CMake in a builder stage, copies the
virtual environment into a smaller runtime stage, starts Xvfb for the linked
OpenGL runtime, then starts `python server.py`.

```bash
docker compose build
docker compose up --build
curl http://localhost:8000/health
```

The `api` service publishes `8000:8000` and persists uploads/job artifacts in
the named `api-data` volume. Existing `app`, `dev`, and optional `gui`
services remain available for the original interactive workflows.

## Existing CLI workflows

The API is additive. Existing command-line entry points remain unchanged:

- `examples/run_joint_plan.py` and `examples/run_multi_plan.py`: physics-based planners using `redmax_py`.
- `baselines/run_joint_plan.py` and `baselines/run_multi_plan.py`: geometric-sampling baselines using native BVH/SDF mesh queries.
- `examples/test_*`: direct native simulation examples.
- `assets/process_mesh.py`, `assets/normalize.py`, and `assets/subdivide.py`: mesh preparation utilities.

## Deployment notes

Run the service behind a reverse proxy with request-size and TLS limits, set
explicit `CORS_ORIGINS`, and put `API_STORAGE_DIR` on persistent storage.
Each planner worker loads meshes and native collision/simulation state, so CPU
and RAM scale with `API_MAX_WORKERS`; start at one worker and measure before
raising it. The shipped planners are CPU/OpenGL-runtime based and do not
require CUDA/GPU. The container must retain its Linux OpenGL/X11 libraries;
Xvfb is started automatically for headless execution.
