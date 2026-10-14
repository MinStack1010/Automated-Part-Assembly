# `joint_assembly` collection

## Status

The full `joint_assembly` dataset is **external data, not tracked in git**.
The repository previously carried only a symlink,
`assets/joint_assembly -> /Users/ploggvn/Downloads/joint_assembly`,
whose target does not exist on a fresh machine (and never existed inside the
Docker image), so every `collection=joint_assembly` request failed with
`ASSEMBLY_NOT_FOUND`. No download URL, token, or checksum for the original
dataset exists anywhere in this repository (checked: README/API/ARCH docs,
`examples/`, `baselines/`, git history, and the GitHub project page), so
there is nothing to re-download verbatim.

What was done instead (2026-10-14):

- Removed the broken absolute-path symlink.
- Made `assets/joint_assembly/` a real directory so the collection resolves
  both on the host and in Docker.
- Bundled one verified working sample, `screw_nut_demo/` (copies of the
  simplified `assets/my_parts` pair: 194/60-vert watertight meshes,
  0.4 clearance, straight-pullable along Z).

## Layout

Each assembly is a folder with ≥ 2 OBJ files named by part id:

```text
assets/joint_assembly/<assembly_id>/screw.obj
assets/joint_assembly/<assembly_id>/nut.obj
assets/joint_assembly/<assembly_id>/translation.json   # optional
```

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/jobs/joint-plan \
  -H 'content-type: application/json' \
  -d '{"assembly":{"collection":"joint_assembly","assembly_id":"screw_nut_demo"}, \
       "engine":"physics","planner":"bfs","move_id":"screw","still_ids":["nut"], \
       "max_time":120,"save_artifacts":true}'
```

## Adding the full external dataset later

Drop the real assembly folders next to the sample (or anywhere and point
`JOINT_ASSEMBLY_DIR` at them):

```bash
JOINT_ASSEMBLY_DIR=$HOME/Downloads/joint_assembly docker compose up -d api
```

`docker-compose.yml` mounts that directory over `/app/assets/joint_assembly`
in every service, so external data shadows nothing else and the sample keeps
working when the variable is unset.
