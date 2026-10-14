"""Package a job's artifacts into one downloadable zip: artifact/{npy,json,gif}/."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = "artifact"
RESULT_JSON = "result.json"
_FRAME_COUNT = 40
_BUILD_LOCK = threading.Lock()


def _load_frames(path: Path) -> Optional[list[dict]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict) and "matrix" in data[0]:
        return data
    return None


def _frame_json_files(artifact_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in artifact_dir.rglob("*.json")
        if path.is_file() and path.parent.name not in {"npy", "gif"}
    )


def npy_target_for(artifact_dir: Path, json_path: Path) -> Path:
    """Return the cached .npy location for one frames-style path.json.

    joint-plan writes a single ``path/path.json`` -> the historical
    ``npy/path.npy`` is kept for compatibility. Multi-plan writes one
    ``path.json`` per sequence step (``<assembly>/<seq>_<move>/path.json``);
    every stem is ``path``, so a flat ``npy/path.npy`` would collide and only
    the first step would survive. Those map to
    ``npy/<assembly>_<seq>_<move>_path.npy`` instead.
    """
    try:
        relative = json_path.relative_to(artifact_dir)
    except ValueError:
        relative = Path(json_path.name)
    if len(relative.parts) == 2 and relative.parts[0] == "path" and json_path.stem == "path":
        return artifact_dir / "npy" / "path.npy"
    flat = "_".join(relative.with_suffix("").parts)
    return artifact_dir / "npy" / f"{flat}.npy"


def _ensure_npy(artifact_dir: Path) -> None:
    """Convert every frames-style path.json into a cached npy array (no collisions)."""
    for json_path in _frame_json_files(artifact_dir):
        frames = _load_frames(json_path)
        if frames is None:
            continue
        target = npy_target_for(artifact_dir, json_path)
        if target.is_file():
            continue
        array = np.asarray([frame["matrix"] for frame in frames], dtype=np.float64)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}.tmp"
        with open(tmp, "wb") as handle:
            np.save(handle, array)
        os.replace(tmp, target)
        logger.info("artifact npy written %s shape=%s", target.name, array.shape)


def resolve_artifact(artifact_dir: Path, relative: str) -> Path:
    """Resolve a user-supplied artifact path, blocking directory traversal escapes."""
    candidate = (artifact_dir / relative).resolve()
    try:
        candidate.relative_to(artifact_dir.resolve())
    except ValueError:
        raise ValueError(f"artifact path escapes the job folder: {relative!r}")
    return candidate


def _parse_step_id(relative: Path) -> dict[str, Any]:
    """Split a multi-plan step path ``<assembly>/<seq>_<move>/path.json``.

    joint-plan's ``path/path.json`` has no sequence info; anything else that
    does not match the multi-plan layout keeps ``sequence_index``/``move``
    empty and the frontend still gets the raw artifact path.
    """
    info: dict[str, Any] = {"id": str(relative.parent), "sequence_index": None, "assembly_id": None}
    parts = relative.parts
    if len(parts) == 3 and parts[2] == "path.json":
        info["assembly_id"] = parts[0]
        head, _, tail = parts[1].partition("_")
        if head.isdigit():
            info["sequence_index"] = int(head)
            info["move_id"] = tail
    return info


def list_animation_steps(artifact_dir: Path) -> list[dict[str, Any]]:
    """Describe every playable animation step in a job's artifact folder.

    Each entry carries the part id (``frames[0].name``), the frame count, the
    source ``path.json`` relative path and the derived ``.npy``/``.gif``
    relative paths when those files exist. The 4x4 matrices follow the
    ``assets/save.py:save_path`` contract: world transform of the moving part
    per frame; still parts stay at their loaded pose.
    """
    steps: list[dict[str, Any]] = []
    if not artifact_dir.is_dir():
        return steps
    for json_path in _frame_json_files(artifact_dir):
        frames = _load_frames(json_path)
        if not frames:
            continue
        try:
            relative = json_path.relative_to(artifact_dir)
        except ValueError:
            continue
        npy_path = npy_target_for(artifact_dir, json_path)
        move_id = str(frames[0].get("name") or json_path.stem)
        gif_path = artifact_dir / "gif" / f"{move_id}.gif"
        step = _parse_step_id(relative)
        step.update(
            {
                "move_id": step.get("move_id") or move_id,
                "artifact": str(relative),
                "frame_count": len(frames),
                "npy_artifact": str(npy_path.relative_to(artifact_dir)) if npy_path.is_file() else None,
                "gif_artifact": str(gif_path.relative_to(artifact_dir)) if gif_path.is_file() else None,
            }
        )
        steps.append(step)
    steps.sort(key=lambda entry: (entry["sequence_index"] is None, entry["sequence_index"] or 0, entry["id"]))
    return steps


def _ensure_gif(artifact_dir: Path, assembly_dir: Optional[Path]) -> None:
    """Render artifact/gif/<move>.gif from path.json on first use (best effort)."""
    if assembly_dir is None or not assembly_dir.is_dir():
        return
    from utils.render_path_gif import render_gif

    for json_path in _frame_json_files(artifact_dir):
        frames = _load_frames(json_path)
        if frames is None:
            continue
        name = str(frames[0].get("name") or json_path.stem)
        target = artifact_dir / "gif" / f"{name}.gif"
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{name}.tmp.gif"
        try:
            render_gif(str(assembly_dir), str(json_path), str(tmp), n_frame=_FRAME_COUNT)
        except Exception:
            logger.warning("gif render failed for %s", json_path, exc_info=True)
            tmp.unlink(missing_ok=True)
            continue
        os.replace(tmp, target)
        logger.info("artifact gif written %s", target.name)


def _group_files(artifact_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {"npy": [], "json": [], "gif": []}
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name == RESULT_JSON and path.parent == artifact_dir:
            continue
        extension = path.suffix.lower()
        if extension == ".npy":
            groups["npy"].append(path)
        elif extension == ".json":
            groups["json"].append(path)
        elif extension == ".gif":
            groups["gif"].append(path)
    return groups


def _write_zip(
    artifact_dir: Path,
    assembly_dir: Optional[Path],
    extra: dict[str, str],
) -> bytes:
    groups: dict[str, list[Path]] = {"npy": [], "json": [], "gif": []}
    with _BUILD_LOCK:
        if artifact_dir.is_dir():
            _ensure_npy(artifact_dir)
            _ensure_gif(artifact_dir, assembly_dir)
            groups = _group_files(artifact_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in extra.items():
            archive.writestr(name, text)
        for group, files in groups.items():
            archive.writestr(f"{ROOT}/{group}/", "")
            used: set[str] = set()
            for path in files:
                name = path.name
                if name in used:
                    name = "_".join(path.relative_to(artifact_dir).parts)
                used.add(name)
                archive.write(path, f"{ROOT}/{group}/{name}")
    return buffer.getvalue()


def build_artifact_zip(artifact_dir: Path, assembly_dir: Optional[Path] = None) -> bytes:
    """Zip artifact/{npy,json,gif}/..., converting path.json and rendering the GIF on first use."""
    return _write_zip(artifact_dir, assembly_dir, extra={})


def ensure_npy(artifact_dir: Path) -> None:
    """Materialize the .npy motion arrays so artifact listings include them right away.

    Called by the job watcher when a job completes: path.json -> npy/path.npy is
    cheap (milliseconds), while the GIF stays lazy inside the zip download.
    """
    if not artifact_dir.is_dir():
        return
    with _BUILD_LOCK:
        _ensure_npy(artifact_dir)
