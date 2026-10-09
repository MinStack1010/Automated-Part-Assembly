"""Package a job's artifacts into one downloadable zip: artifact/{npy,json,gif}/."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = "artifact"
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


def _ensure_npy(artifact_dir: Path) -> None:
    """Convert every frames-style path.json into a cached artifact/npy/*.npy array."""
    for json_path in _frame_json_files(artifact_dir):
        frames = _load_frames(json_path)
        if frames is None:
            continue
        target = artifact_dir / "npy" / f"{json_path.stem}.npy"
        if target.is_file():
            continue
        array = np.asarray([frame["matrix"] for frame in frames], dtype=np.float64)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}.tmp"
        with open(tmp, "wb") as handle:
            np.save(handle, array)
        os.replace(tmp, target)
        logger.info("artifact npy written %s shape=%s", target.name, array.shape)


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
        extension = path.suffix.lower()
        if extension == ".npy":
            groups["npy"].append(path)
        elif extension == ".json":
            groups["json"].append(path)
        elif extension == ".gif":
            groups["gif"].append(path)
    return groups


def build_artifact_zip(artifact_dir: Path, assembly_dir: Optional[Path] = None) -> bytes:
    """Zip artifact/{npy,json,gif}/..., converting path.json and rendering the GIF on first use."""
    groups: dict[str, list[Path]] = {"npy": [], "json": [], "gif": []}
    with _BUILD_LOCK:
        if artifact_dir.is_dir():
            _ensure_npy(artifact_dir)
            _ensure_gif(artifact_dir, assembly_dir)
            groups = _group_files(artifact_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
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
