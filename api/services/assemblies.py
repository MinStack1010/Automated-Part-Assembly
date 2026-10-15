"""Safe resolution and upload of Assemble-Them-All OBJ assemblies."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from api.config import Settings, settings
from api.errors import ApiError
from api.schemas import AssemblyMetadata, AssemblyRef

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ALLOWED_UPLOAD_NAMES = {"translation.json"}
_ALLOWED_SUFFIXES = {".obj"}
# Stems that would collide with assembly-level helpers downstream.
_RESERVED_PART_STEMS = {"assembly", "translation"}
_MAX_PART_NAME_LEN = 64


def _validate_segment(value: str, field: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value) or value in {".", ".."}:
        raise ApiError(400, "INVALID_INPUT", f"{field} must be a simple file-system-safe identifier")
    return value


def _sanitize_part_name(raw: str, used: set[str]) -> str:
    """Map one embedded mesh name to a safe, unique part stem.

    OBJ object names are free text (spaces, slashes, duplicates); part IDs
    must satisfy _SAFE_SEGMENT because they become file names, URL segments
    and planner IDs downstream.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", (raw or "").strip())
    cleaned = cleaned.strip("._-")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"part_{cleaned}" if cleaned else "part"
    cleaned = cleaned[:_MAX_PART_NAME_LEN].rstrip("._-") or "part"
    if cleaned in _RESERVED_PART_STEMS:
        cleaned = f"part_{cleaned}"
    candidate, index = cleaned, 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = f"{cleaned[:_MAX_PART_NAME_LEN - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _declared_object_names(obj_path: Path) -> list[str]:
    """Scan `o` declarations so empty objects can be reported, not silently dropped."""
    names: list[str] = []
    try:
        with obj_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("o ") or stripped.startswith("o\t"):
                    name = stripped[1:].strip()
                    if name:
                        names.append(name)
    except OSError:
        pass
    return names


def _split_single_obj(obj_path: Path, target_dir: Path) -> list[str]:
    """Split one multi-mesh OBJ into per-part `<part_id>.obj` files.

    Splitting follows the file's own `o` object boundaries via trimesh
    (``group_material=False`` keeps objects separate); connected meshes are
    never fragmented. Each part keeps its scene-graph world transform, so the
    spatial relationships from the input file are preserved while every part
    stays independently plannable. Returns the part stems.
    """
    import numpy as np
    import trimesh

    try:
        loaded = trimesh.load(str(obj_path), process=False, group_material=False)
    except Exception as exc:
        raise ApiError(400, "INVALID_INPUT", f"the uploaded OBJ could not be parsed: {exc}") from exc

    nodes: list[tuple[str, object]] = []
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            try:
                nodes.append((str(node), loaded.graph[node]))
            except KeyError:
                continue
        if not nodes:
            nodes = [(str(name), (np.eye(4), name)) for name in loaded.geometry]
    elif isinstance(loaded, trimesh.Trimesh):
        nodes = [("part", (np.eye(4), None))]
    else:
        raise ApiError(400, "INVALID_INPUT", "the uploaded OBJ could not be parsed as mesh data")

    declared = _declared_object_names(obj_path)
    extracted: list[str] = []
    skipped_empty: list[str] = []
    used: set[str] = set()
    for node_name, entry in nodes:
        transform, geometry_name = entry
        key = geometry_name if geometry_name is not None else node_name
        try:
            source = loaded.geometry[key] if isinstance(loaded, trimesh.Scene) else loaded
        except KeyError:
            skipped_empty.append(str(node_name))
            continue
        if len(source.vertices) == 0 or len(source.faces) == 0:
            skipped_empty.append(str(node_name))
            continue
        mesh = source.copy()
        mesh.apply_transform(np.asarray(transform, dtype=float))
        label = str(geometry_name or node_name)
        if label in {"default", "material", "mesh"} and str(node_name) not in {"default", "material", "mesh"}:
            label = str(node_name)
        part_id = _sanitize_part_name(label, used)
        mesh.export(str(target_dir / f"{part_id}.obj"))
        extracted.append(part_id)

    if len(extracted) < 2:
        detail = f"found {len(extracted)} usable mesh(es)"
        if declared:
            detail += f" (declared objects: {', '.join(declared[:8])})"
        if skipped_empty:
            detail += f"; skipped empty: {', '.join(skipped_empty[:8])}"
        raise ApiError(400, "INVALID_ASSEMBLY", f"a single OBJ must contain at least two usable meshes; {detail}")

    obj_path.unlink()
    return extracted


def _is_child(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_assembly(ref: AssemblyRef, config: Settings = settings) -> Path:
    """Resolve an assembly reference without allowing paths outside API roots."""
    if ref.upload_id is not None:
        try:
            upload_id = str(uuid.UUID(ref.upload_id))
        except ValueError as exc:
            raise ApiError(400, "INVALID_INPUT", "upload_id must be a UUID") from exc
        path = (config.uploads_dir / upload_id).resolve()
        parent = config.uploads_dir
    else:
        assert ref.collection is not None and ref.assembly_id is not None
        collection = _validate_segment(ref.collection, "collection")
        assembly_id = _validate_segment(ref.assembly_id, "assembly_id")
        path = (config.assets_dir / collection / assembly_id).resolve()
        parent = config.assets_dir

    if not _is_child(path, parent):
        raise ApiError(400, "INVALID_INPUT", "assembly path escapes the configured storage root")
    if not path.is_dir():
        raise ApiError(404, "ASSEMBLY_NOT_FOUND", "The requested assembly does not exist")
    return path


def get_metadata(ref: AssemblyRef, config: Settings = settings) -> AssemblyMetadata:
    from assets.load import load_part_ids

    assembly_dir = resolve_assembly(ref, config)
    part_ids = load_part_ids(str(assembly_dir))
    if len(part_ids) < 2:
        raise ApiError(400, "INVALID_ASSEMBLY", "An assembly must contain at least two OBJ part files")
    return AssemblyMetadata(
        source=ref,
        part_ids=part_ids,
        part_count=len(part_ids),
        has_translation=(assembly_dir / "translation.json").is_file(),
    )


async def save_upload(files: list[UploadFile], config: Settings = settings) -> AssemblyMetadata:
    """Persist a multipart assembly after validating names and total size.

    Two shapes are accepted: two or more `.obj` files (one part each, the
    historical behavior) or a single `.obj` holding at least two meshes,
    which is split into per-part files (see :func:`_split_single_obj`).
    An optional `translation.json` is allowed in both cases.
    """
    if not files:
        raise ApiError(400, "INVALID_INPUT", "at least one OBJ file is required")

    upload_id = str(uuid.uuid4())
    target_dir = config.uploads_dir / upload_id
    total_bytes = 0
    max_bytes = config.max_upload_mb * 1024 * 1024
    object_count = 0
    seen_names: set[str] = set()
    obj_paths: list[Path] = []

    try:
        target_dir.mkdir(parents=True, exist_ok=False)
        for upload in files:
            original_name = upload.filename or ""
            file_name = Path(original_name).name
            if file_name != original_name or not file_name:
                raise ApiError(400, "INVALID_INPUT", "uploaded filenames must not contain directory components")
            suffix = Path(file_name).suffix.lower()
            if suffix not in _ALLOWED_SUFFIXES and file_name not in _ALLOWED_UPLOAD_NAMES:
                raise ApiError(400, "INVALID_INPUT", "only .obj files and optional translation.json are accepted")
            stem = Path(file_name).stem
            if suffix == ".obj" and not _SAFE_SEGMENT.fullmatch(stem):
                raise ApiError(400, "INVALID_INPUT", "OBJ filenames must use safe part identifiers")
            if file_name in seen_names:
                raise ApiError(400, "INVALID_INPUT", f"duplicate upload filename: {file_name}")
            seen_names.add(file_name)

            target_path = target_dir / file_name
            with target_path.open("wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ApiError(413, "PAYLOAD_TOO_LARGE", f"upload exceeds API_MAX_UPLOAD_MB ({config.max_upload_mb} MB)")
                    destination.write(chunk)
            if suffix == ".obj":
                object_count += 1
                obj_paths.append(target_dir / file_name)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        for upload in files:
            await upload.close()

    if object_count == 0:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ApiError(400, "INVALID_ASSEMBLY", "an assembly must contain at least one OBJ file")

    if object_count == 1:
        # A single file must carry at least two meshes; split it into parts.
        # The original container file is removed so downstream code (which
        # reads one part per *.obj) never mistakes it for an extra part.
        try:
            _split_single_obj(obj_paths[0], target_dir)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        _validate_translation_keys(target_dir)

    return get_metadata(AssemblyRef(upload_id=upload_id), config)


def _validate_translation_keys(target_dir: Path) -> None:
    """Ensure an uploaded translation.json covers every extracted part.

    Without this, planning would fail later with an obscure KeyError when a
    part has no translation entry. Only enforced for single-file splits, whose
    part IDs are assigned by the server and cannot be known upfront.
    """
    translation_path = target_dir / "translation.json"
    if not translation_path.is_file():
        return
    import json

    from assets.load import load_part_ids

    try:
        entries = json.loads(translation_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ApiError(400, "INVALID_INPUT", f"translation.json is not valid JSON: {exc}") from exc
    if not isinstance(entries, dict):
        raise ApiError(400, "INVALID_INPUT", "translation.json must map part IDs to translations")
    part_ids = set(load_part_ids(str(target_dir)))
    missing = sorted(part_ids - set(entries))
    if missing:
        raise ApiError(
            400,
            "INVALID_ASSEMBLY",
            f"translation.json is missing entries for extracted parts: {', '.join(missing)}",
        )
