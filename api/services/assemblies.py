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


def _validate_segment(value: str, field: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value) or value in {".", ".."}:
        raise ApiError(400, "INVALID_INPUT", f"{field} must be a simple file-system-safe identifier")
    return value


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


def list_assemblies(collection: str, config: Settings = settings) -> list[AssemblyMetadata]:
    collection = _validate_segment(collection, "collection")
    collection_dir = (config.assets_dir / collection).resolve()
    if not _is_child(collection_dir, config.assets_dir) or not collection_dir.is_dir():
        raise ApiError(404, "COLLECTION_NOT_FOUND", "The requested assembly collection does not exist")

    results: list[AssemblyMetadata] = []
    for assembly_dir in sorted(path for path in collection_dir.iterdir() if path.is_dir()):
        ref = AssemblyRef(collection=collection, assembly_id=assembly_dir.name)
        try:
            results.append(get_metadata(ref, config))
        except ApiError:
            # A collection may contain non-assembly directories; they are not API resources.
            continue
    return results


async def save_upload(files: list[UploadFile], config: Settings = settings) -> AssemblyMetadata:
    """Persist a multipart assembly after validating names and total size."""
    if not files:
        raise ApiError(400, "INVALID_INPUT", "at least two OBJ files are required")

    upload_id = str(uuid.uuid4())
    target_dir = config.uploads_dir / upload_id
    total_bytes = 0
    max_bytes = config.max_upload_mb * 1024 * 1024
    object_count = 0
    seen_names: set[str] = set()

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
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        for upload in files:
            await upload.close()

    if object_count < 2:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ApiError(400, "INVALID_ASSEMBLY", "an assembly must contain at least two OBJ files")

    return get_metadata(AssemblyRef(upload_id=upload_id), config)
