"""Automatic screw-thread detection and simplification for planner inputs.

Detection reuses the helical pitch estimator in :mod:`api.services.helical`
(axis profile cross-correlation). A candidate is accepted only when its radial
ripple looks like a thread: a small, periodic envelope variation on a surface of
revolution (a hex prism sliced along a flat, for example, has a ripple as large
as its radius and is rejected).

Simplification collapses the radial envelope of the threaded axial region onto
its *inner* boundary, cell by cell:

- external thread (``profile="max"``): the crests are cut back to the root
  cylinder, so the part only loses material inside the detected ripple band;
- internal thread (``profile="min"``): the grooves are filled up to the crest
  cylinder, so material is only added inside the hole and never outside the
  original bounds.

Only the helical shape of the thread is removed. Vertices are moved radially
about the detected axis by at most the ripple amplitude, the topology is
untouched, the axis placement and ``translation.json`` are never rewritten, and
the source OBJ files are only read. Every failure (non-watertight mesh,
re-entrant interference, failed validation) is reported instead of silently
producing a mesh that looks processed.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import trimesh

from api.errors import ApiError

logger = logging.getLogger(__name__)

# Acceptance limits for a candidate to be treated as a real thread.
MAX_RIPPLE_RATIO = 0.4  # ripple amplitude / radius: threads ripple a little, prisms a lot
MAX_PITCH_VARIATION = 0.35  # coefficient of variation of the implied row-to-row pitches
MIN_PERIODIC_PAIRS = 4
# Acceptance limits for a simplified part.
MAX_VOLUME_DELTA = 0.30
BOUNDS_TOL = 1e-6
MIN_RIPPLE = 1e-6

_PROFILE_KIND = {"min": "internal", "max": "external"}


def _row_statistics(profile: np.ndarray, row_filled: np.ndarray) -> Optional[dict[str, Any]]:
    """Recreate the active-row selection the helical estimator uses."""
    from api.services.helical import ROW_RADIUS_FRACTION, ROW_VARIANCE_FRACTION

    row_max = profile.max(axis=1)
    row_min = profile.min(axis=1)
    spread = row_max - row_min
    filled_values = row_max[row_filled]
    if len(filled_values) == 0:
        return None
    radius_floor = ROW_RADIUS_FRACTION * float(np.median(filled_values))
    spread_floor = ROW_VARIANCE_FRACTION * float(np.median(spread[row_filled]))
    active = row_filled & (row_max >= radius_floor) & (spread >= spread_floor)
    active_rows = np.flatnonzero(active)
    if len(active_rows) == 0:
        return None
    return {
        "active": active,
        "active_rows": active_rows,
        "row_max": row_max,
        "row_min": row_min,
        "spread": spread,
        "ripple": float(np.median(spread[active_rows])),
        "radius": float(np.median(row_max[active_rows])),
    }


def _implied_pitches(
    z_centers: np.ndarray, profile: np.ndarray, active_rows: np.ndarray, extent: float
) -> list[float]:
    """Pitch implied by the angular shift between adjacent active rows."""
    from api.services.helical import N_THETA_BINS, _2PI

    pitches: list[float] = []
    tolerance_min = 0.02 * extent
    tolerance_max = 0.5 * extent
    for first_row, second_row in zip(active_rows[:-1], active_rows[1:]):
        if second_row != first_row + 1:
            continue
        delta_z = float(z_centers[second_row] - z_centers[first_row])
        if delta_z <= 0:
            continue
        left = profile[first_row] - profile[first_row].mean()
        right = profile[second_row] - profile[second_row].mean()
        if left.std() < 1e-12 or right.std() < 1e-12:
            continue
        correlation = np.array([float(np.dot(left, np.roll(right, -shift))) for shift in range(N_THETA_BINS)])
        shift = int(np.argmax(correlation))
        alpha = _2PI * shift / N_THETA_BINS
        if alpha > np.pi:
            alpha -= _2PI
        if abs(alpha) < 1e-9:
            continue
        pitch = _2PI * delta_z / abs(alpha)
        if tolerance_min <= pitch <= tolerance_max:
            pitches.append(float(pitch))
    return pitches


def _classify_candidate(mesh: trimesh.Trimesh, candidate: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return thread geometry for a candidate, or None when it does not look like a thread."""
    from api.services.helical import _axis_profiles

    axis = np.asarray(candidate["axis"], dtype=float)
    kind = candidate["profile"]
    z_centers, profile, row_filled, extent = _axis_profiles(mesh, axis, kind)
    stats = _row_statistics(profile, row_filled)
    if stats is None or extent <= 0:
        return None

    ripple = stats["ripple"]
    radius = stats["radius"]
    if ripple <= MIN_RIPPLE or radius <= MIN_RIPPLE:
        return None
    ripple_ratio = ripple / radius
    if ripple_ratio > MAX_RIPPLE_RATIO:
        return None

    pitches = _implied_pitches(z_centers, profile, stats["active_rows"], extent)
    if len(pitches) < MIN_PERIODIC_PAIRS:
        return None
    pitch_values = np.asarray(pitches, dtype=float)
    pitch_mean = float(np.mean(pitch_values))
    if pitch_mean <= 0:
        return None
    pitch_cv = float(np.std(pitch_values) / pitch_mean)
    if pitch_cv > MAX_PITCH_VARIATION:
        return None

    return {
        "candidate": candidate,
        "kind": _PROFILE_KIND[kind],
        "profile": kind,
        "axis": axis,
        "z_centers": z_centers,
        "profile_grid": profile,
        "active": stats["active"],
        "active_rows": stats["active_rows"],
        "ripple": ripple,
        "ripple_ratio": ripple_ratio,
        "pitch_cv": pitch_cv,
        "pitch_pairs": len(pitches),
        "extent": extent,
    }


def detect_thread(mesh: trimesh.Trimesh) -> Optional[dict[str, Any]]:
    """Best accepted thread hypothesis for a part, or None when it is not threaded."""
    from api.services.helical import estimate_thread_candidates

    for candidate in estimate_thread_candidates(mesh):
        try:
            classified = _classify_candidate(mesh, candidate)
        except Exception:
            logger.warning("thread candidate classification failed", exc_info=True)
            continue
        if classified is None:
            continue
        return {
            **classified,
            "pitch": float(candidate["pitch"]),
            "axis_point": list(candidate["axis_point"]),
            "confidence": float(candidate["confidence"]),
            "inliers": int(candidate["inliers"]),
            "pairs": int(candidate["pairs"]),
            "handedness": int(candidate["handedness"]),
        }
    return None


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Snap a detected axis to the nearest world axis so radial moves stay
    inside the perpendicular plane and world-space bounds stay exact."""
    index = int(np.argmax(np.abs(axis)))
    snapped = np.zeros(3)
    snapped[index] = 1.0 if axis[index] >= 0 else -1.0
    return snapped


def _rolling_min(values: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    return np.minimum.reduce(
        [padded[offset : offset + len(values)] for offset in range(2 * radius + 1)]
    )


def _simplify(mesh: trimesh.Trimesh, thread: dict[str, Any]) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Collapse the threaded region onto its inner radial envelope."""
    from api.services.helical import (
        N_THETA_BINS,
        N_Z_BINS,
        _axis_profiles,
        _orthonormal_basis,
    )

    axis = _canonical_axis(np.asarray(thread["axis"], dtype=float))
    kind = thread["profile"]
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = vertices.mean(axis=0)
    offset = vertices - center
    along = offset @ axis
    first, second = _orthonormal_basis(axis)
    perp = offset - np.outer(along, axis)
    radius = np.linalg.norm(perp, axis=1)
    angle = np.arctan2(offset @ first, offset @ second)

    # The inner envelope drives the target radius for both thread kinds: the
    # min profile is the crest cylinder of an internal thread and the root
    # cylinder of an external one.
    _, inner_grid, inner_filled, _ = _axis_profiles(mesh, axis, "min")
    _, kind_grid, kind_filled, _ = _axis_profiles(mesh, axis, kind)
    stats = _row_statistics(kind_grid, kind_filled)
    if stats is None:
        return mesh.copy(), {"applied": False, "reason": "no_active_rows", "vertices_moved": 0}

    active = stats["active"]
    z_centers = np.linspace(along.min(), along.max(), N_Z_BINS + 1)
    z_centers = 0.5 * (z_centers[:-1] + z_centers[1:])
    z_bin = (z_centers[1] - z_centers[0]) if len(z_centers) > 1 else 0.0

    ripple = stats["ripple"]
    margin = max(0.05 * ripple, 1e-9)
    band_lo = float(np.percentile(inner_grid[active], 2)) - margin
    band_hi = float(np.percentile(kind_grid[active], 98)) + margin
    if band_hi - band_lo <= MIN_RIPPLE:
        return mesh.copy(), {"applied": False, "reason": "no_ripple", "vertices_moved": 0}

    # The threaded block grows through neighbouring rows as long as they stay
    # inside the ripple band (a lead-in tip, a plain shank) but stops at any
    # larger feature such as a head or a hex body.
    active_rows = np.flatnonzero(active)
    kind_row_max = np.where(kind_filled, kind_grid.max(axis=1), np.inf)
    band_rows = kind_filled & (kind_row_max <= band_hi)
    start_row, end_row = int(active_rows[0]), int(active_rows[-1])
    while start_row > 0 and band_rows[start_row - 1]:
        start_row -= 1
    while end_row + 1 < N_Z_BINS and band_rows[end_row + 1]:
        end_row += 1
    region_start = z_centers[start_row] - 0.5 * z_bin
    region_end = z_centers[end_row] + 0.5 * z_bin

    # One robust inner radius per axial row, smoothed against sparsely sampled
    # rows, so a stray near-axis vertex (a cap centre) cannot collapse a row.
    row_min = np.where(inner_filled, inner_grid.min(axis=1), np.inf)
    row_target = np.clip(_rolling_min(row_min), band_lo, band_hi)
    row_target = np.where(np.isfinite(row_target), row_target, band_hi)

    edges_z = np.linspace(along.min(), along.max(), N_Z_BINS + 1)
    z_index = np.clip(np.digitize(along, edges_z) - 1, 0, N_Z_BINS - 1)
    theta_index = np.clip(
        ((angle + np.pi) / (2.0 * np.pi) * N_THETA_BINS).astype(int), 0, N_THETA_BINS - 1
    )
    target = np.minimum(row_target[z_index], radius)

    # Only surfaces that live inside the detected ripple band are threaded
    # surfaces: everything else (heads, hex flats, outer walls) stays untouched.
    in_region = (along >= region_start) & (along <= region_end)
    in_band = (radius >= band_lo) & (radius <= band_hi)
    movable = in_region & in_band & (radius > 1e-12) & (np.abs(radius - target) > 1e-12)

    moved = int(np.count_nonzero(movable))
    report: dict[str, Any] = {
        "applied": moved > 0,
        "vertices_moved": moved,
        "axis": [round(float(value), 6) for value in axis],
        "band_radius": [round(band_lo, 6), round(band_hi, 6)],
        "target_radius": [round(float(np.min(row_target[active])), 6), round(float(np.max(row_target[active])), 6)],
        "ripple_radius": round(float(ripple), 6),
        "region_along_axis": [round(float(region_start), 6), round(float(region_end), 6)],
        "ripple_ratio": round(float(ripple / stats["radius"]), 4),
        "pitch_cv": round(float(thread["pitch_cv"]), 4),
        "pitch_pairs": int(thread["pitch_pairs"]),
        "max_vertex_move": 0.0,
    }
    if moved == 0:
        report["reason"] = "no_change"
        return mesh.copy(), report

    scale = np.ones(len(vertices))
    scale[movable] = target[movable] / radius[movable]
    new_vertices = center + np.outer(along, axis) + perp * scale[:, None]

    processed = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces.copy(), process=False)
    report["max_vertex_move"] = round(float(np.max(radius[movable] - target[movable])), 6)
    report["watertight"] = bool(processed.is_watertight)
    return processed, report


def _validate(
    original: trimesh.Trimesh, processed: trimesh.Trimesh, max_move: float
) -> Optional[str]:
    """Return a rejection reason, or None when the simplified part is acceptable."""
    if len(processed.vertices) != len(original.vertices) or len(processed.faces) != len(original.faces):
        return "topology_changed"
    if not np.isfinite(processed.vertices).all():
        return "non_finite_vertices"
    if bool(processed.is_watertight) != bool(original.is_watertight):
        return "watertightness_changed"
    lower_new, upper_new = processed.bounds
    lower_old, upper_old = original.bounds
    if (lower_new < lower_old - BOUNDS_TOL).any() or (upper_new > upper_old + BOUNDS_TOL).any():
        return "bounds_grew"
    # No vertex may travel further than the displacement the simplification
    # itself measured, so a bug cannot silently rewrite the part silhouette.
    if ((lower_new - lower_old) > max_move + BOUNDS_TOL).any():
        return "bounds_shrank_beyond_move"
    if ((upper_old - upper_new) > max_move + BOUNDS_TOL).any():
        return "bounds_shrank_beyond_move"
    if original.is_watertight:
        before = abs(float(original.volume))
        after = abs(float(processed.volume))
        if before > 0 and abs(after - before) / before > MAX_VOLUME_DELTA:
            return "volume_delta_too_large"
    return None


def _interference(parts: dict[str, trimesh.Trimesh]) -> tuple[set[frozenset[str]], bool]:
    from api.services.meshes import _overlap_pairs

    return _overlap_pairs(parts)


def _load_parts(assembly_dir: Path) -> dict[str, trimesh.Trimesh]:
    """Load every OBJ exactly the way the planners do, without applying translation.json."""
    from assets.load import load_assembly

    meshes, names = load_assembly(str(assembly_dir), translate=False, return_names=True)
    return {name.replace(".obj", ""): mesh for mesh, name in zip(meshes, names)}


def process_parts(parts: dict[str, trimesh.Trimesh]) -> tuple[dict[str, trimesh.Trimesh], dict[str, Any]]:
    """Detect and simplify threads on every part; returns the parts actually usable by planners."""
    report: dict[str, Any] = {"parts": {}, "simplified_parts": []}
    processed: dict[str, trimesh.Trimesh] = {}
    before_pairs, overlap_available = _interference(parts)

    for part_id, mesh in parts.items():
        entry: dict[str, Any] = {"detected": False, "applied": False, "watertight": bool(mesh.is_watertight)}
        try:
            thread = detect_thread(mesh)
        except Exception as exc:
            logger.warning("thread detection failed for part=%s", part_id, exc_info=True)
            entry.update(reason="detection_failed", detail=str(exc))
            processed[part_id] = mesh
            report["parts"][part_id] = entry
            continue
        if thread is None:
            entry["reason"] = "no_thread_detected"
            processed[part_id] = mesh
            report["parts"][part_id] = entry
            continue

        entry.update(
            detected=True,
            kind=thread["kind"],
            axis=[round(float(value), 6) for value in thread["axis"]],
            axis_point=[round(float(value), 6) for value in thread["axis_point"]],
            pitch=round(thread["pitch"], 6),
            confidence=round(thread["confidence"], 4),
            inliers=thread["inliers"],
            profile=thread["profile"],
        )
        if not mesh.is_watertight:
            entry.update(reason="not_watertight")
            logger.warning("part=%s has a thread candidate but is not watertight; leaving it unchanged", part_id)
            processed[part_id] = mesh
            report["parts"][part_id] = entry
            continue

        simplified, detail = _simplify(mesh, thread)
        entry.update(detail)
        if not detail.get("applied"):
            processed[part_id] = mesh
            report["parts"][part_id] = entry
            continue

        reason = _validate(mesh, simplified, float(detail.get("max_vertex_move", 0.0)))
        if reason is not None:
            entry.update(applied=False, reason=reason)
            logger.warning("thread simplification rejected for part=%s: %s", part_id, reason)
            processed[part_id] = mesh
            report["parts"][part_id] = entry
            continue

        entry["volume_delta_pct"] = round(
            100.0 * (abs(float(simplified.volume)) - abs(float(mesh.volume))) / max(abs(float(mesh.volume)), 1e-12), 3
        )
        try:
            residual = detect_thread(simplified)
        except Exception:
            logger.warning("residual thread detection failed for part=%s", part_id, exc_info=True)
            residual = None
        entry["residual_candidates"] = 1 if residual is not None else 0
        entry["applied"] = True
        processed[part_id] = simplified
        report["parts"][part_id] = entry
        report["simplified_parts"].append(part_id)

    after_pairs, _ = _interference(processed)
    if overlap_available:
        regressions = sorted(
            sorted(pair) for pair in after_pairs - before_pairs
        )
        report["interference"] = {
            "available": True,
            "before": sorted(sorted(pair) for pair in before_pairs),
            "after": sorted(sorted(pair) for pair in after_pairs),
            "regressions": regressions,
        }
        if regressions:
            logger.warning("thread simplification created new interference between %s; reverting", regressions)
            for part_id in report["simplified_parts"]:
                report["parts"][part_id].update(applied=False, reason="interference_regressed")
            report["simplified_parts"] = []
            processed = dict(parts)
    else:
        report["interference"] = {"available": False, "before": [], "after": [], "regressions": []}

    return processed, report


def _write_processed(parts: dict[str, trimesh.Trimesh], originals: dict[str, trimesh.Trimesh], assembly_dir: Path) -> Path:
    """Copy the assembly into a scratch directory and overwrite only the simplified parts."""
    scratch = Path(tempfile.mkdtemp(prefix="atlas-plan-"))
    try:
        for source in sorted(assembly_dir.iterdir()):
            if source.is_file() and source.suffix.lower() != ".sdf":
                shutil.copy2(source, scratch / source.name)
        for part_id in parts:
            if parts[part_id] is originals.get(part_id):
                continue
            parts[part_id].export(
                str(scratch / f"{part_id}.obj"), file_type="obj", header=None, include_color=False
            )
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return scratch


@contextlib.contextmanager
def planning_assembly(
    assembly_dir: Path, *, enabled: bool = True, skip_reason: Optional[str] = None
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield the directory planners should read plus the automatic thread report.

    The source assembly is never modified; a scratch copy is written only when
    at least one part was actually simplified, and it is deleted on exit.
    """
    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "source": str(assembly_dir),
        "source_modified": False,
        "simplified_parts": [],
        "parts": {},
        "warnings": [],
    }
    scratch: Optional[Path] = None
    if not enabled:
        report["reason"] = skip_reason or "disabled"
        yield assembly_dir, report
        return

    try:
        parts = _load_parts(assembly_dir)
    except ApiError as exc:
        report.update(reason="load_failed", warnings=[str(exc.message)])
        logger.warning("thread preprocessing skipped for %s: %s", assembly_dir, exc.message)
        yield assembly_dir, report
        return
    except Exception as exc:
        report.update(reason="load_failed", warnings=[str(exc)])
        logger.warning("thread preprocessing skipped for %s", assembly_dir, exc_info=True)
        yield assembly_dir, report
        return

    t_removal = time.perf_counter()
    processed, detail = process_parts(parts)
    report.update(detail)
    logger.info(
        "THREAD_REMOVAL seconds=%.3f simplified=%s",
        time.perf_counter() - t_removal,
        ",".join(detail.get("simplified_parts", [])) or "-",
    )
    changed = [part_id for part_id in report["simplified_parts"] if processed[part_id] is not parts[part_id]]
    if not changed:
        report["reason"] = report.get("reason") or "no_simplification_needed"
        yield assembly_dir, report
        return

    try:
        scratch = _write_processed(processed, parts, assembly_dir)
    except Exception as exc:
        report.update(reason="write_failed", warnings=[str(exc)], simplified_parts=[])
        logger.warning("thread preprocessing fell back to the source assembly for %s", assembly_dir, exc_info=True)
        yield assembly_dir, report
        return

    logger.info(
        "THREAD SIMPLIFICATION assembly=%s parts=%s", assembly_dir.name, ",".join(changed)
    )
    try:
        yield scratch, report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
