"""Service-level tests for clearance diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from api.errors import ApiError
from api.services.meshes import mesh_gap

COLLECTION = "mesh_test"
ASSEMBLY_ID = "00001"


def _box(offset: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box()
    mesh.apply_translation([offset, 0.0, 0.0])
    return mesh


class MeshServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.assembly_dir = root / COLLECTION / ASSEMBLY_ID
        self.assembly_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, mesh: trimesh.Trimesh) -> None:
        mesh.export(str(self.assembly_dir / f"{name}.obj"), file_type="obj", header=None, include_color=False)

    def test_gap_flags_overlapping_parts_and_raises_the_threshold(self) -> None:
        self._write("0", _box(0.0))
        self._write("1", _box(0.5))

        report = mesh_gap(self.assembly_dir)

        self.assertTrue(report["overlapping"])
        self.assertTrue(report["touching"])
        self.assertEqual(report["suggested_collision_threshold"], 0.1)
        self.assertEqual(len(report["pairs"]), 1)

    def test_gap_accepts_separated_parts(self) -> None:
        self._write("0", _box(0.0))
        self._write("1", _box(4.0))

        report = mesh_gap(self.assembly_dir, move_id="0", still_ids=["1"])

        self.assertEqual(report["mode"], "move")
        self.assertFalse(report["touching"])
        self.assertEqual(report["suggested_collision_threshold"], 0.01)
        self.assertEqual(report["pairs"][0]["gap"], 3.0)

    def test_gap_rejects_unknown_parts(self) -> None:
        self._write("0", _box(0.0))
        self._write("1", _box(4.0))

        with self.assertRaises(ApiError):
            mesh_gap(self.assembly_dir, move_id="7")

    def test_gap_rejects_large_all_pairs_assemblies(self) -> None:
        for index in range(11):
            self._write(f"{index}", _box(float(index)))

        with self.assertRaises(ApiError):
            mesh_gap(self.assembly_dir)


def _rippled_cylinder(radius: float, height: float, radial: int, axial: int, z0: float) -> trimesh.Trimesh:
    """Thread-like dense mesh: radius ripples along the axis (no fixture files needed)."""
    angles = np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)
    heights = np.linspace(0.0, height, axial)
    ripple = 0.15 * np.sin(heights * 2.0 * np.pi / 1.25)
    vertices = []
    for zi, h in enumerate(heights):
        r = radius + ripple[zi]
        for a in angles:
            vertices.append([r * np.cos(a), r * np.sin(a), z0 + h])
    vertices = np.asarray(vertices)
    faces = []
    for zi in range(axial - 1):
        for ai in range(radial):
            a0 = zi * radial + ai
            a1 = zi * radial + (ai + 1) % radial
            b0 = (zi + 1) * radial + ai
            b1 = (zi + 1) * radial + (ai + 1) % radial
            faces.extend([[a0, b0, b1], [a0, b1, a1]])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


class DenseMeshGapTests(unittest.TestCase):
    """Regression: gap diagnostics must not OOM-kill the worker on dense threaded meshes.

    The rtree-backed trimesh proximity fallback exhausts memory (SIGKILL/-9) on
    100k+ vertex meshes; the FCL BVH path must be used instead.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.assembly_dir = Path(self._tmp.name)
        # Coaxial rippled pair with 0.4 radial clearance (outer r=4, inner r=3.4).
        outer = _rippled_cylinder(4.0, 8.0, 96, 96, 0.0)
        inner = _rippled_cylinder(3.4, 8.0, 96, 96, 0.0)
        outer.export(str(self.assembly_dir / "outer.obj"))
        inner.export(str(self.assembly_dir / "inner.obj"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dense_threaded_pair_returns_finite_gap(self) -> None:
        import time

        started = time.perf_counter()
        report = mesh_gap(self.assembly_dir, move_id="inner", still_ids=["outer"])
        elapsed = time.perf_counter() - started
        self.assertFalse(report["overlapping"])
        self.assertTrue(np.isfinite(report["gap"]))
        self.assertGreaterEqual(report["gap"], 0.0)
        # Must stay far below the multi-minute OOM blowup of the old path.
        self.assertLess(elapsed, 120.0)

    def test_proximity_sampling_is_capped(self) -> None:
        from api.services.meshes import _MAX_PROXIMITY_POINTS, _sampled_vertices

        big = trimesh.Trimesh(
            vertices=np.zeros((_MAX_PROXIMITY_POINTS + 5000, 3)),
            faces=np.zeros((1, 3), dtype=np.int64),
            process=False,
        )
        self.assertLessEqual(len(_sampled_vertices(big)), _MAX_PROXIMITY_POINTS)


if __name__ == "__main__":
    unittest.main()
