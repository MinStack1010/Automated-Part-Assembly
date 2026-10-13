"""Service-level tests for clearance diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
