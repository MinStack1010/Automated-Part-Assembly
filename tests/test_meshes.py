"""Service-level tests for mesh repair, preprocessing, and clearance diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

from api.config import Settings, settings
from api.errors import ApiError
from api.schemas import AssemblyRef
from api.services.meshes import mesh_gap, preprocess_assembly

COLLECTION = "mesh_test"
ASSEMBLY_ID = "00001"


def _open_sphere() -> trimesh.Trimesh:
    """An icosphere with faces removed so it is no longer watertight."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.update_faces(np.arange(len(mesh.faces)) < 250)
    mesh.remove_unreferenced_vertices()
    assert not mesh.is_watertight
    return mesh


def _box(offset: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box()
    mesh.apply_translation([offset, 0.0, 0.0])
    return mesh


class MeshServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.config: Settings = replace(
            settings, assets_dir=root / "assets", storage_dir=root / "storage"
        )
        self.config.uploads_dir.mkdir(parents=True)
        self.config.jobs_dir.mkdir(parents=True)
        self.assembly_dir = self.config.assets_dir / COLLECTION / ASSEMBLY_ID
        self.assembly_dir.mkdir(parents=True)
        self.ref = AssemblyRef(collection=COLLECTION, assembly_id=ASSEMBLY_ID)

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

    def test_preprocess_repairs_writes_and_reports(self) -> None:
        self._write("0", _open_sphere())
        self._write("1", _box(4.0))
        (self.assembly_dir / "translation.json").write_text(json.dumps({"0": [1.0, 2.0, 3.0]}))

        result = preprocess_assembly(self.ref, config=self.config)

        self.assertTrue(result["all_watertight"])
        self.assertTrue(result["parts"][0]["watertight_before"] is False)
        self.assertTrue(result["parts"][0]["repaired"])
        self.assertTrue(result["parts"][0]["watertight_after"])

        upload_id = result["output"]["source"]["upload_id"]
        output_dir = self.config.uploads_dir / upload_id
        self.assertEqual(result["output"]["part_count"], 2)
        self.assertTrue((output_dir / "translation.json").is_file())
        self.assertNotEqual(json.loads((output_dir / "translation.json").read_text())["0"], [1.0, 2.0, 3.0])
        for part_id in ("0", "1"):
            written = trimesh.load_mesh(str(output_dir / f"{part_id}.obj"), process=False, maintain_order=True)
            self.assertTrue(written.is_watertight, f"{part_id} was not written watertight")
        self.assertIsNotNone(result["gap_after"])
        self.assertEqual(self.ref.model_dump(mode="json"), result["source"])

    def test_preprocess_without_repair_keeps_the_mesh_open(self) -> None:
        self._write("0", _open_sphere())
        self._write("1", _box(4.0))

        result = preprocess_assembly(self.ref, repair=False, normalize=False, config=self.config)

        self.assertFalse(result["all_watertight"])
        self.assertIn("warning", result)
        self.assertFalse(result["parts"][0]["repaired"])

    def test_preprocess_shrinks_radially_and_reports_straight_pull(self) -> None:
        self._write("0", _box(0.0))
        self._write("1", _box(5.0))

        result = preprocess_assembly(self.ref, shrink_parts={"0": 0.5}, config=self.config)

        self.assertEqual(result["actions"]["shrink_parts"], {"0": 0.5})
        self.assertEqual(result["parts"][0]["radial_scale"], 0.5)
        pull = result["straight_pull"]["0"]
        self.assertTrue(pull["free"])
        self.assertIsNotNone(pull["direction"])
        upload_id = result["output"]["source"]["upload_id"]
        written = trimesh.load_mesh(
            str(self.config.uploads_dir / upload_id / "0.obj"), process=False, maintain_order=True
        )
        # radial shrink halves x/y only; z keeps the normalized length
        self.assertAlmostEqual(float(written.extents[0]), 0.5 * float(written.extents[2]), places=4)
        self.assertAlmostEqual(float(written.extents[1]), 0.5 * float(written.extents[2]), places=4)

    def test_preprocess_rejects_invalid_shrink_parts(self) -> None:
        self._write("0", _box(0.0))
        self._write("1", _box(5.0))

        with self.assertRaises(ApiError):
            preprocess_assembly(self.ref, shrink_parts={"0": 1.5}, config=self.config)
        with self.assertRaises(ApiError):
            preprocess_assembly(self.ref, shrink_parts={"missing": 0.9}, config=self.config)
        with self.assertRaises(ApiError):
            preprocess_assembly(self.ref, verify_part_id="missing", config=self.config)

    def test_preprocess_subdivides_into_the_new_upload(self) -> None:
        self._write("0", _box(0.0))
        self._write("1", _box(4.0))

        result = preprocess_assembly(self.ref, subdivide=True, max_edge=0.4, config=self.config)

        upload_id = result["output"]["source"]["upload_id"]
        written = trimesh.load_mesh(str(self.config.uploads_dir / upload_id / "0.obj"), process=False, maintain_order=True)
        self.assertGreater(len(written.faces), 12)
        self.assertTrue(written.is_watertight)

    def test_gap_rejects_large_all_pairs_assemblies(self) -> None:
        for index in range(11):
            self._write(f"{index}", _box(float(index)))

        with self.assertRaises(ApiError):
            mesh_gap(self.assembly_dir)


if __name__ == "__main__":
    unittest.main()
