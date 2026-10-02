"""Unit tests for multi-plan motion artifact consolidation."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from api.services.planning import _consolidate_motion_artifacts


def _write_frame(step_dir: Path, frame: int, value: float) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    np.save(step_dir / f"{frame}.npy", np.eye(4) * value)


def _artifacts(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


class ConsolidateMotionArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_merges_attempt_folders_into_a_single_path_sequence(self) -> None:
        assembly_dir = self.root / "8df9e4ab-09f5-43c2-9c3e-3df242239b63"
        _write_frame(assembly_dir / "0_cylinder", 0, 1.0)
        _write_frame(assembly_dir / "0_cylinder", 1, 2.0)
        _write_frame(assembly_dir / "2_peg", 0, 3.0)
        _write_frame(assembly_dir / "2_peg", 1, 4.0)
        _write_frame(assembly_dir / "1_cylinder", 0, 5.0)

        motion = _consolidate_motion_artifacts(self.root)

        self.assertEqual(motion["artifact_directory"], "path")
        self.assertEqual(motion["path_state_count"], 5)
        self.assertEqual(
            motion["path_parts"],
            [
                {"part_id": "cylinder", "attempt": 0, "frame_start": 0, "frame_end": 1},
                {"part_id": "cylinder", "attempt": 1, "frame_start": 2, "frame_end": 2},
                {"part_id": "peg", "attempt": 2, "frame_start": 3, "frame_end": 4},
            ],
        )
        self.assertEqual(_artifacts(self.root), [f"path/{frame}.npy" for frame in range(5)])
        np.testing.assert_allclose(np.load(self.root / "path" / "2.npy"), np.eye(4) * 5.0)
        np.testing.assert_allclose(np.load(self.root / "path" / "3.npy"), np.eye(4) * 3.0)

    def test_flat_attempt_folders_are_removed_without_touching_path(self) -> None:
        _write_frame(self.root / "0_peg", 0, 1.0)
        _write_frame(self.root / "1_peg", 0, 2.0)

        motion = _consolidate_motion_artifacts(self.root)

        self.assertEqual(motion["path_state_count"], 2)
        self.assertEqual(_artifacts(self.root), ["path/0.npy", "path/1.npy"])
        self.assertTrue((self.root / "path").is_dir())

    def test_ignores_folders_that_are_not_motion_exports(self) -> None:
        (self.root / "multi_assembly").mkdir()
        (self.root / "0_peg" / "notes").mkdir(parents=True)
        (self.root / "0_peg" / "notes" / "readme.txt").write_text("not a frame")

        self.assertEqual(_consolidate_motion_artifacts(self.root), {})
        self.assertEqual(_artifacts(self.root), ["0_peg/notes/readme.txt"])


if __name__ == "__main__":
    unittest.main()
