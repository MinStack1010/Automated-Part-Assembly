import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from api.services.artifacts import build_artifact_zip

ASSEMBLY_DIR = Path(__file__).resolve().parent.parent / "assets" / "my_parts"


class ArtifactZipTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self._tmp.name)
        path_dir = self.artifact_dir / "path"
        path_dir.mkdir()
        frames = [
            {"name": "screw", "matrix": np.eye(4).tolist()},
            {"name": "screw", "matrix": np.eye(4).tolist()},
        ]
        frames[1]["matrix"][0][3] = 5.0
        (path_dir / "path.json").write_text(json.dumps(frames))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_zip_layout_converts_json_to_npy(self) -> None:
        payload = build_artifact_zip(self.artifact_dir)

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            self.assertIn("artifact/npy/", names)
            self.assertIn("artifact/json/", names)
            self.assertIn("artifact/gif/", names)
            self.assertIn("artifact/npy/path.npy", names)
            self.assertIn("artifact/json/path.json", names)
            with archive.open("artifact/npy/path.npy") as handle:
                array = np.load(io.BytesIO(handle.read()))

        self.assertEqual(array.shape, (2, 4, 4))
        self.assertEqual(float(array[1][0][3]), 5.0)

    def test_gif_is_rendered_and_cached_with_assembly(self) -> None:
        payload = build_artifact_zip(self.artifact_dir, ASSEMBLY_DIR)

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("artifact/gif/screw.gif", archive.namelist())
        self.assertTrue((self.artifact_dir / "gif" / "screw.gif").is_file())

        # second build reuses the cached gif
        payload = build_artifact_zip(self.artifact_dir, ASSEMBLY_DIR)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("artifact/gif/screw.gif", archive.namelist())


if __name__ == "__main__":
    unittest.main()
