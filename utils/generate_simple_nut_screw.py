"""Generate the simplified demo nut/screw pair in assets/my_parts.

Task-2 rationale: the original threaded pair (22k/10k verts, helical flanks)
exists to exercise automatic thread removal, but the default demo needs a
stable, low-risk assembly. This script builds, from scratch:

- screw: a lathe (shaft r=3 z 0..30 + head r=6 z 30..34, watertight, genus 0)
- nut: a hex prism (circumradius 7, z 10..17.5) with a smooth round
  through-hole (r=3.4, watertight, genus 1) coaxial with the screw shaft.

Radial clearance is 0.4, so the pair loads without interpenetration and a
straight pull along -z is collision-free without any thread preprocessing.
No translation.json is used; both parts are authored in final coordinates,
exactly like the meshes they replace.

Usage (inside the project container, which has trimesh)::
    python utils/generate_simple_nut_screw.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(project_base_dir)

import trimesh

MY_PARTS = os.path.join(project_base_dir, "assets", "my_parts")

# Shared dimensions (model units, Z is the assembly axis).
SHAFT_R = 3.0
SHAFT_TOP = 30.0
HEAD_R = 6.0
HEAD_TOP = 34.0
NUT_FLAT_R = 7.0  # hex circumradius
NUT_Z0, NUT_Z1 = 10.0, 17.5
HOLE_R = 3.4  # > SHAFT_R: 0.4 radial clearance, no interpenetration
SECTIONS = 48


def build_screw() -> trimesh.Trimesh:
    profile = np.array(
        [
            [0.0, 0.0],
            [SHAFT_R, 0.0],
            [SHAFT_R, SHAFT_TOP],
            [HEAD_R, SHAFT_TOP],
            [HEAD_R, HEAD_TOP],
            [0.0, HEAD_TOP],
        ]
    )
    mesh = trimesh.creation.revolve(profile, sections=SECTIONS)
    return mesh


def build_nut() -> trimesh.Trimesh:
    n_hex, n_hole = 6, 24
    hex_angles = np.linspace(0.0, 2.0 * np.pi, n_hex, endpoint=False)
    hole_angles = np.linspace(0.0, 2.0 * np.pi, n_hole, endpoint=False)
    outer = np.stack([NUT_FLAT_R * np.cos(hex_angles), NUT_FLAT_R * np.sin(hex_angles)], axis=1)
    inner = np.stack([HOLE_R * np.cos(hole_angles), HOLE_R * np.sin(hole_angles)], axis=1)

    # Vertex layout: outer-bottom(0:6), outer-top(6:12), inner-bottom(12:36), inner-top(36:60).
    bottom = np.hstack([np.vstack([outer, inner]), np.full((n_hex + n_hole, 1), NUT_Z0)])
    top = np.hstack([np.vstack([outer, inner]), np.full((n_hex + n_hole, 1), NUT_Z1)])
    vertices = np.vstack([bottom, top])
    ob, ib = 0, n_hex
    ot, it = n_hex + n_hole, 2 * n_hex + n_hole
    faces: list[list[int]] = []

    # Outer walls.
    for j in range(n_hex):
        j2 = (j + 1) % n_hex
        faces.append([ob + j, ot + j2, ot + j])
        faces.append([ob + j, ob + j2, ot + j2])
    # Inner walls.
    for i in range(n_hole):
        i2 = (i + 1) % n_hole
        faces.append([ib + i, it + i, it + i2])
        faces.append([ib + i, it + i2, ib + i2])
    # Top annulus (+z): fan each hex sector from its leading corner across
    # the sector's hole vertices.
    per = n_hole // n_hex
    for j in range(n_hex):
        j2 = (j + 1) % n_hex
        ring = [(j * per + k) % n_hole for k in range(per + 1)]
        corner = ot + j
        for k in range(per):
            faces.append([corner, it + ring[k + 1], it + ring[k]])
        faces.append([corner, ot + j2, it + ring[per]])
    # Bottom annulus (-z): mirrored fan.
    for j in range(n_hex):
        j2 = (j + 1) % n_hex
        ring = [(j * per + k) % n_hole for k in range(per + 1)]
        corner = ob + j
        for k in range(per):
            faces.append([corner, ib + ring[k], ib + ring[k + 1]])
        faces.append([corner, ib + ring[per], ob + j2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    # Repair-consistent outward normals: orient all faces uniformly, then pick
    # the sign with positive signed volume.
    trimesh.repair.fix_winding(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def main() -> None:
    os.makedirs(MY_PARTS, exist_ok=True)
    screw = build_screw()
    nut = build_nut()
    for name, mesh in (("screw", screw), ("nut", nut)):
        assert mesh.is_watertight, f"{name} is not watertight"
        assert np.isfinite(mesh.volume) and mesh.volume > 0, f"{name} has no volume"
        path = os.path.join(MY_PARTS, f"{name}.obj")
        mesh.export(path)
        print(f"wrote {path}: verts={len(mesh.vertices)} faces={len(mesh.faces)} "
              f"watertight={mesh.is_watertight} euler={mesh.euler_number}")
    print(f"screw bounds={screw.bounds.tolist()}")
    print(f"nut bounds={nut.bounds.tolist()}")


if __name__ == "__main__":
    main()
