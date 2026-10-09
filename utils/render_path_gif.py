'''
Render a saved path.json animation into a GIF.

path.json is the artifact written by assets/save.py:
    [{"name": "<obj_id>", "matrix": [[...4x4...]}, ...]
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import argparse
import json
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from assets.load import load_assembly
from assets.save import interpolate_path


def _face_colors(mesh):
    colors = np.asarray(mesh.visual.face_colors)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(mesh.faces), 1))
    if colors.shape[1] == 4 and colors.max() > 1.0:
        colors = colors / 255.0
    return colors[:, :3]


def load_frames(path_json):
    with open(path_json, 'r') as fp:
        return json.load(fp)


def render_gif(assembly_dir, path_json, output, n_frame=None, fps=10, dpi=96, elev=22, azim=-60):
    frames = load_frames(path_json)
    if not frames:
        raise ValueError(f'{path_json} contains no frames')

    meshes, names = load_assembly(assembly_dir, return_names=True)
    mesh_by_id = {name.replace('.obj', ''): mesh for mesh, name in zip(meshes, names)}
    move_id = str(frames[0]['name'])
    if move_id not in mesh_by_id:
        raise ValueError(f'object "{move_id}" is not part of {assembly_dir}')

    sampled = interpolate_path(frames, n_frame)
    move_mesh = mesh_by_id[move_id]

    # one vertex cloud per frame, so the camera can stay fixed
    frame_vertices = []
    for frame in sampled:
        vertices = []
        for obj_id, mesh in mesh_by_id.items():
            if obj_id == str(frame['name']):
                vertices.append(transform_vertices(mesh.vertices, frame['matrix']))
            else:
                vertices.append(mesh.vertices)
        frame_vertices.append(np.vstack(vertices))

    all_vertices = np.vstack(frame_vertices)
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.max(upper - lower)) / 2.0 * 1.15 + 1e-6
    limits = [(c - radius, c + radius) for c in center]

    move_tris = move_mesh.vertices[move_mesh.faces] # (n, 3, 3)
    move_colors = _face_colors(move_mesh)
    still_tris = []
    still_colors = []
    for obj_id, mesh in mesh_by_id.items():
        if obj_id == move_id:
            continue
        still_tris.append(mesh.vertices[mesh.faces])
        still_colors.append(_face_colors(mesh))
    still_tris = np.vstack(still_tris) if still_tris else None
    still_colors = np.vstack(still_colors) if still_colors else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        png_paths = []
        for index, (frame, vertices) in enumerate(zip(sampled, frame_vertices)):
            fig = plt.figure(figsize=(6, 6), dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            if still_tris is not None:
                ax.add_collection3d(Poly3DCollection(still_tris, facecolors=still_colors, edgecolors='none'))
            moved_tris = transform_vertices(move_tris.reshape(-1, 3), frame['matrix']).reshape(-1, 3, 3)
            ax.add_collection3d(Poly3DCollection(moved_tris, facecolors=move_colors, edgecolors='none'))

            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_zlim(*limits[2])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(f'{len(sampled)} frames | object {frame["name"]} | frame {index + 1}', fontsize=9)

            png_path = os.path.join(tmp_dir, f'{index:04d}.png')
            # no bbox_inches='tight': every frame must keep the same pixel size
            fig.savefig(png_path, facecolor='white')
            plt.close(fig)
            png_paths.append(png_path)

        images = [Image.open(path).convert('RGB') for path in png_paths]
        size = images[0].size
        images = [image if image.size == size else image.resize(size) for image in images]
        # one shared palette: per-frame palettes make GIF diffs flicker
        palette = images[0].convert('P', palette=Image.ADAPTIVE, colors=256)
        images = [image.quantize(palette=palette, dither=0) for image in images]
        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
            optimize=False,
        )
    return output, len(images)


def transform_vertices(vertices, matrix):
    points = np.hstack([np.asarray(vertices, dtype=float), np.ones((len(vertices), 1))])
    return (np.asarray(matrix, dtype=float) @ points.T).T[:, :3]


def main():
    parser = argparse.ArgumentParser(description='Render path.json to GIF')
    parser.add_argument('--assembly-dir', required=True, help='directory holding the assembly .obj parts')
    parser.add_argument('--path-json', required=True, help='path.json written by assets/save.py')
    parser.add_argument('--output', required=True, help='output .gif path')
    parser.add_argument('--n-frame', type=int, default=40, help='max number of frames to render')
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--elev', type=float, default=22)
    parser.add_argument('--azim', type=float, default=-60)
    args = parser.parse_args()

    output, count = render_gif(
        args.assembly_dir, args.path_json, args.output,
        n_frame=args.n_frame, fps=args.fps, elev=args.elev, azim=args.azim,
    )
    print(f'Saved {count} frames to {output}')


if __name__ == '__main__':
    main()
