"""
Visualize the point cloud and point-wise instance masks.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from plyfile import PlyData, PlyElement


def visualize_two_pc(pc1, pc2, save_file):
    """
    Write point clouds of two consecutive frames into .ply file, each frame with a unique color.
    """
    n_points = pc1.shape[0] + pc2.shape[0]
    vertices = np.empty(n_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # First point cloud as green
    colors1 = np.zeros_like(pc1)
    colors1[:, 1] = 255
    # Second point cloud as blue
    colors2 = np.zeros_like(pc2)
    colors2[:, 2] = 255

    # Save
    pc = np.concatenate((pc1, pc2), 0)
    colors = np.concatenate((colors1, colors2), 0)
    vertices['x'] = pc[:, 0].astype('f4')
    vertices['y'] = pc[:, 1].astype('f4')
    vertices['z'] = pc[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(save_file)


def visualize_obj_pc(pc, obj_mask, save_file,
                     color_list=None):
    """
    Write a point cloud into .ply file, each object inside with a unique color.
    """
    assert pc.shape[0] == obj_mask.shape[0], "Mismatch between point cloud and object mask!"
    n_points = pc.shape[0]
    vertices = np.empty(n_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Generate color list by hand
    if color_list is None:
        n_objects = obj_mask.max()
        colors = plt.cm.get_cmap('hsv', n_objects)
        color_list = [(55, 55, 55)]     # Gray background
        for i in range(n_objects):
            color = (255 * np.array(colors(i)[:3])).astype(int)
            color_list.append(color)

    # First point cloud as green
    colors = np.zeros_like(pc)
    for n in range(n_points):
        colors[n] = color_list[obj_mask[n]]

    # Save
    vertices['x'] = pc[:, 0].astype('f4')
    vertices['y'] = pc[:, 1].astype('f4')
    vertices['z'] = pc[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(save_file)