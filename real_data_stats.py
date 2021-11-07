"""
Apply motion segmentation on GT/estimated scene flow.
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from datasets.flyingthings3d_subset import FlyingThings3DSubset
from datasets.kitti import KITTI


def count_object_points(dataset, n_scenes=None):
    if n_scenes is None:
        n_scenes = len(dataset)

    obj_point_count = []
    obj_point_ratio = []
    for sid in range(n_scenes):
        pc1, pc2, labels, src_data_path = dataset[sid]
        n_points = labels.shape[0]
        # print (src_data_path)

        _, count_per_obj = np.unique(labels, return_counts=True)
        ratio_per_obj = count_per_obj / n_points
        obj_point_count.append(count_per_obj)
        obj_point_ratio.append(ratio_per_obj)

    obj_point_count = np.sort(np.concatenate(obj_point_count))
    obj_point_ratio = np.sort(np.concatenate(obj_point_ratio))

    return obj_point_count, obj_point_ratio


if __name__ == '__main__':
    # Fix randomness for debug
    np.random.seed(10)

    DATASET = 'KITTI'  # 'FT3D' / ‘KITTI'

    # Params and configurations
    if DATASET == 'FT3D':
        data_root = '/media/SSD/ziyang/Datasets/FlyingThings3D'
        vis_stats_path = 'real_data_stats/flythings3d'
    elif DATASET == 'KITTI':
        data_root = '/media/SSD/ziyang/Datasets/KITTI_sceneflow_2015'
        vis_stats_path = 'real_data_stats/kitti'
    else:
        raise ValueError('Unrecognized dataset!')
    os.makedirs(vis_stats_path, exist_ok=True)

    if DATASET == 'FT3D':
        dataset = FlyingThings3DSubset(train=False,
                                       data_root=data_root)
        n_scenes = 200  # For Flythings3D, use first 200 scenes for reference
    elif DATASET == 'KITTI':
        dataset = KITTI(train=False,
                        data_root=data_root)
        n_scenes = len(dataset)


    # Statistics about number of points per object
    obj_point_count, obj_point_ratio = count_object_points(dataset, n_scenes)
    plt.plot(obj_point_count)
    plt.savefig(os.path.join(vis_stats_path, 'object_point_count.png'))
    plt.clf()
    plt.plot(obj_point_ratio)
    plt.savefig(os.path.join(vis_stats_path, 'object_point_ratio.png'))

    for thresh in [0.01, 0.02, 0.05, 0.1]:
        print ((obj_point_ratio > thresh).astype(float).sum() / obj_point_ratio.shape[0])