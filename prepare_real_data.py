"""
Apply motion segmentation on GT/estimated scene flow.
"""

import os
import os.path as osp
import numpy as np
import pickle

from ssc import SSC

from datasets.flyingthings3d_subset import FlyingThings3DSubset
from datasets.kitti import KITTI


def save_data(points, labels, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump({
            'points': points,
            'labels': labels,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Fix randomness for debug
    np.random.seed(10)

    DATASET = 'KITTI'  # 'FT3D' / ‘KITTI'

    # Params and configurations
    if DATASET == 'FT3D':
        data_root = '/media/SSD/ziyang/Datasets/FlyingThings3D'
        save_path = 'real_data/flythings3d'
        save_path_noless01 = 'real_data/flythings3d_noless01'
        save_path_noless02 = 'real_data/flythings3d_noless02'
    elif DATASET == 'KITTI':
        data_root = '/media/SSD/ziyang/Datasets/KITTI_sceneflow_2015'
        save_path = 'real_data/kitti'
        save_path_noless01 = 'real_data/kitti_noless01'
        save_path_noless02 = 'real_data/kitti_noless02'
    else:
        raise ValueError('Unrecognized dataset!')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_noless01, exist_ok=True)
    os.makedirs(save_path_noless02, exist_ok=True)

    if DATASET == 'FT3D':
        dataset = FlyingThings3DSubset(train=False,
                                       data_root=data_root)
        n_scenes = 200    # For Flythings3D, use first 200 scenes for reference
    elif DATASET == 'KITTI':
        dataset = KITTI(train=False,
                        data_root=data_root)
        n_scenes = len(dataset)


    # Subsample points before SSC to speed up
    DOWNSAMPLE_POINTS = 1024

    for sid in range(n_scenes):
        pc1, pc2, labels, src_data_path = dataset[sid]
        n_points = pc1.shape[0]
        scene_name = src_data_path.split('/')[-1]
        print (scene_name)

        # Subsample original point cloud for speed-up
        sampled_indices = np.random.choice(n_points, DOWNSAMPLE_POINTS, replace=False, p=None)
        pc1_sampled = pc1[sampled_indices]
        pc2_sampled = pc2[sampled_indices]
        labels_sampled = labels[sampled_indices]
        points_sampled = np.concatenate((pc1_sampled, pc2_sampled), 1)

        # Save all subsampled data
        save_file = os.path.join(save_path, scene_name+'.pkl')
        save_data(points_sampled, labels_sampled, save_file)

        # Remove objects occupying few points in the scene
        cluster_ids, count_per_cluster = np.unique(labels_sampled, return_counts=True)
        ratio_per_cluster = count_per_cluster / DOWNSAMPLE_POINTS

        cluster_ids_noless01 = cluster_ids[ratio_per_cluster > 0.01]
        point_idx_noless01 = np.in1d(labels_sampled, cluster_ids_noless01)
        points_sampled_noless01 = points_sampled[point_idx_noless01]
        labels_sampled_noless01 = labels_sampled[point_idx_noless01]
        save_file_noless01 = os.path.join(save_path_noless01, scene_name+'.pkl')
        save_data(points_sampled_noless01, labels_sampled_noless01, save_file_noless01)

        cluster_ids_noless02 = cluster_ids[ratio_per_cluster > 0.02]
        point_idx_noless02 = np.in1d(labels_sampled, cluster_ids_noless02)
        points_sampled_noless02 = points_sampled[point_idx_noless02]
        labels_sampled_noless02 = labels_sampled[point_idx_noless02]
        save_file_noless02 = os.path.join(save_path_noless02, scene_name+'.pkl')
        save_data(points_sampled_noless02, labels_sampled_noless02, save_file_noless02)