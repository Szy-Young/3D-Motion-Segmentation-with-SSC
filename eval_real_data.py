"""
Apply motion segmentation on GT/estimated scene flow.
"""

import os
import os.path as osp
import numpy as np
import pickle
import argparse

from ssc import SSC
from ssc_utils.eval import ss_recovery_error, best_map, accumulate_AP, calculate_AP

from visualize.visualize_flythings3d import visualize_obj_pc


parser = argparse.ArgumentParser(description='SSC on toy data.')

parser.add_argument('--data_path', type=str, default='real_data/kitti')
parser.add_argument('--affine', dest='affine', default=False, action='store_true')
parser.add_argument('--n_cluster', dest='n_cluster', default=False, action='store_true')
parser.add_argument('--proj_dim', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='Lasso')
parser.add_argument('--alpha', type=float, default='800')
parser.add_argument('--top_k', type=int, default=4)
parser.add_argument('--min_n_cluster', type=int, default=2)
parser.add_argument('--max_n_cluster', type=int, default=10)
parser.add_argument('--normalize', dest='normalize', default=False, action='store_true')

args = parser.parse_args()


def preproc_data(points, affine=True, normalize=True):
    """
    De-centralize the data in each frame to make the scale of problem more friendly.
    Normalize the data onto a sphere if needed.
    """
    n_frames = points.shape[0] // 3
    points = np.split(points, n_frames, axis=0)
    points_proc = []
    for points_t in points:
        points_t = points_t - np.mean(points_t, 1, keepdims=True)
        points_proc.append(points_t)
    points = np.concatenate(points_proc, 0)

    # SSC on augmented linear subsapce
    if not affine:
        points = np.concatenate((points, np.ones((1, points.shape[1]))), 0)

    # Normalize to (-1, 1)
    if normalize:
        if affine:
            scale = np.absolute(points).max()
            points = points / scale
        else:
            points = points / (np.linalg.norm(points, axis=0) + 1e-10)

    return points


if __name__ == '__main__':
    # Fix randomness for debug
    np.random.seed(10)

    DATA_PATH = args.data_path

    VIS_PATH = os.path.join('eval_results', DATA_PATH.split('/')[-1])
    os.makedirs(VIS_PATH, exist_ok=True)

    # Hyperparams for SSC
    AFFINE = args.affine
    N_CLUSTER = args.n_cluster  # Not given number of clusters and estimate it self-adaptively
    PROJ_DIM = args.proj_dim  # No projection for dimension reduction
    OPTIMIZER = args.optimizer
    ALPHA = args.alpha
    TOP_K = args.top_k
    MIN_N_CLUSTER = args.min_n_cluster
    MAX_N_CLUSTER = args.max_n_cluster
    NORMALIZE = args.normalize
    OUTLIER_LABEL = 999

    print (args)


    TP, FP, FN = [], [], []
    COEF_ERROR = []
    POINT_ERROR = []

    data_files = sorted(os.listdir(DATA_PATH))
    for data_file in data_files:
        with open(os.path.join(DATA_PATH, data_file), 'rb') as f:
            data = pickle.load(f)
        points = data['points'].T
        labels = data['labels']

        # Maintain 1st-frame point cloud for later visualization
        pc1 = points[:3].T

        if N_CLUSTER:
            n_cluster = np.unique(labels).shape[0]
        else:
            n_cluster = 0

        # Data preprocessing
        points = preproc_data(points,
                              affine=AFFINE,
                              normalize=NORMALIZE)

        if AFFINE:
            labels_est, coef_mat, outlier_idx = SSC(points,
                                                    n_cluster=n_cluster,
                                                    proj_dim=PROJ_DIM,
                                                    affine=1,
                                                    optimizer=OPTIMIZER,
                                                    alpha=ALPHA,
                                                    top_k=TOP_K,
                                                    min_n_cluster=MIN_N_CLUSTER,
                                                    max_n_cluster=MAX_N_CLUSTER,
                                                    outlier_label=OUTLIER_LABEL)
        else:
            labels_est, coef_mat, outlier_idx = SSC(points,
                                                    n_cluster=n_cluster,
                                                    proj_dim=PROJ_DIM,
                                                    affine=0,
                                                    optimizer=OPTIMIZER,
                                                    alpha=ALPHA,
                                                    top_k=TOP_K,
                                                    min_n_cluster=MIN_N_CLUSTER,
                                                    max_n_cluster=MAX_N_CLUSTER,
                                                    outlier_label=OUTLIER_LABEL)

        # Count subspace-sparse recovery array
        coef_error = ss_recovery_error(coef_mat, labels, outlier_idx)
        COEF_ERROR.append(coef_error)

        # Count point-wise clustering error
        labels_est = best_map(labels, labels_est)
        point_error = float(np.sum(labels != labels_est)) / labels.shape[0]
        POINT_ERROR.append(point_error)

        tp, fp, fn = accumulate_AP(labels, labels_est, iou_thresh=0.5, outlier_idx=outlier_idx)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)

        print (data_file, coef_error, point_error, tp, fp, fn)

        # Visualize sampled point cloud
        labels_est = labels_est.astype(int)
        visualize_obj_pc(pc1, labels, save_file=osp.join(VIS_PATH, data_file[:-4]+'.ply'))
        visualize_obj_pc(pc1, labels_est, save_file=osp.join(VIS_PATH, data_file[:-4]+'_est.ply'))


    # Accumulate
    coef_error = np.mean(COEF_ERROR)
    point_error = np.mean(POINT_ERROR)
    print ('Subspace-sparse recovery error:', coef_error)
    print ('Point-wise clustering error:', point_error)

    TP, FP, FN = np.sum(TP), np.sum(FP), np.sum(FN)
    prec, recall, AP = calculate_AP(TP, FP, FN)
    print ('Precision@0.5:', prec)
    print ('Recall@0.5:', recall)
    print ('AP@0.5:', AP)

    print (args)