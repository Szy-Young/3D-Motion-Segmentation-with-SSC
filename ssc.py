"""
Test SSC on a toy problem with multiple rigid transforms (affine subspaces).
"""

import os
import pickle
import numpy as np

from ssc_utils.sparse_optim import project_data, sparse_optim, detect_outlier
from ssc_utils.spectral_cluster import build_adjacency, selftune_spectral_cluster


def SSC(data, n_cluster=0, proj_dim=0, affine=1, optimizer='Lasso', alpha=800, top_k=4, min_n_cluster=2, max_n_cluster=10, outlier_label=999):
    """
    A complete Sparse Subspace Clustering (SSC) pipeline.
    :param n_cluster: Number of clusters (if known).
    :param proj_dim: Dimension to project original data, 0 if not project.
    :param affine: Whether apply affine constraint.
    :param optimizer: One of {'L1Perfect','L1Noise','Lasso','L1ED'}.
    :param alpha: Hyperparam to control Noise-term in 'Lasso'.
    :param top_k: Number of top-K efficients to build the similarity graph, 0 if use the whole coefficients.
    :param min_n_cluster: Minimum number of clusters for Self-Tuning Spectral Clustering search.
    :param max_n_cluster: Maximum number of clusters for Self-Tuning Spectral Clustering search.
    :param outlier_label: Label id used to mark outliers (prefer a large integer).
    """

    # Dimension reduction if needed
    if proj_dim > 0:
        data = project_data(data, proj_dim, 'NormalProj')

    if optimizer == 'Lasso':
        # Self-adaptive lambda
        mu = np.absolute(data.T @ data)
        mu = mu - np.diag(np.diag(mu))
        mu = np.amax(mu, 1).min()
        lmbda = alpha / mu
        coef_mat = sparse_optim(data, affine, optimizer, lmbda)
    else:
        coef_mat = sparse_optim(data, affine, optimizer)

    # Make small values to 0 and find outliers
    eps = np.finfo(float).eps
    coef_mat[np.abs(coef_mat) < eps] = 0
    coef_mat, outlier_idx = detect_outlier(coef_mat)

    # Remove outliers from final clustering
    coef_mat_proc = np.delete(coef_mat, outlier_idx, 0)
    coef_mat_proc = np.delete(coef_mat_proc, outlier_idx, 1)

    # Spectral clustering
    affn_mat = build_adjacency(coef_mat_proc, top_k)
    if n_cluster > 0:
        labels_est = selftune_spectral_cluster(affn_mat, numC=n_cluster)
    else:
        labels_est = selftune_spectral_cluster(affn_mat, minC=min_n_cluster, maxC=max_n_cluster)

    # Add outliers back to clustering results with specified id
    n_points = data.shape[1]
    labels_est_full = np.zeros(n_points, dtype=int)
    labels_est_full[outlier_idx] = outlier_label
    outlier_idx_aug = [-1] + outlier_idx + [n_points]
    for i in range(len(outlier_idx_aug)-1):
        start_idx = outlier_idx_aug[i] + 1
        end_idx = outlier_idx_aug[i+1]
        labels_est_full[start_idx:end_idx] = labels_est[(start_idx-i):(end_idx-i)]

    return labels_est_full, coef_mat, outlier_idx