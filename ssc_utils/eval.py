import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(A):
    _, col_ind = linear_sum_assignment(A)
    # Cost can be found as A[row_ind, col_ind].sum()
    return col_ind


def best_map(L1, L2):
    # bestmap: permute labels of L2 to match L1 as good as possible
    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = hungarian(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        try:
            newL2[L2 == Label2[i]] = Label1[c[i]]
        except:
            continue
    return newL2


def ss_recovery_error(coef_mat, labels, outlier_idx=None):
    """
    Subspace-sparse recovery error.
    Given coefficients, count the weights of points from other subspaces as error.
    """
    # Not take outliers into consideration
    if outlier_idx is not None:
        labels = np.delete(labels, outlier_idx, 0)
        coef_mat = np.delete(coef_mat, outlier_idx, 0)
        coef_mat = np.delete(coef_mat, outlier_idx, 1)

    n_points = labels.shape[0]
    errors = []
    for n in range(n_points):
        coef = coef_mat[:, n]
        intraspace_coef = coef[labels == labels[n]]
        error = 1 - np.linalg.norm(intraspace_coef, ord=1) / (np.linalg.norm(coef, ord=1) + 1e-10)
        errors.append(error)

    return np.mean(error)


def accumulate_AP(L1, L2, iou_thresh=0.5, outlier_idx=None):
    """
    Calculate number of TP, FP, FN on each sample with given IoU threshold.
    """
    # Not take outliers into consideration
    if outlier_idx is not None:
        L1 = np.delete(L1, outlier_idx, 0)
        L2 = np.delete(L2, outlier_idx, 0)

    label_gt = np.unique(L1)
    nclass_gt = label_gt.shape[0]
    label_pred = np.unique(L2)
    nclass_pred = label_pred.shape[0]

    # Compute IoU
    iou = np.zeros((nclass_gt, nclass_pred))
    for i in range(nclass_gt):
        for j in range(nclass_pred):
            intersect = np.sum(np.logical_and(L1 == label_gt[i], L2 == label_pred[j]))
            union = np.sum(np.logical_or(L1 == label_gt[i], L2 == label_pred[j]))
            iou[i, j] = intersect / union

    tp = 0
    fp = nclass_pred
    fn = nclass_gt
    while True:
        idx = np.argmax(iou)
        i, j = idx // nclass_pred, idx % nclass_pred
        if iou[i, j] < iou_thresh:
            break
        tp += 1
        fp -= 1
        fn -= 1
        iou[i] = 0
        iou[:, j] = 0
    return tp, fp, fn


def calculate_AP(tp, fp, fn):
    """
    Given TP, FP, FN on whole dataset, calculate precision, recall & AP.
    """
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)

    # AP with only one (precision, recall) pair
    AP = ((1 + prec) * recall + prec * (1 - recall)) / 2

    return prec, recall, AP