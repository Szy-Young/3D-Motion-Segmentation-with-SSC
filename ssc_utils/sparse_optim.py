"""
First step in the SSC-based motion segmentation pipeline: solving a sparse optimization.
"""

import numpy as np
import math
import cvxpy as cvx


def project_data(X, r, type='NormalProj'):
    """
    This function takes the D x N data matrix with columns indicating
    different data points and project the D dimensional data into the r
    dimensional space.
    :param X: D x N data matrix of N data points
    :param r: dimension of the space to project the data to
    :param type:
        (1) Projection using PCA
        (2) Projection using random projections with iid elements from N(0,1/r)
        (3) Projection using random projections with iid elements from symmetric
            bernoulli distribution: +1/sqrt(r),-1/sqrt(r) elements with same probability
    :return:
        Xp: r x N data matrix of N projectred data points
    """
    Xp = None
    D, N = X.shape
    if r == 0:
        Xp = X
    else:
        if type == 'PCA':
            isEcon = False
            if D > N:
                isEcon = True
            U, S, V = np.linalg.svd(X.T, full_matrices=isEcon)
            Xp = U[:, 0:r].T
        if type == 'NormalProj':
            normP = (1.0 / math.sqrt(r)) * np.random.randn(r * D, 1)
            PrN = normP.reshape(r, D, order='F')
            Xp = np.matmul(PrN, X)
        if type == 'BernoulliProj':
            bp = np.random.rand(r * D, 1)
            Bp = (1.0 / math.sqrt(r)) * (bp >= .5) - (1.0 / math.sqrt(r)) * (bp < .5)
            PrB = Bp.reshape(r, D, order='F')
            Xp = np.matmul(PrB, X)
    return Xp


def sparse_optim(Xp, cst=0, Opt='Lasso', lmbda=0.001):
    """
    This function takes the D x N matrix of N data points and write every
    point as a sparse linear combination of other points.
    :param Xp: D x N matrix of N data points
    :param cst: 1 if using the affine constraint sum(c)=1, else 0
    :param Opt: type of optimization, {'L1Perfect','L1Noisy','Lasso','L1ED'}
    :param lmbda: regularizartion parameter of LASSO, typically between 0.001 and 0.1 or the noise level for 'L1Noise'
    :return:
        CMat: N x N matrix of coefficients, column i correspond to the sparse coefficients of data point in column i of Xp
    """
    D, N = Xp.shape
    CMat = np.zeros([N, N])
    for i in range(0, N):
        y = Xp[:, i]
        if i == 0:
            Y = Xp[:, i + 1:]
        elif i > 0 and i < N - 1:
            Y = np.concatenate((Xp[:, 0:i], Xp[:, i + 1:N]), axis=1)
        else:
            Y = Xp[:, 0:N - 1]

        try:
            if cst == 1:
                if Opt == 'Lasso':
                    c = cvx.Variable(N - 1)
                    obj = cvx.Minimize(cvx.norm(c, 1) + lmbda * cvx.norm(Y @ c - y))
                    constraint = [cvx.sum(c) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif Opt == 'L1Perfect':
                    c = cvx.Variable(N - 1)
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [Y @ c == y, cvx.sum(c) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif Opt == 'L1Noise':
                    c = cvx.Variable(N - 1)
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [(Y @ c - y) <= lmbda, cvx.sum(c) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif Opt == 'L1ED':
                    c = cvx.Variable(N - 1 + D, 1)
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [np.concatenate((Y, np.identity(D)), axis=1)
                                  * c == y, cvx.sum(c[0:N - 1]) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()

            else:
                if Opt == 'Lasso':
                    c = cvx.Variable(N - 1)
                    obj = cvx.Minimize(cvx.norm(c, 1) + lmbda * cvx.norm(Y @ c - y))
                    prob = cvx.Problem(obj)
                    prob.solve()
                elif Opt == 'L1Perfect':
                    c = cvx.Variable(N - 1)
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [Y @ c == y]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif Opt == 'L1Noise':
                    c = cvx.Variable(N - 1)
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [(Y @ c - y) <= lmbda]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif Opt == 'L1ED':
                    c = cvx.Variable(N - 1 + D, 1)
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [np.concatenate((Y, np.identity(D)), axis=1) * c == y]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
            coef = c.value
        except:
            # Solver may fail on some points
            coef = np.empty(N-1)
            coef.fill(np.nan)

        if i == 0:
            CMat[0, 0] = 0
            CMat[1:N, 0] = coef[0: N - 1]
        elif i > 0 and i < N - 1:
            CMat[0:i, i] = coef[0:i]
            CMat[i, i] = 0
            CMat[i + 1:N, i] = coef[i:N - 1]
        else:
            CMat[0:N - 1, N - 1] = coef[0:N - 1]
            CMat[N - 1, N - 1] = 0
    return CMat


def detect_outlier(CMat):
    """
    This function takes the coefficient matrix resulted from sparse
    representation using \ell_1 minimization. If a point cannot be written as
    a linear combination of other points, it should be an outlier. The
    function detects the indices of outliers and modifies the coefficient
    matrix and the ground-truth accordingly.
    :param CMat: NxN coefficient matrix
    :return:
        CMatC: coefficient matrix after eliminating Nans
    """
    _, N = CMat.shape
    OutlierIndx = list()

    for i in range(0, N):
        c = CMat[:, i]
        if np.sum(np.isnan(c)) >= 1:
            OutlierIndx.append(i)
    CMatC = CMat.astype(float)
    CMatC[OutlierIndx, :] = np.nan
    CMatC[:, OutlierIndx] = np.nan

    return CMatC, OutlierIndx