import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import identity
from scipy.linalg import eigh

# from stsc_utils import get_rotation_matrix
from ssc_utils.stsc_utils_manopt import get_rotation_matrix


def build_adjacency(CMat, K):
    """
    This function takes a NxN coefficient matrix and returns a NxN adjacency
    matrix by choosing only the K strongest connections in the similarity graph.
    :param CMat: NxN coefficient matrix
    :param K: number of strongest edges to keep; if K=0 use all the coefficients
    :return:
        CKSym: NxN symmetric adjacency matrix
    """
    CMat = CMat.astype(float)
    CKSym = None
    N, _ = CMat.shape
    CAbs = np.absolute(CMat).astype(float)
    for i in range(0, N):
        c = CAbs[:, i]
        PInd = np.flip(np.argsort(c), 0)
        CAbs[:, i] = CAbs[:, i] / float(np.absolute(c[PInd[0]]))
    CSym = np.add(CAbs, CAbs.T).astype(float)
    if K != 0:
        Ind = np.flip(np.argsort(CSym, axis=0), 0)
        CK = np.zeros([N, N]).astype(float)
        for i in range(0, N):
            for j in range(0, K):
                CK[Ind[j, i], i] = CSym[Ind[j, i], i] / float(np.absolute(CSym[Ind[0, i], i]))
        CKSym = np.add(CK, CK.T)
    else:
        CKSym = CSym
    return CKSym


def spectral_cluster(CKSym, n):
    # This is direct port of JHU vision lab code. Could probably use sklearn SpectralClustering.
    CKSym = CKSym.astype(float)
    N, _ = CKSym.shape
    MAXiter = 1000  # Maximum number of iterations for KMeans
    REPlic = 20  # Number of replications for KMeans

    DN = np.diag(np.divide(1, np.sqrt(np.sum(CKSym, axis=0) + np.finfo(float).eps)))
    LapN = identity(N).toarray().astype(float) - np.matmul(np.matmul(DN, CKSym), DN)
    _, _, vN = np.linalg.svd(LapN)
    vN = vN.T
    kerN = vN[:, N - n:N]
    normN = np.sqrt(np.sum(np.square(kerN), axis=1))
    kerNS = np.divide(kerN, normN.reshape(len(normN), 1) + np.finfo(float).eps)
    km = KMeans(n_clusters=n, n_init=REPlic, max_iter=MAXiter).fit(kerNS)
    return km.labels_


def selftune_spectral_cluster(CKSym, numC=None, minC=2, maxC=2, relax=0.0001):
    """
    Self-tuning spectral clustering to determine the number of clusters adaptively.
    """
    MAXiter = 1000  # Maximum number of iterations for KMeans
    REPlic = 20  # Number of replications for KMeans

    DN = np.diag(np.divide(1, np.sqrt(np.sum(CKSym, axis=0) + np.finfo(float).eps)))
    LapN = np.matmul(np.matmul(DN, CKSym), DN)
    eigenv, eigenvecs = eigh(LapN)

    if numC is None:
        # Search for the number of clusters with the lowest cost.
        record = []
        for c in range(minC, maxC+1):
            vecs = eigenvecs[:, -c:]
            cost, rot = get_rotation_matrix(vecs, c)
            record.append((cost, vecs.dot(rot)))
            # print('n_cluster: %d \t cost: %f' % (c, cost))

        # Find the largest cluster number giving minimal cost (relaxation for ambiguity)
        sorted_record = sorted(record, key=lambda x: x[0])
        min_cost, Z_best = sorted_record[0]
        for c in range(1, len(sorted_record)):
            cost, Z = sorted_record[c]
            if cost > (1+relax) * min_cost:
                break
            elif Z.shape[1] > Z_best.shape[1]:
                Z_best = Z

        nC = Z_best.shape[1]
        km = KMeans(n_clusters=nC, n_init=REPlic, max_iter=MAXiter).fit(Z_best)

    else:
        vecs = eigenvecs[:, -numC:]
        cost, rot = get_rotation_matrix(vecs, numC)
        Z = vecs.dot(rot)
        km = KMeans(n_clusters=numC, n_init=REPlic, max_iter=MAXiter).fit(Z)

    return km.labels_


