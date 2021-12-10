import numpy as np
import scipy.spatial.distance as spdist
import scipy.sparse as spsparse


def distance_matrix(X):
    distance_matrix = spdist.squareform(spdist.pdist(X))
    return distance_matrix


def node_degrees(W):
    if isinstance(W, np.ndarray):
        D = W.sum(axis=1) - np.diag(W)
    else:
        D = W.sum(axis=1).A1 - W.diagonal()
    return D


def kernel_exponential(a, eps=1.0, d=1, cutoff=None):
    result = (eps ** -d) * np.exp(-((a / eps) ** 2))
    if not cutoff is None:
        result[result < cutoff] = 0.0
    result = spsparse.csr_matrix(result)

    return result
