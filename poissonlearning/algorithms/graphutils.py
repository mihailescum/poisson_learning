import numpy as np
import scipy.spatial as spspacial
import scipy.sparse as spsparse

import sys


def distance_matrix(X):
    distance_matrix = spspacial.distance.squareform(spspacial.distance.pdist(X))
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


def kernel_indicator(a, eps=1.0, d=1, radius=1.0):
    result = a.copy()
    result[result >= radius * eps] = 0.0
    result[result > 1e-10] = 1.0
    np.fill_diagonal(result, 1.0)  # inplace operation
    result = (eps ** -d) * spsparse.csr_matrix(result)

    return result


def epsilon_ball(data, epsilon, kernel="gaussian", eta=None):
    """Epsilon ball weight matrix

    This is an adaptation of Jeff Calders implementation licensed
    under the MIT licence. For the original source code see 
    `https://github.com/jwcalder/GraphLearning/blob/master/graphlearning/ssl.py`.
    ======

    General function for constructing a sparse epsilon-ball weight matrix, whose weights have the form
    \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
    when \\(\\|x_i - x_j\\|\\leq \\varepsilon\\), and \\(w_{i,j}=0\\) otherwise.
    This type of weight matrix is only feasible in relatively low dimensions.
   
    Parameters
    ----------
    data : (n,m) numpy array
        n data points, each of dimension m
    epsilon : float
        Connectivity radius
    kernel : string (optional), {'uniform','gaussian','singular','distance'}, default='gaussian'
        The choice of kernel in computing the weights between \\(x_i\\) and \\(x_j\\) when
        \\(\\|x_i-x_j\\|\\leq \\varepsilon\\). The choice 'uniform' corresponds to \\(w_{i,j}=1\\) 
        and constitutes an unweighted graph, 'gaussian' corresponds to
        \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
        'distance' corresponds to
        \\[ w_{i,j} = \\|x_i - x_j\\|, \\]
        and 'singular' corresponds to 
        \\[ w_{i,j} = \\frac{1}{\\|x_i - x_j\\|}, \\]
        when \\(i\\neq j\\) and \\(w_{i,i}=1\\).
    eta : python function handle (optional)
        If provided, this overrides the kernel option and instead uses the weights
        \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right). \\]

    Returns
    -------
    W : (n,n) scipy sparse matrix, float 
        Sparse weight matrix.
    """
    n = data.shape[0]  # Number of points

    # Rangesearch to find nearest neighbors
    Xtree = spspacial.cKDTree(data)
    M = Xtree.query_pairs(epsilon, output_type="ndarray")

    # Differences between points and neighbors
    V = data[M[:, 0], :] - data[M[:, 1], :]
    dists = np.sum(V * V, axis=1)

    # If eta is None, use kernel keyword
    if eta is None:

        if kernel == "uniform":
            weights = np.ones_like(dists)
            fzero = 1
        elif kernel == "gaussian":
            weights = np.exp(-4 * dists / (epsilon * epsilon))
            fzero = 1
        elif kernel == "distance":
            weights = np.sqrt(dists)
            fzero = 0
        elif kernel == "singular":
            weights = np.sqrt(dists)
            weights[dists == 0] = 1
            weights = 1 / weights
            fzero = 1
        else:
            sys.exit("Invalid choice of kernel: " + kernel)

    # Else use user-defined eta
    else:
        weights = eta(dists / (epsilon * epsilon))
        fzero = eta(0)

    # Weights

    # Symmetrize weights and add diagonal entries
    weights = np.concatenate((weights, weights, fzero * np.ones(n,)))
    M1 = np.concatenate((M[:, 0], M[:, 1], np.arange(0, n)))
    M2 = np.concatenate((M[:, 1], M[:, 0], np.arange(0, n)))

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = spsparse.coo_matrix((weights, (M1, M2)), shape=(n, n))

    return W.tocsr()
