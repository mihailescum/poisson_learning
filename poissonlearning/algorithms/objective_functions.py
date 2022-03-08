import numpy as np
import scipy.sparse as spsparse


def objective_p_laplace(u, W, b, p):
    """Minimizers of this objective function solve the p-Laplace equation with `b` as
    RHS.
    $ 1/(2p) \sum_{ij}^n W_{ij} |u_i - u_j|^p - u * b $
    """
    difference_matrix = np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** p
    Ju = 0.5 / p * W.multiply(difference_matrix).sum() - np.dot(u, b)
    return Ju


def objective_p_laplace_gradient(u, W, b, p):
    """Gradient of the p-Laplace objective function. Critical points satisfy 
    $ -div(|\grad u|^{p-2} \grad u) = b $
    For a derivation of the vector notation used here, see [1].

    References
    ----------
    [1] "Algorithms for l^p-based semi-supervised Learning on Graphs" by MF Rios, J Calder
    and G Lerman; Preprint (arXiv); https://arxiv.org/abs/1901.05031
    """
    if p != 2:
        A = W.multiply(np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** (p - 2))
    else:
        A = W
    # TODO: use get_node_degrees here?
    D = spsparse.diags(A.sum(axis=1).A1)
    grad_Ju = 1.0 * (D - A) @ u - b
    return grad_Ju


def objective_weighted_mean(u, A):
    mean = np.dot(u, A)
    return mean


def objective_weighted_mean_gradient(u, A):
    return A
