import numpy as np
import scipy.sparse as spsparse


def objective_p_laplace(u, W, b, p):
    difference_matrix = np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** p
    Ju = 0.5 / p * W.multiply(difference_matrix).sum() - np.dot(u, b)
    return Ju


def objective_p_laplace_gradient(u, W, b, p):
    A = W.multiply(np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** (p - 2))
    # TODO: use get_node_degrees here?
    D = spsparse.diags(A.sum(axis=1).A1)
    grad_Ju = 0.5 * (D - A) @ u - b
    return grad_Ju


def objective_weighted_mean(u, A):
    mean = np.dot(u, A)
    return mean


def objective_weighted_mean_gradient(u, A):
    return A
