import numpy as np
import logging

import graphlearning as gl
from examples.line import LOGGER
import poissonlearning as pl


LOGGER = logging.getLogger(name=__name__)


def estimate_epsilon(n, d):
    if d == 1:
        epsilon = 15 * np.log(n) / n
    else:
        raise ValueError("Unsupported dimension")

    return epsilon


def build_weight_matrix(dataset, experiment):
    LOGGER.info("Creating weight matrix...")
    d = dataset.data.shape[1]

    W = gl.weightmatrix.epsilon_ball(
        dataset.data, experiment["eps"], kernel=experiment["kernel"]
    )
    W *= experiment["eps"] ** (-d)
    return W


def get_normalization_constant(kernel, d):
    if kernel == "uniform":
        if d == 1:
            # integrate -1 to 1: t^(d+2) dt
            sigma = 2 / 3
        elif d == 2:
            pass
    elif kernel == "gaussian":
        if d == 1:
            # integrate -1 to 1: exp(-4t^2)t^2 dt
            sigma = 0.10568126
        elif d == 2:
            pass
    return sigma


def get_rhs(dataset, experiment):
    LOGGER.info("Solving Poisson problem...")

    train_ind = experiment["train_indices"]
    train_labels = dataset.labels[train_ind]

    if isinstance(experiment["bump"], float):
        rhs = pl.algorithms.rhs.bump(
            dataset.data, train_ind, train_labels, bump_width=experiment["bump"]
        )
    elif experiment["bump"] == "dirac":
        rhs = None
    return rhs
