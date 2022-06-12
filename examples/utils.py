import numpy as np
import logging
import copy

import graphlearning as gl
import poissonlearning as pl


LOGGER = logging.getLogger(name=__name__)


def estimate_epsilon(n, d):
    if d == 1:
        epsilon = 15 * np.log(n) / n
    elif d == 2:
        conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
        epsilon = 2.0 * np.log(n) ** (1 / 15) * conn_radius
    else:
        raise ValueError("Unsupported dimension")

    return epsilon


def build_weight_matrix(dataset, experiment, normalize=True):
    LOGGER.info("Creating weight matrix...")
    d = dataset.data.shape[1]

    W = pl.algorithms.epsilon_ball(
        data=dataset.data, epsilon=experiment["eps"], kernel=experiment["kernel"],
    )

    # Remove sigularities by only keeping the largest connected component
    G = gl.graph(W)
    Grestricted, indices_largest_component = G.largest_connected_component()
    W = Grestricted.weight_matrix

    if normalize:
        W *= experiment["eps"] ** (-d)

    return W, indices_largest_component


def get_normalization_constant(kernel, d, p=2):
    sigma = None
    if kernel == "uniform":
        if p == 2:
            if d == 1:
                # integrate -1 to 1: t^(d+2) dt
                sigma = 2 / 3
    elif kernel == "gaussian":
        if p == 2:
            if d == 1:
                # integrate -1 to 1: exp(-4t^2)t^2 dt
                sigma = 0.10568126
            elif d == 2:
                # integrate B_1(0): exp(-4r^2)(r*cos(t))^2 dtdr
                sigma = np.pi * 0.052840632061

    if sigma is None:
        raise ValueError("Unsupported combination of inputs")
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


def run_experiment_poisson(dataset, experiment, scale, tol=1e-3, max_iter=1e3):
    dataset = copy.deepcopy(dataset)

    LOGGER.info(
        "Experiment: {}".format({k: v for k, v in experiment.items() if k != "results"})
    )
    train_ind = experiment["train_indices"]
    train_labels = dataset.labels[train_ind]

    W, indices_largest_component = build_weight_matrix(dataset, experiment)
    dataset.data = dataset.data[indices_largest_component]
    dataset.labels = dataset.labels[indices_largest_component]

    rhs = get_rhs(dataset, experiment)

    poisson = pl.algorithms.Poisson(
        W,
        p=1,
        scale=scale,
        solver="conjugate_gradient",
        normalization="combinatorial",
        tol=tol,
        max_iter=max_iter,
        rhs=rhs,
    )
    solution = poisson.fit(train_ind, train_labels)[:, 0]
    return solution, indices_largest_component
