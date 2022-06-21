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
                # integrate -1 to 1: t^2 dt
                sigma = 2 / 3
    elif kernel == "gaussian":
        if p == 2:
            if d == 1:
                # integrate -1 to 1: exp(-4t^2)t^2 dt
                sigma = 0.10568126
            elif d == 2:
                # integrate B_1(0): exp(-4r^2)(r*cos(t))^2 r dtdr
                sigma = np.pi * (1 - 5 * np.e ** (-4)) / 32.0

    if sigma is None:
        raise ValueError("Unsupported combination of inputs")
    return sigma


def get_rhs(dataset, train_ind, bump):
    LOGGER.info(f"Computing RHS with bump='{bump}'...")

    train_labels = dataset.labels[train_ind]

    if isinstance(bump, float):
        rhs = pl.algorithms.rhs.bump(
            dataset.data, train_ind, train_labels, bump_width=bump
        )
    elif bump == "dirac":
        rhs = None
    return rhs


def run_experiment_poisson(dataset, experiment, scale, tol=1e-3, max_iter=1e3):
    dataset = copy.deepcopy(dataset)

    LOGGER.info(
        "Experiment: {}".format({k: v for k, v in experiment.items() if k != "results"})
    )
    label_locations = experiment["label_locations"]
    train_ind = np.arange(len(label_locations))
    train_labels = dataset.labels[train_ind]

    W, indices_largest_component = build_weight_matrix(dataset, experiment)
    dataset.data = dataset.data[indices_largest_component]
    dataset.labels = dataset.labels[indices_largest_component]

    bumps = experiment["bump"]
    if not isinstance(bumps, list):
        bumps = [bumps]

    solution = []
    for bump in bumps:
        rhs = get_rhs(dataset, train_ind, bump)

        LOGGER.info("Solving Poisson problem...")
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
        fit = poisson.fit(train_ind, train_labels)[:, 0]
        solution.append({"bump": bump, "solution": fit})

    return solution, indices_largest_component
