import numpy as np
import scipy.sparse.linalg as splinalg
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


def build_weight_matrix(dataset, experiment, eps=None, n_neighbors=None):
    LOGGER.info("Creating weight matrix...")
    d = dataset.data.shape[1]

    if eps is not None:
        W = pl.algorithms.epsilon_ball(
            data=dataset.data, epsilon=eps, kernel=experiment["kernel"],
        )
    elif n_neighbors is not None:
        W = gl.weightmatrix.knn(
            data=dataset.data, k=n_neighbors, kernel=experiment["kernel"]
        )
    else:
        raise ValueError("Must specify either `eps` or `n_neighbors`.")

    # Remove sigularities by only keeping the largest connected component
    G = gl.graph(W)
    return G


def construct_ilu_preconditioner(G):
    LOGGER.info("Constructing ILU preconditioner")
    L = G.laplacian(normalization="combinatorial").tocsr()

    preconditioner = splinalg.spilu(L.tocsc())
    return preconditioner


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


def run_experiment_poisson(dataset, experiment, rho2=1, tol=1e-3, max_iter=1e3):
    LOGGER.info(
        "Experiment: {}".format({k: v for k, v in experiment.items() if k != "results"})
    )
    label_locations = experiment["label_locations"]
    train_ind = np.arange(len(label_locations))
    train_labels = dataset.labels[train_ind]

    bumps = x if isinstance(x := experiment["bump"], list) else [x]

    if "eps" in experiment:
        epslist = x if isinstance(x := experiment["eps"], list) else [x]
    else:
        epslist = []

    if "n_neighbors" in experiment:
        n_neighbors_list = (
            x if isinstance(x := experiment["n_neighbors"], list) else [x]
        )
    else:
        n_neighbors_list = []

    solution = []
    for eps in epslist:
        solution.extend(
            run_experiment_graphconfig(
                dataset=dataset.copy(),
                experiment=experiment,
                train_ind=train_ind,
                train_labels=train_labels,
                bumps=bumps,
                eps=eps,
                rho2=rho2,
                tol=tol,
                max_iter=max_iter,
            )
        )

    for n_neighbors in n_neighbors_list:
        solution.extend(
            run_experiment_graphconfig(
                dataset=dataset.copy(),
                experiment=experiment,
                train_ind=train_ind,
                train_labels=train_labels,
                bumps=bumps,
                n_neighbors=n_neighbors,
                rho2=rho2,
                tol=tol,
                max_iter=max_iter,
            )
        )

    return solution


def run_experiment_graphconfig(
    dataset,
    experiment,
    bumps,
    train_ind,
    train_labels,
    eps=None,
    n_neighbors=None,
    rho2=1,
    tol=1e-3,
    max_iter=1e3,
):
    LOGGER.info(f"Using eps={eps} and n_neighbors={n_neighbors}")
    n, d = dataset.data.shape

    G = build_weight_matrix(dataset, experiment, eps=eps, n_neighbors=n_neighbors)
    G, indices_largest_component = G.largest_connected_component()
    W = G.weight_matrix

    if eps is not None:
        W *= eps ** (-d)
        sigma = get_normalization_constant(experiment["kernel"], d)
        scale = 0.5 * sigma * rho2 * eps ** 2 * n ** 2
    elif n_neighbors is not None:
        scale = None
    else:
        raise ValueError("Must specify either `eps` or `n_neighbors`.")

    dataset.data = dataset.data[indices_largest_component]
    dataset.labels = dataset.labels[indices_largest_component]

    preconditioner = construct_ilu_preconditioner(G)

    result = []
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
            preconditioner=preconditioner,
        )
        fit = poisson.fit(train_ind, train_labels)[:, 0]
        item = {
            "bump": bump,
            "solution": fit,
            "largest_component": indices_largest_component,
        }
        if eps is not None:
            item["eps"] = eps
        elif n_neighbors is not None:
            item["n_neighbors"] = n_neighbors
        else:
            raise ValueError("Must specify either `eps` or `n_neighbors`.")

        result.append(item)

    return result
