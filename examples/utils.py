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


def build_graph(dataset, experiment, eps=None, n_neighbors=None, eta=None):
    LOGGER.info("Creating weight matrix...")
    d = dataset.data.shape[1]

    if eps is not None:
        W = pl.algorithms.epsilon_ball(
            data=dataset.data, epsilon=eps, kernel=experiment["kernel"], eta=eta,
        )
    elif n_neighbors is not None:
        W = gl.weightmatrix.knn(
            data=dataset.data, k=n_neighbors, kernel=experiment["kernel"]
        )
        LOGGER.info(
            f"#Nodes in connected component: {gl.graph(W).largest_connected_component()[1].sum()}"
        )
    else:
        raise ValueError("Must specify either `eps` or `n_neighbors`.")

    # Remove sigularities by only keeping the largest connected component
    G = gl.graph(W)
    G, indices_largest_component = G.largest_connected_component()
    return G, indices_largest_component


def construct_ilu_preconditioner(G):
    LOGGER.info("Constructing ILU preconditioner")
    L = G.laplacian(normalization="combinatorial").tocsr()

    preconditioner = splinalg.spilu(L.tocsc())
    return preconditioner


def get_normalization_constant(kernel, d, p=[2]):
    p = np.array(p)
    result = np.zeros(p.size)
    for i, p_single in enumerate(p):
        result[i] = _get_normalization_constant(kernel, d, p_single)

    if result.size == 1:
        result = result[0]

    return result


def _get_normalization_constant(kernel, d, p):
    sigma_all = {}
    if kernel == "uniform":
        if d == 1:
            # integrate -1 to 1: t^2 dt
            sigma_all = {
                2: 2 / 3,
            }
    elif kernel == "gaussian":
        if d == 1:
            sigma_all = {
                2: 0.10568,
                4: 0.035052,
                8: 0.010583,
                12: 0.00549624,
                16: 0.00358198,
                20: 0.002624,
                26: 0.00185778,
                32: 0.00143257,
            }
        elif d == 2:
            sigma_all = {
                # integrate B_1(0): exp(-4r^2)(r*cos(t))^2 r dtdr
                2: np.pi * (1 - 5 * np.e ** (-4)) / 32.0,
                3: 8 / 3 * 0.0175258,
                4: 3 * np.pi / 4 * 0.011905,
                5: 32 / 15 * 0.0086642,
                6: 5 * np.pi / 8 * 0.0066390,
                8: 35 * np.pi / 64 * 0.0043496,
                10: 63 * np.pi / 128 * 0.0031475,
                12: 231 * np.pi / 512 * 0.0024318,
                16: 1.2339 * 0.00164294,
                20: 1.1071 * 0.00122844,
                24: 1.0127 * 0.000976852,
                28: 0.939 * 0.00080914,
                32: 0.87933 * 0.000689817,
                36: 0.82976 * 0.000600784,
                40: 0.78773 * 0.000531897,
                45: 0.743192 * 0.000464927,
                50: 0.70545 * 0.000404119,
                60: 0.644518 * 0.00033741,
                70: 0.597063 * 0.000285109,
                80: 0.55875 * 0.0002468102,
                90: 0.526978 * 0.000217563,
                100: 0.500074 * 0.0001945035,
            }

    sigma = sigma_all.get(p, None)
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


def run_experiment_poisson(dataset, experiment, rho2=1):
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
):
    LOGGER.info(f"Using eps={eps} and n_neighbors={n_neighbors}")
    n, d = dataset.data.shape

    p_homotopy = experiment.get("p", None)
    p = 2 if p_homotopy is None else p_homotopy[-1]
    solver = "conjugate_gradient" if p_homotopy is None else "variational"
    eta = None if p_homotopy is None else lambda x: np.exp(-x)

    G, indices_largest_component = build_graph(
        dataset, experiment, eps=eps, n_neighbors=n_neighbors, eta=eta
    )
    W = G.weight_matrix

    if eps is not None:
        W *= eps ** (-d)
        if p_homotopy is None:
            sigma = get_normalization_constant(experiment["kernel"], d)
            scale = 0.5 * sigma * rho2 * (eps ** 2) * n ** 2
        else:
            sigma = get_normalization_constant(experiment["kernel"], d, p_homotopy)
            scale = 0.5 * sigma * rho2 * (eps ** p_homotopy) * n ** 2
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
            p=(p - 1),
            scale=scale,
            solver=solver,
            normalization="combinatorial",
            tol=experiment["tol"],
            max_iter=experiment["max_iter"],
            rhs=rhs,
            preconditioner=preconditioner,
            homotopy_steps=p_homotopy,
        )
        fit = poisson.fit(train_ind, train_labels)

        if p_homotopy is None:
            fit = fit[:, 0]
        else:
            fit = fit[1]

        item = {
            "bump": bump,
            "solution": fit,
            "largest_component": indices_largest_component,
            "tol": experiment["tol"],
            "max_iter": experiment["max_iter"],
        }
        if eps is not None:
            item["eps"] = eps
        elif n_neighbors is not None:
            item["n_neighbors"] = n_neighbors
        else:
            raise ValueError("Must specify either `eps` or `n_neighbors`.")

        result.append(item)

    return result
