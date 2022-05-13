import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import logging

import poissonlearning as pl
import graphlearning as gl

from plotting import plot_graph_function_with_triangulation

logger = logging.getLogger("ex.one_circle")
logging.basicConfig(level="INFO")


def estimate_epsilon(n):
    factor = 0.4
    conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
    epsilon = factor * np.log(n) ** (1 / 15) * conn_radius
    return epsilon


# `n` will be overwritten with the number of nodes from the largest connected component
experiments = [
    {"n": 10000, "eps": 0.02452177, "bump": "dirac"},
    {"n": 30000, "eps": 0.01552237, "bump": "dirac"},
    {"n": 50000, "eps": 0.01250796, "bump": "dirac"},
    {"n": 100000, "eps": 0.00930454, "bump": "dirac"},
    {"n": 300000, "eps": 0.00578709, "bump": "dirac"},
    {"n": 700000, "eps": 0.00399516, "bump": "dirac"},
    # {"n": 1000000, "eps": 0.00341476, "bump": "dirac"},
]
NUM_PLOTTING_POINTS = 10000


for experiment in experiments:
    print(f"Experiment: {experiment}")

    # Load the one_circle dataset
    dataset = pl.datasets.Dataset.load("one_circle", "raw", experiment["n"])

    train_ind = np.array([0, 1])
    train_labels = dataset.labels[train_ind]

    # Build the weight matrix
    print("Creating weight matrix...")
    W = gl.weightmatrix.epsilon_ball(
        dataset.data, experiment["eps"], eta=lambda x: np.exp(-x)
    )

    # Remove sigularities by only keeping the largest connected component
    G = gl.graph(W)
    Grestricted, indices = G.largest_connected_component()
    dataset.data = dataset.data[indices]
    dataset.labels = dataset.labels[indices]
    W = Grestricted.weight_matrix
    n, d = dataset.data.shape
    experiment["n"] = n

    # normalization constant, integrate B_1(0): eta(r)(r*cos(t))^2 dtdr
    sigma = np.pi * (np.e - 2) / (2 * np.e)

    # Solve the poisson problem with dirac RHS
    print("Solving Poisson problem...")
    if isinstance(experiment["bump"], float):
        rhs = pl.algorithms.rhs.bump(
            dataset.data, train_ind, train_labels, bump_width=experiment["bump"]
        )
    elif experiment["bump"] == "dirac":
        rhs = None
    else:
        raise ValueError("Invalid bump width, must be either float or 'dirac'.")

    poisson = pl.algorithms.Poisson(
        W,
        p=1,
        scale=0.5 * sigma * experiment["eps"] ** (d + 2) * n ** 2,
        solver="conjugate_gradient",
        normalization="combinatorial",
        spectral_cutoff=50,
        tol=1e-3,
        max_iter=100,
        rhs=rhs,
    )
    solution = poisson.fit(train_ind, train_labels)[:, 0]

    result = pd.DataFrame(columns=["x", "y", "z"])
    result["x"] = dataset.data[:, 0]
    result["y"] = dataset.data[:, 1]
    result["z"] = solution
    experiment["result"] = result

print("Plotting...")

# Compute errors
print("Computing errors...")


def get_analytic_solution(xy, z1, z2):
    # Compute the analytic continuum limit
    green_first_label = pl.datasets.one_circle.greens_function(x=xy, z=z1)
    green_second_label = pl.datasets.one_circle.greens_function(x=xy, z=z2)
    solution_analytic = 0.5 * green_first_label - 0.5 * green_second_label
    return solution_analytic


for e in experiments:
    xy = e["result"][["x", "y"]].to_numpy()
    z = e["result"]["z"].to_numpy()

    solution_analytic = get_analytic_solution(
        xy, dataset.data[train_ind[0]], dataset.data[train_ind[1]]
    )
    mask_infty = np.isfinite(solution_analytic)

    error_L1_unscaled = np.abs(z[mask_infty] - solution_analytic[mask_infty]).mean()
    e["L1_unscaled"] = error_L1_unscaled

    scale = (solution_analytic[mask_infty] / z[mask_infty]).mean()
    print(f"scale for {e['n']}: {scale}")
    z_scaled = scale * z
    error_L1_scaled = np.abs(
        z_scaled[mask_infty] - solution_analytic[mask_infty]
    ).mean()
    e["L1_scaled"] = error_L1_scaled


print("Plotting...")
# Plot solution
n = 50000  # 1000000
bump_width = "dirac"
n_max = max([e["n"] for e in experiments])
ex_max = [e for e in experiments if e["n"] == n_max and e["bump"] == bump_width][0]
sample = ex_max["result"].sample(NUM_PLOTTING_POINTS, random_state=1)

xy = sample[["x", "y"]].to_numpy()
dist = cdist(xy, xy, metric="euclidean")

fig_results = plt.figure()
ax_solution = fig_results.add_subplot(1, 2, 1, projection="3d")
plot_graph_function_with_triangulation(
    ax_solution, xy, sample["z"].to_numpy(), dist=dist, max_dist=0.1,
)
ax_solution.set_title(f"Computed discrete solution; n: {n}; RHS: {bump_width}")


solution_analytic = get_analytic_solution(
    xy, dataset.data[train_ind[0]], dataset.data[train_ind[1]]
)
ax_analytic = fig_results.add_subplot(1, 2, 2, projection="3d")
plot_graph_function_with_triangulation(
    ax_analytic, xy, solution_analytic, dist=dist, max_dist=0.1,
)
ax_analytic.set_title(f"Analytic solution to continuum problem")

# Plot errors
bump_width = "dirac"

fig_error, ax_error = plt.subplots(1, 1)
ex_error = [e for e in experiments if e["bump"] == bump_width]
for s in ["unscaled", "scaled"]:
    errors = [e[f"L1_{s}"] for e in ex_error]
    n = [e["n"] for e in ex_error]
    ax_error.plot(n, errors, marker="x", ls="-", label=s)
    logger.info(f"L1 error {s}: {errors}")
ax_error.set_xscale("log")
ax_error.set_title(f"L1 Error compared with RHS {bump_width} to analytic solution")
ax_error.grid()
ax_error.legend()

plt.show()
