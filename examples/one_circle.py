import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import logging

import poissonlearning as pl
import graphlearning as gl

from plotting import plot_graph_function_with_triangulation

logger = logging.getLogger("ex.one_circle")

NUM_TRAINING_POINTS = [10000, 30000, 50000, 100000, 300000, 700000, 1000000]
BUMP_WIDTHS = ["dirac"]
NUM_PLOTTING_POINTS = 10000

logging.basicConfig(level="INFO")


def estimate_epsilon(data):
    n = data.shape[0]
    conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
    epsilon = 0.36 * np.log(n) ** (1 / 15) * conn_radius

    return epsilon


results = {}
for training_points in NUM_TRAINING_POINTS:
    print(f"\n# training points: {training_points}")
    results[training_points] = {}

    # Load the one_circle dataset
    dataset = pl.datasets.Dataset.load("one_circle", "raw", training_points)

    train_ind = np.array([0, 1])
    train_labels = dataset.labels[train_ind]

    # Build the weight matrix
    print("Creating weight matrix...")
    epsilon = estimate_epsilon(dataset.data)
    W = gl.weightmatrix.epsilon_ball(
        dataset.data, epsilon, eta=lambda x: np.exp(-x)
    )  # kernel="gaussian")

    # Remove sigularities by only keeping the largest connected component
    G = gl.graph(W)
    Grestricted, indices = G.largest_connected_component()
    dataset.data = dataset.data[indices]
    dataset.labels = dataset.labels[indices]
    W = Grestricted.weight_matrix
    n, d = dataset.data.shape
    print(f"n: {n}; epsilon: {epsilon}")

    # W = epsilon_ball_test(dataset.data, epsilon, kernel="uniform")
    # W *= epsilon ** (-d)
    # normalization constant, integrate B_1(0): eta(r)(r*cos(t))^2 dtdr
    sigma = np.pi * (np.e - 2) / (2 * np.e)

    # Solve the poisson problem with dirac RHS
    print("Solving Poisson problem...")
    for bump_width in BUMP_WIDTHS:
        print(f"Bump width: {bump_width}")
        if isinstance(bump_width, float):
            rhs = pl.algorithms.rhs.bump(
                dataset.data, train_ind, train_labels, bump_width=bump_width
            )
        elif bump_width == "dirac":
            rhs = None
        else:
            raise ValueError("Invalid bump width, must be either float or 'dirac'.")

        poisson = pl.algorithms.Poisson(
            W,
            p=1,
            scale=0.5 * sigma * epsilon ** (d + 2) * n ** 2,
            solver="conjugate_gradient",
            normalization="combinatorial",
            spectral_cutoff=50,
            tol=1.5e-3,
            max_iter=1e3,
            rhs=rhs,
        )
        solution = poisson.fit(train_ind, train_labels)[:, 0]

        result = pd.DataFrame(columns=["x", "y", "z"])
        result["x"] = dataset.data[:, 0]
        result["y"] = dataset.data[:, 1]
        result["z"] = solution
        results[training_points][bump_width] = result

print("Plotting...")

# Plot solution
n = max(NUM_TRAINING_POINTS)
bump_width = "dirac"
sample = results[n][bump_width].sample(NUM_PLOTTING_POINTS, random_state=1)
xy = sample[["x", "y"]].to_numpy()
dist = cdist(xy, xy, metric="euclidean",)
fig_results = plt.figure()
ax_solution = fig_results.add_subplot(1, 2, 1, projection="3d")
plot_graph_function_with_triangulation(
    ax_solution, xy, sample["z"].to_numpy(), dist=dist, max_dist=0.1,
)
ax_solution.set_title(f"n: {results[n][bump_width].shape[0]}; RHS: {bump_width}")


def get_analytic_solution(xy, z1, z2):
    # Compute the analytic continuum limit
    green_first_label = pl.datasets.one_circle.greens_function(x=xy, z=z1)
    green_second_label = pl.datasets.one_circle.greens_function(x=xy, z=z2)
    solution_analytic = 0.5 * green_first_label - 0.5 * green_second_label
    return solution_analytic


solution_analytic = get_analytic_solution(
    xy, dataset.data[train_ind[0]], dataset.data[train_ind[1]]
)
ax_analytic = fig_results.add_subplot(1, 2, 2, projection="3d")
plot_graph_function_with_triangulation(
    ax_analytic, xy, solution_analytic, dist=dist, max_dist=0.1,
)
ax_analytic.set_title(f"eps: {epsilon:.4f}; Analytic solution to continuum problem")

# Calculate errors
bump_width = "dirac"
errors = {"scaled": {}, "unscaled": {}}
for n in NUM_TRAINING_POINTS:
    solution = results[n][bump_width]
    xy = solution[["x", "y"]].to_numpy()
    z = solution["z"].to_numpy()

    solution_analytic = get_analytic_solution(
        xy, dataset.data[train_ind[0]], dataset.data[train_ind[1]]
    )
    mask_infty = np.isfinite(solution_analytic)

    error_L1_unscaled = np.abs(z[mask_infty] - solution_analytic[mask_infty]).mean()
    errors["unscaled"][n] = error_L1_unscaled

    scale = (solution_analytic[mask_infty] / z[mask_infty]).mean()
    print(f"Scale for {n}: {scale}")
    z_scaled = scale * z
    error_L1_scaled = np.abs(
        z_scaled[mask_infty] - solution_analytic[mask_infty]
    ).mean()
    errors["scaled"][n] = error_L1_scaled

fig_error, ax_error = plt.subplots(1, 1)
for s in ["unscaled", "scaled"]:
    ax_error.plot(
        list(errors[s].keys()), list(errors[s].values()), marker="x", ls="-", label=s,
    )
    logger.info(f"L1 error {s}: {errors[s]}")
ax_error.set_xscale("log")
ax_error.set_title(f"L1 Error compared with RHS {bump_width} to analytic solution")
ax_error.grid()
ax_error.legend()

plt.show()
