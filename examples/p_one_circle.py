import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import logging

import poissonlearning as pl
import graphlearning as gl

from plotting import plot_graph_function_with_triangulation

logger = logging.getLogger("ex.one_circle")

NUM_TRAINING_POINTS = [10000]  # , 30000, 50000, 100000, 300000, 700000, 1000000]
BUMP_WIDTH = "dirac"
HOMOTOPY_STEPS = [3, 4, 5, 6, 8, 10, 12, 16]
NUM_PLOTTING_POINTS = 10000

logging.basicConfig(level="INFO")


def estimate_epsilon(n):
    factor = 0.7
    conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
    epsilon = factor * np.log(n) ** (1 / 15) * conn_radius
    return epsilon


results = {}
for training_points in NUM_TRAINING_POINTS:
    print(f"\n# training points: {training_points}")
    results[training_points] = {}

    # Load the one_circle dataset
    dataset = pl.datasets.Dataset.load("one_circle", "raw", training_points)

    train_ind = np.array([0, 1, 2])
    train_labels = dataset.labels[train_ind]

    # Build the weight matrix
    print("Creating weight matrix...")
    epsilon = estimate_epsilon(dataset.data.shape[0])
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
    print(f"Bump width: {BUMP_WIDTH}")
    if isinstance(BUMP_WIDTH, float):
        rhs = pl.algorithms.rhs.bump(
            dataset.data, train_ind, train_labels, bump_width=BUMP_WIDTH
        )
    elif BUMP_WIDTH == "dirac":
        rhs = None
    else:
        raise ValueError("Invalid bump width, must be either float or 'dirac'.")

    p = HOMOTOPY_STEPS[-1]
    poisson = pl.algorithms.Poisson(
        W,
        p=p - 1,
        scale=None,
        solver="conjugate_gradient",
        normalization="combinatorial",
        spectral_cutoff=50,
        tol=1e-3,
        max_iter=200,
        rhs=rhs,
        homotopy_steps=HOMOTOPY_STEPS,
    )
    _, homotopy_solutions = poisson.fit(train_ind, train_labels)
    for p_homotopy, solution_homotopy in homotopy_solutions.items():
        scale = 0.5 * sigma * epsilon ** (d + p_homotopy) * n ** 2
        solution_homotopy = scale ** (1 / p_homotopy) * solution_homotopy

        result = pd.DataFrame(columns=["x", "y", "z"])
        result["x"] = dataset.data[:, 0]
        result["y"] = dataset.data[:, 1]
        result["z"] = solution_homotopy
        results[training_points][p_homotopy] = result

print("Plotting...")

# Plot solution
n = max(NUM_TRAINING_POINTS)
sample_size = NUM_PLOTTING_POINTS

fig_results = plt.figure()
for i, p_homotopy in enumerate(results[n], start=1):
    ax_solution = fig_results.add_subplot(
        int(np.floor(np.sqrt(len(results[n])))),
        int(np.floor(np.sqrt(len(results[n])))),
        i,
        projection="3d",
    )

    sample = results[n][p_homotopy].sample(sample_size, random_state=1)
    xy = sample[["x", "y"]].to_numpy()

    dist = cdist(xy, xy, metric="euclidean",)
    plot_graph_function_with_triangulation(
        ax_solution, xy, sample["z"].to_numpy(), dist=dist, max_dist=0.1,
    )
    ax_solution.set_title(f"p={p_homotopy}; n={n}")


plt.show()
