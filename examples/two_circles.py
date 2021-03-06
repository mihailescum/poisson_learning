import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import spdiags

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import poissonlearning as pl
import graphlearning as gl

from plotting import plot_graph_function_with_triangulation, plot_data_with_labels

import logging

logging.basicConfig(level="INFO")

NUM_TRAINING_POINTS = 100000
NUM_PLOTTING_POINTS = 10000
if NUM_PLOTTING_POINTS > NUM_TRAINING_POINTS:
    NUM_PLOTTING_POINTS = NUM_TRAINING_POINTS

# Load the two_circles dataset
dataset = pl.datasets.Dataset.load("two_circles", "raw", NUM_TRAINING_POINTS)
n, d = dataset.data.shape

train_ind = np.array([0, 1])
train_labels = dataset.labels[train_ind]


def estimate_epsilon(data, d):
    min = data.min(axis=0)
    max = data.max(axis=0)
    volume = np.prod(np.abs(max - min))

    n = data.shape[0]
    if d == 2:
        conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
    elif d >= 3:
        conn_radius = (np.log(n) / n) ** (1 / d)
    else:
        raise ValueError("Dimension not supported")

    epsilon = 0.7 * np.log(n) ** (1 / 8) * conn_radius

    return epsilon


# Build the weight matrix

# W = gl.weightmatrix.knn(dataset.data, k=5, symmetrize=True)
# print(W.count_nonzero())
epsilon = estimate_epsilon(dataset.data, d=d)
print(f"Epsilon: {epsilon}")
W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, eta=lambda x: np.exp(-4 * x))
sigma = ((1 - 5.0 / (np.e ** 4)) * np.pi) / 32.0
# print(W.count_nonzero())

# Plot raw data
plot_labels = np.full(shape=NUM_PLOTTING_POINTS, fill_value=-1)
plot_labels[train_ind] = train_labels
# _, ax = plt.subplots()
# plot_data_with_labels(ax, dataset.data[:NUM_PLOTTING_POINTS], plot_labels)
# plt.show()


p = 2
# Solve the poisson problem with dirac RHS
poisson_dirac = pl.algorithms.Poisson(
    W,
    p=(p - 1),
    scale=0.5 * sigma * epsilon ** (d + 2) * n ** 2,
    solver="conjugate_gradient",
    normalization="combinatorial",
    spectral_cutoff=150,
    tol=1e-2,
    max_iter=1e6,
    rhs=None,
)
solution_dirac = poisson_dirac.fit(train_ind, train_labels)

# Solve the poisson problem with bump RHS
bump_width = 1e-1
rhs_bump = pl.algorithms.rhs.bump(
    dataset.data, train_ind, train_labels, bump_width=bump_width
)
poisson_bump = pl.algorithms.Poisson(
    W,
    p=(p - 1),
    scale=0.5 * sigma * epsilon ** (d + 2) * n ** 2,
    solver="conjugate_gradient",
    normalization="combinatorial",
    spectral_cutoff=150,
    tol=1e-2,
    max_iter=1e6,
    rhs=rhs_bump,
)
solution_bump = poisson_bump.fit(train_ind, train_labels)
# solution_bump = np.zeros_like(solution_dirac)

D = gl.graph(W - spdiags(W.diagonal(), 0, n, n)).degree_vector()
print(f"Mean of solution: {np.mean(solution_dirac, axis=0)}")
print(f"Weighted mean of solution: {np.dot(D, solution_dirac)}")

# Plot the solution
dist = cdist(
    dataset.data[:NUM_PLOTTING_POINTS],
    dataset.data[:NUM_PLOTTING_POINTS],
    metric="euclidean",
)
fig = plt.figure()
ax_dirac = fig.add_subplot(1, 2, 1, projection="3d")
plot_graph_function_with_triangulation(
    ax_dirac,
    dataset.data[:NUM_PLOTTING_POINTS],
    solution_dirac[:NUM_PLOTTING_POINTS, 0],
    dist=dist,
    max_dist=0.1,
)
ax_dirac.set_title(f"n: {n}; eps: {epsilon:.4f}; RHS: Dirac")

ax_bump = fig.add_subplot(1, 2, 2, projection="3d")
plot_graph_function_with_triangulation(
    ax_bump,
    dataset.data[:NUM_PLOTTING_POINTS],
    solution_bump[:NUM_PLOTTING_POINTS, 0],
    dist=dist,
    max_dist=0.1,
)
ax_bump.set_title(f"n: {n}; eps: {epsilon:.4f}; RHS: Bump with width {bump_width}")

plt.show()
