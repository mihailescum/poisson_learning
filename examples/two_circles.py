import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import poissonlearning as pl
import graphlearning as gl

from plotting import plot_graph_function_with_triangulation, plot_data_with_labels

NUM_TRAINING_POINTS = 5000
NUM_PLOTTING_POINTS = 10000
if NUM_PLOTTING_POINTS > NUM_TRAINING_POINTS:
    NUM_PLOTTING_POINTS = NUM_TRAINING_POINTS

# Load the two_circles dataset
dataset = pl.datasets.Dataset.load("two_circles", "raw", NUM_TRAINING_POINTS)
n, d = dataset.data.shape

train_ind = np.array([345, 718])
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
print(epsilon)
W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, kernel="gaussian")
# print(W.count_nonzero())

# Plot raw data
plot_labels = np.full(shape=NUM_PLOTTING_POINTS, fill_value=-1)
plot_labels[train_ind] = train_labels
# plot_data_with_labels(dataset.data[:NUM_PLOTTING_POINTS], plot_labels)
# plt.show()

# Build RHS of the poisson equation
bump_width = 3e-2
rhs = None  # pl.algorithms.rhs.bump(dataset.data, train_ind, train_labels, bump_width=bump_width)

# Solve the poisson problem
p = 2
poisson_dirac = pl.algorithms.Poisson(
    W,
    p=(p - 1),
    solver="conjugate_gradient",
    normalization="combinatorial",
    spectral_cutoff=150,
    tol=1e-3,
    max_iter=1e6,
    rhs=rhs,
)
solution = poisson_dirac.fit(train_ind, train_labels)
# Normalize solution
mu = n * epsilon ** (p + d)
solution = mu ** (1 / p - 1) * solution

D = gl.graph(W).degree_vector()
print(f"Mean of solution: {solution[:,0].mean()}")  # np.dot(solution[:, 0], D)}")

# Plot the solution
dist = cdist(
    dataset.data[:NUM_PLOTTING_POINTS],
    dataset.data[:NUM_PLOTTING_POINTS],
    metric="euclidean",
)
fig, ax = plot_graph_function_with_triangulation(
    dataset.data[:NUM_PLOTTING_POINTS],
    solution[:NUM_PLOTTING_POINTS, 0],
    dist=dist,
    max_dist=0.1,
)
if rhs is not None:
    fig.suptitle(f"n: {n}; eps: {epsilon}; RHS: Bump with width {bump_width}")
else:
    fig.suptitle(f"n: {n}; eps: {epsilon}; RHS: Dirac")

plt.show()
