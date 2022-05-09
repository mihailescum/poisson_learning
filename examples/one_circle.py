import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import poissonlearning as pl
import graphlearning as gl

from plotting import plot_graph_function_with_triangulation

NUM_TRAINING_POINTS = 200000
NUM_PLOTTING_POINTS = 10000

# Load the two_circles dataset
dataset = pl.datasets.Dataset.load("one_circle", "raw", NUM_TRAINING_POINTS)

train_ind = np.array([0, 1])
train_labels = dataset.labels[train_ind]


def estimate_epsilon(data):
    n = data.shape[0]
    conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
    epsilon = 0.5 * np.log(n) ** (1 / 9) * conn_radius

    return epsilon


# Build the weight matrix
print("Creating weight matrix...")
epsilon = estimate_epsilon(dataset.data)
W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, kernel="gaussian")

# Remove sigularities by only keeping the largest connected component
G = gl.graph(W)
Grestricted, indices = G.largest_connected_component()
dataset.data = dataset.data[indices]
dataset.labels = dataset.labels[indices]
epsilon = estimate_epsilon(dataset.data)
W = Grestricted.weight_matrix
n, d = dataset.data.shape
print(f"Epsilon: {epsilon}")

NUM_PLOTTING_POINTS = min(n, NUM_PLOTTING_POINTS)

# W = epsilon_ball_test(dataset.data, epsilon, kernel="uniform")
# W *= epsilon ** (-d)
# normalization constant, integrate B_1(0): eta(r)(r*cos(t))^p dtdr
sigma = np.pi * 1 / 32 * (1 - 5 / np.e ** 4)
# sigma = np.pi * 0.25
print(W.count_nonzero() / n ** 2 * 100)

# Plot raw data
plot_labels = np.full(shape=NUM_PLOTTING_POINTS, fill_value=-1)
plot_labels[train_ind] = train_labels
# _, ax = plt.subplots()
# plot_data_with_labels(ax, dataset.data[:NUM_PLOTTING_POINTS], plot_labels)
# plt.show()


# Solve the poisson problem with dirac RHS
print("Solving Poisson problem...")
p = 2
bump_width = "dirac"
if isinstance(bump_width, float):
    rhs = pl.algorithms.rhs.bump(
        dataset.data, train_ind, train_labels, bump_width=bump_width
    )
elif bump_width == "dirac":
    rhs = None
else:
    raise ValueError("Invalid bump widht, must be either float or 'dirac'.")
poisson = pl.algorithms.Poisson(
    W,
    p=(p - 1),
    scale=1.24 * 0.5 * sigma * epsilon ** (p + d) * n ** 2,
    solver="conjugate_gradient",
    normalization="combinatorial",
    spectral_cutoff=50,
    tol=1e-1,
    max_iter=1e6,
    rhs=rhs,
)
solution = poisson.fit(train_ind, train_labels)[:, 0]

# Remove values at the labels for plotting
solution[train_ind] = np.where(train_labels == 0, np.inf, -np.inf)

# Compute the analytic continuum limit
print("Computing analytic solution...")
green_first_label = pl.datasets.one_circle.greens_function(
    x=dataset.data, z=dataset.data[train_ind[0]],
)
green_second_label = pl.datasets.one_circle.greens_function(
    x=dataset.data, z=dataset.data[train_ind[1]],
)
solution_analytic = 0.5 * green_first_label - 0.5 * green_second_label

# Plot the solution
print("Plotting...")
dist = cdist(
    dataset.data[:NUM_PLOTTING_POINTS],
    dataset.data[:NUM_PLOTTING_POINTS],
    metric="euclidean",
)
fig = plt.figure()
ax_solution = fig.add_subplot(1, 3, 1, projection="3d")
plot_graph_function_with_triangulation(
    ax_solution,
    dataset.data[:NUM_PLOTTING_POINTS],
    solution[:NUM_PLOTTING_POINTS],
    dist=dist,
    max_dist=0.1,
)
ax_solution.set_title(f"n: {n}; eps: {epsilon:.4f}; RHS: {bump_width}")

ax_analytic = fig.add_subplot(1, 3, 2, projection="3d")
plot_graph_function_with_triangulation(
    ax_analytic,
    dataset.data[:NUM_PLOTTING_POINTS],
    solution_analytic[:NUM_PLOTTING_POINTS],
    dist=dist,
    max_dist=0.1,
)
ax_analytic.set_title(f"eps: {epsilon:.4f}; Analytic solution to continuum problem")

ax_error = fig.add_subplot(1, 3, 3, projection="3d")
solution[np.isposinf(solution)] = solution[np.isfinite(solution)].max()
solution[np.isneginf(solution)] = solution[np.isfinite(solution)].min()
solution_analytic[np.isposinf(solution_analytic)] = solution_analytic[
    np.isfinite(solution_analytic)
].max()
solution_analytic[np.isneginf(solution_analytic)] = solution_analytic[
    np.isfinite(solution_analytic)
].min()
plot_graph_function_with_triangulation(
    ax_error,
    dataset.data[:NUM_PLOTTING_POINTS],
    np.abs(solution_analytic[:NUM_PLOTTING_POINTS] - solution[:NUM_PLOTTING_POINTS]),
    dist=dist,
    max_dist=0.1,
)
ax_error.set_title(f"eps: {epsilon:.4f}; Error")

print(
    np.mean(solution_analytic[np.isfinite(solution)] / solution[np.isfinite(solution)])
)

print(
    f"L1 error: {np.abs(solution_analytic[np.isfinite(solution)] - solution[np.isfinite(solution)]).mean()}"
)

plt.show()
