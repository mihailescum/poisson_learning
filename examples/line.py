import numpy as np
import scipy.sparse

import matplotlib.pyplot as plt

import poissonlearning as pl
import graphlearning as gl

from plotting import get_photocopy_colors


NUM_TRAINING_POINTS = [1000]  # , 20000, 30000]
BUMP_WIDTHS = ["dirac"]  # [1e-1, 1e-2, 1e-3, "dirac"]
NUM_PLOTTING_POINTS = 1000


def estimate_epsilon(data):
    n = data.shape[0]
    epsilon = 10 * np.log(n) / n

    return epsilon


results = {}
for training_points in NUM_TRAINING_POINTS:
    print(f"# training points: {training_points}")
    results[training_points] = {}

    dataset = pl.datasets.Dataset.load("line", "raw", training_points - 2)

    dataset.data = np.concatenate([np.array([[0.5], [0.8]]), dataset.data])
    dataset.labels = np.concatenate([np.array([0, 1]), dataset.labels])

    n, d = dataset.data.shape

    train_ind = np.array([0, 1])
    train_labels = dataset.labels[train_ind]

    print("Creating weight matrix...")
    epsilon = estimate_epsilon(dataset.data)
    print(f"Epsilon: {epsilon}")
    W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, kernel="uniform")
    # W = epsilon_ball_test(dataset.data, epsilon, kernel="uniform")
    W *= epsilon ** (-d)
    # normalization constant, integrate -1 to 1: t^p dt
    sigma = 2 / 3

    print("Solving Poisson problem...")
    p = 4
    for bump_width in BUMP_WIDTHS:
        print(f"Bump width: {bump_width}")
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
            scale=0.5 * sigma * epsilon ** p * n ** 2,
            eps_scale=epsilon,
            solver="conjugate_gradient",
            normalization="combinatorial",
            spectral_cutoff=150,
            tol=1e-2,
            max_iter=1e5,
            rhs=rhs,
        )
        solution = poisson.fit(train_ind, train_labels)[:, 0]
        results[training_points][bump_width] = solution

print("Computing analytic solution...")
# Compute the analytic continuum limit
green_first_label = pl.datasets.line.greens_function(
    x=dataset.data, z=dataset.data[train_ind[0]],
)
green_second_label = pl.datasets.line.greens_function(
    x=dataset.data, z=dataset.data[train_ind[1]],
)
solution_analytic = 0.5 * green_first_label - 0.5 * green_second_label
results["analytic"] = solution_analytic


print("Plotting...")
plot_indices = np.argsort(dataset.data[:NUM_PLOTTING_POINTS, 0])
x = dataset.data[plot_indices, 0]

# Convergence of dirac solutions
colors = get_photocopy_colors(n=len(NUM_TRAINING_POINTS) + 1)
fig_dirac, ax_dirac = plt.subplots(1, 1)
ax_dirac.plot(
    x, results["analytic"][plot_indices], label=f"analytic solution", c=colors[0],
)
for i, training_points in enumerate(NUM_TRAINING_POINTS):
    ax_dirac.plot(
        x,
        results[training_points]["dirac"][plot_indices],
        label=f"n={training_points}",
        c=colors[i + 1],
    )

ax_dirac.set_title("Convergence of Dirac RHS")
ax_dirac.legend()
ax_dirac.grid()

# Convergence of bumps
colors = get_photocopy_colors(n=len(BUMP_WIDTHS))
fig_bump, ax_bump = plt.subplots(1, 1)
max_training_points = max(NUM_TRAINING_POINTS)
for i, bump_width in enumerate(BUMP_WIDTHS):
    ax_bump.plot(
        x,
        results[max_training_points][bump_width][plot_indices],
        label=f"bump width={bump_width}",
        c=colors[i],
    )

ax_bump.set_title("Convergence of smoothed RHS")
ax_bump.legend()
ax_bump.grid()

# print(f"L1 error: {np.nanmean(np.abs(solution_analytic - solution_dirac[:, 0]))}")

plt.show()
