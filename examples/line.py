import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import poissonlearning as pl
import graphlearning as gl

from plotting import get_plot_colors


NUM_TRAINING_POINTS = [1000, 10000, 30000, 50000, 100000]
BUMP_WIDTHS = ["dirac"]  # [1e-1, 1e-2, 1e-3, "dirac"]
KERNELS = ["uniform", "gaussian"]
NUM_PLOTTING_POINTS = 1000

LABEL_LOCATIONS = np.array([[0.2], [0.8]])


def estimate_epsilon(data):
    n = data.shape[0]
    epsilon = 15 * np.log(n) / n

    return epsilon


results = {}
for kernel in KERNELS:
    print(f"\nKernel: {kernel}")
    results[kernel] = {}
    for training_points in NUM_TRAINING_POINTS:
        print(f"# training points: {training_points}")
        results[kernel][training_points] = {}

        dataset = pl.datasets.Dataset.load("line", "raw", training_points - 2)

        dataset.data = np.concatenate([LABEL_LOCATIONS, dataset.data])
        dataset.labels = np.concatenate([np.array([0, 1]), dataset.labels])

        n, d = dataset.data.shape

        train_ind = np.array([0, 1])
        train_labels = dataset.labels[train_ind]

        print("Creating weight matrix...")
        epsilon = estimate_epsilon(dataset.data)
        print(f"Epsilon: {epsilon}")
        W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, kernel=kernel)
        # W = epsilon_ball_test(dataset.data, epsilon, kernel="uniform")
        W *= epsilon ** (-d)
        if kernel == "uniform":
            # normalization constant, integrate -1 to 1: t^2 dt
            sigma = 2 / 3
        elif kernel == "gaussian":
            # integrate -1 to 1: exp(-4t^2)t^2 dt
            sigma = 0.10568126
        else:
            raise ValueError(f"Unknown kernel {kernel}")

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
                scale=0.5 * sigma * epsilon ** 2 * n ** 2,
                eps_scale=epsilon,
                solver="conjugate_gradient",
                normalization="combinatorial",
                spectral_cutoff=150,
                tol=1e-2,
                max_iter=1e5,
                rhs=rhs,
            )
            solution = poisson.fit(train_ind, train_labels)[:, 0]
            result = pd.Series(solution, index=dataset.data[:, 0]).sort_index()
            results[kernel][training_points][bump_width] = result

print("Computing analytic solution...")
# Compute the analytic continuum limit
x = results[KERNELS[0]][max(NUM_TRAINING_POINTS)][BUMP_WIDTHS[0]].index.to_numpy()[
    :, np.newaxis
]
green_first_label = pl.datasets.line.greens_function(x=x, z=dataset.data[train_ind[0]],)
green_second_label = pl.datasets.line.greens_function(
    x=x, z=dataset.data[train_ind[1]],
)
solution_analytic = 0.5 * green_first_label - 0.5 * green_second_label
results["analytic"] = pd.Series(solution_analytic, index=x[:, 0]).sort_index()


print("Plotting...")
kernel_plotting = "gaussian"

# Convergence of dirac solutions
colors = get_plot_colors(n=len(NUM_TRAINING_POINTS) + 1)
fig_dirac, ax_dirac = plt.subplots(1, 1)
ax_dirac.plot(
    results["analytic"].sample(NUM_PLOTTING_POINTS, random_state=1).sort_index(),
    label=f"analytic solution",
    c=colors[0],
)
for i, n in enumerate(NUM_TRAINING_POINTS):
    solution = results[kernel_plotting][n]["dirac"].copy()
    label_values = solution[LABEL_LOCATIONS[:, 0]]
    solution = solution[~solution.index.isin(LABEL_LOCATIONS[:, 0])]
    sample = solution.sample(NUM_PLOTTING_POINTS - label_values.size, random_state=1)
    sample = pd.concat([sample, label_values])
    sample = sample.sort_index()

    ax_dirac.plot(
        sample, label=f"n={n}", c=colors[i + 1],
    )

ax_dirac.set_title(f"Convergence of Dirac RHS with kernel {kernel_plotting}")
ax_dirac.legend()
ax_dirac.grid()

# Convergence of bumps
colors = get_plot_colors(n=len(BUMP_WIDTHS))
fig_bump, ax_bump = plt.subplots(1, 1)
n_max = max(NUM_TRAINING_POINTS)
for i, bump_width in enumerate(BUMP_WIDTHS):
    solution = results[kernel_plotting][n_max][bump_width].copy()
    label_values = solution[LABEL_LOCATIONS[:, 0]]
    solution = solution[~solution.index.isin(LABEL_LOCATIONS[:, 0])]
    sample = solution.sample(NUM_PLOTTING_POINTS - label_values.size, random_state=1)
    sample = pd.concat([sample, label_values])
    sample = sample.sort_index()

    ax_bump.plot(
        sample, label=f"bump width={bump_width}", c=colors[i],
    )

ax_bump.set_title(f"Convergence of smoothed RHS with kernel {kernel_plotting}")
ax_bump.legend()
ax_bump.grid()

# L1 errors
bump_width = "dirac"
errors = {kernel: {} for kernel in KERNELS}
for kernel in KERNELS:
    for n in NUM_TRAINING_POINTS:
        solution = results[kernel][n][bump_width]
        analytic = results["analytic"].reindex(solution.index).interpolate()
        error_L1 = np.abs(solution - analytic).mean()
        errors[kernel][n] = error_L1

fig_error, ax_error = plt.subplots(1, 1)
for kernel in KERNELS:
    ax_error.plot(
        list(errors[kernel].keys()),
        list(errors[kernel].values()),
        marker="x",
        ls="-",
        label=kernel,
    )
ax_error.set_xscale("log")
ax_error.set_title(f"L1 Error compared with RHS {bump_width} to analytic solution")
ax_error.grid()
ax_error.legend()

plt.show()
