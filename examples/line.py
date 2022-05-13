import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import poissonlearning as pl
import graphlearning as gl

from plotting import get_plot_colors


def estimate_epsilon(n):
    epsilon = 15 * np.log(n) / n
    return epsilon


# Set-up experiments
experiments = [
    {"n": 1000, "eps": 0.103616329, "bump": "dirac", "kernel": "uniform"},
    {"n": 10000, "eps": 0.01381551, "bump": "dirac", "kernel": "uniform"},
    {"n": 20000, "eps": 0.007427616, "bump": "dirac", "kernel": "uniform"},
    {"n": 50000, "eps": 0.003245933485, "bump": "dirac", "kernel": "uniform"},
    {"n": 100000, "eps": 0.0017269388197, "bump": "dirac", "kernel": "uniform"},
    {"n": 1000, "eps": 0.103616329, "bump": "dirac", "kernel": "gaussian"},
    {"n": 10000, "eps": 0.01381551, "bump": "dirac", "kernel": "gaussian"},
    {"n": 20000, "eps": 0.007427616, "bump": "dirac", "kernel": "gaussian"},
    {"n": 50000, "eps": 0.003245933485, "bump": "dirac", "kernel": "gaussian"},
    {"n": 100000, "eps": 0.0017269388197, "bump": "dirac", "kernel": "gaussian"},
    {"n": 20000, "eps": 0.007427616, "bump": 2e-1, "kernel": "gaussian"},
    {"n": 20000, "eps": 0.007427616, "bump": 1e-1, "kernel": "gaussian"},
    {"n": 20000, "eps": 0.007427616, "bump": 1e-2, "kernel": "gaussian"},
]

NUM_PLOTTING_POINTS = 5000
LABEL_LOCATIONS = np.array([0.4, 0.8])

# Run experiments
for experiment in experiments:
    print(f"Experiment: {experiment}")
    dataset = pl.datasets.Dataset.load("line", "raw", experiment["n"] - 2)

    dataset.data = np.concatenate([LABEL_LOCATIONS[:, np.newaxis], dataset.data])
    dataset.labels = np.concatenate([np.array([0, 1]), dataset.labels])
    n, d = dataset.data.shape

    train_ind = np.array([0, 1])
    train_labels = dataset.labels[train_ind]

    print("Creating weight matrix...")
    W = gl.weightmatrix.epsilon_ball(
        dataset.data, experiment["eps"], kernel=experiment["kernel"]
    )
    W *= experiment["eps"] ** (-d)

    # normalization constant
    if experiment["kernel"] == "uniform":
        # integrate -1 to 1: t^2 dt
        sigma = 2 / 3
    elif experiment["kernel"] == "gaussian":
        # integrate -1 to 1: exp(-4t^2)t^2 dt
        sigma = 0.10568126

    print("Solving Poisson problem...")
    if isinstance(experiment["bump"], float):
        rhs = pl.algorithms.rhs.bump(
            dataset.data, train_ind, train_labels, bump_width=experiment["bump"]
        )
    elif experiment["bump"] == "dirac":
        rhs = None

    poisson = pl.algorithms.Poisson(
        W,
        p=1,
        scale=0.5 * sigma * experiment["eps"] ** 2 * n ** 2,
        solver="conjugate_gradient",
        normalization="combinatorial",
        spectral_cutoff=150,
        tol=1e-3,
        max_iter=1e5,
        rhs=rhs,
    )
    solution = poisson.fit(train_ind, train_labels)[:, 0]
    result = pd.Series(solution, index=dataset.data[:, 0]).sort_index()
    experiment["result"] = result


# Compute the analytic continuum limit
print("Computing analytic solution...")
x = np.linspace(0, 1, 100000)
green_first_label = pl.datasets.line.greens_function(x=x, z=LABEL_LOCATIONS[0])
green_second_label = pl.datasets.line.greens_function(x=x, z=LABEL_LOCATIONS[1])
solution_analytic = pd.Series(
    0.5 * green_first_label - 0.5 * green_second_label, index=x
)

# Compute errors
print("Computing errors...")
for experiment in experiments:
    solution = experiment["result"]
    analytic = solution_analytic.reindex(
        solution.index, method="nearest", limit=1
    ).interpolate()
    error_L1 = np.abs(solution - analytic).mean()
    experiment["error_L1"] = error_L1

print("Plotting...")
# Convergence of dirac solutions
colors = get_plot_colors(n=5)
fig_dirac, ax_dirac = plt.subplots(1, 1)
ax_dirac.plot(
    solution_analytic.sample(NUM_PLOTTING_POINTS, random_state=1).sort_index(),
    label=f"analytic solution",
    c=colors[0],
)

kernel = "gaussian"
ex_convergence = [
    e
    for e in experiments
    if e["bump"] == "dirac" and e["kernel"] == kernel and e["n"] >= 10000
]
for i, e in enumerate(ex_convergence, start=1):
    solution = e["result"].copy()
    label_values = solution[LABEL_LOCATIONS]
    solution = solution[~solution.index.isin(LABEL_LOCATIONS)]
    sample = solution.sample(NUM_PLOTTING_POINTS - label_values.size, random_state=1)
    sample = pd.concat([sample, label_values])
    sample = sample.sort_index()

    ax_dirac.plot(
        sample, label=f"n={e['n']}", c=colors[i],
    )

ax_dirac.set_title(f"Convergence of Dirac RHS with kernel '{kernel}'")
ax_dirac.legend()
ax_dirac.grid()

# Convergence of bumps
colors = get_plot_colors(n=4)
fig_bump, ax_bump = plt.subplots(1, 1)
n = 20000
kernel = "gaussian"
ex_bumps = [e for e in experiments if e["n"] == n and e["kernel"] == kernel]
for i, e in enumerate(ex_bumps):
    solution = e["result"].copy()
    label_values = solution[LABEL_LOCATIONS]
    solution = solution[~solution.index.isin(LABEL_LOCATIONS)]
    sample = solution.sample(NUM_PLOTTING_POINTS - label_values.size, random_state=1)
    sample = pd.concat([sample, label_values])
    sample = sample.sort_index()

    ax_bump.plot(
        sample, label=f"bump width={e['bump']}", c=colors[i],
    )

ax_bump.set_title(f"Convergence of smoothed RHS with kernel '{kernel}', n={n}")
ax_bump.legend()
ax_bump.grid()

# L1 errors
fig_error, ax_error = plt.subplots(1, 1)
ex_dirac_uniform = [
    e for e in experiments if e["kernel"] == "uniform" and e["bump"] == "dirac"
]
ex_dirac_gaussian = [
    e for e in experiments if e["kernel"] == "gaussian" and e["bump"] == "dirac"
]

ax_error.plot(
    [e["n"] for e in ex_dirac_uniform],
    [e["error_L1"] for e in ex_dirac_uniform],
    label="kernel=uniform",
)
ax_error.plot(
    [e["n"] for e in ex_dirac_gaussian],
    [e["error_L1"] for e in ex_dirac_gaussian],
    label="kernel=gaussian",
)

ax_error.set_xscale("log")
ax_error.set_title("L1 error of solution with dirac RHS")
ax_error.legend()
ax_error.grid()

plt.show()
