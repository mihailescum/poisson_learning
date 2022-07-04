import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import poissonlearning as pl
import graphlearning as gl

from plotting import get_plot_colors

import logging

logging.basicConfig(level="INFO")

NUM_TRAINING_POINTS = 3000

NUM_PLOTTING_POINTS = 2000
LABEL_LOCATIONS = np.array([[0.4], [0.8]])

P = [2, 4, 8, 12, 16, 20, 26, 32]
P_HOMOTOPY = [
    2.5,
    3,
    3.5,
    4,
    4.75,
    5.5,
    6.25,
    7,
    8,
    9,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
]
SIGMA = {
    2: 0.10568,
    4: 0.035052,
    8: 0.010583,
    12: 0.00549624,
    16: 0.00358198,
    20: 0.002624,
    26: 0.00185778,
    32: 0.00143257,
}

dataset = pl.datasets.Dataset.load("line", "raw", NUM_TRAINING_POINTS - 2)
dataset.data = np.concatenate([LABEL_LOCATIONS, dataset.data])
dataset.labels = np.concatenate([np.array([0, 1]), dataset.labels])

n, d = dataset.data.shape

train_ind = np.array([0, 1])
train_labels = dataset.labels[train_ind]


def estimate_epsilon(data):
    n = data.shape[0]
    epsilon = 15 * np.log(n) / n

    return epsilon


print("Creating weight matrix...")
epsilon = estimate_epsilon(dataset.data)
print(f"Epsilon: {epsilon}")
W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, kernel="gaussian")
W *= epsilon ** (-d)

print("Solving Poisson problem...")
bump_width = "dirac"
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
    p=(max(P) - 1),
    homotopy_steps=P_HOMOTOPY,
    scale=None,
    eps_scale=epsilon,
    solver="variational",
    normalization="combinatorial",
    spectral_cutoff=150,
    tol=1e-6,
    max_iter=500,
    rhs=rhs,
)
solution_homotopy = poisson.fit(train_ind, train_labels)[1]
results = {}
for p in P:
    result = solution_homotopy[p]
    scale = 0.5 * SIGMA[p] * n ** 2 * epsilon ** (p + d)
    result = scale ** (1 / (p - 1)) * result
    results[p] = pd.Series(result, index=dataset.data[:, 0]).sort_index()

print("Plotting...")
colors = get_plot_colors(n=len(P))
fig, ax = plt.subplots(1, 1)
for i, p in enumerate(P):
    solution = results[p].copy()
    label_values = solution[LABEL_LOCATIONS[:, 0]]
    solution = solution[~solution.index.isin(LABEL_LOCATIONS[:, 0])]
    sample = solution.sample(NUM_PLOTTING_POINTS - label_values.size, random_state=1)
    sample = pd.concat([sample, label_values])
    sample = sample.sort_index()

    ax.plot(sample, c=colors[i], label=f"p={p}")
ax.grid()
ax.legend()

plt.show()
