from http.client import NON_AUTHORITATIVE_INFORMATION
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import copy
import logging

import poissonlearning as pl

import utils
import storage

LOGGER = logging.getLogger("ex.one_circle")
logging.basicConfig(level="INFO")

NUM_TRIALS = 1
NUM_THREADS = 4


def estimate_epsilon(n):
    factor = 0.7
    conn_radius = np.log(n) ** (3 / 4) / np.sqrt(n)
    epsilon = factor * np.log(n) ** (1 / 15) * conn_radius
    return epsilon


def run_trial(experiments, seed):
    LOGGER.info(f"Running trial with seed='{seed}'")
    rng = np.random.default_rng(seed=seed)

    data, labels = pl.datasets.one_circle.generate(
        center=np.array([0, 0]), r=1, size=1000000, rng=rng,
    )
    # Add two label points at the beginning of the data set

    label_locations = experiments[0]["label_locations"]
    data[0] = label_locations[0]
    data[1] = label_locations[1]
    labels[0] = 0
    labels[1] = 1

    trial_result = []
    for experiment in experiments:
        n = experiment["n"]
        dataset = pl.datasets.Dataset(data[:n].copy(), labels[:n].copy(), metric="raw")

        rho2 = 1.0 / (np.pi * np.pi)
        solution = utils.run_experiment_poisson(
            dataset, experiment, rho2=rho2, tol=1e-5, max_iter=200,
        )

        for s in solution:
            indices_largest_component = s["largest_component"]

            for p, homotopy_solution in s["solution"].items():
                result = pd.DataFrame(columns=["x", "y", "z"])
                result["x"] = dataset.data[indices_largest_component, 0]
                result["y"] = dataset.data[indices_largest_component, 1]
                result["z"] = homotopy_solution

                item = copy.deepcopy(experiment)
                item["bump"] = s["bump"]
                item["p"] = p

                if "eps" in s:
                    item["eps"] = s["eps"]
                    item.pop("n_neighbors", None)
                elif "n_neighbors" in s:
                    item["n_neighbors"] = s["n_neighbors"]
                    item.pop("eps", None)

                item["seed"] = seed
                item["solution"] = result
                trial_result.append(item)

    return trial_result


if __name__ == "__main__":
    experiments = storage.load_experiments("p_one_circle", "examples/experiments")

    NUM_THREADS = min(NUM_THREADS, NUM_TRIALS)
    func = partial(run_trial, experiments)
    if NUM_THREADS > 1:
        pool = multiprocessing.Pool(NUM_THREADS)
        trial_results = pool.map(func, range(NUM_TRIALS))
    else:
        trial_results = [func(seed) for seed in range(NUM_TRIALS)]
    results = [x for flatten in trial_results for x in flatten]

    storage.save_results(results, name="one_circle", folder="results")


""" results = []
for training_points in NUM_TRAINING_POINTS:
    print(f"\n# training points: {training_points}")

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

        solution = pd.DataFrame(columns=["x", "y", "z"])
        solution["x"] = dataset.data[:, 0]
        solution["y"] = dataset.data[:, 1]
        solution["z"] = solution_homotopy

        item = {}
        item["n"] = training_points
        item["p"] = p_homotopy
        item["solution"] = solution
        item["eps"] = epsilon
        item["bump"] = BUMP_WIDTH
        item["label_loations"] = dataset.data[train_ind]
        results.append(item)

storage.save_results(results, "p_one_circle", "results")
print("Plotting...")

# Plot solution
n = max(NUM_TRAINING_POINTS)
sample_size = NUM_PLOTTING_POINTS

fig_results = plt.figure()
for i, p_homotopy in enumerate(results[n], start=1):
    ax_solution = fig_results.add_subplot(
        2,  # int(np.floor(np.sqrt(len(results[n])))),
        5,  # int(np.floor(np.sqrt(len(results[n])))),
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
 """
