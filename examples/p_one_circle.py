import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import copy
import logging

import poissonlearning as pl

import utils
import storage

LOGGER = logging.getLogger("ex.p_one_circle")
logging.basicConfig(level=logging.INFO)
logging.getLogger("pl.numerics").setLevel(logging.WARNING)

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
        solution = utils.run_experiment_poisson(dataset, experiment, rho2=rho2)

        for s in solution:
            indices_largest_component = s["largest_component"]

            for p, homotopy_solution in s["solution"].items():
                result = pd.DataFrame(columns=["x", "y", "z"])
                result["x"] = dataset.data[indices_largest_component, 0]
                result["y"] = dataset.data[indices_largest_component, 1]
                result["z"] = homotopy_solution[:, 0]

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

    storage.save_results(results, name="p_one_circle", folder="results")
