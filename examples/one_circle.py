import numpy as np
import pandas as pd
import logging
import multiprocessing
from functools import partial
import copy

import poissonlearning as pl

import utils
import storage

LOGGER = logging.getLogger("ex.one_circle")
logging.basicConfig(level="INFO")


# `n` will be overwritten with the number of nodes from the largest connected component
# experiments =
"""
    {
        "n": 1000000,
        "eps": 0.00341476,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
]"""

NUM_TRIALS = 12


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
        dataset = pl.datasets.Dataset(data[:n], labels[:n], metric="raw")
        d = dataset.data.shape[1]

        sigma = utils.get_normalization_constant(experiment["kernel"], d)
        rho2 = 1.0 / (np.pi * np.pi)  # Density of the probability distribution
        scale = 0.5 * sigma * rho2 * experiment["eps"] ** 2 * n ** 2
        solution, indices_largest_component = utils.run_experiment_poisson(
            dataset, experiment, scale, tol=1e-8,
        )

        for s in solution:
            result = pd.DataFrame(columns=["x", "y", "z"])
            result["x"] = dataset.data[indices_largest_component, 0]
            result["y"] = dataset.data[indices_largest_component, 1]
            result["z"] = s["solution"]

            subresult = copy.deepcopy(experiment)
            subresult["bump"] = s["bump"]
            subresult["seed"] = seed
            subresult["solution"] = result
            trial_result.append(subresult)
    return trial_result


if __name__ == "__main__":
    experiments = storage.load_experiments("one_circle", "examples/experiments")

    func = partial(run_trial, experiments)
    pool = multiprocessing.Pool(4)
    trial_results = pool.map(func, range(NUM_TRIALS))
    # trial_results = [func(seed) for seed in range(NUM_TRIALS)]
    results = [x for flatten in trial_results for x in flatten]

    storage.save_results(results, name="one_circle", folder="results")
