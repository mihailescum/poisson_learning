import numpy as np
import pandas as pd
import logging
import multiprocessing

import poissonlearning as pl

import utils
import storage

LOGGER = logging.getLogger(name="ex.line")
logging.basicConfig(level="INFO")

LABEL_LOCATIONS = np.array([0.4, 0.8])
# Set-up experiments
experiments = [
    {
        "n": 1000,
        "eps": 0.103616329,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 10000,
        "eps": 0.01381551,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 20000,
        "eps": 0.007427616,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 35000,
        "eps": 0.004484187146,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 50000,
        "eps": 0.003245933485,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 70000,
        "eps": 0.0023906251116,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 100000,
        "eps": 0.0017269388197,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
]
"""
    {
        "n": 300000,
        "eps": 0.000630576889,
        "bump": "dirac",
        "kernel": "uniform",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 1000,
        "eps": 0.103616329,
        "bump": "dirac",
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 10000,
        "eps": 0.01381551,
        "bump": "dirac",
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 20000,
        "eps": 0.007427616,
        "bump": "dirac",
        "kernel": "gaussian",
        "traiexperiment_experimentsn_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 50000,
        "eps": 0.003245933485,
        "bump": "dirac",
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 100000,
        "eps": 0.0017269388197,
        "bump": "dirac",
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 20000,
        "eps": 0.007427616,
        "bump": 2e-1,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 20000,
        "eps": 0.007427616,
        "bump": 1e-1,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
    {
        "n": 20000,
        "eps": 0.007427616,
        "bump": 1e-2,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "label_locations": LABEL_LOCATIONS,
    },
]"""


NUM_TRIALS = 4


def run_trial(seed):
    rng = np.random.default_rng(seed=seed)

    data, labels = pl.datasets.line.generate(0.0, 1.0, 1000000, rng=rng)
    data = np.concatenate([LABEL_LOCATIONS[:, np.newaxis], data])
    labels = np.concatenate([np.array([0, 1]), labels])

    trial_result = []
    for experiment in experiments:
        n = experiment["n"]
        dataset = pl.datasets.Dataset(data[:n], labels[:n], metric="raw")
        d = dataset.data.shape[1]

        sigma = utils.get_normalization_constant(experiment["kernel"], d, p=2)
        scale = 0.5 * sigma * experiment["eps"] ** 2 * n ** 2
        solution, indices_largest_component = utils.run_experiment_poisson(
            dataset, experiment, scale, tol=1e-8,
        )

        result = pd.Series(
            solution, index=dataset.data[indices_largest_component, 0]
        ).sort_index()

        trial_result.append({"seed": seed, "solution": result})
    return trial_result


# Run experiments
if __name__ == "__main__":
    pool = multiprocessing.Pool(4)
    trial_results = pool.map(run_trial, range(NUM_TRIALS))

    for i, experiment in enumerate(experiments):
        experiment["results"] = [trial[i] for trial in trial_results]

    storage.save_experiments(experiments, name="line", folder="results")
