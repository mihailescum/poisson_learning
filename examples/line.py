from cProfile import run
from cmath import exp
import numpy as np
import pandas as pd
import logging
import copy

import matplotlib.pyplot as plt

import poissonlearning as pl
import graphlearning as gl

import setup
import storage

LOGGER = logging.getLogger(name=__name__)
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
]
"""{
        "n": 20000,
        "eps": 0.007427616,
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
        "n": 100000,
        "eps": 0.0017269388197,
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
        "traiexperiment_setupn_indices": [0, 1],
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


def run_experiment(dataset, experiment):
    dataset = copy.deepcopy(dataset)

    LOGGER.info(f"Experiment: {experiment}")
    n, d = dataset.data.shape

    train_ind = experiment["train_indices"]
    train_labels = dataset.labels[train_ind]

    W = setup.build_weight_matrix(dataset, experiment)
    sigma = setup.get_normalization_constant(experiment["kernel"], d)
    rhs = setup.get_rhs(dataset, experiment)

    scale = 0.5 * sigma * experiment["eps"] ** 2 * n ** 2
    poisson = pl.algorithms.Poisson(
        W,
        p=1,
        scale=scale,
        solver="conjugate_gradient",
        normalization="combinatorial",
        tol=1e-3,
        max_iter=1e5,
        rhs=rhs,
    )
    solution = poisson.fit(train_ind, train_labels)[:, 0]
    return solution


NUM_TRIALS = 1

# Run experiments
if __name__ == "__main__":
    for trial in range(NUM_TRIALS):
        seed = trial
        rng = np.random.default_rng(seed=seed)

        data, labels = pl.datasets.line.generate(0.0, 1.0, 1000000, rng=rng)
        data = np.concatenate([LABEL_LOCATIONS[:, np.newaxis], data])
        labels = np.concatenate([np.array([0, 1]), labels])

        for experiment in experiments:
            n = experiment["n"]
            dataset = pl.datasets.Dataset(data[:n], labels[:n], metric="raw")
            solution = run_experiment(dataset, experiment)

            result = pd.Series(solution, index=dataset.data[:, 0]).sort_index()
            if "results" not in experiment:
                experiment["results"] = []

            experiment["results"].append({"seed": seed, "result": result})

    storage.save_experiments(experiments, name="line", folder="results")
