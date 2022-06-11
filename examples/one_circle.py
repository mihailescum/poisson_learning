import numpy as np
import pandas as pd
import logging

import poissonlearning as pl

import setup
import storage

LOGGER = logging.getLogger("ex.one_circle")
logging.basicConfig(level="INFO")


# `n` will be overwritten with the number of nodes from the largest connected component
experiments = [
    {
        "n": 10000,
        "eps": 0.02452177,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
    {
        "n": 30000,
        "eps": 0.01552237,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
]
"""    {
        "n": 50000,
        "eps": 0.01250796,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
    {
        "n": 100000,
        "eps": 0.00930454,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
    {
        "n": 300000,
        "eps": 0.00578709,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
    {
        "n": 700000,
        "eps": 0.00399516,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
    {
        "n": 1000000,
        "eps": 0.00341476,
        "kernel": "gaussian",
        "train_indices": [0, 1],
        "bump": "dirac",
    },
]"""

NUM_TRIALS = 1

if __name__ == "__main__":
    for trial in range(NUM_TRIALS):
        seed = trial
        rng = np.random.default_rng(seed=seed)

        data, labels = pl.datasets.one_circle.generate(
            center=np.array([0, 0]), r=1, size=1000000, rng=rng,
        )
        # Add two label points at the beginning of the data set
        data[0] = np.array([[-2 / 3 * 1, 0]])
        data[1] = np.array([[2 / 3 * 1, 0]])
        labels[0] = 0
        labels[1] = 1

        for experiment in experiments:
            n = experiment["n"]
            dataset = pl.datasets.Dataset(data[:n], labels[:n], metric="raw")
            d = dataset.data.shape[1]

            sigma = setup.get_normalization_constant(experiment["kernel"], d, p=2)
            scale = 0.5 * sigma * experiment["eps"] ** 2 * n ** 2
            solution, indices_largest_component = setup.run_experiment_poisson(
                dataset, experiment, scale
            )

            if "results" not in experiment:
                experiment["results"] = []

            result = pd.DataFrame(columns=["x", "y", "z"])
            result["x"] = dataset.data[indices_largest_component, 0]
            result["y"] = dataset.data[indices_largest_component, 1]
            result["z"] = solution

            experiment["results"].append({"seed": seed, "solution": result})

    storage.save_experiments(experiments, name="one_circle", folder="results")
