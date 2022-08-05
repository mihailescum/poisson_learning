import numpy as np
import pandas as pd
import logging
import multiprocessing
from functools import partial
import copy

import poissonlearning as pl

import utils
import storage

LOGGER = logging.getLogger(name="ex.line")
logging.basicConfig(level="INFO")


NUM_TRIALS = 1


def run_trial(experiments, seed):
    LOGGER.info(f"Running trial with seed='{seed}'")
    rng = np.random.default_rng(seed=seed)

    label_locations = experiments[0]["label_locations"]

    data, labels = pl.datasets.line.generate(0.0, 1.0, 1000000, rng=rng)
    data = np.concatenate([label_locations[:, np.newaxis], data])
    labels = np.concatenate([np.array([0, 1]), labels])

    trial_result = []
    for experiment in experiments:
        n = experiment["n"]
        dataset = pl.datasets.Dataset(data[:n].copy(), labels[:n].copy(), metric="raw")

        rho2 = 1  # Density of the probability distribution
        solution = utils.run_experiment_poisson(dataset, experiment, rho2=rho2,)

        for s in solution:
            indices_largest_component = s["largest_component"]
            result = pd.Series(
                s["solution"], index=dataset.data[indices_largest_component, 0]
            ).sort_index()

            item = copy.deepcopy(experiment)
            item["bump"] = s["bump"]

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


# Run experiments
if __name__ == "__main__":
    experiments = storage.load_experiments("line", "examples/experiments")

    func = partial(run_trial, experiments)
    # pool = multiprocessing.Pool(2)
    # trial_results = pool.map(func, range(NUM_TRIALS))
    trial_results = [func(seed) for seed in range(NUM_TRIALS)]
    results = [x for flatten in trial_results for x in flatten]

    storage.save_results(results, name="line", folder="results")
