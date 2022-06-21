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


NUM_TRIALS = 10


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
        dataset = pl.datasets.Dataset(data[:n], labels[:n], metric="raw")
        d = dataset.data.shape[1]

        sigma = utils.get_normalization_constant(experiment["kernel"], d, p=2)
        rho2 = 1  # Density of the probability distribution
        scale = 0.5 * sigma * rho2 * experiment["eps"] ** 2 * n ** 2
        solution, indices_largest_component = utils.run_experiment_poisson(
            dataset, experiment, scale, tol=1e-8,
        )

        for s in solution:
            result = pd.Series(
                s["solution"], index=dataset.data[indices_largest_component, 0]
            ).sort_index()

            subresult = copy.deepcopy(experiment)
            subresult["bump"] = s["bump"]
            subresult["seed"] = seed
            subresult["solution"] = result
            trial_result.append(subresult)
    return trial_result


# Run experiments
if __name__ == "__main__":
    # pool = multiprocessing.Pool(2)
    experiments = storage.load_experiments("line", "examples/experiments")

    func = partial(run_trial, experiments)
    # trial_results = pool.map(func, range(NUM_TRIALS))
    trial_results = [func(seed) for seed in range(NUM_TRIALS)]
    results = [x for flatten in trial_results for x in flatten]

    storage.save_results(results, name="line", folder="results")
