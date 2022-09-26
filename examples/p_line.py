import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import copy
import logging

import poissonlearning as pl

import utils
import storage

LOGGER = logging.getLogger("ex.p_line")
logging.basicConfig(level=logging.INFO)
logging.getLogger("pl.numerics").setLevel(logging.WARNING)

NUM_TRIALS = 12
NUM_THREADS = 4


def run_trial(experiments, seed):
    LOGGER.info(f"Running trial with seed='{seed}'")
    rng = np.random.default_rng(seed=seed)

    data = rng.uniform(-1.0, 1.0, size=50000)[:, np.newaxis]
    labels = np.sign(data)

    trial_result = []
    for experiment in experiments:
        n = experiment["n"]
        data_local = data[:n].copy()
        labels_local = labels[:n].copy()
        labels_local -= np.mean(labels_local)
        dataset = pl.datasets.Dataset(data_local, labels_local, metric="raw")

        rho2 = 0.25
        experiment["label_locations"] = dataset.labels
        solution = utils.run_experiment_poisson(dataset, experiment, rho2=rho2)

        for s in solution:
            indices_largest_component = s["largest_component"]

            for p, homotopy_solution in s["solution"].items():
                result = pd.Series(
                    homotopy_solution[:, 0],
                    index=dataset.data[indices_largest_component, 0],
                ).sort_index()

                item = copy.deepcopy(experiment)
                item["bump"] = s["bump"]
                item["p"] = p
                item.pop("label_locations", None)

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
    experiments = storage.load_experiments("p_line", "examples/experiments")

    NUM_THREADS = min(NUM_THREADS, NUM_TRIALS)
    func = partial(run_trial, experiments)
    if NUM_THREADS > 1:
        pool = multiprocessing.Pool(NUM_THREADS)
        trial_results = pool.map(func, range(NUM_TRIALS))
    else:
        trial_results = [func(seed) for seed in range(NUM_TRIALS)]
    results = [x for flatten in trial_results for x in flatten]

    storage.save_results(results, name="p_line", folder="results")
