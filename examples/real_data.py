import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

import graphlearning as gl
import poissonlearning as pl

import logging

import storage
import utils

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)-10s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger("ex.real_data")
logging.getLogger("pl.numerics").setLevel(logging.WARNING)
logging.getLogger("pl.poisson").setLevel(logging.WARNING)

NUM_TRIALS = 4
SEED_START = 2
NUM_THREADS = 2


def run_trial(experiments, seed):
    LOGGER.info(f"Running trial with seed='{seed}'")
    rng = np.random.default_rng(seed=seed)

    trial_results = []
    for experiment in experiments:
        dataset = pl.datasets.Dataset.load(
            experiment["dataset"], metric=experiment["dataset_metric"],
        )
        LOGGER.info(
            f"Dataset: {experiment['dataset']}, metric: {experiment['dataset_metric']}"
        )
        dataset_sample = dataset.sample(size=experiment["n"], rng=rng)
        G, indices_largest_component = utils.build_graph(
            dataset_sample, experiment, n_neighbors=experiment["n_neighbors"]
        )
        W = G.weight_matrix

        dataset_sample.data = dataset_sample.data[indices_largest_component]
        dataset_sample.labels = dataset_sample.labels[indices_largest_component]

        labels_per_class = (
            experiment["labels_per_class"]
            if isinstance(experiment["labels_per_class"], (list, np.ndarray))
            else [experiment["labels_per_class"]]
        )

        for num_train_labels in labels_per_class:
            num_train_labels = int(num_train_labels)
            train_ind = gl.trainsets.generate(
                dataset_sample.labels, rate=num_train_labels, seed=seed,
            )
            train_labels = dataset_sample.labels[train_ind]

            LOGGER.info(f"Fitting model..., labels per class: {num_train_labels}")
            model = pl.algorithms.Poisson(
                W,
                p=(max(experiment["p"]) - 1),
                homotopy_steps=experiment["p"],
                solver="variational",
                normalization="combinatorial",
                tol=experiment["tol"],
                max_iter=experiment["max_iter"],
            )
            _, fit = model.fit(train_ind, train_labels)
            for p_homotopy, u_homotopy in fit.items():
                result = pd.DataFrame(
                    columns=["x", "y"] + [f"z{k}" for k in range(u_homotopy.shape[1])]
                )
                result["x"] = dataset_sample.data[:, 0]
                result["y"] = dataset_sample.data[:, 1]
                result["true_labels"] = dataset_sample.labels
                for k in range(u_homotopy.shape[1]):
                    result[f"z{k}"] = u_homotopy[:, k]

                item = experiment.copy()
                item["labels_per_class"] = num_train_labels
                item["p"] = p_homotopy
                item["solution"] = result
                item["seed"] = seed
                trial_results.append(item)

    return trial_results


if __name__ == "__main__":
    experiments = storage.load_experiments("real_data", "examples/experiments")

    NUM_THREADS = min(NUM_THREADS, NUM_TRIALS)
    func = partial(run_trial, experiments)
    if NUM_THREADS > 1:
        pool = multiprocessing.Pool(NUM_THREADS)
        trial_results = pool.map(func, range(SEED_START, SEED_START + NUM_TRIALS))
    else:
        trial_results = [
            func(seed) for seed in range(SEED_START, SEED_START + NUM_TRIALS)
        ]
    results = [x for flatten in trial_results for x in flatten]

    storage.save_results(results, name="real_data", folder="results")
