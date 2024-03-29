from cmath import exp
import numpy as np
import pandas as pd

import hashlib
import os
import json
import re


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _compute_run_hash(experiment):
    s = "_".join([str(v) for _, v in experiment.items()])
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
    return h


def load_experiments(name, folder):
    def _convert_arrays(x):
        if "label_locations" in x:
            x["label_locations"] = np.array(x["label_locations"], dtype="float64")
        if "p" in x:
            if isinstance(x["p"], list):
                x["p"] = np.array(x["p"], dtype="float64")
        if "labels_per_class" in x:
            if isinstance(x["labels_per_class"], list):
                x["labels_per_class"] = np.array(x["labels_per_class"], dtype="int64")

        return x

    with open(os.path.join(folder, name + ".json"), mode="r") as file:
        experiments = json.load(file, object_hook=_convert_arrays)

    return experiments


def load_results(name, folder):
    experiments = load_experiments(name, folder)

    hdf = pd.HDFStore(os.path.join(folder, name + ".hd5"), mode="r")

    for experiment in experiments:
        if f"results/hash_{experiment['hash']}" in hdf:
            solution = hdf.get(f"results/hash_{experiment['hash']}")
            experiment["solution"] = solution
        else:
            experiment["solution"] = None

    hdf.close()
    return experiments


def save_results(results, name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    hdf = pd.HDFStore(os.path.join(folder, name + ".hd5"), mode="w")

    experiment_runs = []
    for result in results:
        run = {}
        if "n" in result:
            run["n"] = result["n"]
        if "bump" in result:
            run["bump"] = result["bump"]
        if "kernel" in result:
            run["kernel"] = result["kernel"]
        if "label_locations" in result:
            run["label_locations"] = result["label_locations"]
        if "seed" in result:
            run["seed"] = result["seed"]
        if "p" in result:
            run["p"] = result["p"]
        if "eps" in result:
            run["eps"] = result["eps"]
        if "n_neighbors" in result:
            run["n_neighbors"] = result["n_neighbors"]
        if "labels_per_class" in result:
            run["labels_per_class"] = result["labels_per_class"]
        if "tol" in result:
            run["tol"] = result["tol"]
        if "max_iter" in result:
            run["max_iter"] = result["max_iter"]
        if "dataset" in result:
            run["dataset"] = result["dataset"]
        if "dataset_metric" in result:
            run["dataset_metric"] = result["dataset_metric"]
        if "error" in result:
            run["error"] = result["error"]

        hash = _compute_run_hash(run)
        run["hash"] = hash

        solution = result.get("solution", None)
        if solution is not None:
            hdf.put(f"results/hash_{hash}", solution)
        experiment_runs.append(run)

    hdf.close()

    with open(os.path.join(folder, name + ".json"), mode="w") as file:
        file.write(json.dumps(experiment_runs, cls=_NumpyEncoder, indent=2))


def join_results(names, folder, output_name):
    results_all = [load_results(name, folder) for name in names]
    results = [e for results_sub in results_all for e in results_sub]
    save_results(results, output_name, folder)


if __name__ == "__main__":
    join_results(
        ["p_one_circle", "p_one_circle_2"], "results", "p_one_circle_joined",
    )
