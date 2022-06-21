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

        return x

    with open(os.path.join(folder, name + ".json"), mode="r") as file:
        experiments = json.load(file, object_hook=_convert_arrays)

    return experiments


def load_results(name, folder):
    experiments = load_experiments(name, folder)

    hdf = pd.HDFStore(os.path.join(folder, name + ".hd5"), mode="r")

    for experiment in experiments:
        solution = hdf.get(f"results/hash_{experiment['hash']}")
        experiment["solution"] = solution

    hdf.close()
    return experiments


def save_results(results, name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    hdf = pd.HDFStore(os.path.join(folder, name + ".hd5"), mode="w")

    experiment_runs = []
    for result in results:
        run = {
            "n": result["n"],
            "eps": result["eps"],
            "bump": result["bump"],
            "kernel": result["kernel"],
            "label_locations": result["label_locations"],
            "seed": result["seed"],
        }
        hash = _compute_run_hash(run)
        run["hash"] = hash

        hdf.put(f"results/hash_{hash}", result["solution"])
        experiment_runs.append(run)

    hdf.close()

    with open(os.path.join(folder, name + ".json"), mode="w") as file:
        file.write(json.dumps(experiment_runs, cls=_NumpyEncoder, indent=2))
