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


def _compute_expeiment_hash(experiment):
    s = "_".join([str(v) for k, v in experiment.items() if k != "results"])
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
    return h


def load_experiments(name, folder):
    def _convert_arrays(x):
        if "train_indices" in x:
            x["train_indices"] = np.array(x["train_indices"], dtype="int64")

        if "label_locations" in x:
            x["label_locations"] = np.array(x["label_locations"], dtype="float64")

        return x

    with open(os.path.join(folder, name + ".json"), mode="r") as file:
        experiments = json.load(file, object_hook=_convert_arrays)
    hdf = pd.HDFStore(os.path.join(folder, name + ".hd5"), mode="r")
    keys = [
        m.groups() for k in hdf.keys() if (m := re.match("\/results\/E(.*)_(.*)", k))
    ]

    for experiment in experiments:
        seeds = [s for h, s in keys if h == experiment["hash"]]
        results = [
            {"seed": s, "solution": hdf.get(f"/results/E{experiment['hash']}_{s}"),}
            for s in seeds
        ]
        experiment["results"] = results

    hdf.close()
    return experiments


def save_experiments(experiments, name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    hdf = pd.HDFStore(os.path.join(folder, name + ".hd5"), mode="w")

    for experiment in experiments:
        experiment["hash"] = _compute_expeiment_hash(experiment)
        for result in experiment["results"]:
            key = f"E{experiment['hash']}_{result['seed']}"
            hdf.put(f"results/{key}", result["solution"])

    hdf.close()

    experiments_dump = [
        {k: v for k, v in e.items() if k != "results"} for e in experiments
    ]
    with open(os.path.join(folder, name + ".json"), mode="w") as file:
        file.write(json.dumps(experiments_dump, cls=_NumpyEncoder, indent=2))
