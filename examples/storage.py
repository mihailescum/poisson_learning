import numpy as np
import pandas as pd

import hashlib
import os


def load_experiments(name, folder):
    hdf = pd.HDFStore(os.path.join(folder, name), mode="r")


def save_experiments(experiments, name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    hdf = pd.HDFStore(os.path.join(folder, name), mode="w")

    for experiment in experiments:
        for result in experiment["results"]:
            s = "_".join([str(v) for k, v in experiment.items() if k != "results"])
            h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

            key = f"h{h}_{result['seed']}"
            hdf.put(f"results/{key}", result["result"])
            hdf.put(
                f"experiments/{key}",
                pd.Series({k: v for k, v in experiment.items() if k != "results"}),
            )
