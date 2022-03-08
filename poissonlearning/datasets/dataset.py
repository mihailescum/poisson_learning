import numpy as np
from dataclasses import dataclass
import graphlearning as gl


@dataclass
class Dataset:
    data: np.ndarray
    labels: np.ndarray
    metric: str

    def load(dataset, metric="raw"):
        data, labels = gl.datasets.load(dataset, metric)
        result = Dataset(data, labels, metric)
        return result
