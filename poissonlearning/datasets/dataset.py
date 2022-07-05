import numpy as np
from dataclasses import dataclass
import graphlearning as gl


@dataclass
class Dataset:
    data: np.ndarray
    labels: np.ndarray
    metric: str

    def load(dataset, metric="raw", cutoff=None):
        data, labels = gl.datasets.load(dataset, metric)
        result = Dataset(data[:cutoff], labels[:cutoff], metric)
        return result

    def copy(self):
        return Dataset(
            data=self.data.copy(), labels=self.labels.copy(), metric=self.metric
        )
