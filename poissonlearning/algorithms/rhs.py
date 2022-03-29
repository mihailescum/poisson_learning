from scipy.spatial.distance import cdist
import numpy as np

import graphlearning as gl


def _exp_bump(x):
    if isinstance(x, float):
        result = np.exp(-1 / (1 - x ** 2)) if np.abs(x) < 1 else 0
    elif isinstance(x, np.ndarray):
        result = np.zeros_like(x)
        result[np.abs(x) < 1] = np.exp(-1 / (1 - x[np.abs(x) < 1] ** 2))
    else:
        raise ValueError("Type of `x` is not supported")

    # normalize s.t. integral equals to 1
    result = result / 0.443993818
    return result


def bump(data, train_ind, train_labels, bump_width=1.0):
    n, d = data.shape
    onehot = gl.utils.labels_to_onehot(train_labels)
    label_weights = onehot - np.mean(onehot, axis=0)
    num_labels = onehot.shape[1]

    dist_to_labels = cdist(data[train_ind], data)
    bumps = _exp_bump(dist_to_labels / bump_width).T
    bumps *= bump_width ** (-d)

    # Normalize the bumps to one.
    # Asymptotically, the sum will converge to $\rho(x_j)$,
    # where x_j is the j-th labeled node. This has to be divided
    # out for asymptotic consistency.
    bumps /= bumps.sum(axis=0)

    source = np.zeros((n, num_labels))
    for i in range(num_labels):
        source[:, i] = (bumps * label_weights[i]).sum(axis=1)

    return source
