from random import gammavariate
import numpy as np
import graphlearning as gl


def generate(low, high, size, rng=None):
    if rng is None:
        data = np.random.uniform(low, high, size)
    else:
        data = rng.uniform(low, high, size)

    data[0] = low + 0.1 * (high - low)
    data[1] = low + 0.9 * (high - low)

    data = data[:, np.newaxis]
    labels = np.where(data[:, 0] > 0.5, 1, 0)

    return data, labels


def greens_function(x, z):
    phi = -0.5 * np.abs(x - z)
    avg = -0.5 * z ** 2 + 0.5 * z - 0.25
    g = phi - avg
    return g


if __name__ == "__main__":
    data, labels = generate(0, 1, 1000000)
    gl.datasets.save(data, labels, "line", overwrite=True)
