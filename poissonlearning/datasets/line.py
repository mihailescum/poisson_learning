from random import gammavariate
import numpy as np
import graphlearning as gl


def generate(low, high, size):
    data = np.random.uniform(low, high, size)
    data[0] = low + 0.1 * (high - low)
    data[1] = low + 0.9 * (high - low)
    data = data[:, np.newaxis]
    return data


def greens_function(x, z):
    x = x[:, 0]
    phi = -0.5 * np.abs(x - z) + 0.5 * (z ** 2) - 0.5 * z + 0.25
    gamma = -0.5 * (x ** 2) + 0.5 * x - 1.0 / 12.0
    g = phi + gamma
    return g


if __name__ == "__main__":
    data = generate(0, 1, 1000000)
    labels = np.where(data[:, 0] > 0.5, 1, 0)
    gl.datasets.save(data, labels, "line", overwrite=True)
