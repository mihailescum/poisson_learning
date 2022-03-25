import numpy as np
import graphlearning as gl


def generate(center, r, size):
    radii = np.random.uniform(0, r, size)
    radii = radii ** (1 / 2)  # number of points within r scales as r^2
    degrees = np.random.uniform(0, 2 * np.pi, size)

    x = radii * np.sin(degrees)
    y = radii * np.cos(degrees)
    result = center + np.vstack([x, y]).T

    # Add two label points at the beginning of the data set
    result[0] = center + np.array([[2 / 3 * r, 0]])
    result[1] = center - np.array([[2 / 3 * r, 0]])

    return result


if __name__ == "__main__":
    data = generate(np.array([[0, 0]]), 1, 1000000)
    labels = np.where(data[:, 0] > 0, 1, 0)
    gl.datasets.save(data, labels, "one_circle", overwrite=True)
