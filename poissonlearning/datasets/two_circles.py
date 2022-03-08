import numpy as np
import graphlearning as gl


def generate(center1, r1, center2, r2, size):
    circle = np.random.randint(0, 2, size=size)  # From which circle to sample?

    radii = np.empty(size, dtype="float64")
    radii[circle == 0] = np.random.uniform(0, r1, (circle == 0).sum())
    radii[circle == 1] = np.random.uniform(0, r2, (circle == 1).sum())
    radii = radii ** (1 / 2)  # number of points within r scales as r^2
    degrees = np.random.uniform(0, 2 * np.pi, size)

    center = np.empty((size, 2), dtype="float64")
    center[circle == 0] = center1
    center[circle == 1] = center2

    x = radii * np.sin(degrees)
    y = radii * np.cos(degrees)
    result = center + np.vstack([x, y]).T
    return result


if __name__ == "__main__":
    data = generate([0.95, 0], 1, [-0.95, 0], 1, 1000000)
    labels = np.where(data[:, 0] > 0, 1, 0)
    gl.datasets.save(data, labels, "two_circles")
