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


def greens_function(x, z):
    """See Benedikt Wirth (2020) "Green's Function for the Neumann–Poisson Problem on n-Dimensional Balls", 
    The American Mathematical Monthly, 127:8, 737-743, DOI: 10.1080/00029890.2020.179091"""
    w2inv = 2 / np.pi
    z_norm = np.linalg.norm(z)

    with np.errstate(divide="ignore"):
        phi = w2inv * np.log(np.linalg.norm(x - z, axis=1))
        gamma = w2inv * np.log(np.linalg.norm(z_norm * x - z / z_norm, axis=1))

    remainder = 0.5 * w2inv * np.linalg.norm(x, axis=1) ** 2
    g = phi + gamma - remainder
    g -= np.mean(g[~np.isinf(g)])

    return g


if __name__ == "__main__":
    data = generate(np.array([[0, 0]]), 1, 1000000)
    labels = np.where(data[:, 0] > 0, 1, 0)
    gl.datasets.save(data, labels, "one_circle", overwrite=True)
