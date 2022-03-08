import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import poissonlearning as pl
import graphlearning as gl


def plot_graph_function_with_triangulation(x, y, z, dist, max_dist):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    t = mtri.Triangulation(x, y)
    xind, yind, zind = t.triangles.T
    xy = dist[xind, yind] ** 2
    xz = dist[xind, zind] ** 2
    yz = dist[yind, zind] ** 2
    mask = np.any(np.vstack([xy, xz, yz]).T > max_dist, axis=1)
    t.set_mask(mask)

    ax.plot_trisurf(t, z, cmap="viridis")
    plt.show()


cutoff = 1000000
dataset = pl.datasets.Dataset.load("two_circles", "raw")
W = gl.weightmatrix.knn(dataset.data[:cutoff], k=5, symmetrize=True)

train_ind = gl.trainsets.generate(dataset.labels[:cutoff], rate=1)
train_labels = dataset.labels[train_ind]

poisson_dirac = gl.ssl.poisson(W, p=1, solver="conjugate_gradient")
solution = poisson_dirac.fit(train_ind, train_labels)

dist = cdist(dataset.data[:cutoff], dataset.data[:cutoff], metric="euclidean")
plot_graph_function_with_triangulation(
    dataset.data[:cutoff, 0],
    dataset.data[:cutoff, 1],
    solution[:, 0],
    dist=dist,
    max_dist=0.01,
)
