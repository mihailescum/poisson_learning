import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np


def plot_graph_function_with_triangulation(ax, data, z, dist, max_dist):
    ax.view_init(elev=10, azim=-90)

    t = mtri.Triangulation(data[:, 0], data[:, 1])
    xind, yind, zind = t.triangles.T
    xy = dist[xind, yind] ** 2
    xz = dist[xind, zind] ** 2
    yz = dist[yind, zind] ** 2
    mask = np.any(np.vstack([xy, xz, yz]).T > max_dist, axis=1)
    t.set_mask(mask)

    ax.plot_trisurf(t, z, cmap="viridis")


def plot_data_with_labels(ax, data, labels):
    ax.scatter(data[labels == -1, 0], data[labels == -1, 1], c="grey", s=1)
    ax.scatter(
        data[labels >= 0, 0], data[labels >= 0, 1], c=labels[labels >= 0], cmap="Set3",
    )
