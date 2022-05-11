import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors
import matplotlib.cm as cmx

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

    # Remove values where `z` is infinity for plotting purposes
    z_masked_infty = z.copy()
    z_masked_infty[np.isposinf(z)] = np.max(z[~np.isposinf(z)])
    z_masked_infty[np.isneginf(z)] = np.min(z[~np.isneginf(z)])

    # TODO: remove
    z_masked_infty[np.isnan(z_masked_infty)] = 0.0

    ax.plot_trisurf(t, z_masked_infty, cmap="viridis")


def plot_data_with_labels(ax, data, labels):
    ax.scatter(data[labels == -1, 0], data[labels == -1, 1], c="grey", s=1)
    ax.scatter(
        data[labels >= 0, 0], data[labels >= 0, 1], c=labels[labels >= 0], cmap="Set3",
    )


def get_plot_colors(n):
    cmap = plt.get_cmap("Dark2")
    cNorm = colors.Normalize(vmin=0, vmax=n - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    result = [scalarMap.to_rgba(i) for i in range(n)]
    return result
