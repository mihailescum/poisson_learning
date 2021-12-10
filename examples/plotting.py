import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np


def plot_graph_function_with_triangulation(x, y, z, dist=None, max_dist=None):
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
