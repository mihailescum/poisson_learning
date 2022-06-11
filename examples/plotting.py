import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np
import pandas as pd


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
    # cmap = plt.get_cmap("Dark2")
    cmap = plt.get_cmap("binary")
    cNorm = colors.Normalize(vmin=-(n - 1) * 0.3 * 2, vmax=n - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    result = [scalarMap.to_rgba(i) for i in range(n)]
    return result


def get_linestyles():
    """See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html"""
    styles = [
        "solid",
        "dashed",
        "dotted",
        "dashdot",
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        (0, (5, 5)),  # loosely dashed
        (0, (1, 5)),  # loosely dotted
    ]
    return styles


def error_plot(experiments, ax):
    linestyles = get_linestyles()
    for ls, (label, value) in zip(linestyles, experiments.items()):
        x = [v["n"] for v in value]
        y = [v["err_mean"] for v in value]
        lower_error = [v["err_lower"] for v in value]
        upper_error = [v["err_upper"] for v in value]

        ax.errorbar(
            x, y, yerr=[lower_error, upper_error], label=label, ls=ls, c="black",
        )

    ax.grid(linestyle="dashed")


def results_1D(experiments, ax, truth=None, num_plotting_points=5000):
    colors = get_plot_colors(n=len(experiments))
    linestyles = get_linestyles()

    if truth is not None:
        ax.plot(
            truth.sample(num_plotting_points, random_state=1).sort_index(),
            label=f"Ground Truth",
            c="red",
            linestyle="-",
        )

    for c, ls, (label, e) in zip(colors, linestyles, experiments.items()):
        solution = e["results"][0]["solution"].copy()
        label_values = solution[e["label_locations"]]
        solution = solution[~solution.index.isin(label_values.index)]
        sample_size = min(num_plotting_points, solution.shape[0]) - label_values.size
        sample = solution.sample(sample_size, random_state=1)
        sample = pd.concat([sample, label_values])
        sample = sample.sort_index()

        ax.plot(sample, label=label, c="black", ls=ls)
