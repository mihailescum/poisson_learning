import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np
import pandas as pd
import scipy.optimize
import logging

# line cyclers adapted to colourblind people
from cycler import cycler

line_cycler = cycler(
    color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
) + cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."])
marker_cycler = (
    cycler(
        color=[
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#F0E442",
        ]
    )
    + cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"])
    + cycler(marker=["4", "2", "3", "1", "+", "x", "."])
)

LOGGER = logging.getLogger(__name__)


def setup(latex=False):
    plt.rc("axes", labelsize="xx-large", titlesize="xx-large")
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rc("legend", fontsize="x-large")

    if latex:
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def plot_graph_function_with_triangulation(ax, data, z, dist, max_dist):
    ax.view_init(elev=20, azim=-110)

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


def plot_graph_function_scatter(ax, data, z, dist, max_dist):
    ax.view_init(elev=20, azim=-110)

    xs = data[:, 0]
    ys = data[:, 1]

    # Remove values where `z` is infinity for plotting purposes
    mask_z = np.isfinite(z)
    xs = xs[mask_z]
    ys = ys[mask_z]
    z = z[mask_z]

    ax.scatter(xs, ys, z, s=3, c=z, cmap="viridis", alpha=0.5)


def plot_data_with_labels(ax, data, labels):
    ax.scatter(data[labels == -1, 0], data[labels == -1, 1], c="grey", s=1)
    ax.scatter(
        data[labels >= 0, 0], data[labels >= 0, 1], c=labels[labels >= 0], cmap="Set3",
    )


def get_plot_colors(n):
    cmap = plt.get_cmap("Dark2")
    # cmap = plt.get_cmap("binary")
    cNorm = colors.Normalize(vmin=-(n - 1) * 0.3 * 2, vmax=n - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    result = [scalarMap.to_rgba(i) for i in range(n)]
    return result


def get_linestyles():
    """See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html"""
    styles = [
        "solid",
        "dashed",
        (0, (1, 1)),  # dotted
        "dashdot",
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        (0, (5, 5)),  # loosely dashed
        (0, (1, 5)),  # loosely dotted
    ]
    return styles


def error_plot(errors, ax, fit=None, label=None, c=None):
    for (label, value) in errors.items():
        x = list(value.keys())
        if len(x) == 0:
            continue

        y = [value[n]["mean"] for n in x]
        lower_error = [value[n]["mean"] - value[n]["min"] for n in x]
        upper_error = [value[n]["max"] - value[n]["mean"] for n in x]

        eb = ax.errorbar(
            x,
            y,
            yerr=[lower_error, upper_error],
            label="errors" if label is None else label,
            c=c,
            capsize=5,
        )
        eb[-1][0].set_linestyle("--")

        # Fit exponential error curve
        def _func_exp(X, a, c):
            return a * np.exp(-c * X)

        def _func_polynomial(X, a, n):
            return a * (X ** (-n))

        xplot = np.linspace(np.min(x), np.max(x), 1000)
        if fit == "exponential":
            popt, _ = scipy.optimize.curve_fit(_func_exp, x, y, p0=(1, 1e-5))
            yplot = _func_exp(xplot, *popt)
        elif fit == "polynomial":
            popt, _ = scipy.optimize.curve_fit(_func_polynomial, x, y, p0=(1, 1))
            yplot = _func_polynomial(xplot, *popt)

        if fit is not None:
            LOGGER.info(f"Fitted parameters: {popt}")
            ax.plot(xplot, yplot, c="red", label=f"{fit} fit")

    ax.grid(linestyle="dashed")


def results_1D(experiments, ax, truth=None, num_plotting_points=5000):
    if truth is not None:
        ax.plot(
            truth.sample(num_plotting_points, random_state=1).sort_index(),
            label=f"Ground Truth",
            c="red",
            linestyle="-",
        )

    for (label, e) in experiments.items():
        label_locations = e["label_locations"][:, np.newaxis]
        dist_to_labels = np.abs(e["solution"].index.to_numpy() - label_locations)
        close_to_labels = np.any(dist_to_labels < 1e-4, axis=0)

        sample_values = e["solution"][~close_to_labels]
        sample_size = (
            min(num_plotting_points, sample_values.shape[0]) - close_to_labels.sum()
        )
        sample = sample_values.sample(sample_size, random_state=1)
        sample = pd.concat([sample, e["solution"][close_to_labels]])
        sample = sample.sort_index()

        ax.plot(sample, label=label)
