import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import logging

import matplotlib.pyplot as plt


import poissonlearning as pl

import storage
import plotting

LOGGER = logging.getLogger(name=__name__)
logging.basicConfig(level="INFO")

plotting.setup(latex=True)

SAVE_PLOTS = False
SHOW_PLOTS = True
NUM_PLOTTING_POINTS = 5000


def compute_analytic_solution(xy, z1, z2):
    # Compute the analytic continuum limit
    green_first_label = pl.datasets.one_circle.greens_function(x=xy, z=z1)
    green_second_label = pl.datasets.one_circle.greens_function(x=xy, z=z2)
    solution_analytic = -0.5 * green_first_label + 0.5 * green_second_label
    return solution_analytic


def compute_errors(experiments):
    for experiment in experiments:
        solution = experiment["solution"]
        xy = solution[["x", "y"]].to_numpy()
        z = solution["z"].to_numpy()

        z1 = experiment["label_locations"][0]
        z2 = experiment["label_locations"][1]
        analytic = compute_analytic_solution(xy, z1, z2)

        mask_infty = np.isfinite(analytic)

        error_L1 = np.abs(z[mask_infty] - analytic[mask_infty]).mean()
        experiment["error_L1"] = error_L1


if __name__ == "__main__":
    LOGGER.info("Loading experiments")
    experiments = storage.load_results(name="one_circle", folder="results")

    # Compute errors
    LOGGER.info("Computing errors...")
    compute_errors(experiments)

    LOGGER.info("Plotting...")
    # Plot solution

    bump_width = "dirac"
    n = max([e["n"] for e in experiments])
    ex_plot = list(
        filter(lambda x: x["n"] == n and x["bump"] == bump_width, experiments,)
    )[0]
    solution_plot = ex_plot["solution"]
    sample = solution_plot.sample(NUM_PLOTTING_POINTS, random_state=1)

    xy = sample[["x", "y"]].to_numpy()
    dist = cdist(xy, xy, metric="euclidean")

    fig_results = plt.figure()
    ax_solution = fig_results.add_subplot(1, 2, 1, projection="3d")
    plotting.plot_graph_function_scatter(
        ax_solution, xy, sample["z"].to_numpy(), dist=dist, max_dist=0.1,
    )

    z1 = ex_plot["label_locations"][0]
    z2 = ex_plot["label_locations"][1]
    analytic = compute_analytic_solution(xy, z1, z2)
    ax_analytic = fig_results.add_subplot(1, 2, 2, projection="3d")
    plotting.plot_graph_function_with_triangulation(
        ax_analytic, xy, analytic, dist=dist, max_dist=0.1,
    )
    # fig_results.suptitle(f"Discrete solution [n={n}] vs analytic solution")
    fig_results.tight_layout()

    # Plot errors
    ex_error = {
        kernel: list(
            filter(
                lambda x: x["kernel"] == kernel and x["bump"] == "dirac", experiments,
            )
        )
        for kernel in ["gaussian"]
    }

    n_error = sorted(list(set([e["n"] for e in list(ex_error.values())[0]])))
    error = {kernel: {} for kernel in ex_error.keys()}
    for kernel, ex_error_kernel in ex_error.items():
        for n in n_error:
            ex = list(filter(lambda x: x["n"] == n, ex_error_kernel))
            error[kernel][n] = {}
            error[kernel][n]["mean"] = np.mean([e["error_L1"] for e in ex])
            error[kernel][n]["max"] = np.max([e["error_L1"] for e in ex])
            error[kernel][n]["min"] = np.min([e["error_L1"] for e in ex])

    fig_error, ax_error = plt.subplots(1, 1)
    plotting.error_plot(error, ax_error, fit="polynomial")
    # ax_error.set_title(f"L1 Error compared with RHS {bump_width} to analytic solution")
    ax_error.legend()
    ax_error.set_xlabel(r"$n$")
    ax_error.set_ylabel(r"$\lVert u_n - u \rVert_{L^1 \left(B_1 \right)}$")
    fig_error.tight_layout()

    if SAVE_PLOTS:
        fig_results.savefig("plots/one_circle_convergence.svg", bbox_inches="tight")
        fig_error.savefig("plots/one_circle_error.svg", bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
