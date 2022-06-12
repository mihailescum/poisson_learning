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

SAVE_PLOTS = True
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
        for result in experiment["results"]:
            solution = result["solution"]
            xy = solution[["x", "y"]].to_numpy()
            z = solution["z"].to_numpy()

            z1 = xy[experiment["train_indices"][0]]
            z2 = xy[experiment["train_indices"][1]]
            analytic = compute_analytic_solution(xy, z1, z2)

            mask_infty = np.isfinite(analytic)

            error_L1_unscaled = np.abs(z[mask_infty] - analytic[mask_infty]).mean()
            result["L1_unscaled"] = error_L1_unscaled

            scale = (analytic[mask_infty] / z[mask_infty]).mean()
            LOGGER.info(f"scale for {experiment['n']}:{result['seed']}: {scale}")
            z_scaled = scale * z
            error_L1_scaled = np.abs(z_scaled[mask_infty] - analytic[mask_infty]).mean()
            result["L1_scaled"] = error_L1_scaled

    for experiment in experiments:
        L1_errors_unscaled = [r["L1_unscaled"] for r in experiment["results"]]
        experiment["err_unscaled_mean"] = np.mean(L1_errors_unscaled)
        experiment["err_unscaled_upper"] = (
            np.max(L1_errors_unscaled) - experiment["err_unscaled_mean"]
        )
        experiment["err_unscaled_lower"] = experiment["err_unscaled_mean"] - np.min(
            L1_errors_unscaled
        )

        L1_errors_scaled = [r["L1_scaled"] for r in experiment["results"]]
        experiment["err_scaled_mean"] = np.mean(L1_errors_scaled)
        experiment["err_scaled_upper"] = (
            np.max(L1_errors_scaled) - experiment["err_scaled_mean"]
        )
        experiment["err_scaled_lower"] = experiment["err_scaled_mean"] - np.min(
            L1_errors_scaled
        )


if __name__ == "__main__":
    LOGGER.info("Loading experiments")
    experiments = storage.load_experiments(name="one_circle", folder="results")

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
    solution_plot = ex_plot["results"][0]["solution"]
    sample = solution_plot.sample(NUM_PLOTTING_POINTS, random_state=1)

    xy = sample[["x", "y"]].to_numpy()
    dist = cdist(xy, xy, metric="euclidean")

    fig_results = plt.figure()
    ax_solution = fig_results.add_subplot(1, 2, 1, projection="3d")
    plotting.plot_graph_function_with_triangulation(
        ax_solution, xy, sample["z"].to_numpy(), dist=dist, max_dist=0.1,
    )
    ax_solution.set_title(f"Computed discrete solution; n: {n}; RHS: {bump_width}")

    z1 = solution_plot[["x", "y"]].iloc[ex_plot["train_indices"][0]].to_numpy()
    z2 = solution_plot[["x", "y"]].iloc[ex_plot["train_indices"][1]].to_numpy()
    analytic = compute_analytic_solution(xy, z1, z2)
    ax_analytic = fig_results.add_subplot(1, 2, 2, projection="3d")
    plotting.plot_graph_function_with_triangulation(
        ax_analytic, xy, analytic, dist=dist, max_dist=0.1,
    )
    ax_analytic.set_title(f"Analytic solution to continuum problem")

    # Plot errors
    """bump_width = "dirac"

    fig_error, ax_error = plt.subplots(1, 1)
    ex_error = [e for e in experiments if e["bump"] == bump_width]
    for s in ["unscaled", "scaled"]:
        errors = [e[f"L1_{s}"] for e in ex_error]
        n = [e["n"] for e in ex_error]
        ax_error.plot(n, errors, marker="x", ls="-", label=s)
        logger.info(f"L1 error {s}: {errors}")
    ax_error.set_xscale("log")
    ax_error.set_title(f"L1 Error compared with RHS {bump_width} to analytic solution")
    ax_error.grid()
    ax_error.legend()"""

    if SHOW_PLOTS:
        plt.show()
