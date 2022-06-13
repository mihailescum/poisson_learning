import numpy as np
import pandas as pd
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


def compute_analytic_solution(x, label_locations):
    green_first_label = pl.datasets.line.greens_function(x=x, z=label_locations[0])
    green_second_label = pl.datasets.line.greens_function(x=x, z=label_locations[1])
    solution_analytic = pd.Series(
        0.5 * green_first_label - 0.5 * green_second_label, index=x
    )
    return solution_analytic


def compute_errors(experiments):
    for experiment in experiments:
        for result in experiment["results"]:
            solution = result["solution"]
            x = solution.index
            analytic = compute_analytic_solution(x, experiment["label_locations"])
            error_L1 = np.abs(solution - analytic).mean()
            result["error_L1"] = error_L1

    for experiment in experiments:
        L1_errors = [r["error_L1"] for r in experiment["results"]]
        experiment["err_mean"] = np.mean(L1_errors)
        experiment["err_upper"] = np.max(L1_errors) - experiment["err_mean"]
        experiment["err_lower"] = experiment["err_mean"] - np.min(L1_errors)


if __name__ == "__main__":
    LOGGER.info("Loading experiments")
    experiments = storage.load_experiments(name="line", folder="results")

    # Compute errors
    LOGGER.info("Computing errors...")
    compute_errors(experiments)

    # Compute analytic_solution
    solution_analytic = compute_analytic_solution(
        np.linspace(0, 1, NUM_PLOTTING_POINTS), experiments[0]["label_locations"]
    )

    LOGGER.info("Plotting...")

    # Convergence of dirac solutions
    kernel = "uniform"
    ex_convergence = {
        f"n={n}": list(
            filter(
                lambda x: x["kernel"] == kernel
                and x["bump"] == "dirac"
                and x["n"] == n,
                experiments,
            )
        )[0]
        for n in [10000, 20000, 50000]  # , 100000]
    }

    fig_dirac, ax_dirac = plt.subplots(1, 1)
    plotting.results_1D(
        ex_convergence,
        ax=ax_dirac,
        truth=solution_analytic,
        num_plotting_points=NUM_PLOTTING_POINTS,
    )

    ax_dirac.set_title(f"Convergence of Dirac RHS with kernel '{kernel}'")
    ax_dirac.legend()
    ax_dirac.grid(linestyle="dashed")

    # Convergence of bumps
    n = 50000
    kernel = "uniform"
    ex_bumps = {
        f"bump width={bump}": list(
            filter(
                lambda x: x["kernel"] == kernel and x["bump"] == bump and x["n"] == n,
                experiments,
            )
        )[0]
        for bump in ["dirac"]
    }

    fig_bump, ax_bump = plt.subplots(1, 1)
    plotting.results_1D(
        ex_bumps,
        ax=ax_bump,
        truth=solution_analytic,
        num_plotting_points=NUM_PLOTTING_POINTS,
    )
    ax_bump.set_title(f"Convergence of smoothed RHS with kernel '{kernel}', n={n}")
    ax_bump.legend()
    ax_bump.grid(linestyle="dashed")

    # Errors for dirac solutions
    ex_error = {
        kernel: list(
            filter(
                lambda x: x["kernel"] == kernel and x["bump"] == "dirac", experiments,
            )
        )
        for kernel in ["uniform", "gaussian"]
    }
    fig_error, ax_error = plt.subplots(1, 1)
    plotting.error_plot(ex_error, ax_error, fit="exponential")

    # ax_error.set_xscale("log")
    ax_error.set_title("L1 error of solution with dirac RHS")
    ax_error.legend()

    if SHOW_PLOTS:
        plt.show()
