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
        solution = experiment["solution"]
        x = solution.index
        analytic = compute_analytic_solution(x, experiment["label_locations"])
        error_L1 = np.abs(solution - analytic).mean()
        experiment["error_L1"] = error_L1


if __name__ == "__main__":
    LOGGER.info("Loading experiments")
    experiments = storage.load_results(name="line", folder="results")

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
        for n in [10000, 20000, 50000, 100000, 300000]
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
    n = 300000
    kernel = "gaussian"
    ex_bumps = {
        f"bump width={bump}": list(
            filter(
                lambda x: x["kernel"] == kernel and x["bump"] == bump and x["n"] == n,
                experiments,
            )
        )[0]
        for bump in ["dirac", 0.001, 0.01, 0.1]
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
    n_error = [1000, 10000, 20000, 35000, 50000, 70000, 100000, 200000, 300000]
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

    # ax_error.set_xscale("log")
    ax_error.set_title("L1 error of solution with dirac RHS")
    ax_error.legend()

    if SHOW_PLOTS:
        plt.show()
