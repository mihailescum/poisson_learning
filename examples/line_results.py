import numpy as np
import pandas as pd
import logging

import poissonlearning as pl

import storage
import plotting

LOGGER = logging.getLogger(name=__name__)

SAVE_PLOTS = True
SHOW_PLOTS = True


def compute_analytic_solution(x, label_locations):
    LOGGER.info("Computing analytic solution...")
    green_first_label = pl.datasets.line.greens_function(x=x, z=label_locations[0])
    green_second_label = pl.datasets.line.greens_function(x=x, z=label_locations[1])
    solution_analytic = pd.Series(
        0.5 * green_first_label - 0.5 * green_second_label, index=x
    )
    return solution_analytic


def compute_errors(experiments):
    LOGGER.info("Computing errors...")
    for experiment in experiments:
        solution = experiment["result"]
        x = solution.index
        analytic = compute_analytic_solution(x, experiment["label_locations"])
        error_L1 = np.abs(solution - analytic).mean()
        experiment["error_L1"] = error_L1


if __name__ == "__main__":
    LOGGER.info("Loading experiments")
    experiments = storage.load_experiments(name="line", folder="results")

    # Compute the analytic continuum limit
    solution_analytic = compute_analytic_solution(sample_size=1000000)

    # Compute errors
    compute_errors(experiments)

    LOGGER.info("Plotting...")
    # Convergence of dirac solutions
    colors = get_plot_colors(n=5)
    linestyles = get_linestyles()
    fig_dirac, ax_dirac = plt.subplots(1, 1)
    ax_dirac.plot(
        solution_analytic.sample(NUM_PLOTTING_POINTS, random_state=1).sort_index(),
        label=f"analytic solution",
        c=colors[0],
        linestyle=linestyles[0],
    )

    kernel = "gaussian"
    ex_convergence = [
        e
        for e in experiments
        if e["bump"] == "dirac" and e["kernel"] == kernel and e["n"] >= 10000
    ]
    for i, e in enumerate(ex_convergence, start=1):
        solution = e["result"].copy()
        label_values = solution[LABEL_LOCATIONS]
        solution = solution[~solution.index.isin(LABEL_LOCATIONS)]
        sample = solution.sample(
            NUM_PLOTTING_POINTS - label_values.size, random_state=1
        )
        sample = pd.concat([sample, label_values])
        sample = sample.sort_index()

        ax_dirac.plot(
            sample, label=f"n={e['n']}", c=colors[i], linestyle=linestyles[i],
        )

    ax_dirac.set_title(f"Convergence of Dirac RHS with kernel '{kernel}'")
    ax_dirac.legend()
    ax_dirac.grid(linestyle="dashed")

    # Convergence of bumps
    colors = get_plot_colors(n=4)
    linestyles = get_linestyles()

    fig_bump, ax_bump = plt.subplots(1, 1)
    n = 20000
    kernel = "gaussian"
    ex_bumps = [e for e in experiments if e["n"] == n and e["kernel"] == kernel]
    for i, e in enumerate(ex_bumps):
        solution = e["result"].copy()
        label_values = solution[LABEL_LOCATIONS]
        solution = solution[~solution.index.isin(LABEL_LOCATIONS)]
        sample = solution.sample(
            NUM_PLOTTING_POINTS - label_values.size, random_state=1
        )
        sample = pd.concat([sample, label_values])
        sample = sample.sort_index()

        ax_bump.plot(
            sample,
            label=f"bump width={e['bump']}",
            c=colors[i],
            linestyle=linestyles[i],
        )

    ax_bump.set_title(f"Convergence of smoothed RHS with kernel '{kernel}', n={n}")
    ax_bump.legend()
    ax_bump.grid(linestyle="dashed")

    # L1 errors
    linestyles = get_linestyles()

    fig_error, ax_error = plt.subplots(1, 1)
    ex_dirac_uniform = [
        e for e in experiments if e["kernel"] == "uniform" and e["bump"] == "dirac"
    ]
    ex_dirac_gaussian = [
        e for e in experiments if e["kernel"] == "gaussian" and e["bump"] == "dirac"
    ]

    ax_error.plot(
        [e["n"] for e in ex_dirac_uniform],
        [e["error_L1"] for e in ex_dirac_uniform],
        label="kernel=uniform",
        c="black",
        linestyle=linestyles[0],
    )
    ax_error.plot(
        [e["n"] for e in ex_dirac_gaussian],
        [e["error_L1"] for e in ex_dirac_gaussian],
        label="kernel=gaussian",
        c="black",
        linestyle=linestyles[1],
    )

    ax_error.set_xscale("log")
    ax_error.set_title("L1 error of solution with dirac RHS")
    ax_error.legend()
    ax_error.grid(linestyle="dashed")

    plt.show()
