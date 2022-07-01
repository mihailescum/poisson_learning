import numpy as np
import pandas as pd
import logging

import matplotlib.pyplot as plt

import poissonlearning as pl

import storage
import plotting

LOGGER = logging.getLogger(name=__name__)
logging.basicConfig(level="INFO")

SAVE_PLOTS = False
SHOW_PLOTS = True
NUM_PLOTTING_POINTS = 5000

plotting.setup(latex=True)


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

        # Compute error to dirac experiment, if exists
        e_dirac = list(
            filter(
                lambda x: x["seed"] == experiment["seed"]
                and x["n"] == experiment["n"]
                and np.isclose(x["eps"], experiment["eps"])
                and np.allclose(x["label_locations"], experiment["label_locations"])
                and x["kernel"] == experiment["kernel"]
                and x["bump"] == "dirac",
                experiments,
            )
        )
        if e_dirac:
            error_L1_dirac = np.abs(solution - e_dirac[0]["solution"]).mean()
            experiment["error_L1_dirac"] = error_L1_dirac


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
    colors = plotting.get_plot_colors(n=5)
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
        colors=colors,
    )

    # ax_dirac.set_title(f"Convergence of Dirac RHS with kernel '{kernel}'")
    ax_dirac.set_xlabel(r"$x$")
    ax_dirac.set_ylabel(r"$u_n$")
    ax_dirac.legend()
    ax_dirac.grid(linestyle="dashed")
    fig_dirac.tight_layout()

    fig_dirac_zoom, ax_dirac_zoom = plt.subplots(1, 1)
    plotting.results_1D(
        ex_convergence,
        ax=ax_dirac_zoom,
        truth=solution_analytic,
        num_plotting_points=NUM_PLOTTING_POINTS,
        colors=colors,
    )

    # ax_dirac.set_title(f"Convergence of Dirac RHS with kernel '{kernel}'")
    ax_dirac_zoom.set_xlabel(r"$x$")
    ax_dirac_zoom.set_ylabel(r"$u_n$")
    ax_dirac_zoom.legend()
    ax_dirac_zoom.grid(linestyle="dashed")
    ax_dirac_zoom.set_xlim(0.39995, 0.40015)
    ax_dirac_zoom.set_ylim(0.075, 0.24)
    ax_dirac_zoom.set_xticks(np.linspace(0.39995, 0.40015, 5))
    fig_dirac_zoom.tight_layout()

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
        colors=colors,
    )
    # ax_bump.set_title(f"Convergence of smoothed RHS with kernel '{kernel}', n={n}")
    ax_bump.set_xlabel(r"$x$")
    ax_bump.set_ylabel(fr"$u_{ {n} }^\delta$")
    ax_bump.legend()
    ax_bump.grid(linestyle="dashed")
    fig_bump.tight_layout()

    fig_bump_zoom, ax_bump_zoom = plt.subplots(1, 1)
    plotting.results_1D(
        ex_bumps,
        ax=ax_bump_zoom,
        truth=solution_analytic,
        num_plotting_points=NUM_PLOTTING_POINTS,
        colors=colors,
    )
    # ax_bump.set_title(f"Convergence of smoothed RHS with kernel '{kernel}', n={n}")
    ax_bump_zoom.set_xlabel(r"$x$")
    ax_bump_zoom.set_ylabel(fr"$u_{ {n} }^\delta$")
    ax_bump_zoom.legend()
    ax_bump_zoom.grid(linestyle="dashed")
    ax_bump_zoom.set_xlim(0.38, 0.42)
    ax_bump_zoom.set_ylim(0.07, 0.085)
    fig_bump_zoom.tight_layout()

    # Errors for dirac solutions
    ex_error = {
        kernel: list(
            filter(
                lambda x: x["kernel"] == kernel and x["bump"] == "dirac", experiments,
            )
        )
        for kernel in ["uniform", "gaussian"]
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

    figs_error = {}
    for kernel, e in error.items():
        fig_error, ax_error = plt.subplots(1, 1)
        plotting.error_plot({kernel: e}, ax_error, fit="polynomial")

        # ax_error.set_xscale("log")
        # ax_error.set_title("L1 error of solution with dirac RHS")
        ax_error.legend()
        ax_error.set_xlabel(r"$n$")
        ax_error.set_ylabel(r"$\lVert u_n - u \rVert_{L^1 \left((0, 1) \right)}$")

        fig_error.tight_layout()
        figs_error[kernel] = fig_error

    # Errors for bump convergence
    n = 300000
    ex_error_bump = {
        kernel: list(
            filter(lambda x: x["kernel"] == kernel and x["n"] == n, experiments,)
        )
        for kernel in ["gaussian"]
    }

    bump_error = list(set([e["bump"] for e in list(ex_error_bump.values())[0]]))
    bump_error = sorted([b for b in bump_error if isinstance(b, float)])
    error_bump = {kernel: {} for kernel in ex_error_bump.keys()}
    for kernel, ex_error_kernel in ex_error_bump.items():
        for bump in bump_error:
            ex = list(filter(lambda x: x["bump"] == bump, ex_error_kernel))
            bump_inv = 1.0 / bump
            error_bump[kernel][bump_inv] = {}
            error_bump[kernel][bump_inv]["mean"] = np.mean(
                [e["error_L1_dirac"] for e in ex]
            )
            error_bump[kernel][bump_inv]["max"] = np.max(
                [e["error_L1_dirac"] for e in ex]
            )
            error_bump[kernel][bump_inv]["min"] = np.min(
                [e["error_L1_dirac"] for e in ex]
            )

    figs_error_bump = {}
    for kernel, e in error_bump.items():
        fig_error, ax_error = plt.subplots(1, 1)
        plotting.error_plot({kernel: e}, ax_error, fit="exponential")

        # ax_error.set_xscale("log")
        # ax_error.set_title("L1 error of solution with dirac RHS")
        ax_error.legend()
        ax_error.set_xlabel(r"$1/\delta$")
        ax_error.set_ylabel(fr"$\lVert u_{ {n} } - u_{ {n} }^\delta \rVert_1$")
        ax_error.set_xscale("log")

        fig_error.tight_layout()
        figs_error_bump[kernel] = fig_error

    if SAVE_PLOTS:
        fig_dirac.savefig("plots/line_convergence.svg", bbox_inches="tight")
        fig_dirac_zoom.savefig("plots/line_convergence_zoom.svg", bbox_inches="tight")
        fig_bump.savefig("plots/line_bump.svg", bbox_inches="tight")
        fig_bump_zoom.savefig("plots/line_bump_zoom.svg", bbox_inches="tight")

        for kernel, fig in figs_error.items():
            fig.savefig(f"plots/line_error_{kernel}.svg", bbox_inches="tight")

        for kernel, fig in figs_error_bump.items():
            fig.savefig(f"plots/line_error_bump_{kernel}.svg", bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
