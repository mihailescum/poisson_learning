import numpy as np
import pandas as pd
import logging

import storage

LOGGER = logging.getLogger(name=__name__)

if __name__ == "__main__":
    LOGGER.info("Loading experiments")
    experiments = storage.load_experiments(name="one_circle", folder="results")

    print("Plotting...")

    # Compute errors
    print("Computing errors...")

    def get_analytic_solution(xy, z1, z2):
        # Compute the analytic continuum limit
        green_first_label = pl.datasets.one_circle.greens_function(x=xy, z=z1)
        green_second_label = pl.datasets.one_circle.greens_function(x=xy, z=z2)
        solution_analytic = 0.5 * green_first_label - 0.5 * green_second_label
        return solution_analytic

    for e in experiments:
        xy = e["result"][["x", "y"]].to_numpy()
        z = e["result"]["z"].to_numpy()

        solution_analytic = get_analytic_solution(
            xy, dataset.data[train_ind[0]], dataset.data[train_ind[1]]
        )
        mask_infty = np.isfinite(solution_analytic)

        error_L1_unscaled = np.abs(z[mask_infty] - solution_analytic[mask_infty]).mean()
        e["L1_unscaled"] = error_L1_unscaled

        scale = (solution_analytic[mask_infty] / z[mask_infty]).mean()
        print(f"scale for {e['n']}: {scale}")
        z_scaled = scale * z
        error_L1_scaled = np.abs(
            z_scaled[mask_infty] - solution_analytic[mask_infty]
        ).mean()
        e["L1_scaled"] = error_L1_scaled

    print("Plotting...")
    # Plot solution
    n = 50000  # 1000000
    bump_width = "dirac"
    n_max = max([e["n"] for e in experiments])
    ex_max = [e for e in experiments if e["n"] == n_max and e["bump"] == bump_width][0]
    sample = ex_max["result"].sample(NUM_PLOTTING_POINTS, random_state=1)

    xy = sample[["x", "y"]].to_numpy()
    dist = cdist(xy, xy, metric="euclidean")

    fig_results = plt.figure()
    ax_solution = fig_results.add_subplot(1, 2, 1, projection="3d")
    plot_graph_function_with_triangulation(
        ax_solution, xy, sample["z"].to_numpy(), dist=dist, max_dist=0.1,
    )
    ax_solution.set_title(f"Computed discrete solution; n: {n}; RHS: {bump_width}")

    solution_analytic = get_analytic_solution(
        xy, dataset.data[train_ind[0]], dataset.data[train_ind[1]]
    )
    ax_analytic = fig_results.add_subplot(1, 2, 2, projection="3d")
    plot_graph_function_with_triangulation(
        ax_analytic, xy, solution_analytic, dist=dist, max_dist=0.1,
    )
    ax_analytic.set_title(f"Analytic solution to continuum problem")

    # Plot errors
    bump_width = "dirac"

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
    ax_error.legend()

    plt.show()
