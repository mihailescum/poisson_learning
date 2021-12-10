import numpy as np
from scipy import sparse as spsparse
from scipy import optimize as spoptimize

from .objective_functions import (
    objective_p_laplace,
    objective_p_laplace_gradient,
    objective_weighted_mean,
    objective_weighted_mean_gradient,
)


class PoissonSolver:
    def __init__(
        self,
        eps,
        p,
        rescale_minimizer=True,
        method="homotopy",
        rhs="dirac_delta",
        stepsize=5,
        tol=1e-5,
        maxiter=100,
        disp=False,
    ):
        self.eps = eps
        self.p = p

        self.rescale_minimizer = rescale_minimizer
        self.method = method
        if method == "iterative" and p != 2:
            raise ValueError("For iterative scheme `p` has to equal 2!")

        self.rhs_method = rhs
        self.tol = tol
        self.maxiter = maxiter
        self.disp = disp

        self._output = None

    def fit(self, W, y):
        W = spsparse.csr_matrix(W)
        encoded_labels = self._encode_labels(y)

        n_samples = W.shape[0]
        n_classes = encoded_labels.shape[1]

        if self.rhs_method.lower() == "dirac_delta":
            f = self._rhs_dirac_delta(W, encoded_labels)

        self._output = np.full((n_samples, n_classes), np.nan)

        for c in range(n_classes):
            if n_classes == 2 and c == 1:
                result = -self._output[:, 0]
            else:
                if self.method == "minimizer":
                    result = self._solve_using_minimizer(u0=f[:, c], W=W, b=f[:, c],)
                elif self.method == "iterative":
                    result = self._solve_using_iteration(W=W, b=f[:, c])
                elif self.method == "homotopy":
                    raise NotImplementedError()
                else:
                    raise ValueError(f"Method '{self.method}' not understood.")

            self._output[:, c] = result

        # Reweight minimizer
        if self.rescale_minimizer:
            mu = n_samples * (self.eps ** self.p)
            self._output = mu ** (1.0 / (self.p - 1)) * self._output

    def _encode_labels(self, y):
        n_classes = np.unique(y).size
        encoded = np.zeros((y.size, n_classes))
        encoded[np.arange(y.size), y] = 1.0
        return encoded

    def _rhs_dirac_delta(self, W, encoded_labels):
        n_samples = W.shape[0]
        n_classes = encoded_labels.shape[1]
        recentered_labels = encoded_labels - encoded_labels.mean(axis=0)

        rhs = np.zeros((n_samples, n_classes))
        rhs[: recentered_labels.shape[0]] = recentered_labels
        return rhs

    def get_node_degrees(W):
        D = W.sum(axis=1).A1
        return D

    def _solve_using_minimizer(self, u0, W, b):
        W = W.copy()
        W = W - spsparse.diags(W.diagonal())
        D = PoissonSolver.get_node_degrees(W)
        result = spoptimize.minimize(
            objective_p_laplace,
            u0,
            args=(W, b, self.p),
            jac=objective_p_laplace_gradient,
            constraints=(
                {
                    "type": "eq",
                    "fun": objective_weighted_mean,
                    "jac": objective_weighted_mean_gradient,
                    "args": (D,),
                }
            ),
            options={"maxiter": self.maxiter, "disp": self.disp},
            tol=self.tol,
        )
        return result.x

    def _solve_using_iteration(self, W, b):
        n = b.size

        degrees = PoissonSolver.get_node_degrees(W)
        D = spsparse.diags(degrees)
        degrees_inv = 1.0 / degrees

        L = D - W

        u = np.zeros(n)
        u_prev = np.full(n, np.inf)

        p = np.zeros(n)
        p[np.abs(b) > 1e-8] = 1.0
        p_infty = W @ np.ones(n) / (np.ones(n).T @ W @ np.ones(n))

        it = 0
        mixing_time_reached = False
        while np.abs(u - u_prev).max() >= self.tol:
            if np.abs(p - p_infty).max() < 1 / n:
                mixing_time_reached = True
                break
            else:
                u_prev = u

            u = u + degrees_inv * (b - L @ u)
            p = W @ (degrees_inv * p)
            it += 1

        if self.disp:
            if not mixing_time_reached:
                print("Iterative approach terminated successfully")
            else:
                print("Iterative approach reached mixing time")

            print("\tCurrent function value:", objective_p_laplace(u, W, b, 2))
            print("\tIterations:", it)

        return u