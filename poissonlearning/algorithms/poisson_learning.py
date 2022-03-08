import numpy as np
from scipy import sparse as spsparse
from scipy import optimize as spoptimize

from .graphutils import node_degrees
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
        """ Initialized a solver for poisson learning [1].

        Parameters
        ----------
        eps
            Epsilon used when constructing the graph

        p
            Power of the p-Laplacian

        rescale_minimizer
            

        method
            Method you want to use to solve the poisson learning problem. Can be
            "iterative": Solves the problem for `p=2` in using an iterative approach
                described in Section 3 of [1].
            "minimizer": Solves the problem for general `p` by minimizing the objective
                Function outlined in Section 2.2 of [1] (the variational approach).
            "homotopy": Uses the idea of p-homotopy as described in [2]: First, solves
                the problem for `p=2` using the iterative approach for speed and uses this
                as initial guess for an ascending sequence of p-values for the minimizing
                approach, until the desired `p` is reached. This yields faster convergence.

        rhs
            Function on the right hand side of the laplace equation. Can be
            "dirac_delta": $ \sum_j^M (y_j - \bar y) \delta_ij $, where $(y_j)_{j=1}^M$
                are the label vectors, and $\bar y$ is the average label value.

        stepsize


        tol
            Desired tolerance in the approximation shemes

        maxiter
            Maximum number of iterations

        disp
            Display convergence messages

        References
        ----------
        [1] "Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label Rates"
        by J Calder, B Cook, M Thorpe, D Slepčev; Preprint (arXiv); 
        https://arxiv.org/abs/2006.11184


        [2] "Algorithms for l^p-based semi-supervised Learning on Graphs" by MF Rios, J Calder
        and G Lerman; Preprint (arXiv); https://arxiv.org/abs/1901.05031
        """
        self.eps = eps
        self.p = p
        if p <= 1:
            raise ValueError("`p` has to be strictly larger than 1.")

        self.rescale_minimizer = rescale_minimizer
        self.method = method
        if method == "iterative" and p != 2:
            raise ValueError("For iterative scheme `p` has to equal 2.")

        self.stepsize = stepsize

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
                b = f[:, c]
                if self.method == "minimizer":
                    result = self._solve_using_minimizer(W=W, b=b)
                elif self.method == "iterative":
                    result = self._solve_using_iteration(W=W, b=b)
                elif self.method == "homotopy":
                    result = self._solve_using_homotopy(W=W, b=b)
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

    def _solve_using_homotopy(self, W, b):
        u0 = self._solve_using_iteration(W, b)

        if self.p < 2:
            u0 = self._solve_using_minimizer(W, b, u0=u0)
        else:
            p_current = 2
            while p_current < self.p:
                p_current = min(p_current + self.stepsize, self.p)
                u0 = self._solve_using_minimizer(W, b, u0=u0, p=p_current)

                if self.disp:
                    print(f"Finished homotopy iteration with p={p_current}")

        return u0

    def _solve_using_minimizer(self, W, b, u0=None, p=None):
        if u0 is None:
            u0 = np.zeros(b.size)
        if p is None:
            p = self.p

        D = node_degrees(W)
        W = W - spsparse.diags(W.diagonal())

        result = spoptimize.minimize(
            objective_p_laplace,
            u0,
            args=(W, b, p),
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

        degrees = node_degrees(W)
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
