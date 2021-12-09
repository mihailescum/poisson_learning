import numpy as np
from scipy import sparse as spsparse
from scipy import optimize as spoptimize


class PoissonSolver:
    def __init__(
        self,
        eps,
        p,
        rescale_minimizer=True,
        method="minimization",
        rhs="dirac_delta",
        tol=1e-5,
        maxiter=100,
    ):
        self.eps = eps
        self.p = p

        self.rescale_minimizer = rescale_minimizer
        self.method = method
        self.rhs_method = rhs
        self.tol = tol
        self.maxiter = maxiter

        self._output = None

    def fit(self, W, y):
        W = spsparse.csr_matrix(W)
        encoded_labels = self._encode_labels(y)

        n_samples = encoded_labels.shape[0]
        n_classes = encoded_labels.shape[1]

        if self.rhs_method.lower() == "dirac_delta":
            f = self._rhs_dirac_delta(W, encoded_labels)

        self._output = np.full((n_samples, n_classes), np.nan)

        for c in range(n_classes):
            if n_classes == 2 and c == 1:
                result = -self._output[:, 0]
            else:
                result = self._solve_using_minimizer(
                    u0=f[:, c],
                    W=W,
                    b=f[:, c],
                    p=self.p,
                    tol=self.tol,
                    maxiter=self.maxiter,
                )

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

    def J(u, W, b, p):
        difference_matrix = np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** p
        Ju = 1.0 / p * W.multiply(difference_matrix).sum() - np.dot(u, b)
        return Ju

    def grad_J(u, W, b, p):
        A = W.multiply(np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** (p - 2))
        # TODO: use _get_degrees here?
        D = spsparse.diags(A.sum(axis=1).A1)
        grad_Ju = (D - A) @ u - b
        return grad_Ju

    def g(u, D):
        g = np.dot(u, D)
        return g

    def grad_g(u, D):
        return D

    def get_node_degrees(W):
        D = W.sum(axis=1).A1
        return D

    def _solve_using_minimizer(self, u0, W, b, p, tol, maxiter):
        W = W.copy()
        W = W - spsparse.diags(W.diagonal())
        D = PoissonSolver.get_node_degrees(W)
        result = spoptimize.minimize(
            PoissonSolver.J,
            u0,
            args=(W, b, p),
            jac=PoissonSolver.grad_J,
            constraints=(
                {
                    "type": "eq",
                    "fun": PoissonSolver.g,
                    "jac": PoissonSolver.grad_g,
                    "args": (D,),
                }
            ),
            options={"maxiter": maxiter, "disp": True},
            tol=tol,
        )
        return result.x

