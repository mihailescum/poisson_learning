import numpy as np
from scipy import linalg


class PoissonSolver:
    def __init__(self, W, eps, p, method="minimization", rhs="dirac_delta"):
        self.method = method
        self.rhs_method = rhs
        self.W = W
        self.eps = eps
        self.p = p

        self._output = None
        self._error = None
        self._iterations = None

    def fit(self, X, y):
        encoded_labels = self._encode_labels(y)
        n_samples = X.shape[0]
        n_classes = encoded_labels.shape[1]

        if self.rhs_method.lower() == "dirac_delta":
            f = self._rhs_dirac_delta(X, encoded_labels)

        self._output = np.full((n_samples, n_classes), np.nan)

        D = np.sum(self.W, axis=1) - np.diag(self.W)
        for c in range(n_classes):
            result, error, it = self._compute_newton_approximation(
                u0=f[:, c], W=self.W, f=f[:, c], p=self.p, eps=self.eps
            )
            mean = np.dot(D, result)
            result -= mean / D.sum()

            self._output[:, c] = result

    def _encode_labels(self, y):
        n_classes = np.unique(y).size
        encoded = np.zeros((y.size, n_classes))
        encoded[np.arange(y.size), y] = 1.0
        return encoded

    def _rhs_dirac_delta(self, X, encoded_labels):
        n_samples = X.shape[0]
        n_classes = encoded_labels.shape[1]
        recentered_labels = encoded_labels - encoded_labels.mean(axis=0)

        rhs = np.zeros((n_samples, n_classes))
        rhs[: recentered_labels.shape[0]] = recentered_labels
        return rhs

    def _compute_newton_approximation(self, u0, W, f, p, eps, tol=1e-5, max_iter=100):
        u = u0.copy()
        n = u.shape[0]
        error = np.inf

        for it in range(max_iter):
            if error <= tol:
                break

            Lu = W * np.abs(u[:, np.newaxis] - u[np.newaxis, :]) ** (p - 2)
            Lu_inv = linalg.inv(Lu)

            u_new = (p - 2) / (p - 1) * u + eps ** p * n / (p - 1) * (Lu_inv @ f)
            error = np.sum(np.abs(u - u_new))
            u = u_new

        return u, error, it

