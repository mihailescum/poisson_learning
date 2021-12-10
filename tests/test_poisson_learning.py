import numpy as np
import scipy.sparse as spsparse

import pytest
import numpy.testing as npt

from poissonlearning import PoissonSolver


@pytest.fixture(params=[(2, 1)])
def solver(request):
    p, eps = request.param
    return PoissonSolver(eps=eps, p=p, disp=True, tol=1e-10, maxiter=100)


@pytest.mark.parametrize(
    "y, expected",
    [
        (
            np.array([0, 0, 1, 2, 0]),
            np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        )
    ],
)
def test_encode_labels(y, solver, expected):
    output = solver._encode_labels(y)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "W, encoded_labels, expected",
    [
        (
            spsparse.csr_matrix(
                np.array(
                    [
                        [0.0, 0.5, 0.0, 0.0],
                        [0.5, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.5, 0.0],
                    ]
                )
            ),
            np.array([[1, 0], [1, 0], [0, 1]]),
            np.array([[1 / 3, -1 / 3], [1 / 3, -1 / 3], [-2 / 3, 2 / 3], [0, 0]]),
        )
    ],
)
def test_rhs_dirac_delta(W, encoded_labels, solver, expected):
    output = solver._rhs_dirac_delta(W, encoded_labels)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "u0, W, b, p, expected",
    [
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
            spsparse.csr_matrix(
                np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0],
                    ]
                )
            ),
            np.array([1.0, 0.0, 0.0, -1.0]),
            2,
            np.array([1.125, 0.125, -0.20833333, -0.54166667]),
        )
    ],
)
def test_solve_using_minimizer(u0, W, b, p, solver, expected):
    if p != solver.p:
        pytest.skip("`p` values don't match.")

    output = solver._solve_using_minimizer(u0=u0, W=W, b=b)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "W, b, p, expected",
    [
        (
            spsparse.csr_matrix(
                np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0],
                    ]
                )
            ),
            np.array([1.0, 0.0, 0.0, -1.0]),
            2,
            np.array([1.125, 0.125, -0.20833333, -0.54166667]),
        )
    ],
)
def test_solve_using_iteration(W, b, p, solver, expected):
    if p != solver.p:
        pytest.skip("`p` values don't match.")

    output = solver._solve_using_iteration(W=W, b=b)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "W, expected",
    [
        (
            spsparse.csr_matrix(
                np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            ),
            np.array([2.0, 2.0, 1.0]),
        ),
        (
            spsparse.csr_matrix(
                np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.1, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.5, 0.0],
                    ]
                )
            ),
            np.array([1.0, 1.1, 2.0, 2.0, 0.5]),
        ),
    ],
)
def test_get_node_degrees(W, expected):
    output = PoissonSolver.get_node_degrees(W)
    npt.assert_allclose(expected, output)
