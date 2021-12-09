import numpy as np
import scipy.sparse as spsparse

import pytest
import numpy.testing as npt

from poisson_learning import PoissonSolver


@pytest.fixture(params=[(2, 1)])
def solver(request):
    p, eps = request.param
    return PoissonSolver(eps, p)


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
    "u, W, b, p, expected",
    [
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
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
            np.array([1.0, 0.0, 0.0, 1.0]),
            2,
            0.0,
        ),
        (
            np.array([-2.0, 0.0, 1.0, -1.0]),
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
            np.array([1.0, 0.0, 0.0, 1.0]),
            2,
            9,
        ),
        (
            np.array([-2.0, 0.0, 1.0, -1.0]),
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
            np.array([1.0, 0.0, 0.0, 1.0]),
            4,
            13.5,
        ),
    ],
)
def test_J(u, W, b, p, expected):
    output = PoissonSolver.J(u, W, b, p)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "u, W, b, p, expected",
    [
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
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
            np.array([1.0, 0.0, 0.0, 1.0]),
            2,
            np.array([-1.0, 0.0, 0.0, -1.0]),
        ),
        (
            np.array([-2.0, 0.0, 1.0, -1.0]),
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
            np.array([1.0, 0.0, 0.0, 1.0]),
            2,
            np.array([-2.0, 0.0, 3.0, -2.0]),
        ),
        (
            np.array([-2.0, 0.0, 1.0, -1.0]),
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
            np.array([1.0, 0.0, 0.0, 1.0]),
            4,
            np.array([-5.0, 3.0, 9.0, -5.0]),
        ),
    ],
)
def test_grad_J(u, W, b, p, expected):
    output = PoissonSolver.grad_J(u, W, b, p)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "u, D, expected",
    [
        (np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.0, 1.0]), 0.0),
        (np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 2.0, 2.0, 1.0]), 6.0),
        (np.array([1.0, -1.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.0, 1.0]), -1.0),
    ],
)
def test_g(u, D, expected):
    output = PoissonSolver.g(u, D)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "u, D, expected",
    [
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 2.0, 1.0]),
            np.array([1.0, 2.0, 2.0, 1.0]),
        ),
        (
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 2.0, 1.0]),
            np.array([1.0, 2.0, 2.0, 1.0]),
        ),
        (
            np.array([1.0, -1.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 2.0, 1.0]),
            np.array([1.0, 2.0, 2.0, 1.0]),
        ),
    ],
)
def test_grad_g(u, D, expected):
    output = PoissonSolver.grad_g(u, D)
    npt.assert_allclose(expected, output)
