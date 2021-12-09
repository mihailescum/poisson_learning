import numpy as np
import scipy.sparse as spsparse

import pytest
import numpy.testing as npt

from poisson_learning import PoissonSolver


@pytest.fixture(params=[(2, 1)])
def solver(request):
    p, eps = request.param
    return PoissonSolver(eps=eps, p=p)


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
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            ),
            np.array([1.0, 0.0, 0.0, -1.0]),
            2,
            np.array([1.5, 0.5, -0.5, -1.5]),
        )
    ],
)
def test_solve_using_minimizer(u0, W, b, p, solver, expected):
    output = solver._solve_using_minimizer(
        u0=u0, W=W, b=b, p=p, tol=1e-16, maxiter=1000
    )

    # Assert that `output` has mean zero
    D = PoissonSolver.get_node_degrees(W)
    weighted_mean = np.dot(output, D)
    npt.assert_allclose(weighted_mean, 0.0)

    # Assert that output solves L_p(u)=b
    grad_J = PoissonSolver.grad_J(output, W, b, p)
    npt.assert_allclose(grad_J, 0.0)

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


@pytest.mark.parametrize(
    "u, W, b, p",
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
        ),
    ],
)
def test_J(u, W, b, p):
    n = u.size
    expected = 0.0
    for i in range(n):
        for j in range(n):
            expected += 0.5 / p * W[i, j] * np.abs(u[i] - u[j]) ** p

        expected -= b[i] * u[i]

    output = PoissonSolver.J(u, W, b, p)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "u, W, b, p",
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
            np.array([1.0, 0.0, 0.0, 0.5]),
            4,
        ),
    ],
)
def test_grad_J(u, W, b, p):
    n = u.size
    expected = np.zeros(n)
    for i in range(n):
        for j in range(n):
            expected[i] += (
                0.5 * W[i, j] * np.abs(u[i] - u[j]) ** (p - 2) * (u[i] - u[j])
            )

        expected[i] -= b[i]

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
