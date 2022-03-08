import numpy as np
import scipy.sparse as spsparse

import pytest
import numpy.testing as npt

from poissonlearning.objective_functions import (
    objective_p_laplace,
    objective_p_laplace_gradient,
    objective_weighted_mean,
    objective_weighted_mean_gradient,
)


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
def test_objective_p_laplace(u, W, b, p):
    n = u.size
    expected = 0.0
    for i in range(n):
        for j in range(n):
            expected += 0.5 / p * W[i, j] * np.abs(u[i] - u[j]) ** p

        expected -= b[i] * u[i]

    output = objective_p_laplace(u, W, b, p)
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
def test_objective_p_laplace_gradient(u, W, b, p):
    n = u.size
    expected = np.zeros(n)
    for i in range(n):
        for j in range(n):
            expected[i] += W[i, j] * np.abs(u[i] - u[j]) ** (p - 2) * (u[i] - u[j])

        expected[i] -= b[i]

    output = objective_p_laplace_gradient(u, W, b, p)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "u, D, expected",
    [
        (np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.0, 1.0]), 0.0),
        (np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 2.0, 2.0, 1.0]), 6.0),
        (np.array([1.0, -1.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.0, 1.0]), -1.0),
    ],
)
def test_objective_weighted_mean(u, D, expected):
    output = objective_weighted_mean(u, D)
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
def test_objective_weighted_mean_gradient(u, D, expected):
    output = objective_weighted_mean_gradient(u, D)
    npt.assert_allclose(expected, output)
