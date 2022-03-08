import numpy as np
import scipy.sparse as spsparse

import pytest
import numpy.testing as npt

from poissonlearning.graphutils import node_degrees, distance_matrix, kernel_exponential


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
        (
            spsparse.csr_matrix(
                np.array(
                    [
                        [10.0, 1.0, 0.0, 0.0, 0.0],
                        [0.1, 5.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.9, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.5, 0.0],
                    ]
                )
            ),
            np.array([1.0, 1.1, 2.0, 2.0, 0.5]),
        ),
    ],
)
def test_node_degrees(W, expected):
    output = node_degrees(W)
    npt.assert_allclose(expected, output)


@pytest.mark.parametrize(
    "X, expected",
    [
        (
            np.array([[1.0, 2.0], [1.0, 3.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array(
                [
                    [0.0, 1.0, np.sqrt(5), np.sqrt(5)],
                    [1.0, 0.0, np.sqrt(10), np.sqrt(10)],
                    [np.sqrt(5), np.sqrt(10), 0.0, 0.0],
                    [np.sqrt(5), np.sqrt(10), 0.0, 0.0],
                ]
            ),
        )
    ],
)
def test_ditance_matrix(X, expected):
    output = distance_matrix(X)
    npt.assert_allclose(expected, output)


@pytest.mark.skip
def test_kernel_exponential(a, eps, d):
    pass
