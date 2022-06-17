import numpy as np
import scipy.sparse.linalg as splinalg

import logging

logger = logging.getLogger("pl.numerics")


def conjgrad(A, b, x0=None, max_iter=1e5, tol=1e-10, preconditioner=None):
    """Conjugate Gradient Method
    ======
    Conjugate gradient method for solving the linear equation
    \\[Ax = b\\]
    where \\(A\\in \\mathbb{R}^{n\\times n}\\) is symmetric and positive semi-definite, 
    \\(x\\in \\mathbb{R}^{n\\times k}\\) and \\(b\\in \\mathbb{R}^{n\\times k}\\).
    Parameters
    ----------
    A : (n,n) numpy array or scipy sparse matrix
        Left hand side of linear equation.
    b : (n,k) numpy array
        Right hand side of linear equation.
    x0 : (n,k) numpy array (optional)
        Initial guess. If not provided, then x0=0.
    max_iter : int (optional), default = 1e5
        Maximum number of iterations.
    tol : float (optional), default = 1e-10
        Tolerance for stopping conjugate gradient iterations.
    preconditioner : str (optional), default = None
        Preconditioner to use. Can be one of ["diagonal", "ilu"]

    Returns
    -------
    x : (n,k) numpy array
        Solution of \\(Ax=b\\) with conjugate gradient

    Notestion analysis once, and then re-use it to efficiently decompose many matrices with the same pattern of non-zero entries.
In-place ‘update’ and ‘downdate’ operations, for computing the Cholesky decomposition of a rank-k update of A
and of product AA′. So, the result is the Cholesky decomposition of A+CC′ (or AA′+CC′
). The last case is useful when the columns of A become available incrementally (e.g., due to memory constraints), or when many matrices with similar but non-identical columns must be factored.
Convenience functions for computing the (log) determinant of the matrix that has been factored.
    -----
    Original Code by Jeff Calder, licencesed under MIT license.
    See `https://github.com/jwcalder/GraphLearning/blob/master/graphlearning/utils.py`

    For algorithmic description see 
    `http://www.math.iit.edu/~fass/477577_Chapter_16.pdf`
    """

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    A = A.tocsc()

    logger.info("CG - Constructing preconditioner")
    if isinstance(preconditioner, splinalg.SuperLU):
        M = preconditioner
    elif preconditioner == "diagonal":
        M = A.diagonal()
        M[(M >= 0) & (M < 1e-10)] = 1e-10
        M[(M < 0) & (M > -1e-10)] = -1e-10
        Minv = 1 / M

        Minv = np.tile(Minv, (2, 1)).T
    elif preconditioner == "ilu":
        M = splinalg.spilu(A)

    r = b - A @ x

    if preconditioner is None:
        r_tilde = r
    elif preconditioner == "diagonal":
        r_tilde = Minv * r
    elif preconditioner == "ilu" or isinstance(preconditioner, splinalg.SuperLU):
        r_tilde = M.solve(r)

    p = r_tilde

    rsold = np.dot(r.T, r_tilde)
    if isinstance(rsold, np.ndarray):
        rsold = np.diagonal(rsold)

    err = np.sqrt(np.sum(rsold))
    i = 0

    logger.info(f"CG - It: {i}; error: {err}")
    while (err > tol) and (i < max_iter):

        i += 1
        Ap = A @ p
        alpha = rsold / np.sum(p * Ap, axis=0)
        x += alpha * p
        r -= alpha * Ap

        if preconditioner is None:
            r_tilde = r
        elif preconditioner == "diagonal":
            r_tilde = Minv * r
        elif preconditioner == "ilu" or isinstance(preconditioner, splinalg.SuperLU):
            r_tilde = M.solve(r)

        rsnew = np.dot(r.T, r_tilde)
        if isinstance(rsnew, np.ndarray):
            rsnew = np.diagonal(rsnew)

        beta = rsnew / rsold
        p = r_tilde + beta * p

        err = np.sqrt(np.sum(rsnew))
        rsold = rsnew
        logger.info(f"CG - It: {i}; error: {err}")

    return x
