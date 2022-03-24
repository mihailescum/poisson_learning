""" This is an adaptation of Jeff Calders implementation of the poisson learning algorithm licensed
under the MIT licence. For the original source code see 
`https://github.com/jwcalder/GraphLearning/blob/master/graphlearning/ssl.py`.
"""
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg as spcg

import graphlearning as gl


class Poisson(gl.ssl.ssl):
    def __init__(
        self,
        W=None,
        rhs=None,
        scale=None,
        class_priors=None,
        solver="conjugate_gradient",
        p=1,
        normalization="combinatorial",
        use_cuda=False,
        min_iter=50,
        max_iter=1000,
        tol=1e-3,
        spectral_cutoff=10,
    ):
        """Poisson Learning
        ===================
        Semi-supervised learning via the solution of the Poisson equation
        \\[L^p u = \\sum_{j=1}^m \\delta_j(y_j - \\overline{y})^T,\\]
        where \\(L=D-W\\) is the combinatorial graph Laplacian,
        \\(y_j\\) are the label vectors, \\(\\overline{y} = \\frac{1}{m}\\sum_{i=1}^m y_j\\)
        is the average label vector, \\(m\\) is the number of training points, and
        \\(\\delta_j\\) are standard basis vectors. See the reference for more details.
        Implements 3 different solvers, spectral, gradient_descent, and conjugate_gradient.
        GPU acceleration is available for gradient descent. See [1] for details.
        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        rhs : numpy array (optional), default=None
            (fill with details)
        scale: float (optional), default=None
            (fill with details)
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        solver : {'spectral', 'conjugate_gradient', 'gradient_descent'} (optional), default='conjugate_gradient'
            Choice of solver for Poisson learning.
        p : int (optional), default=1
            Power for Laplacian, can be any positive real number. Solver will default to 'spectral' if p!=1.
        noralization : str (optional), default="combinatorial"
            Normalization of he graph laplacian
        use_cuda : bool (optional), default=False
            Whether to use GPU acceleration for gradient descent solver.
        min_iter : int (optional), default=50
            Minimum number of iterations of gradient descent before checking stopping condition.
        max_iter : int (optional), default=1000
            Maximum number of iterations of gradient descent.
        tol : float (optional), default=1e-3
            Tolerance for conjugate gradient solver.
        spectral_cutoff : int (optional), default=10
            Number of eigenvectors to use for spectral solver.
        Examples
        --------
        Poisson learning works on directed (i.e., nonsymmetric) graphs with the gradient descent solver: [poisson_directed.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/poisson_directed.py).
        ```py
        import numpy as np
        import graphlearning as gl
        import matplotlib.pyplot as plt
        import sklearn.datasets as datasets
        X,labels = datasets.make_moons(n_samples=500,noise=0.1)
        W = gl.weightmatrix.knn(X,10,symmetrize=False)
        train_ind = gl.trainsets.generate(labels, rate=5)
        train_labels = labels[train_ind]
        model = gl.ssl.poisson(W, solver='gradient_descent')
        pred_labels = model.fit_predict(train_ind, train_labels)
        accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, len(train_ind))
        print("Accuracy: %.2f%%"%accuracy)
        plt.scatter(X[:,0],X[:,1], c=pred_labels)
        plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
        plt.show()
        ```
        Reference
        ---------
        [1] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html),
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.
        """
        super().__init__(W, class_priors)
        self.rhs = rhs
        self.scale = scale

        if solver not in ["conjugate_gradient", "spectral", "gradient_descent"]:
            sys.exit("Invalid Poisson solver")
        self.solver = solver
        self.p = p
        if p != 1:
            self.solver = "spectral"
        self.normalization = normalization
        self.use_cuda = use_cuda
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.spectral_cutoff = spectral_cutoff

        # Setup accuracy filename
        fname = "_poisson"
        if self.p != 1:
            fname += "_p%.2f" % p
        if self.solver == "spectral":
            fname += "_N%d" % self.spectral_cutoff
            self.requries_eig = True
        self.accuracy_filename = fname

        # Setup Algorithm name
        self.name = "Poisson Learning"

    def _fit(self, train_ind, train_labels, all_labels=None):

        n = self.graph.num_nodes
        unique_labels = np.unique(train_labels)
        k = len(unique_labels)

        # Zero out diagonal for faster convergence
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(), 0, n, n)
        G = gl.graph(W)

        # Poisson source term
        if self.rhs is None:
            onehot = gl.utils.labels_to_onehot(train_labels)
            source = np.zeros((n, onehot.shape[1]))
            source[train_ind] = onehot - np.mean(onehot, axis=0)
        else:
            source = self.rhs

        if self.solver == "conjugate_gradient":  # Conjugate gradient solver

            L = G.laplacian(normalization=self.normalization)
            if self.normalization == "combinatorial":
                u = gl.utils.conjgrad(L, source, tol=self.tol, max_iter=self.max_iter)
                # u = np.empty_like(source, dtype="float64")
                # for i in range(u.shape[1]):
                #    u[:, i], _ = spcg(
                #        L, source[:, i], tol=self.tol, maxiter=self.max_iter
                #    )
            elif self.normalization == "normalized":
                D = G.degree_matrix(p=-0.5)
                u = gl.utils.conjgrad(
                    L, D * source, tol=self.tol, max_iter=self.max_iter
                )
                u = D * u
            else:
                raise ValueError(
                    f"Normalization `{self.normalization}` not supported with \
                    solver `conjugate_gradient`."
                )

        elif self.solver == "gradient_descent":

            # Setup matrices
            D = G.degree_matrix(p=-1)
            P = D * W.transpose()
            Db = D * source

            # Invariant distribution
            v = np.zeros(n)
            v[train_ind] = 1
            v = v / np.sum(v)
            deg = G.degree_vector()
            vinf = deg / np.sum(deg)
            RW = W.transpose() * D
            u = np.zeros((n, k))

            # Number of iterations
            T = 0
            if self.use_cuda:
                import torch

                Pt = gl.utils.torch_sparse(P).cuda()
                ut = torch.from_numpy(u).float().cuda()
                Dbt = torch.from_numpy(Db).float().cuda()

                while (T < self.min_iter or np.max(np.absolute(v - vinf)) > 1 / n) and (
                    T < self.max_iter
                ):
                    ut = torch.sparse.addmm(Dbt, Pt, ut)
                    v = RW * v
                    T = T + 1

                # Transfer to CPU and convert to numpy
                u = ut.cpu().numpy()

            else:  # Use CPU

                while (T < self.min_iter or np.max(np.absolute(v - vinf)) > 1 / n) and (
                    T < self.max_iter
                ):
                    u = Db + P * u
                    v = RW * v
                    T = T + 1

                    # Compute accuracy if all labels are provided
                    if all_labels is not None:
                        self.prob = u
                        labels = self.predict()
                        acc = gl.ssl.ssl_accuracy(labels, all_labels, len(train_ind))
                        print("%d,Accuracy = %.2f" % (T, acc))

        # Use spectral solver
        elif self.solver == "spectral":

            vals, vecs = G.eigen_decomp(
                normalization=self.normalization, k=self.spectral_cutoff + 1
            )
            V = vecs[:, 1:]
            vals = vals[1:]
            if self.p != 1:
                vals = vals ** self.p
            L = sparse.spdiags(1 / vals, 0, self.spectral_cutoff, self.spectral_cutoff)
            u = V @ (L @ (V.T @ source))

        else:
            sys.exit("Invalid Poisson solver " + self.solver)

        # Scale solution
        if self.scale is not None:
            u = self.scale ** (1 / self.p - 1) * u

        return u
