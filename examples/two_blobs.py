import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import sparse

import matplotlib.pyplot as plt

from poisson_learning import PoissonSolver

from graphlearning import poisson2

np.random.seed(1)

n = 1000
d = 2
X_labeled = np.array([[-2.0, -2.0], [2.0, 2.0]])
Y_labeled = np.array([0, 1])
X1 = np.random.normal(
    loc=X_labeled[0], scale=1, size=((n - X_labeled.shape[0]) // 2, 2)
)
X2 = np.random.normal(
    loc=X_labeled[1], scale=1.5, size=((n - X_labeled.shape[0]) // 2, 2)
)
X = np.concatenate([X_labeled, X1, X2])
y = Y_labeled

dist = squareform(pdist(X))
eps = 1  # 10 * (np.log(n) / np.sqrt(n)) ** d
W = (eps ** -d) * np.exp(-((dist / eps) ** 2))
W[W < 1e-4] = 0
plt.plot((W.sum(axis=1)) * eps ** d)
plt.show()
print("eps:", eps)

plt.imshow(W - np.diag(np.diag(W)), interpolation="none")
plt.colorbar()
plt.show()

solver = PoissonSolver(eps=eps, p=2, method="iterative", maxiter=1000, disp=True)
solver.fit(W, y)
output = solver._output[:, 0]

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(
    X[:, 0], X[:, 1], output, c=output, cmap="viridis",
)
plt.show()
