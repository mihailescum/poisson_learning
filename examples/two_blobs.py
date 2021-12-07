import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from poisson_learning import PoissonSolver

n = 10
d = 2
X_labeled = np.array([[-10.0, -10.0], [10, 10]])
Y_labeled = np.array([0, 1])

X1 = np.random.normal(
    loc=[-10.0, -10.0], scale=1, size=((n - X_labeled.shape[0]) // 2, 2)
)
X2 = np.random.normal(
    loc=[10.0, 10.0], scale=1, size=((n - X_labeled.shape[0]) // 2, 2)
)
X = np.concatenate([X_labeled, X1, X2])
y = Y_labeled

dist = squareform(pdist(X))
eps = 1  # (np.log(n) ** 2 / n) ** d
W = np.exp(-((dist / eps) ** 2))
print("eps:", eps)

plt.imshow(W - np.diag(np.diag(W)), interpolation="none")
plt.colorbar()
plt.show()

# plt.scatter(X[:, 0], X[:, 1], c="grey")
# plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=Y_labeled)
# plt.show()

solver = PoissonSolver(W, eps, p=2)
solver.fit(X, y)
output = solver._output[:, 0]

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(
    X[:, 0], X[:, 1], output, c=output, cmap="viridis",
)
plt.show()
