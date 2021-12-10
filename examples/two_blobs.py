import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from poissonlearning import PoissonSolver

from graphlearning import poisson2

np.random.seed(1)

n = 1000
d = 2
X_labeled = np.array([[-1.0, -1.0], [1.0, 1.0]])
Y_labeled = np.array([0, 1])
X1 = np.random.uniform(
    low=[-1.8, -1.8], high=[0.2, 0.2], size=((n - X_labeled.shape[0]) // 2, 2)
)
X2 = np.random.uniform(
    low=[-0.2, -0.2], high=[1.8, 1.8], size=((n - X_labeled.shape[0]) // 2, 2)
)
X = np.concatenate([X_labeled, X1, X2])
y = Y_labeled

dist = squareform(pdist(X))
eps = 0.1  # 10 * (np.log(n) / np.sqrt(n)) ** d
W = (eps ** -d) * np.exp(-((dist / eps) ** 2))
W[W < 1e-4] = 0
plt.plot((W.sum(axis=1)) * eps ** d)
plt.show()
print("eps:", eps)

plt.imshow(W - np.diag(np.diag(W)), interpolation="none")
plt.colorbar()
plt.show()

solver = PoissonSolver(eps=eps, p=16, method="minimizer", maxiter=1000, disp=True)
solver.fit(W, y)
output = solver._output[:, 0]

fig = plt.figure()
ax = plt.axes(projection="3d")

t = mtri.Triangulation(X[:, 0], X[:, 1])
xind, yind, zind = t.triangles.T
xy = dist[xind, yind] ** 2
xz = dist[xind, zind] ** 2
yz = dist[yind, zind] ** 2
mask = np.any([xy > 0.5, xz > 0.5, yz > 0.5], axis=0)
t.set_mask(mask)

ax.plot_trisurf(t, output, cmap="viridis")

# ax.scatter3D(
#    X[:, 0], X[:, 1], output, c=output, cmap="viridis",
# )
plt.show()
