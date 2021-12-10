import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import poissonlearning as pl

import plotting

np.random.seed(1)

n = 1000
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

dist = pl.distance_matrix(X)
eps = 0.1  # 10 * (np.log(n) / np.sqrt(n)) ** d
print("eps:", eps)

W = pl.kernel_exponential(dist, eps, d=2, cutoff=1e-4)

fig, ax = plt.subplots(2, 1)
ax[0].plot(pl.node_degrees(W))

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
im = ax[1].imshow((W - np.diag(W.diagonal())) * (eps ** 2), interpolation="none")
fig.colorbar(im, cax=cax, orientation="vertical")

plt.show()

solver = pl.PoissonSolver(eps=eps, p=2, method="minimizer", maxiter=1000, disp=True)
solver.fit(W, y)
output = solver._output[:, 0]

plotting.plot_graph_function_with_triangulation(
    X[:, 0], X[:, 1], output, dist ** 2, max_dist=0.5,
)

plt.show()
