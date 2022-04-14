import numpy as np

import matplotlib.pyplot as plt

import poissonlearning as pl
import graphlearning as gl

NUM_TRAINING_POINTS = 50000
NUM_PLOTTING_POINTS = 100000
if NUM_PLOTTING_POINTS > NUM_TRAINING_POINTS:
    NUM_PLOTTING_POINTS = NUM_TRAINING_POINTS

# Load the two_circles dataset
dataset = pl.datasets.Dataset.load("line", "raw", NUM_TRAINING_POINTS)
# dataset.data = np.linspace(0, 1, NUM_TRAINING_POINTS)[:, np.newaxis]
# dataset.labels = np.zeros(NUM_TRAINING_POINTS)
# dataset.labels[dataset.data[:, 0] > 0.5] = 1

dataset.data = np.concatenate([np.array([[0.4], [0.6]]), dataset.data])
dataset.labels = np.concatenate([np.array([0, 1]), dataset.labels])

n, d = dataset.data.shape

# train_ind = np.array(
#    [(int)(NUM_TRAINING_POINTS * 0.3), (int)(NUM_TRAINING_POINTS * 0.7)]
# )
train_ind = np.array([0, 1])
train_labels = dataset.labels[train_ind]


def estimate_epsilon(data):
    min = data.min(axis=0)
    max = data.max(axis=0)
    volume = np.prod(np.abs(max - min))

    n = data.shape[0]
    epsilon = 2 * np.log(n) / n

    return epsilon


# Build the weight matrix
def epsilon_ball_test(data, epsilon, kernel="gaussian", eta=None):
    """Epsilon ball weight matrix
    ======

    General function for constructing a sparse epsilon-ball weight matrix, whose weights have the form 
   
    Parameters
    ----------
    data : (n,m) numpy array(weights, (M1, M2)), shape=(n, n)
        n data points, each of dimension m
    epsilon : float
        Connectivity radius
    kernel : string (optional), {'uniform','gaussian','singular','distance'}, default='gaussian'
        The choice of kernel in computing the weights between \\(x_i\\) and \\(x_j\\) when
        \\(\\|x_i-x_j\\|\\leq \\varepsilon\\). The choice 'uniform' corresponds to \\(w_{i,j}=1\\) 
        and constitutes an unweighted graph, 'gaussian' corresponds to
        \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
        'distance' corresponds to
        \\[ w_{i,j} = \\|x_i - x_j\\|, \\]
        and 'singular' corresponds to 
        \\[ w_{i,j} = \\frac{1}{\\|x_i - x_j\\|}, \\]
        when \\(i\\neq j\\) a(weights, (M1, M2)), shape=(n, n)nd \\(w_{i,i}=1\\).
    eta : python function handle (optional)
        If provided, this overrides the kernel option and instead uses the weights
        \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right). \\]

    Returns
    -------
    W : (n,n) scipy sparse matrix, float 
        Sparse weight matrix.
    """
    n = data.shape[0]  # Number of points

    # Rangesearch to find nearest neighbors
    from scipy.spatial.distance import cdist

    weights = cdist(data, data)
    weights[weights > epsilon] = 0.0
    weights[weights > 0] = 1.0
    np.fill_diagonal(weights, np.sum(weights, axis=1))

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    import scipy.sparse as sparse

    W = sparse.coo_matrix(weights)

    return W.tocsr()


# W = gl.weightmatrix.knn(dataset.data, k=5, symmetrize=True)
# print(W.count_nonzero())
epsilon = estimate_epsilon(dataset.data)
print(f"Epsilon: {epsilon}")
W = gl.weightmatrix.epsilon_ball(dataset.data, epsilon, kernel="uniform")
# W = epsilon_ball_test(dataset.data, epsilon, kernel="uniform")
W *= epsilon ** (-d)
# normalization constant, integrate 0 to 1: exp(-t^2/4)*t^3 dt
sigma = 1 / 3  # 8 - 10 * (np.e ** -(1 / 4))
# print(W.count_nonzero())


p = 2
# Solve the poisson problem with dirac RHS
poisson_dirac = pl.algorithms.Poisson(
    W,
    p=(p - 1),
    scale=n ** 2 * epsilon ** (p),
    solver="conjugate_gradient",
    normalization="combinatorial",
    spectral_cutoff=150,
    tol=1e-10,
    max_iter=1e7,
    rhs=None,
)
solution_dirac = poisson_dirac.fit(train_ind, train_labels)

D = gl.graph(W).degree_vector()
print(f"Mean of solution: {solution_dirac[:,0].mean()}")  # np.dot(solution[:, 0], D)}")

# Compute the analytic continuum limit
green_first_label = pl.datasets.line.greens_function(
    x=dataset.data, z=dataset.data[train_ind[0]],
)
green_second_label = pl.datasets.line.greens_function(
    x=dataset.data, z=dataset.data[train_ind[1]],
)
solution_analytic = 1 / sigma * (0.5 * green_first_label - 0.5 * green_second_label)
print(
    "Multiplicative offset: {}".format(
        np.abs(solution_analytic / solution_dirac[:, 0]).mean()
    )
)

plot_indices = np.argsort(dataset.data[:NUM_PLOTTING_POINTS, 0])

# Plot the solution
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    dataset.data[plot_indices, 0], solution_dirac[plot_indices, 0], label="RHS: Dirac",
)
ax.plot(
    dataset.data[plot_indices, 0], solution_analytic[plot_indices], label="Analytic",
)
# ax_bump.set_title(f"eps: {epsilon:.4f}; Analytic solution to continuum problem")
ax.set_title(f"n: {n}; eps: {epsilon:.4f}")  # ; RHS: Dirac")
ax.legend()
ax.grid()

print(f"L1 error: {np.nanmean(np.abs(solution_analytic - solution_dirac[:, 0]))}")

plt.show()
