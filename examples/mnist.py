import numpy as np

import graphlearning as gl
import poissonlearning as pl

import logging

LOGGER = logging.getLogger("mnist")
logging.basicConfig(level="INFO")

LOGGER.info("Loading dataset...")
labels = gl.datasets.load("mnist", labels_only=True)
W = gl.weightmatrix.knn("mnist", 10, metric="vae")

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

LOGGER.info(f"Fitting model...")
P = [2, 3, 4, 6, 8]
P_HOMOTOPY = [2, 2.5, 3, 3.5, 4, 4.75, 5.5, 6, 7, 8]

model = pl.algorithms.Poisson(
    W,
    p=(max(P) - 1),
    homotopy_steps=P_HOMOTOPY,
    solver="variational",
    normalization="combinatorial",
    tol=1e-3,
    max_iter=200,
)
_, result = model.fit(train_ind, train_labels)

LOGGER.info("Computing accuracy...")
for p in P:
    prob = result[p]
    scores = prob - np.min(prob)
    scores = scores / np.max(scores)

    # Check if scores are similarity or distance
    pred_labels = np.argmax(scores, axis=1)
    accuracy = gl.ssl.ssl_accuracy(labels, pred_labels, len(train_ind))
    print("p=" + p + " : %.2f%%" % accuracy)
