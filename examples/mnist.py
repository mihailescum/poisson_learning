import numpy as np

import graphlearning as gl
import poissonlearning as pl

import logging

LOGGER = logging.getLogger("mnist")
logging.basicConfig(level="INFO")

SEED = 0
N_NEIGHBORS = 10
LABELS_PER_CLASS = 5
P = [2, 3, 4]  # , 5, 6, 7, 8]

LOGGER.info("Loading dataset...")
dataset = pl.datasets.Dataset.load("mnist", metric="vae")
rng = np.random.default_rng(seed=SEED)
dataset_sample = dataset.sample(size=10000, rng=rng)
W = gl.weightmatrix.knn(data=dataset_sample.data, k=N_NEIGHBORS, kernel="gaussian")

train_ind = gl.trainsets.generate(dataset_sample.labels, rate=LABELS_PER_CLASS)
train_labels = dataset_sample.labels[train_ind]

LOGGER.info(f"Fitting model...")

model = pl.algorithms.Poisson(
    W,
    p=(max(P) - 1),
    homotopy_steps=P,
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
    accuracy = gl.ssl.ssl_accuracy(dataset_sample.labels, pred_labels, len(train_ind))
    print(f"Accuracy: p={p} : {accuracy:.2f}")
