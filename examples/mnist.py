import graphlearning as gl

labels = gl.datasets.load("mnist", labels_only=True)
W = gl.weightmatrix.knn("mnist", 10, metric="vae")

print(labels)
