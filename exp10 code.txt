import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate dataset
X, y = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

# Apply EM algorithm using Gaussian Mixture Model
model = GaussianMixture(
    n_components=3,
    random_state=42,
    max_iter=100
)

model.fit(X)

# Predict clusters
labels = model.predict(X)

# Display results
print("Expectation-Maximization Algorithm")
print("Number of Clusters:", model.n_components)
print("Number of Iterations:", model.n_iter_)
print("Converged:", model.converged_)
print("Log-Likelihood:", round(model.score(X) * len(X), 2))

# Plot original dataset
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# Plot clustered dataset
plt.figure(figsize=(7, 5))
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap="viridis",
    s=40
)

plt.scatter(
    model.means_[:, 0],
    model.means_[:, 1],
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.title("EM Algorithm - Gaussian Mixture Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()