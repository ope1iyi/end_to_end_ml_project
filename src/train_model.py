from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data  # features

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Cluster centers
print("Cluster centers:\n", kmeans.cluster_centers_)

# Predicted cluster labels
labels = kmeans.labels_
print("Cluster labels:\n", labels[:10])  # first 10 labels