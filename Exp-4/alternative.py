import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Predefined dataset with many 2D data points
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6],
    [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0], [6.0, 4.0],
    [3.0, 3.5], [4.5, 5.0], [4.0, 4.0], [6.0, 5.0], [3.5, 4.5],
    [7.0, 6.0], [2.0, 2.5], [11.0, 3.0], [4.5, 4.5], [5.5, 5.5],
    [6.0, 2.0], [7.5, 8.5], [2.5, 0.5], [3.5, 2.0], [7.0, 9.5]
])

# Define number of clusters
k = 3
# KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
# Cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters
colors = ['red', 'green', 'blue', 'purple', 'orange']
for i in range(k):
    cluster = X[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')

# Plotting centroids
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=250, label='Centroids')

# Graph details
plt.title("K-Means Clustering Result (k=3)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend()
plt.show()

