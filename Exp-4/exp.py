import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# -------------------------------
# 1. Load CSV and select features
# -------------------------------
df = pd.read_csv('E:/4th Year 2nd Semester\Machine Learning Lab\ML Lab Code\Exp-4\income.csv')
X = df[['Age', 'Income($)']].values
rows = X.shape[0]
# -------------------------------
# 2. Scale the data
# -------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# -------------------------------
# 3. Initialize centroids (by row index)
# -------------------------------
centroids = [
    X_scaled[5],
    X_scaled[10],
    X_scaled[15]
]
# -------------------------------
# 4. Assign dummy "true labels" based on income for evaluation
# -------------------------------
labels = np.array([
    0 if income < 0.3 else 1 if income < 0.7 else 2
    for _, income in X_scaled
])
# -------------------------------
# 5. K-Means Loop
# -------------------------------
max_iter = 90
for _ in range(max_iter):
    clusters = {0: [], 1: [], 2: []}
    class_labels = {0: [], 1: [], 2: []}

    # Assign points to closest centroid
    for i in range(rows):
        point = X_scaled[i]
        cl = labels[i]
        distances = [np.linalg.norm(point - c) for c in centroids]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)
        class_labels[cluster_idx].append(cl)

    # Update centroids
    for i in range(3):
        if clusters[i]:
            centroids[i] = np.mean(clusters[i], axis=0)
# -------------------------------
# 6. Evaluation Function
# -------------------------------
def classify_correction(y):
    counts = {val: y.count(val) for val in set(y)}
    majority_class = max(counts, key=counts.get)
    correct = counts[majority_class]
    incorrect = len(y) - correct
    return correct, incorrect, majority_class
# -------------------------------
# 7. Show Result
# -------------------------------
for i in range(3):
    corr, incorr, cl = classify_correction(class_labels[i])
    print(f"Cluster {i+1} → Correct: {corr}, Incorrect: {incorr}, Class: {cl}")
    print("*" * 40)
# -------------------------------
# 8. Plot Clusters
# -------------------------------
plt.figure(figsize=(10, 6))
colors = ['green', 'red', 'blue']
for i in range(3):
    cluster = np.array(clusters[i])
    if len(cluster) > 0:
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')

for c in centroids:
    plt.scatter(c[0], c[1], color='black', marker='x', s=200)

plt.xlabel('Age (scaled)')
plt.ylabel('Income ($, scaled)')
plt.title('Manual K-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()
