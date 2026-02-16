import pandas as pd
from sklearn.cluster import KMeans

# Unsupervised data (no target column)
data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 60],
    "Salary": [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary"]]

# KMeans clustering with 2 clusters
model = KMeans(n_clusters=2, random_state=42)

model.fit(X)

# Cluster labels assigned to each data point
labels = model.labels_

print("Cluster Labels:", labels)
