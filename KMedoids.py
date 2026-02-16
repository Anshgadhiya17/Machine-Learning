import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids

# Create sample dataset (no target column)

data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 60],
    "Salary": [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
}

# DataFrame stores structured tabular data
df = pd.DataFrame(data)

# Features for clustering
X = df[["Age", "Salary"]]

# Create K-Medoids model
# n_clusters = number of groups
model = KMedoids(n_clusters=2, random_state=42)

# Train the model (fit on data)
model.fit(X)

# Get cluster labels assigned to each data point
labels = model.labels_

# Get medoid centers (actual data points)
medoids = model.cluster_centers_

print("Cluster Labels:", labels)
print("\nMedoid Centers:")
print(medoids)
