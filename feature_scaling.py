import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [25, 30, 35, 40, 45],
    "Salary": [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary"]]

# StandardScaler scales data to mean=0 and std=1
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original Data:\n", X)
print("\nScaled Data:\n", X_scaled)
