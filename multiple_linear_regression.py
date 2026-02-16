import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset with multiple features
data = {
    "Experience": [1, 2, 3, 4, 5, 6],
    "Age": [22, 25, 30, 35, 40, 45],
    "Salary": [30000, 40000, 50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

# Multiple input features
X = df[["Experience", "Age"]]

# Target variable
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
