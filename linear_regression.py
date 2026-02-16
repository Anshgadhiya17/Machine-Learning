import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create sample dataset using DataFrame
# DataFrame stores data in tabular format (like Excel)
data = {
    "Experience": [1, 2, 3, 4, 5, 6],
    "Salary": [30000, 40000, 50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

# X = independent variable (input feature)
X = df[["Experience"]]

# y = dependent variable (target output)
y = df["Salary"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Calculate error between actual and predicted values
mse = mean_squared_error(y_test, predictions)

print("Predictions:", predictions)
print("Mean Squared Error:", mse)
