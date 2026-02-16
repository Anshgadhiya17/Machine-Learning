import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 60],
    "Salary": [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],
    "Purchased": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary"]]
y = df["Purchased"]

model = LogisticRegression()

# 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())
