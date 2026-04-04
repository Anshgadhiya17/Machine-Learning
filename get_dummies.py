"""
get_dummies.py

This script demonstrates One-Hot Encoding using pandas get_dummies().
"""

import pandas as pd

# Sample Data
data = {
    "Name": ["Ansh", "Raj", "Priya", "Neha"],
    "City": ["Ahmedabad", "Mumbai", "Delhi", "Mumbai"]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Apply get_dummies
df_encoded = pd.get_dummies(df, columns=["City"])

print("\nEncoded DataFrame using get_dummies:")
print(df_encoded)