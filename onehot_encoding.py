"""
onehot_encoding.py

This script demonstrates One-Hot Encoding using sklearn.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample Data
data = {
    "Name": ["Ansh", "Raj", "Priya", "Neha"],
    "City": ["Ahmedabad", "Mumbai", "Delhi", "Mumbai"]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and Transform
encoded_data = encoder.fit_transform(df[["City"]])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(["City"]))

# Combine with original DataFrame
final_df = pd.concat([df, encoded_df], axis=1)

# Drop original column
final_df.drop("City", axis=1, inplace=True)

print("\nOne-Hot Encoded DataFrame:")
print(final_df)