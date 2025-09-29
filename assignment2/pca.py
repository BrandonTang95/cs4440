# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: pca.py
# SPECIFICATION: Perform PCA on a dataset, removing one feature at a time to find the feature whose removal maximizes the variance of the first principal component (PC1).
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
# --> add your Python code here
df = pd.read_csv("heart_disease_dataset.csv")

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Get the number of features
# --> add your Python code here
num_features = df.shape[1]

# Store results
results = []

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA(n_components=1)
    pca.fit(reduced_data)

    # Store PC1 variance and the feature removed
    # Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_variance = pca.explained_variance_ratio_[0]
    feature_removed = df.columns[i]
    results.append((pc1_variance, feature_removed))
    print(f"PC1 variance when removing {feature_removed}: {pc1_variance:.6f}")

# Find the maximum PC1 variance
# --> add your Python code here
best_pc1, best_feature = max(results, key=lambda x: x[0])

# Print results
# Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {best_pc1:.6f} when removing {best_feature}")
