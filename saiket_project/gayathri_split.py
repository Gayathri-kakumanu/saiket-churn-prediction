import pandas as pd
from sklearn.model_selection import train_test_split

print("\n=============================================")
print("  GAYATHRI'S PROJECT - PHASE 2: DATA SPLIT   ")
print("=============================================")

# Load data backbone
dataframe = pd.read_csv("processed_backbone.csv")

# Separate features from target labels
X_matrix = dataframe.drop(columns=['Churn'])
y_vector = dataframe['Churn'].map({'Yes': 1, 'No': 0})

# Use an 80/20 train/test partition ratio
X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_vector, test_size=0.20, random_state=42)

# Export the temporary splits
X_train.to_csv("train_features.csv", index=False)
X_test.to_csv("test_features.csv", index=False)
y_train.to_csv("train_labels.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)

print(">>> Success: Data splits saved to your workspace.\n")