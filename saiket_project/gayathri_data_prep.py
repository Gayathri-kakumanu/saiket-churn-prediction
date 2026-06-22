import pandas as pd

print("\n=============================================")
print("  GAYATHRI'S PROJECT - PHASE 1: DATA PREP    ")
print("=============================================")

# Load the raw dataset
raw_data = pd.read_csv("Telco_Customer_Churn_Dataset  (1).csv")

# Isolate the precise columns requested in the SKS internship manual
target_columns = ['tenure', 'MonthlyCharges', 'Contract', 'TotalCharges', 'Churn']
filtered_matrix = raw_data[target_columns].copy()

# Fix hidden empty text string bugs in TotalCharges
filtered_matrix['TotalCharges'] = pd.to_numeric(filtered_matrix['TotalCharges'].replace(' ', '0'))

# Save clean data
filtered_matrix.to_csv("processed_backbone.csv", index=False)
print(">>> Success: Cleaned data exported to 'processed_backbone.csv'\n")