import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from gayathri_features import configure_features_and_algorithm

# 1. Fetch algorithm configurations from your script
predictive_model = configure_features_and_algorithm()

print("\n=============================================")
print("  GAYATHRI'S PROJECT - PHASE 5 & 6: ENGINE   ")
print("=============================================")

# 2. Load ready parameters
X_train_ready = pd.read_csv("finalized_train_inputs.csv")
y_train_ready = pd.read_csv("train_labels.csv").values.ravel()

X_test_raw = pd.read_csv("test_features.csv")
X_test_ready = pd.get_dummies(X_test_raw, columns=['Contract'])
y_test_ready = pd.read_csv("test_labels.csv").values.ravel()

# Adjust testing array layout to perfectly reflect the model's dimensions
X_test_ready = X_test_ready.reindex(columns=X_train_ready.columns, fill_value=0)

# 3. Fit pipeline 
# Adding class_weight='balanced' here will natively fix the data imbalance issue!
predictive_model.set_params(class_weight='balanced')
predictive_model.fit(X_train_ready, y_train_ready)
print(">>> Success: Random Forest Classifier trained successfully.")

# 4. Extract predictive answers
class_predictions = predictive_model.predict(X_test_ready)
probability_predictions = predictive_model.predict_proba(X_test_ready)[:, 1]

# 5. Final Report Outputs
print("\n=============================================")
print("        SAIKET SYSTEMS EVALUATION REPORT     ")
print("=============================================")
print(f" ACCURACY METRIC  : {accuracy_score(y_test_ready, class_predictions):.4f}")
print(f" PRECISION METRIC : {precision_score(y_test_ready, class_predictions):.4f}")
print(f" RECALL METRIC    : {recall_score(y_test_ready, class_predictions):.4f}")
print(f" F1-SCORE METRIC  : {f1_score(y_test_ready, class_predictions):.4f}")
print(f" ROC-AUC MATRIX   : {roc_auc_score(y_test_ready, probability_predictions):.4f}")
print("=============================================\n")

# --- EXTRA CREDIT: FEATURE IMPORTANCE EXTRACTION ---
print(">>> Extracting Top Decision Drivers for SaiKet Systems Managers...")

# Get importance scores from our trained random forest
importances = predictive_model.feature_importances_
features_list = X_train_ready.columns

# Sort them in descending order
indices = np.argsort(importances)[::-1]

# Plot the graph
plt.figure(figsize=(10, 6))
plt.title("Customer Churn Predictive Drivers (Gayathri's Model)")
plt.bar(range(X_train_ready.shape[1]), importances[indices], align="center", color="teal")
plt.xticks(range(X_train_ready.shape[1]), [features_list[i] for i in indices], rotation=45)
plt.tight_layout()

# Save the chart as an image in your folder
plt.savefig("churn_drivers_chart.png")
print(">>> Success: Visual chart generated and saved as 'churn_drivers_chart.png'!")
plt.show()