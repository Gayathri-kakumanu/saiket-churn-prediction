import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set up the webpage layout and title
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("### Created by Gayathri | SaiKet Systems Internship Solution")
st.write("Adjust the customer attributes below to predict the likelihood of churn in real-time.")

# 1. Load the processed data and train the model inside the app
@st.cache_resource
def train_app_model():
    # Points directly to your subfolder files on GitHub
    X_train = pd.read_csv("saiket_project/finalized_train_inputs.csv")
    y_train = pd.read_csv("saiket_project/train_labels.csv").values.ravel()
    
    # Train the Random Forest using balanced class weights to fix data skew
    model = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=100)
    model.fit(X_train, y_train)
    
    return model, X_train.columns.tolist()

# Run the training function
predictive_model, trained_columns = train_app_model()

# 2. Build the User Input Form on the UI
st.sidebar.header("👤 Customer Profile Inputs")

# Sidebar sliders and dropdowns for the user to change values
tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
total_charges = st.sidebar.slider("Total Charges ($)", min_value=18.0, max_value=8500.0, value=800.0)

contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech_support = st.sidebar.selectbox("Has Tech Support?", ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Has Online Security?", ["No", "Yes", "No internet service"])

# 3. Process the UI inputs to match the model's training columns
input_data = {col: 0 for col in trained_columns}

# Map numerical inputs directly
if "tenure" in input_data: input_data["tenure"] = tenure
if "MonthlyCharges" in input_data: input_data["MonthlyCharges"] = monthly_charges
if "TotalCharges" in input_data: input_data["TotalCharges"] = total_charges

# Map categorical inputs to their respective one-hot encoded flags
if f"Contract_{contract_type}" in input_data:
    input_data[f"Contract_{contract_type}"] = 1
if f"TechSupport_{tech_support}" in input_data:
    input_data[f"TechSupport_{tech_support}"] = 1
if f"OnlineSecurity_{online_security}" in input_data:
    input_data[f"OnlineSecurity_{online_security}"] = 1

# Convert dictionary to a DataFrame matching the model columns layout
input_df = pd.DataFrame([input_data])[trained_columns]

# 4. Predict and display results on the main screen
st.subheader("🔮 Churn Risk Evaluation")

if st.button("Analyze Risk Profile", type="primary"):
    # Generate prediction probabilities
    prediction_prob = predictive_model.predict_proba(input_df)[0][1]
    risk_percentage = prediction_prob * 100
    
    st.markdown("---")
    if risk_percentage >= 50:
        st.error(f"🚨 **High Risk Warning:** This customer has a **{risk_percentage:.1f}%** probability of churning!")
        st.write("👉 **Retention Recommendation:** Offer a long-term contract upgrade or provide discount incentives on their Monthly Charges.")
    else:
        st.success(f"✅ **Low Risk Profile:** This customer is stable with only a **{risk_percentage:.1f}%** probability of churning.")
        st.write("👉 **Status:** Customer shows strong signs of loyalty. Keep monitoring behavior via regular service standard updates.")
