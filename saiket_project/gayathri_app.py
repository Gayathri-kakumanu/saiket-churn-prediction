import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Created by Gayathri | SaiKet Systems Internship Solution")
st.write("---")

# --- Step 1: Background Model Training (Simulated for App) ---
# We load the exact data processed in your earlier tasks to keep it perfectly aligned
@st.cache_resource
def train_app_model():
    X_train = pd.read_csv("finalized_train_inputs.csv")
    y_train = pd.read_csv("train_labels.csv").values.ravel()
    
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns

predictive_model, trained_columns = train_app_model()

# --- Step 2: Sidebar Input Form for Business Users ---
st.sidebar.header("User Input Features")

tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=780.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# --- Step 3: Format Input into Model Columns ---
input_data = {
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract_Month-to-month': [1 if contract_type == "Month-to-month" else 0],
    'Contract_One year': [1 if contract_type == "One year" else 0],
    'Contract_Two year': [1 if contract_type == "Two year" else 0]
}

input_df = pd.DataFrame(input_data)
# Ensure columns match training alignment perfectly
input_df = input_df.reindex(columns=trained_columns, fill_value=0)

# --- Step 4: Live Prediction Engine ---
st.subheader("🔮 Real-Time Prediction Analytics")

if st.button("Analyze Risk Profile"):
    prediction = predictive_model.predict(input_df)[0]
    probability = predictive_model.predict_proba(input_df)[0][1]
    
    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ **High Attrition Risk Detected!**")
        st.write(f"This customer has a **{probability*100:.1f}%** probability of canceling their service.")
    else:
        st.success(f"✅ **Low Attrition Risk (Loyal Customer)**")
        st.write(f"This customer has only a **{probability*100:.1f}%** probability of leaving.")
        
    # Display individual input properties back to the user
    st.dataframe(input_df)