# 📊 Customer Churn Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](PASTE_YOUR_LIVE_STREAMLIT_URL_HERE)

### Developed by Gayathri | SaiKet Systems Machine Learning Internship Solution

An end-to-end, production-ready machine learning pipeline and interactive web application built to predict telecom customer attrition. This project transitions a basic flat-file dataset into a modular, enterprise-grade data science workflow, complete with feature driver visualization and a live user interface.

---

## 🚀 Key Features & Project Upgrades
* **Modular Pipeline Architecture:** The project is broken down into clean, production-standard scripts for isolation of tasks (Preparation ➡️ Splitting ➡️ Feature Engineering ➡️ Evaluation).
* **Imbalance Class Mitigation:** Handled target skew natively by incorporating balanced class weights within the core algorithm.
* **Feature Importance Analytics:** Extracts and plots the top decision drivers showing what causes customer churn.
* **Interactive Dashboard UI:** Built a web app with live sliders and input fields allowing managers to test customer profiles and see real-time risk percentages.

---

## 🛠️ Project Architecture

The project consists of 5 main files working together sequentially:

1. **`gayathri_data_prep.py`** - Cleans the raw dataset, handles missing data values, and corrects structural column types (e.g., parsing empty spaces in `TotalCharges`).
2. **`gayathri_split.py`** - Segments the records cleanly into an 80/20 train/test layout while maintaining feature synchronization.
3. **`gayathri_features.py`** - Encodes categorical text variables into analytical binary flags and initializes the `RandomForestClassifier` engine.
4. **`gayathri_evaluation.py`** - Fits the model to training data, predicts outcomes, exports a professional performance metrics report, and generates a visual feature importance bar chart.
5. **`gayathri_app.py`** - The interactive UI engine built with Streamlit to bring the machine learning model to a web interface.

---

## 📈 Performance Evaluation Metrics
The pipeline automatically computes and evaluates the baseline across five mandatory enterprise metrics:
* **Accuracy:** Overall correctness rate of predictions.
* **Precision:** The accuracy score specifically for flagged high-risk profiles.
* **Recall:** The model's ability to successfully capture and catch all actual churn cases.
* **F1-Score:** The balanced harmonic mean between Precision and Recall.
* **ROC-AUC:** Tells how well the model separates the two customer classes (staying vs. leaving).

---

## 💻 How to Run the Project Locally

### 1. Prerequisites
Ensure you have Python installed, then set up the required business libraries using your terminal:
```bash
pip install -r requirements.txt
