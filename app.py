"""
app.py
------
Streamlit interface for the Employee Salary Predictor.
Run with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/salary_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    numeric_cols = joblib.load("model/numeric_cols.pkl")
    best_model_name = joblib.load("model/best_model_name.pkl")
    category_options = joblib.load("model/category_options.pkl")
    return model, encoders, scaler, feature_cols, numeric_cols, best_model_name, category_options


try:
    model, encoders, scaler, feature_cols, numeric_cols, best_model_name, category_options = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Please run `python generate_dataset.py` "
        "and then `python train_model.py` first to train and save the model."
    )
    st.stop()

st.title("💼 Employee Salary Predictor")
st.write(
    "Enter an employee's details below to predict their expected annual salary. "
    f"Powered by a **{best_model_name}** model trained on employee data."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=30)
    experience = st.slider("Years of Experience", min_value=0, max_value=45, value=5)
    gender = st.selectbox("Gender", category_options["Gender"])
    education = st.selectbox("Education Level", category_options["Education Level"])

with col2:
    department = st.selectbox("Department", category_options["Department"])
    job_title = st.selectbox("Job Title", category_options["Job Title"])
    location = st.selectbox("Location", category_options["Location"])

if experience > (age - 18):
    st.warning("Years of experience seems high relative to age — prediction may be less reliable.")

st.divider()

if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
    input_dict = {
        "Age": age,
        "Years of Experience": experience,
        "Gender": gender,
        "Education Level": education,
        "Department": department,
        "Job Title": job_title,
        "Location": location,
    }
    input_df = pd.DataFrame([input_dict])

    # Encode categoricals using the saved LabelEncoders
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Scale numeric columns using the saved scaler
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    input_df = input_df[feature_cols]
    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Annual Salary: ₹{prediction:,.0f}")
    st.caption(
        "This is an estimate based on patterns in the training data, "
        "not a guaranteed figure."
    )

with st.expander("📊 About this model"):
    try:
        comparison = pd.read_csv("model/model_comparison.csv")
        st.write("Model comparison from training (on held-out test data):")
        st.dataframe(comparison.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}))
    except FileNotFoundError:
        st.write("Run `train_model.py` to generate a model comparison table.")
    st.write(
        "The dataset includes Age, Gender, Education Level, Department, Job Title, "
        "Years of Experience, and Location as predictors of Salary."
    )
