import streamlit as st
import pandas as pd
import joblib

st.title("💳 Credit Card Fraud Detection")

model = joblib.load("fraud_model.pkl")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)

    data["Prediction"] = predictions

    st.write(data)

    fraud_count = predictions.sum()

    st.success(f"Fraud Transactions Detected: {fraud_count}")