"""
Simple Streamlit app to demo churn predictions.

It expects an exported pipeline at exported_model/rf_churn_pipe.joblib.
If not found, it will attempt to look in exported_model_ci/ for CI runs.
"""

import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATHS = [
    "exported_model/rf_churn_pipe.joblib",
    "exported_model_ci/rf_churn_pipe.joblib",
    "exported_model/rf_churn_pipe.pkl",
    "exported_model/model.joblib"
]

@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            model = joblib.load(p)
            st.session_state["model_path"] = p
            return model
    return None

st.set_page_config(page_title="Churn Demo", layout="centered")
st.title("Online Retail — Customer Churn Demo")

model = load_model()

if model is None:
    st.error("No trained model found. Please run train_and_export.py and ensure `exported_model/rf_churn_pipe.joblib` exists.")
    st.text("Example: python train_and_export.py --sample 0.05 --model-dir exported_model")
    st.stop()

st.markdown(f"**Loaded model from:** `{st.session_state.get('model_path')}`")

st.write("Enter customer features (RFM + avg unit price + total quantity)")

col1, col2 = st.columns(2)
with col1:
    recency = st.number_input("Recency (days since last purchase)", min_value=0.0, value=30.0, step=1.0)
    frequency = st.number_input("Frequency (unique invoices)", min_value=0.0, value=3.0, step=1.0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0, step=1.0)
with col2:
    avg_price = st.number_input("Avg Unit Price", min_value=0.0, value=20.0, step=0.5)
    total_qty = st.number_input("Total Quantity", min_value=0.0, value=10.0, step=1.0)

if st.button("Predict churn probability"):
    X = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AvgUnitPrice": avg_price,
        "TotalQuantity": total_qty
    }])
    try:
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(model.predict(X)[0])
        st.metric("Churn probability", f"{proba:.3f}")
        st.write("Predicted class:", "Churn" if pred == 1 else "Not churn")
    except Exception as e:
        st.error(f"Model failed to predict: {e}")
