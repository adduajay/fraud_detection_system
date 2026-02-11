import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🚨 Fraud Detection Analytics Dashboard")

# =========================
# Upload Dataset
# =========================
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(data.head())

    # =========================
    # Basic Cleaning
    # =========================
    data = data.dropna()

    # =========================
    # Select numeric features
    # =========================
    numeric_cols = data.select_dtypes(include=np.number).columns
    X = data[numeric_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # Anomaly Detection Model
    # =========================
    model = IsolationForest(contamination=0.02, random_state=42)
    data["Anomaly"] = model.fit_predict(X_scaled)

    # Convert: -1 = Fraud, 1 = Normal
    data["Fraud_Prediction"] = data["Anomaly"].apply(lambda x: 1 if x == -1 else 0)

    # Risk Score
    scores = model.decision_function(X_scaled)
    data["Risk_Score"] = np.round((1 - scores) * 100, 2)

    # =========================
    # Dashboard Metrics
    # =========================
    total = len(data)
    frauds = data["Fraud_Prediction"].sum()
    normal = total - frauds

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected", frauds)
    col3.metric("Normal Transactions", normal)

    # =========================
    # Charts
    # =========================

    st.subheader("📈 Fraud Distribution")

    pie = px.pie(
        names=["Normal", "Fraud"],
        values=[normal, frauds],
        title="Fraud vs Normal"
    )
    st.plotly_chart(pie, use_container_width=True)

    # Risk score distribution
    st.subheader("📉 Risk Score Distribution")
    hist = px.histogram(data, x="Risk_Score")
    st.plotly_chart(hist, use_container_width=True)

    # Amount vs Risk (if exists)
    if "Amount" in data.columns:
        st.subheader("💳 Amount vs Risk Score")
        scatter = px.scatter(
            data,
            x="Amount",
            y="Risk_Score",
            color="Fraud_Prediction"
        )
        st.plotly_chart(scatter, use_container_width=True)

    # =========================
    # Suspicious Transactions
    # =========================
    st.subheader("⚠️ Suspicious Transactions")
    suspicious = data[data["Fraud_Prediction"] == 1]
    st.dataframe(suspicious.head(20))

    # Download option
    st.download_button(
        "Download Fraud Results",
        suspicious.to_csv(index=False),
        file_name="fraud_transactions.csv"
    )

else:
    st.info("Upload a Kaggle dataset CSV to start analysis.")
