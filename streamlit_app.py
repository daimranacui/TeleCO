import json
import os
import pickle

import pandas as pd
import streamlit as st

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    with open(os.path.join(HERE, "churn_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(HERE, "model_meta.json")) as f:
        meta = json.load(f)
    return model, meta


model, meta = load()
enc = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
OPT = {c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL, enc.categories_)}

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉")
st.title("Telco Customer Churn Predictor")
st.caption("Enter a customer's details to estimate the probability that they will leave.")

with st.form("customer"):
    c1, c2, c3 = st.columns(3)
    tenure = c1.number_input("Tenure (months)", 0, 100, 3)
    monthly = c2.number_input("Monthly charges ($)", 1.0, 500.0, 85.5, step=0.5)
    total = c3.number_input("Total charges ($, 0 = estimate)", 0.0, 20000.0, 0.0, step=1.0)

    c1, c2, c3, c4 = st.columns(4)
    senior = c1.selectbox("Senior citizen", ["No", "Yes"])
    gender = c2.selectbox("Gender", OPT["gender"])
    partner = c3.selectbox("Partner", OPT["Partner"])
    dependents = c4.selectbox("Dependents", OPT["Dependents"])

    c1, c2, c3 = st.columns(3)
    contract = c1.selectbox("Contract", OPT["Contract"])
    payment = c2.selectbox("Payment method", OPT["PaymentMethod"])
    paperless = c3.selectbox("Paperless billing", OPT["PaperlessBilling"], index=1)

    c1, c2, c3 = st.columns(3)
    phone = c1.selectbox("Phone service", OPT["PhoneService"], index=1)
    lines = c2.selectbox("Multiple lines", OPT["MultipleLines"])
    internet = c3.selectbox("Internet service", OPT["InternetService"], index=1)

    st.caption("If a customer has no internet, choose 'No internet service' for the add-ons below.")
    c1, c2, c3 = st.columns(3)
    sec = c1.selectbox("Online security", OPT["OnlineSecurity"])
    backup = c2.selectbox("Online backup", OPT["OnlineBackup"])
    device = c3.selectbox("Device protection", OPT["DeviceProtection"])
    c1, c2, c3 = st.columns(3)
    tech = c1.selectbox("Tech support", OPT["TechSupport"])
    tv = c2.selectbox("Streaming TV", OPT["StreamingTV"])
    movies = c3.selectbox("Streaming movies", OPT["StreamingMovies"])

    submitted = st.form_submit_button("Predict churn")

if submitted:
    row = {
        "SeniorCitizen": 1 if senior == "Yes" else 0, "tenure": tenure,
        "MonthlyCharges": monthly, "TotalCharges": total if total > 0 else tenure * monthly,
        "gender": gender, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    }
    p = float(model.predict_proba(pd.DataFrame([row])[NUMERIC + CATEGORICAL])[0, 1])
    st.metric("Churn probability", f"{p * 100:.1f}%")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= 0.5:
        st.error("Likely to churn (high risk)")
    elif p >= 0.3:
        st.warning("Medium churn risk")
    else:
        st.success("Likely to stay (low risk)")

st.divider()
st.caption(f"Model: {meta['algorithm']} | {meta['cv_folds']}-fold CV ROC-AUC {meta['cv_scores']['ROC-AUC']}, "
           f"Accuracy {meta['cv_scores']['Accuracy']}, F1 {meta['cv_scores']['F1']} | "
           f"Trained on {meta['n_customers']:,} customers")
