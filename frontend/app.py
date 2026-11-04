import os

import requests
import streamlit as st

API_URL = f"{os.getenv('API_URL', 'http://127.0.0.1:8000').rstrip('/')}/predict"

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="centered",
)
st.title("Customer Churn Prediction")
st.write("Enter customer information to estimate churn risk.")

with st.form("churn_input_form"):
    tenure = st.slider("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input(
        "Monthly charges",
        min_value=0.0,
        max_value=1000.0,
        value=65.0,
    )
    total_charges = st.number_input(
        "Total charges",
        min_value=0.0,
        max_value=100000.0,
        value=780.0,
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )
    paperless = st.selectbox("Paperless billing", ["No", "Yes"])
    submitted = st.form_submit_button("Calculate churn risk")

if submitted:
    payload = {
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Contract_One_year": int(contract == "One year"),
        "Contract_Two_year": int(contract == "Two year"),
        "PaperlessBilling_Yes": int(paperless == "Yes"),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        if error.response is not None:
            try:
                detail = error.response.json().get("detail", detail)
            except (ValueError, AttributeError):
                pass
        st.error(f"Prediction API returned an error: {detail}")
    except requests.RequestException as error:
        st.error(f"Could not get a prediction from the API: {error}")
    else:
        result = response.json()
        probability = result["churn_probability"] * 100
        st.metric("Churn probability", f"{probability:.2f}%")
        if result["churn_prediction"] == 1:
            st.error(result["retention_insight"])
        else:
            st.success(result["retention_insight"])
