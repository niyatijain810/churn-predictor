import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")
st.markdown("**Predict whether a customer will churn using a Random Forest Model**")
st.markdown("---")

st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=70, value=30)
    services_opted = st.slider("Services Opted (1–6)", min_value=1, max_value=6, value=3)
    frequent_flyer = st.selectbox("Frequent Flyer?", ["No", "Yes"])

with col2:
    annual_income = st.selectbox("Annual Income Class",
                                  ["Low Income", "Middle Income", "High Income"])
    account_synced = st.selectbox("Account Synced to Social Media?", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel?", ["No", "Yes"])

st.markdown("---")

# ── Encoding (must match training) ────────────────────────────────────────────
def encode_inputs():
    ff_map = {"No": 0, "Yes": 1}
    income_map = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
    binary_map = {"No": 0, "Yes": 1}

    return pd.DataFrame([[
        age,
        ff_map[frequent_flyer],
        income_map[annual_income],
        services_opted,
        binary_map[account_synced],
        binary_map[booked_hotel]
    ]], columns=[
        'Age', 'FrequentFlyer', 'AnnualIncomeClass',
        'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot'
    ])

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):
    input_df = encode_inputs()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("### 🎯 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN**")
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
        st.info("💡 Tip: Consider offering a loyalty discount or personalized service upgrade.")
    else:
        st.success(f"✅ **This customer is likely to STAY**")
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
        st.info("💡 Great! Keep engaging this customer with rewards programs.")

    with st.expander("📊 View Input Summary"):
        st.dataframe(input_df)

st.markdown("---")
st.caption("🤖 Powered by Random Forest | B.Tech Gen AI Project")
