import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ----------------------------
# Path handling (HF safe)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Load model & preprocessors (cached)
# ----------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model.h5"),
        compile=False
    )

    with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open(os.path.join(BASE_DIR, "onehot_encoder_geo.pkl"), "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open(os.path.join(BASE_DIR, "scalers.pickle"), "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { padding-top: 0rem; }
    .stTitle { text-align: center; color: #1f77b4; font-size: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔮 Customer Churn Prediction")
st.markdown("---")

# ----------------------------
# UI Layout
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Information")
    geography = st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0]
    )
    gender = st.selectbox(
        "Gender",
        label_encoder_gender.classes_
    )
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    st.subheader("💰 Financial Information")
    credit_score = st.number_input(
        "Credit Score", 300, 850, 650, step=10
    )
    balance = st.number_input(
        "Balance ($)", 0, 250000, 100000, step=1000
    )
    estimated_salary = st.number_input(
        "Estimated Salary ($)", 0, 200000, 100000, step=1000
    )

col3, col4 = st.columns(2)

with col3:
    st.subheader("🛍️ Product & Services")
    num_of_products = st.slider("Number of Products", 1, 4, 2)

with col4:
    st.subheader("✅ Membership Status")
    has_cr_card = st.selectbox(
        "Has Credit Card",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    is_active_member = st.selectbox(
        "Is Active Member",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# One-hot encode Geography (SAFE)
geo_encoded = onehot_encoder_geo.transform([[geography]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine features
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale input
input_data_scaled = scaler.transform(input_data)

st.markdown("---")

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(input_data_scaled, verbose=0)
prediction_proba = float(prediction[0][0])

# ----------------------------
# Results
# ----------------------------
st.subheader("🎯 Prediction Results")

col_result1, col_result2 = st.columns(2)

with col_result1:
    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba * 100:.2f}%",
        delta=f"{prediction_proba:.4f}",
        delta_color="inverse"
    )

with col_result2:
    if prediction_proba > 0.5:
        st.error("⚠️ **High Risk** — Customer is likely to churn.")
    else:
        st.success("✅ **Low Risk** — Customer is unlikely to churn.")

st.markdown("### Churn Risk Level")
st.progress(min(prediction_proba, 1.0))

# ----------------------------
# Summary
# ----------------------------
with st.expander("📊 Prediction Summary"):
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Geography:** {geography}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")
        st.write(f"**Tenure:** {tenure} years")
    with c2:
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Balance:** ${balance:,.0f}")
        st.write(f"**Salary:** ${estimated_salary:,.0f}")
        st.write(f"**Products:** {num_of_products}")
