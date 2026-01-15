import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scalers.pickle', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { padding-top: 0rem; }
    .stTitle { text-align: center; color: #1f77b4; font-size: 2.5rem; }
    </style>
    """, unsafe_allow_html=True)

st.title('🔮 Customer Churn Prediction')
st.markdown('---')

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Information")
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key="geo")
    gender = st.selectbox('Gender', label_encoder_gender.classes_, key="gender")
    age = st.slider('Age', 18, 92, 35, key="age")
    tenure = st.slider('Tenure (Years)', 0, 10, 5, key="tenure")
    
with col2:
    st.subheader("💰 Financial Information")
    credit_score = st.number_input('Credit Score', 300, 850, 650, step=10, key="credit")
    balance = st.number_input('Balance ($)', 0, 250000, 100000, step=1000, key="balance")
    estimated_salary = st.number_input('Estimated Salary ($)', 0, 200000, 100000, step=1000, key="salary")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🛍️ Product & Services")
    num_of_products = st.slider('Number of Products', 1, 4, 2, key="products")
    
with col4:
    st.subheader("✅ Membership Status")
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="card")
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="active")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

st.markdown('---')

# Predict churn
prediction = model.predict(input_data_scaled, verbose=0)
prediction_proba = prediction[0][0]

# Display results with visual styling
st.subheader("🎯 Prediction Results")

col_result1, col_result2 = st.columns(2)

with col_result1:
    st.metric(label="Churn Probability", value=f"{prediction_proba*100:.2f}%", 
              delta=f"{prediction_proba:.4f}", delta_color="inverse")

with col_result2:
    if prediction_proba > 0.5:
        st.error(f"⚠️ **High Risk** - The customer is likely to churn.")
    else:
        st.success(f"✅ **Low Risk** - The customer is not likely to churn.")

# Add a progress bar
st.markdown("### Churn Risk Level")
st.progress(float(min(prediction_proba, 1.0)))

# Summary section
with st.expander("📊 Prediction Summary"):
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.write(f"**Geography:** {geography}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")
        st.write(f"**Tenure:** {tenure} years")
    with summary_col2:
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Balance:** ${balance:,.0f}")
        st.write(f"**Salary:** ${estimated_salary:,.0f}")
        st.write(f"**Products:** {num_of_products}")
