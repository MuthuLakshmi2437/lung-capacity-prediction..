import streamlit as st
import pandas as pd
import pickle

# Load model
with open("ridge_model_pickle", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler_pickle", "rb") as f:
    scaler = pickle.load(f)

st.title("🫁 Lung Capacity Prediction")

age = st.number_input("Age", min_value=1, max_value=100, value=10)
height = st.number_input("Height", min_value=20.0, max_value=90.0, value=60.0)

gender = st.selectbox("Gender", ["Female", "Male"])
smoke = st.selectbox("Smoke", ["No", "Yes"])

# One-hot encoding
gender_male = 1 if gender == "Male" else 0
smoke_yes = 1 if smoke == "Yes" else 0

# Input DataFrame (same column names and order as training)
input_df = pd.DataFrame({
    "Age": [age],
    "Height": [height],
    "Gender_male": [gender_male],
    "Smoke_yes": [smoke_yes]
})

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Lung Capacity: {prediction[0]:.2f}")
