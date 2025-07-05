import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# load model and scaler
model = joblib.load('wine_quality_model.pkl')
scaler = joblib.load('wine_quality_scaler.pkl')

st.title("Red Wine Quality Classifier 🍷")
st.write("Input chemical attributes to predict if the wine is **Good** or **Not Good**")


# input fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.001)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.001)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.00001)
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# collect input 

input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, pH, sulphates, alcohol]])
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]
    
    result = "Good 🍷" if prediction == 1 else "Not Good ❌"
    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence Score: **{confidence:.2%}**")

    st.markdown("""
    ⚠️ **Note:**

    The Confidence Score shows how certain the model is about its prediction based on your inputs.

    - A high score (close to 100%) means the model is very confident.
    - A lower score means it is less certain.

    **Important:** A high confidence score does NOT always guarantee the prediction is correct. It only shows the model's certainty based on what it learned from the training data.
    """)


