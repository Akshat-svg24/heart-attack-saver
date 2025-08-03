
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Attack Risk Predictor")
st.write("Enter your health details below:")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])

# Combine input
features = np.array([[age, sex, cp, chol, fbs, thalach, exang]])
scaled_input = scaler.transform(features)

# Prediction
prediction = model.predict(scaled_input)[0]
proba = model.predict_proba(scaled_input)[0][1]

# Display
st.subheader("Prediction Result:")
st.write(f"Predicted Risk Score: {proba*100:.2f}%")
if prediction == 1:
    st.error("⚠️ High Risk of Heart Attack")
else:
    st.success("✅ Low Risk of Heart Attack")
