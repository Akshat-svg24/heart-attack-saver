import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("❤️ Heart Attack Risk Predictor")
st.markdown("**Enter your health details below to check your heart attack risk.**")

# User inputs
age = st.slider("Age", 20, 100, 30)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.slider("Resting Blood Pressure (in mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.slider("Resting ECG Results (0–2)", 0, 2, 1)
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.radio("Exercise Induced Angina?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.slider("Slope of Peak Exercise ST Segment (0–2)", 0, 2, 1)
ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)
thal = st.slider("Thalassemia (0–3)", 0, 3, 1)

# Format input
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale input
scaled_input = scaler.transform(user_input)

# Debug view (can be removed later)
st.write("🔍 Your Input:", user_input)
st.write("🔬 Scaled Input:", scaled_input)

# Predict
if st.button("Predict Risk"):
    try:
        result = model.predict_proba(scaled_input)[0][1]
        risk_percent = round(result * 100, 2)

        # Sanity check
        if not 0 <= risk_percent <= 100:
            st.error("⚠️ Invalid prediction. Please check your inputs or model.")
        elif risk_percent < 40:
            st.success(f"✅ Low Risk: {risk_percent}% chance of heart attack.")
        elif 40 <= risk_percent < 70:
            st.warning(f"⚠️ Moderate Risk: {risk_percent}% chance of heart attack.")
        else:
            st.error(f"🚨 High Risk: {risk_percent}% chance of heart attack. Please consult a doctor.")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
