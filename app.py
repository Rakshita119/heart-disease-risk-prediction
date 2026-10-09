import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
# Load the learned Q-table
q_table = np.load("q_table.npy")

# Define states and actions
states = ["Low Risk", "Moderate Risk", "High Risk"]

actions = [
    "Exercise",
    "Healthy Diet",
    "Better Sleep",
    "Medical Monitoring"
]

print("Model and Q-table loaded successfully!")
# Page title and introduction

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.30),
                    rgba(255, 255, 255, 0.30)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}

        [data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.15);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("hospital.png")
st.title("❤️ AI-Based Heart Disease Risk Prediction")

st.write(
    "This system uses Machine Learning to estimate heart disease risk "
    "and Q-learning to provide a general wellness recommendation."
)

st.info(
    "This is an educational risk-screening prototype, not a medical "
    "diagnosis tool. Please consult a qualified healthcare professional "
    "for medical advice."
)
st.header("Enter Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [1, 2, 3, 4]
)

trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol (chol)",
    min_value=50,
    max_value=700,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG (restecg)",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    [0, 1]
)

oldpeak = st.number_input(
    "ST Depression (oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels (ca)",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [3, 6, 7]
)
if st.button("🔍 Predict Heart Disease Risk"):

    patient_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Predict probability
    probability = model.predict_proba(patient_data)[0][1]

    # Determine risk state
    if probability < 0.33:
        state = 0
    elif probability < 0.66:
        state = 1
    else:
        state = 2

    risk_state = states[state]

    # Select Q-learning recommendation
    best_action = np.argmax(q_table[state])
    recommendation = actions[best_action]

    # Recommendation messages
    messages = {
        "Exercise": "Regular physical activity can support overall heart health.",
        "Healthy Diet": "A balanced diet with nutritious foods can support overall heart health.",
        "Better Sleep": "Maintaining a regular and adequate sleep schedule supports overall health.",
        "Medical Monitoring": "Consider discussing your risk factors with a qualified healthcare professional."
    }

    advice = messages[recommendation]

    st.subheader("Prediction Result")

    st.write("**Risk Probability:**", round(probability, 3))
    st.write("**Risk State:**", risk_state)
    st.write("**Recommended Action:**", recommendation)
    st.write("**Advice:**", advice)
