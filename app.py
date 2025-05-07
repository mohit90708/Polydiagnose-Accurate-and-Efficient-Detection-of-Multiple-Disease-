import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib

# Set page configuration
st.set_page_config(page_title="Polydiagnose - Multi Disease Detection", layout="wide", page_icon="🧑‍⚕️")

# Load models
diabetes_model = joblib.load('diabetes_model_prob.sav')
diabetes_scaler = joblib.load('diabetes_scaler.sav')

heart_model = joblib.load('heart_model_prob.sav')
heart_scaler = joblib.load('heart_scaler.sav')

parkinsons_model = pickle.load(open('C:\\Multiple Disease\\parkinsons_model.sav', 'rb'))

liver_model = joblib.load('liver.pkl')

# Sidebar menu
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction',  ],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'droplet'],
                           default_index=0)

# ================= Diabetes Prediction =================
if selected == 'Diabetes Prediction':
    st.title('🧪 Diabetes Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', 0, 20, 1)
    with col2:
        Glucose = st.number_input('Glucose Level', 0, 200, 110)
    with col3:
        BloodPressure = st.number_input('Blood Pressure', 0, 140, 80)

    with col1:
        SkinThickness = st.number_input('Skin Thickness', 0, 100, 20)
    with col2:
        Insulin = st.number_input('Insulin Level', 0, 900, 80)
    with col3:
        BMI = st.number_input('BMI', 10.0, 70.0, 25.0, format="%.2f")

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5, format="%.2f")
    with col2:
        Age = st.number_input('Age', 1, 100, 30)

    if st.button("🔍 Predict Diabetes"):
        user_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                       BMI, DiabetesPedigreeFunction, Age]]
        scaled_input = diabetes_scaler.transform(user_input)
        prediction = diabetes_model.predict(scaled_input)[0]
        probability = diabetes_model.predict_proba(scaled_input)[0][1] * 100

        if prediction == 1:
            st.warning(f"🩺 The person is **diabetic**.\n\nProbability: **{probability:.2f}%**")
        else:
            st.success(f"✅ The person is **not diabetic**.\n\nProbability: **{probability:.2f}%**")

# ================= Heart Disease Prediction =================
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', 1, 120, 45)
        trestbps = st.number_input('Resting Blood Pressure', 80, 200, 120)
        restecg = st.number_input('Resting ECG (0-2)', 0, 2, 1)
        oldpeak = st.number_input('Oldpeak (ST depression)', 0.0, 6.0, 1.0)
        thal = st.number_input('Thal (0=Normal, 1=Fixed, 2=Reversible)', 0, 2, 1)

    with col2:
        sex = st.number_input('Sex (0=Female, 1=Male)', 0, 1, 1)
        chol = st.number_input('Cholesterol', 100, 600, 200)
        thalach = st.number_input('Max Heart Rate', 60, 220, 150)
        slope = st.number_input('Slope of ST Segment (0-2)', 0, 2, 1)
        ca = st.number_input('Number of Major Vessels (0-4)', 0, 4, 0)

    with col3:
        cp = st.number_input('Chest Pain Type (0-3)', 0, 3, 1)
        fbs = st.number_input('Fasting Blood Sugar >120 mg/dl (0 or 1)', 0, 1, 0)
        exang = st.number_input('Exercise Induced Angina (0 or 1)', 0, 1, 0)

    if st.button("🔍 Predict Heart Disease"):
        user_input = [[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]]
        scaled_input = heart_scaler.transform(user_input)
        prediction = heart_model.predict(scaled_input)[0]
        probability = heart_model.predict_proba(scaled_input)[0][1] * 100

        if prediction == 1:
            st.warning(f"💔 The person is **likely to have heart disease**.\n\nProbability: **{probability:.2f}%**")
        else:
            st.success(f"❤️ The person is **not likely to have heart disease**.\n\nProbability: **{probability:.2f}%**")

# ================= Parkinson’s Prediction =================
if selected == 'Parkinsons Prediction':
    st.title("🧠 Parkinson's Disease Prediction")
    features = [
        "MDVP: Fo(Hz)", "MDVP: Fhi(Hz)", "MDVP: Flo(Hz)",
        "MDVP: Jitter(%)", "MDVP: Jitter(Abs)", "MDVP: RAP", "MDVP: PPQ", "Jitter: DDP",
        "MDVP: Shimmer", "MDVP: Shimmer(dB)", "Shimmer: APQ3", "Shimmer: APQ5", "MDVP: APQ",
        "Shimmer: DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    values = []
    col1, col2, col3 = st.columns(3)
    for i, feat in enumerate(features):
        with [col1, col2, col3][i % 3]:
            val = st.number_input(feat, value=0.0, format="%.4f")
            values.append(val)

    if st.button("🔍 Predict Parkinson's"):
        result = parkinsons_model.predict([values])[0]
        if result == 1:
            st.warning("🧠 The person **has Parkinson's disease**.")
        else:
            st.success("✅ The person **does not have Parkinson's disease**.")
