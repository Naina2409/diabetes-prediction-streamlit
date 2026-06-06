import streamlit as st
import numpy as np
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(
    open("diabetes_model.pkl","rb")
)

scaler = pickle.load(
    open("scaler.pkl","rb")
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("About Project")
st.sidebar.title("DIABETES PREDICTOR SYSTEM")
st.sidebar.info(
"""
Machine Learning Based
Diabetes Prediction System

Algorithm Used:
Support Vector Machine (SVM)

Developed using:
Python
Scikit-Learn
Streamlit
"""
)

# ==========================================
# TITLE
# ==========================================

st.title("🩺 Diabetes Prediction System")

st.write(
"""
This application predicts whether a person
is diabetic based on health parameters.
"""
)

# ==========================================
# SHOW MODEL PERFORMANCE
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.image(
        "diabetes_distribution.png",
        caption="Diabetes Distribution"
    )

with col2:
    st.image(
        "correlation_heatmap.png",
        caption="Correlation Heatmap"
    )

st.divider()

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("Enter Patient Details")

c1, c2 = st.columns(2)

with c1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0.0
    )

    glucose = st.number_input(
        "Glucose Level"
    )

    blood_pressure = st.number_input(
        "Blood Pressure"
    )

    skin_thickness = st.number_input(
        "Skin Thickness"
    )

with c2:

    insulin = st.number_input(
        "Insulin"
    )

    bmi = st.number_input(
        "BMI"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function"
    )

    age = st.number_input(
        "Age"
    )
if st.button("Predict Diabetes"):

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]).reshape(1,-1)

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )

    probability = model.predict_proba(
        input_scaled
    )

    confidence = np.max(
        probability
    ) * 100

    st.divider()

    if prediction[0] == 1:

        st.error(
            f"""
            Patient is likely DIABETIC

            Confidence:
            {confidence:.2f}%
            """
        )

    else:

        st.success(
            f"""
            Patient is likely NON-DIABETIC

            Confidence:
            {confidence:.2f}%
            """
        )

st.divider()

st.subheader("Model Evaluation")

st.image(
    "confusion_matrix.png",
    caption="Confusion Matrix"
)

st.markdown("---")

st.caption(
    "Developed using Streamlit and Scikit-Learn"
)
with st.expander("About the Inputs"):
    st.markdown("""
    **Skin Thickness:** Use 20–30 if unknown.  
    **Insulin:** Use 30.5 if unknown.  
    **DPF:** Family history risk score; use 0.37 if unknown.  
    **Blood Pressure:** Enter the diastolic value (e.g., 80 from 120/80).  
    **BMI:** Weight (kg) / Height² (m²).  
    """)