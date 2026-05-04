import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="30-Day Readmission Predictor", layout="wide")

st.title("🏥 Hospital Readmission Prediction (≤30 Days)")
st.markdown("This project builds a predictive machine learning model in Python to estimate the likelihood of patient readmission within 30 days, utilizing a Random Forest algorithm and the SMOTEENN resampling method to manage class imbalance and improve overall model accuracy.")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("lessthan30_RandomForest_Tuned.Smoteenn.Imputations.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'lessthan30_RandomForest_Tuned.Smoteenn.Imputations.pkl' is in the workspace.")
        return None

# ============================================================
# HELPER FUNCTIONS (FROM NOTEBOOK)
# ============================================================

def icd_to_chapter(code):
    """Convert ICD-9 code to diagnosis chapter."""
    try:
        code = float(code)
    except:
        return "Unknown"

    if 1 <= code <= 139:
        return "Infectious and Parasitic Diseases"
    elif 140 <= code <= 239:
        return "Neoplasms"
    elif 240 <= code <= 279:
        return "Endocrine/Metabolic"
    elif 280 <= code <= 289:
        return "Blood Diseases"
    elif 290 <= code <= 319:
        return "Mental Disorders"
    elif 320 <= code <= 389:
        return "Nervous System"
    elif 390 <= code <= 459:
        return "Circulatory System"
    elif 460 <= code <= 519:
        return "Respiratory System"
    elif 520 <= code <= 579:
        return "Digestive System"
    elif 580 <= code <= 629:
        return "Genitourinary System"
    elif 630 <= code <= 679:
        return "Pregnancy/Childbirth"
    elif 680 <= code <= 709:
        return "Skin/Subcutaneous"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 740 <= code <= 759:
        return "Congenital Anomalies"
    elif 760 <= code <= 779:
        return "Perinatal Conditions"
    elif 780 <= code <= 799:
        return "Symptoms/Ill-defined"
    elif 800 <= code <= 999:
        return "Injury/Poisoning"
    elif code >= 1000:
        return "Unknown"
    else:
        return "Unknown"

def diag_chapter_to_binary(chapter):
    """Convert diagnosis chapter to binary risk indicator."""
    risk_chapters = {
        "Endocrine/Metabolic": 1,
        "Circulatory System": 1,
        "Respiratory System": 1,
        "Neoplasms": 1,
        "Infectious and Parasitic Diseases": 1,
        "Injury/Poisoning": 1,
        "Genitourinary System": 1,
        "Nervous System": 1,
        "Blood Diseases": 1,
        "Mental Disorders": 1,
    }
    return risk_chapters.get(chapter, 0)

def preprocess_input(input_data):
    """Apply all preprocessing steps from the notebook."""
    df = pd.DataFrame([input_data])
    
    # ---- GENDER ENCODING (matches training data) ----
    # Training used actual gender values from the dataset
    gender_mapping = {
        "M": "Male",
        "F": "Female"
    }
    actual_gender = gender_mapping.get(df['gender'].iloc[0], df['gender'].iloc[0])
    
    # Create all possible gender columns
    for g_val in ["Female", "Male", "Unknown/Invalid"]:
        col = f"Gender_{g_val}"
        df[col] = int(actual_gender == g_val)
    
    # ---- AGE ENCODING ----
    age_map = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    
    # ---- ADMISSION TYPE BINARY ----
    admission_type_binary_map = {
        1: 1, 2: 1, 7: 1,  # Emergency/Urgent/Trauma
        3: 0, 4: 0, 5: 0, 6: 0, 8: 0  # Elective/Newborn/NotAvailable/NULL/NotMapped
    }
    df["admission_emergency"] = df["admission_type_id"].map(admission_type_binary_map).fillna(0).astype(int)
    
    # ---- DISCHARGE DISPOSITION RISK ----
    discharge_disposition_binary_map = {
        2: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 19: 0, 20: 0, 21: 0,  # Lower-risk
        1: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 16: 1,
        17: 1, 18: 1, 22: 1, 23: 1, 24: 1, 25: 1, 27: 1, 28: 1, 29: 1, 30: 1  # Higher-risk
    }
    df["DischargeRisk"] = df["discharge_disposition_id"].map(discharge_disposition_binary_map).fillna(0).astype(int)
    
    # ---- DIAGNOSIS CHAPTERS ----
    for col in ["diag_1", "diag_2", "diag_3"]:
        chapter_col = f"{col}_chapter"
        df[chapter_col] = df[col].apply(icd_to_chapter)
        df[chapter_col] = df[chapter_col].apply(diag_chapter_to_binary)
    
    # ---- DIABETIC MEDICATION ----
    df['diabetesMed'] = (df['diabetesMed'].str.lower() == "yes").astype(int)
    
    # ---- RACE ENCODING (matches training data - no spaces) ----
    race_mapping = {
        "Caucasian": "Caucasian",
        "African American": "AfricanAmerican",
        "Hispanic": "Hispanic",
        "Asian": "Asian",
        "Other": "Other",
        "Unknown": "Unknown"
    }
    actual_race = race_mapping.get(df['race'].iloc[0], df['race'].iloc[0])
    
    # Create all possible race columns as they were in training
    for race_val in ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]:
        col_name = f"Race_{race_val}"
        df[col_name] = int(actual_race == race_val)
    
    return df


def get_model_features():
    """Get the exact feature names and order expected by the model."""
    # Try to get feature names from the model itself
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    
    # Fallback to hardcoded list if model doesn't have feature_names_in_
    return [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_outpatient',
        'number_inpatient', 'number_diagnoses', 'age',
        'admission_emergency', 'DischargeRisk',
        'diag_1_chapter', 'diag_2_chapter', 'diag_3_chapter',
        'diabetesMed', 'Gender_Female', 'Gender_Male', 'Gender_Unknown/Invalid',
        'Race_Caucasian', 'Race_AfricanAmerican', 'Race_Hispanic', 'Race_Asian', 'Race_Other', 'Race_Unknown'
    ]

# ============================================================
# STREAMLIT APP
# ============================================================

model = load_model()

if model is not None:
    # ============================================================
    # MAIN CONTENT LAYOUT
    # ============================================================
    
    st.header("📋 Patient Information")
    
    # Demographics Section
    st.subheader("👤 Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_options = [""] + ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        age_range = st.selectbox(
            "Age Range",
            age_options,
            help="Select patient age range"
        )
    
    with col2:
        gender_options = ["", "F", "M"]
        gender = st.selectbox("Gender", gender_options)
    
    with col3:
        race_options = ["", "African American", "Asian", "Caucasian", "Hispanic", "Other", "Unknown"]
        race = st.selectbox(
            "Race",
            race_options
        )
    
    # Hospital Stay Metrics
    st.subheader("🏥 Hospital Stay Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14)
    
    with col2:
        num_lab_procedures = st.slider("Lab Procedures", 0, 100)
    
    with col3:
        num_procedures = st.slider("Procedures", 0, 10)
    
    with col4:
        num_medications = st.slider("Medications", 0, 81)
    
    # Visit History
    st.subheader("📊 Visit History")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        number_emergency = st.slider("Emergency Visits", 0, 50)
    
    with col2:
        number_outpatient = st.slider("Outpatient Visits", 0, 50)
    
    with col3:
        number_inpatient = st.slider("Inpatient Visits", 0, 50)
    
    with col4:
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16)
    
    # Clinical Details
    st.subheader("🏥 Clinical Details")
    col1, col2 = st.columns(2)
    
    with col1:
        admission_type_map = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 
             5: "NotAvailable", 6: "NULL", 7: "TraumaCenter", 8: "NotMapped"}
        admission_options = [""] + sorted([v for k, v in admission_type_map.items()])
        admission_display = st.selectbox(
            "Admission Type",
            admission_options
        )
        admission_type = next((k for k, v in admission_type_map.items() if v == admission_display), None) if admission_display else None
    
    with col2:
        discharge_disposition_map = {
            1: "Home", 2: "Short-Term Hospital", 3: "SNF", 4: "ICF", 5: "Another Inpatient Care",
            6: "Home with Home Health", 7: "Left AMA", 8: "Home with IV Provider", 9: "Admitted to This Hospital",
            10: "Neonate to Mother", 11: "Expired", 12: "Still Patient/Expected Return",
            13: "Cancer Center/Hospital", 14: "Federal Health Care Facility", 15: "Nursing Facility (Medicare)",
            16: "Psychiatric Hospital", 17: "Rehabilitation Facility", 18: "Another Inpatient Care",
            19: "Cardiac Surgery Hospital", 20: "SNF (Medicare) in Acute Hospital", 21: "Outpatient Services",
            22: "Rehabilitation Facility", 23: "Long-Term Care Hospital", 24: "Nursing Facility (Medicare)",
            25: "Not Mapped", 26: "Unknown", 27: "Home with Skilled Nursing", 28: "Medical Facility Exploitation",
            29: "Intermediate Care/Mentally Retarded", 30: "Psychiatric Hospital/Distinct Unit"
        }
        disposition_options = [""] + sorted([v for k, v in discharge_disposition_map.items()])
        disposition_display = st.selectbox(
            "Discharge Disposition",
            disposition_options
        )
        discharge_disposition = next((k for k, v in discharge_disposition_map.items() if v == disposition_display), None) if disposition_display else None
    
    # Diagnosis Codes
    st.subheader("📊 Diagnosis Chapters (ICD-9)")
    
    # Mapping of diagnosis chapters to representative ICD codes
    diagnosis_chapters_unsorted = {
        "Infectious and Parasitic Diseases": 1,
        "Neoplasms": 140,
        "Endocrine/Metabolic": 250,
        "Blood Diseases": 280,
        "Mental Disorders": 290,
        "Nervous System": 320,
        "Circulatory System": 390,
        "Respiratory System": 460,
        "Digestive System": 520,
        "Genitourinary System": 580,
        "Pregnancy/Childbirth": 630,
        "Skin/Subcutaneous": 680,
        "Musculoskeletal": 710,
        "Congenital Anomalies": 740,
        "Perinatal Conditions": 760,
        "Symptoms/Ill-defined": 780,
        "Injury/Poisoning": 800,
        "Unknown": 999
    }
    diagnosis_chapters = diagnosis_chapters_unsorted
    sorted_chapters = [""] + sorted([k for k in diagnosis_chapters.keys()])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diag_1_chapter = st.selectbox("Primary Diagnosis Chapter", sorted_chapters)
        diag_1 = diagnosis_chapters[diag_1_chapter] if diag_1_chapter else None
    
    with col2:
        diag_2_chapter = st.selectbox("Secondary Diagnosis Chapter", sorted_chapters)
        diag_2 = diagnosis_chapters[diag_2_chapter] if diag_2_chapter else None
    
    with col3:
        diag_3_chapter = st.selectbox("Tertiary Diagnosis Chapter", sorted_chapters)
        diag_3 = diagnosis_chapters[diag_3_chapter] if diag_3_chapter else None
    
    # Medication
    st.subheader("💊 Medication")
    diabetesMed_options = ["", "No", "Yes"]
    diabetesMed = st.selectbox("Diabetic Medication Prescribed", diabetesMed_options)
    
    # Prediction Button
    st.markdown("---")
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        predict_button = st.button("🔮 Predict Readmission Risk", use_container_width=True, key="predict_btn")
    
    # ============================================================
    # PREPARE INPUT
    # ============================================================
    input_data = {
        'age': age_range,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_emergency': number_emergency,
        'number_outpatient': number_outpatient,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'admission_type_id': admission_type,
        'discharge_disposition_id': discharge_disposition,
        'race': race,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3,
        'diabetesMed': diabetesMed,
    }
    
    # ============================================================
    # MAIN CONTENT
    # ============================================================
    
    if predict_button:
        # Validate that all required fields are filled
        if not age_range or not gender or not race or not admission_type or not discharge_disposition or not diag_1_chapter or not diag_2_chapter or not diag_3_chapter or not diabetesMed:
            st.error("⚠️ Error: Please fill in all required fields before making a prediction")
            st.stop()
        
        try:
            # Preprocess
            processed_df = preprocess_input(input_data)
            
            # Get feature names that the model expects (in correct order)
            feature_cols = get_model_features()
            
            # Ensure all required features exist in processed_df
            for col in feature_cols:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            # Reorder columns to match training order exactly
            X_input = processed_df[feature_cols].astype(float)
            
            # Verify the data
            if X_input.isnull().any().any():
                st.error("⚠️ Error: NaN values found in features after preprocessing")
                st.stop()
            
            # Make prediction
            prediction_proba = model.predict_proba(X_input)[0]
            prediction = model.predict(X_input)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Prediction",
                    "⚠️ HIGH READMISSION RISK" if prediction_proba[1] > 0.40 else "✅ LOW READMISSION RISK",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Readmission Probability",
                    f"{prediction_proba[1]:.1%}",
                    delta=None
                )
            
            # Input summary
            st.markdown("---")
            st.subheader("👤 Input Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Demographics**")
                st.write(f"- Age: {age_range}")
                st.write(f"- Gender: {gender}")
                st.write(f"- Race: {race}")
            
            with summary_col2:
                st.write("**Clinical Metrics**")
                st.write(f"- Time in Hospital: {time_in_hospital} days")
                st.write(f"- Number of Medications: {num_medications}")
                st.write(f"- Number of Diagnoses: {number_diagnoses}")
            
            st.write("**Clinical Diagnosis Chapters**")
            st.write(f"- Primary: {diag_1_chapter}")
            st.write(f"- Secondary: {diag_2_chapter}")
            st.write(f"- Tertiary: {diag_3_chapter}")
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
            st.warning("Check that all required input fields are correctly filled.")
