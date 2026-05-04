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

def preprocess_input(input_data):
    """Apply all preprocessing steps from the notebook."""
    df = pd.DataFrame([input_data])
    
    # ---- GENDER ENCODING (matches training data) ----
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
    
    # ---- ADMISSION TYPE ONE-HOT ENCODING ----
    admission_type_map = {
        1: "emergency",
        2: "urgent",
        3: "elective",
        4: "newborn",
        5: "notavailable",
        6: "null",
        7: "trauma",
        8: "notmapped"
    }
    df["admission_type"] = df["admission_type_id"].map(admission_type_map)
    
    # One-hot encode admission types
    admission_dummies = pd.get_dummies(
        df["admission_type"],
        prefix="admission"
    ).astype(int)
    df = pd.concat([df, admission_dummies], axis=1)
    
    # ---- DISCHARGE DISPOSITION ONE-HOT ENCODING ----
    discharge_disposition_map = {
        1: "home",
        2: "short_term_hospital",
        3: "snf",
        4: "icf",
        5: "other_inpatient",
        6: "home_health",
        7: "left_ama",
        8: "home_iv_care",
        9: "admitted_this_hospital",
        10: "neonate_another_hospital",
        11: "expired",
        12: "still_patient",
        13: "hospice_home",
        14: "hospice_facility",
        15: "swing_bed",
        16: "outpatient_other_institution",
        17: "outpatient_this_institution",
        18: "null",
        19: "expired_home_medicaid",
        20: "expired_facility_medicaid",
        21: "expired_unknown_medicaid",
        22: "rehab_facility",
        23: "long_term_care_hospital",
        24: "medicaid_nursing_facility",
        25: "not_mapped",
        27: "federal_healthcare_facility",
        28: "psychiatric_hospital",
        29: "critical_access_hospital",
        30: "other_healthcare_institution"
    }
    df["discharge_type"] = df["discharge_disposition_id"].map(discharge_disposition_map)
    
    # One-hot encode discharge types
    discharge_dummies = pd.get_dummies(
        df["discharge_type"],
        prefix="discharge"
    ).astype(int)
    df = pd.concat([df, discharge_dummies], axis=1)
    
    # ---- DIAGNOSIS CHAPTERS WITH ONE-HOT ENCODING ----
    diag_chapter_map = {
        "Endocrine/Metabolic": "endocrine_metabolic",
        "Circulatory System": "circulatory",
        "Respiratory System": "respiratory",
        "Neoplasms": "neoplasms",
        "Infectious and Parasitic Diseases": "infectious_parasitic",
        "Injury/Poisoning": "injury_poisoning",
        "Genitourinary System": "genitourinary",
        "Nervous System": "nervous_system",
        "Blood Diseases": "blood_diseases",
        "Mental Disorders": "mental_disorders",
        "Symptoms/Ill-defined": "symptoms_ill_defined",
        "Skin/Subcutaneous": "skin_subcutaneous",
        "Musculoskeletal": "musculoskeletal",
        "Digestive System": "digestive",
        "Pregnancy/Childbirth": "pregnancy_childbirth",
        "Congenital Anomalies": "congenital_anomalies",
        "Perinatal Conditions": "perinatal_conditions",
        "Unknown": "unknown"
    }
    
    for col in ["diag_1", "diag_2", "diag_3"]:
        chapter_col = f"{col}_chapter"
        df[chapter_col] = df[col].apply(icd_to_chapter)
        df[chapter_col] = df[chapter_col].map(diag_chapter_map)
    
    # One-hot encode diagnosis chapters
    diag_dummies = pd.get_dummies(
        df[["diag_1_chapter", "diag_2_chapter", "diag_3_chapter"]],
        prefix=["diag1", "diag2", "diag3"]
    ).astype(int)
    df = pd.concat([df, diag_dummies], axis=1)
    
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
    
    # Complete list of features from the training data
    return [
        'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_outpatient',
        'number_inpatient', 'number_diagnoses', 'weight_num', 'max_glu_serum_num',
        'A1Cresult_num', 'diabetesMed',
        'Gender_Female', 'Gender_Male', 'Gender_Unknown/Invalid',
        'admission_elective', 'admission_emergency', 'admission_newborn',
        'admission_notavailable', 'admission_notmapped', 'admission_null',
        'admission_trauma', 'admission_urgent',
        'discharge_admitted_this_hospital', 'discharge_expired',
        'discharge_expired_facility_medicaid', 'discharge_expired_home_medicaid',
        'discharge_federal_healthcare_facility', 'discharge_home',
        'discharge_home_health', 'discharge_home_iv_care',
        'discharge_hospice_facility', 'discharge_hospice_home', 'discharge_icf',
        'discharge_left_ama', 'discharge_long_term_care_hospital',
        'discharge_medicaid_nursing_facility', 'discharge_neonate_another_hospital',
        'discharge_not_mapped', 'discharge_null', 'discharge_other_inpatient',
        'discharge_outpatient_other_institution', 'discharge_outpatient_this_institution',
        'discharge_psychiatric_hospital', 'discharge_rehab_facility',
        'discharge_short_term_hospital', 'discharge_snf', 'discharge_still_patient',
        'discharge_swing_bed',
        'diag1_blood_diseases', 'diag1_circulatory', 'diag1_congenital_anomalies',
        'diag1_digestive', 'diag1_endocrine_metabolic', 'diag1_genitourinary',
        'diag1_infectious_parasitic', 'diag1_injury_poisoning', 'diag1_mental_disorders',
        'diag1_musculoskeletal', 'diag1_neoplasms', 'diag1_nervous_system',
        'diag1_pregnancy_childbirth', 'diag1_respiratory', 'diag1_skin_subcutaneous',
        'diag1_symptoms_ill_defined', 'diag1_unknown',
        'diag2_blood_diseases', 'diag2_circulatory', 'diag2_congenital_anomalies',
        'diag2_digestive', 'diag2_endocrine_metabolic', 'diag2_genitourinary',
        'diag2_infectious_parasitic', 'diag2_injury_poisoning', 'diag2_mental_disorders',
        'diag2_musculoskeletal', 'diag2_neoplasms', 'diag2_nervous_system',
        'diag2_pregnancy_childbirth', 'diag2_respiratory', 'diag2_skin_subcutaneous',
        'diag2_symptoms_ill_defined', 'diag2_unknown',
        'diag3_blood_diseases', 'diag3_circulatory', 'diag3_congenital_anomalies',
        'diag3_digestive', 'diag3_endocrine_metabolic', 'diag3_genitourinary',
        'diag3_infectious_parasitic', 'diag3_injury_poisoning', 'diag3_mental_disorders',
        'diag3_musculoskeletal', 'diag3_neoplasms', 'diag3_nervous_system',
        'diag3_pregnancy_childbirth', 'diag3_respiratory', 'diag3_skin_subcutaneous',
        'diag3_symptoms_ill_defined', 'diag3_unknown',
        'Race_Caucasian', 'Race_AfricanAmerican', 'Race_Unknown', 'Race_Other',
        'Race_Asian', 'Race_Hispanic'
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
    
    # Laboratory Values
    st.subheader("🧪 Laboratory Values")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight_num = st.slider("Weight (lbs)", 25, 250)
    
    with col2:
        max_glu_serum_num = st.slider("Max Glucose Serum (mg/dL)", 70, 350)
    
    with col3:
        A1Cresult_num = st.slider("A1C Result (%)", 4.0, 10.0, 0.1)
    
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
        'weight_num': weight_num,
        'max_glu_serum_num': max_glu_serum_num,
        'A1Cresult_num': A1Cresult_num,
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
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
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
            
            with summary_col3:
                st.write("**Laboratory Values**")
                st.write(f"- Weight: {weight_num} lbs")
                st.write(f"- Max Glucose: {max_glu_serum_num} mg/dL")
                st.write(f"- A1C Result: {A1Cresult_num}%")
            
            st.write("**Clinical Diagnosis Chapters**")
            st.write(f"- Primary: {diag_1_chapter}")
            st.write(f"- Secondary: {diag_2_chapter}")
            st.write(f"- Tertiary: {diag_3_chapter}")
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
            st.warning("Check that all required input fields are correctly filled.")
