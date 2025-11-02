import os
import math
import joblib
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from gspread_dataframe import set_with_dataframe
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, brier_score_loss, precision_recall_fscore_support
)
from sklearn.inspection import permutation_importance

# --- Page Configuration ---
st.set_page_config(
    page_title="Cloud-Ready Referral System",
    page_icon="🏥",
    layout="wide"
)

MODEL_PATH = "trained_pipeline.joblib"

# --- Utilities ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# --- Google Sheets & Drive Connection ---
@st.cache_resource
def connect_to_gsheet():
    """Connects to Google Sheets using credentials from st.secrets."""
    creds = st.secrets["connections"]["gcs"]["service_account_info"]
    gc = gspread.service_account_from_dict(creds)
    spreadsheet = gc.open("clinic-referral-app-data")
    worksheet = spreadsheet.worksheet("Clinics")
    return worksheet

@st.cache_data(ttl="5m")
def load_clinics():
    """Loads clinic data from the 'Clinics' worksheet."""
    worksheet = connect_to_gsheet()
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    df = df.dropna(subset=['clinic_id'])
    df['clinic_id'] = df['clinic_id'].astype(int)
    df['capacity'] = df['capacity'].astype(int)
    df['current_load'] = df['current_load'].astype(int)
    return df

def persist_clinics(df):
    """Updates the 'Clinics' worksheet with the provided DataFrame."""
    worksheet = connect_to_gsheet()
    set_with_dataframe(worksheet, df)
    load_clinics.clear()
    st.toast("Clinic loads updated in the cloud.")

@st.cache_resource
def get_gdrive_auth():
    """Authenticate with Google Drive using Service Account credentials."""
    gauth = GoogleAuth()
    creds = st.secrets["connections"]["gcs"]["service_account_info"]
    gauth.credentials = gspread.service_account_from_dict(creds).auth
    return gauth

def upload_snapshot_to_gdrive(snapshot_data):
    """Saves a snapshot locally, uploads to GDrive, then cleans up."""
    try:
        gauth = get_gdrive_auth()
        drive = GoogleDrive(gauth)
        folder_id = st.secrets["google_drive_folder_id"]
        export_time = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_filename = f"export_snapshot_{export_time}.joblib"
        
        joblib.dump(snapshot_data, local_filename)

        file_metadata = {'title': local_filename, 'parents': [{'id': folder_id}]}
        gfile = drive.CreateFile(file_metadata)
        gfile.SetContentFile(local_filename)
        gfile.Upload()

        os.remove(local_filename)
        st.success(f"Snapshot '{local_filename}' saved to your Google Drive!")
    except Exception as e:
        st.error(f"Failed to upload to Google Drive: {e}")

# --- Model Training ---
@st.cache_resource
def train_sophisticated_model(retrain=False):
    if os.path.exists(MODEL_PATH) and not retrain:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    np.random.seed(42)
    num_patients = 4000
    patient_data = {
        'age': np.random.randint(18, 90, size=num_patients),
        'severity': np.random.choice(['low', 'medium', 'high'], size=num_patients, p=[0.55, 0.35, 0.10]),
        'condition': np.random.choice(['flu', 'fracture', 'cardiac', 'respiratory', 'other'], size=num_patients, p=[0.35, 0.25, 0.10, 0.15, 0.15]),
        'vitals_score': np.random.randint(1, 11, size=num_patients),
        'comorbidities': np.random.randint(0, 6, size=num_patients),
        'hospital_load': np.random.randint(10, 100, size=num_patients),
        'patient_lat': 40.0 + np.random.rand(num_patients) * 0.1,
        'patient_lon': -74.0 + np.random.rand(num_patients) * 0.1
    }
    patients_df = pd.DataFrame(patient_data)

    def should_refer(row):
        p = 0.3
        if row['severity'] == 'low' and row['vitals_score'] > 6 and row['condition'] in ['flu', 'fracture', 'other']: p += 0.45
        if row['severity'] == 'high' or row['vitals_score'] <= 3 or row['condition'] == 'cardiac': p -= 0.25
        p += 0.02 * row['comorbidities']
        if row['hospital_load'] > 70: p += 0.25 * ((row['hospital_load'] - 70) / 30)
        p = np.clip(p, 0.02, 0.98)
        return np.random.choice([0, 1], p=[1 - p, p])

    patients_df['referred'] = patients_df.apply(should_refer, axis=1)
    feature_cols = ['age', 'vitals_score', 'comorbidities', 'hospital_load', 'patient_lat', 'patient_lon', 'severity', 'condition']
    X = patients_df[feature_cols]
    y = patients_df['referred']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    
    numeric_features = ['age', 'vitals_score', 'comorbidities', 'hospital_load', 'patient_lat', 'patient_lon']
    categorical_features = ['severity', 'condition']
    
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    
    clf = HistGradientBoostingClassifier(random_state=42, early_stopping=True)
    pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
    
    param_grid = {'clf__max_iter': [100, 200], 'clf__learning_rate': [0.05, 0.1], 'clf__max_depth': [3, 5]}
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    grid.fit(X_train, y_train)
    
    best_pipeline = grid.best_estimator_
    calibrated = CalibratedClassifierCV(best_pipeline, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)

    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    class_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred)
    
    perm_res = permutation_importance(calibrated, X_test, y_test, n_repeats=12, random_state=42, n_jobs=-1)
    
    # FINAL FIX for ValueError: This guarantees the lengths match.
    feat_names = X_test.columns
    importances_df = pd.DataFrame({
        'feature': feat_names,
        'importance_mean': perm_res.importances_mean,
        'importance_std': perm_res.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    model_artifact = {
        'pipeline': calibrated, 'feature_cols': feature_cols, 'X_test': X_test,
        'y_test': y_test, 'accuracy': accuracy, 'auc': auc, 'brier': brier,
        'class_report_df': class_report_df, 'cm': cm, 'importances_df': importances_df
    }
    joblib.dump(model_artifact, MODEL_PATH)
    return model_artifact

# --- UI and Helper Functions ---
def initialize_session_state():
    if 'clinics_df' not in st.session_state:
        st.session_state.clinics_df = load_clinics()
    if 'model' not in st.session_state:
        st.session_state.model_artifact = train_sophisticated_model()
        st.session_state.model = st.session_state.model_artifact['pipeline']

def score_clinics_for_patient(patient_row, clinics_df, specialty_map, weights):
    scores = []
    for _, c in clinics_df.iterrows():
        availability = float(c['capacity'] - c['current_load'])
        distance = haversine_km(patient_row['patient_lat'], patient_row['patient_lon'], c['lat'], c['lon'])
        specialty_match = 1.0 if specialty_map.get(patient_row['condition'], 'general') == c['specialty'] else 0.0
        score = weights['availability'] * (availability / 20.0) - weights['distance'] * (distance / 30.0) + weights['specialty'] * specialty_match
        if availability <= 0:
            score -= 2.0
        scores.append((score, int(c['clinic_id'])))
    scores.sort(reverse=True)
    return scores

def find_best_clinic(patient_row, clinics_df, weights=None):
    if weights is None:
        weights = {'availability': 1.2, 'distance': 0.8, 'specialty': 1.0}
    specialty_map = {'flu': 'general', 'fracture': 'orthopedics', 'cardiac': 'cardiology', 'respiratory': 'pulmonology', 'other': 'general'}
    scored = score_clinics_for_patient(patient_row, clinics_df, specialty_map, weights)
    if not scored: return None
    best_score, best_id = scored[0]
    if best_score < -1.0: return None
    return best_id

def simulate_policy(pipeline, X_sim, y_sim, clinics_start_df, threshold=0.5, weights=None):
    clinics_df = clinics_start_df.copy().reset_index(drop=True)
    results = []
    for _, row in X_sim.iterrows():
        proba = pipeline.predict_proba(row.to_frame().T)[0, 1]
        refer = int(proba >= threshold)
        chosen_clinic = None
        if refer == 1:
            clinic_id = find_best_clinic(row, clinics_df, weights=weights)
            if clinic_id is not None:
                chosen_clinic = clinic_id
                cidx = clinics_df[clinics_df['clinic_id'] == clinic_id].index[0]
                clinics_df.at[cidx, 'current_load'] += 1
        results.append({'true': int(y_sim[row.name]), 'pred': refer, 'proba': float(proba), 'clinic_id': chosen_clinic})
    res_df = pd.DataFrame(results)
    p, r, f1, _ = precision_recall_fscore_support(res_df['true'], res_df['pred'], average='binary', zero_division=0)
    metrics = {
        'accuracy': accuracy_score(res_df['true'], res_df['pred']), 'precision': p, 'recall': r, 'f1': f1,
        'referrals_made': res_df['pred'].sum(), 'successful_referrals': res_df['clinic_id'].notnull().sum()
    }
    return metrics, res_df, clinics_df

# --- Main App ---
initialize_session_state()

st.title("🏥 Advanced Patient Referral System — Sophisticated Policy Mode")
col1, col2 = st.columns((2, 1))

# Sidebar controls
with st.sidebar:
    st.header("Model & Policy Controls")
    if st.button("Retrain model"):
        train_sophisticated_model(retrain=True)
        st.rerun()
    threshold = st.slider("Referral probability threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    st.markdown("##### Routing Weights")
    avail_w = st.slider("Availability weight", 0.0, 3.0, 1.2, 0.1)
    dist_w = st.slider("Distance weight", 0.0, 3.0, 0.8, 0.1)
    spec_w = st.slider("Specialty match weight", 0.0, 3.0, 1.0, 0.1)
    routing_weights = {'availability': avail_w, 'distance': dist_w, 'specialty': spec_w}
    st.markdown("---")
    st.header("Model Diagnostics")
    ma = st.session_state.model_artifact
    st.metric("Model AUC", f"{ma['auc']:.3f}")
    st.metric("Model Brier Score", f"{ma['brier']:.3f}")
    st.subheader("Classification Report")
    st.dataframe(ma['class_report_df'].style.format("{:.2f}"))
    st.subheader("Permutation Importance")
    st.dataframe(ma['importances_df'].head(8))

# Main page content
with col1:
    st.subheader("Patient Intake (Live)")
    with st.form("patient_live_form"):
        p_age = st.slider("Age", 1, 100, 50)
        p_vitals = st.slider("Vitals (1=critical, 10=stable)", 1, 10, 7)
        p_comorbid = st.number_input("Comorbidities", 0, 10, 1)
        p_severity = st.selectbox("Severity", options=['low', 'medium', 'high'])
        p_condition = st.selectbox("Condition", options=['flu', 'fracture', 'cardiac', 'respiratory', 'other'])
        p_hosp_load = st.slider("Hospital load (%)", 0, 100, 60)
        p_lat = st.number_input("Patient latitude", 40.02, 40.12, 40.05, format="%.5f")
        p_lon = st.number_input("Patient longitude", -74.12, -73.98, -74.02, format="%.5f") # Corrected min/max
        
        if st.form_submit_button("Assess and Route"):
            input_df = pd.DataFrame([{'age': p_age, 'vitals_score': p_vitals, 'comorbidities': p_comorbid,
                                      'hospital_load': p_hosp_load, 'patient_lat': p_lat, 'patient_lon': p_lon,
                                      'severity': p_severity, 'condition': p_condition}])
            proba = st.session_state.model.predict_proba(input_df)[0, 1]
            recommend = proba >= threshold
            
            st.markdown(f"#### Assessment: Predicted Referral Probability is **{proba:.1%}**")
            
            if recommend:
                st.warning("Action: **Referral Recommended**")
                clinic_id = find_best_clinic(input_df.iloc[0], st.session_state.clinics_df, weights=routing_weights)
                if clinic_id is not None:
                    cinfo = st.session_state.clinics_df[st.session_state.clinics_df['clinic_id'] == clinic_id].iloc[0]
                    st.success(f"Best Option: Refer to **{cinfo['clinic_name']}** (Specialty: {cinfo['specialty']})")
                    st.write(f"Distance: {haversine_km(p_lat, p_lon, cinfo['lat'], cinfo['lon']):.1f} km | Available Beds: {int(cinfo['capacity'] - cinfo['current_load'])}")
                    
                    idx = st.session_state.clinics_df[st.session_state.clinics_df['clinic_id'] == clinic_id].index[0]
                    st.session_state.clinics_df.at[idx, 'current_load'] += 1
                    persist_clinics(st.session_state.clinics_df)
                else:
                    st.error("No suitable clinic available. Consider on-site treatment.")
            else:
                st.info("Action: **Referral Not Recommended**")
                if p_hosp_load >= 92:
                    st.warning("Policy Override: Hospital is critically overloaded—consider manual referral.")
                st.write("Treat patient at hospital and monitor.")

    st.markdown("---")
    st.subheader("Policy Simulation / Stress Test")
    n_sim = st.number_input("Number of test patients to simulate", 50, 1000, 250, 50)
    if st.button("Run Simulation with Current Settings"):
        ma = st.session_state.model_artifact
        X_test, y_test = ma['X_test'].reset_index(drop=True), ma['y_test'].reset_index(drop=True)
        X_sub, y_sub = X_test.head(n_sim), y_test.head(n_sim)
        
        metrics, res_df, clinics_after = simulate_policy(
            st.session_state.model, X_sub, y_sub, 
            st.session_state.clinics_df, threshold=threshold, weights=routing_weights
        )
        
        st.subheader("Simulation Results")
        m_col1, m_col2, m_col3 = st.columns(3)
        referrals_made = int(metrics['referrals_made'])
        successful_referrals = int(metrics['successful_referrals'])
        
        m_col1.metric("Referrals Recommended", referrals_made)
        m_col2.metric("Successful Referrals", successful_referrals)
        if referrals_made > 0:
            success_rate = (successful_referrals / referrals_made) * 100
            m_col3.metric("Routing Success Rate", f"{success_rate:.1f}%")
        else:
            m_col3.metric("Routing Success Rate", "N/A")

        st.markdown(f"This policy captured **{metrics['recall']:.0%}** of all patients who could have been referred.")
        
        with st.expander("Show detailed simulation results"):
            st.markdown("##### Detailed Classification Metrics")
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            d_col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
            d_col2.metric("Precision", f"{metrics['precision']:.2f}")
            d_col3.metric("Recall", f"{metrics['recall']:.2f}")
            d_col4.metric("F1-Score", f"{metrics['f1']:.2f}")
            
            st.markdown("##### Final Clinic Status (Snapshot)")
            st.dataframe(clinics_after)

with col2:
    st.subheader("Live Clinic Status")
    st.dataframe(st.session_state.clinics_df, use_container_width=True)
    st.caption("Data is live from Google Sheets.")

# --- Footer ---
st.markdown("---")
if st.button("Export & Upload Snapshot to Google Drive"):
    with st.spinner("Exporting snapshot..."):
        snapshot_data = {
            'model_artifact': st.session_state.model_artifact,
            'clinics_snapshot': st.session_state.clinics_df
        }
        upload_snapshot_to_gdrive(snapshot_data)