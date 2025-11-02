import os
import math
import joblib
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread # CHANGED: Import the core gspread library
from gspread_dataframe import set_with_dataframe # CHANGED: Import a helper for writing data

# ... (all sklearn imports remain the same) ...
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
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

## ----------------------------------------------------------------------------------
## CHANGED: New, more robust functions to connect directly using gspread
## This completely bypasses the broken st.connection feature.
## ----------------------------------------------------------------------------------

# Use caching to prevent re-authenticating on every page interaction
@st.cache_resource
def connect_to_gsheet():
    """Connects to Google Sheets using credentials from st.secrets and returns the worksheet."""
    creds = st.secrets["connections"]["gcs"]["service_account_info"]
    gc = gspread.service_account_from_dict(creds)
    # IMPORTANT: Your spreadsheet file must be named "clinic-referral-app-data"
    spreadsheet = gc.open("clinic-referral-app-data")
    # IMPORTANT: The tab at the bottom must be named "Clinics"
    worksheet = spreadsheet.worksheet("Clinics")
    return worksheet

# Use caching to prevent re-downloading the data on every interaction
@st.cache_data(ttl="5m")
def load_clinics():
    """Loads clinic data from the 'Clinics' worksheet."""
    worksheet = connect_to_gsheet()
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Ensure data types are correct
    df = df.dropna(subset=['clinic_id'])
    df['clinic_id'] = df['clinic_id'].astype(int)
    df['capacity'] = df['capacity'].astype(int)
    df['current_load'] = df['current_load'].astype(int)
    return df

def persist_clinics(df):
    """Updates the 'Clinics' worksheet with the provided DataFrame."""
    worksheet = connect_to_gsheet()
    set_with_dataframe(worksheet, df)
    # Clear the cache so the next user sees the updated data
    load_clinics.clear()
    st.toast("Clinic loads updated in the cloud.")

# --- Training and Model Utilities (This whole section is unchanged) ---
@st.cache_resource
def train_sophisticated_model(retrain=False):
    if os.path.exists(MODEL_PATH) and not retrain:
        try:
            data = joblib.load(MODEL_PATH)
            return data
        except Exception:
            pass

    np.random.seed(42)
    num_patients = 4000
    patient_data = {
        'age': np.random.randint(18, 90, size=num_patients),
        'severity': np.random.choice(['low', 'medium', 'high'], size=num_patients, p=[0.55, 0.35, 0.10]),
        'condition': np.random.choice(['flu', 'fracture', 'cardiac', 'respiratory', 'other'],
                                      size=num_patients, p=[0.35, 0.25, 0.10, 0.15, 0.15]),
        'vitals_score': np.random.randint(1, 11, size=num_patients),
        'comorbidities': np.random.randint(0, 6, size=num_patients),
        'hospital_load': np.random.randint(10, 100, size=num_patients),
        'patient_lat': 40.0 + np.random.rand(num_patients) * 0.1,
        'patient_lon': -74.0 + np.random.rand(num_patients) * 0.1
    }
    patients_df = pd.DataFrame(patient_data)
    def should_refer(row):
        p = 0.3
        if row['severity'] == 'low' and row['vitals_score'] > 6 and row['condition'] in ['flu', 'fracture', 'other']:
            p += 0.45
        if row['severity'] == 'high' or row['vitals_score'] <= 3 or row['condition'] == 'cardiac':
            p -= 0.25
        p += 0.02 * row['comorbidities']
        if row['hospital_load'] > 70:
            p += 0.25 * ((row['hospital_load'] - 70) / 30)
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
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')
    clf = HistGradientBoostingClassifier(random_state=42, early_stopping=True)
    pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
    param_grid = {
        'clf__max_iter': [100, 200],
        'clf__learning_rate': [0.05, 0.1],
        'clf__max_depth': [3, 5]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=0)
    grid.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(grid.best_estimator_, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    class_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred)
    perm_res = permutation_importance(calibrated, X_test, y_test, n_repeats=12, random_state=42, n_jobs=-1)
    cat_names = calibrated.calibrated_classifiers_[0].named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feat_names = numeric_features + list(cat_names)
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

# --- The rest of the code is unchanged ---
def initialize_session_state():
    if 'clinics_df' not in st.session_state:
        st.session_state.clinics_df = load_clinics()
    if 'model' not in st.session_state:
        st.session_state.model_artifact = train_sophisticated_model()
        st.session_state.model = st.session_state.model_artifact['pipeline']
initialize_session_state()
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
    if not scored:
        return None
    best_score, best_id = scored[0]
    if best_score < -1.0:
        return None
    return best_id
def simulate_policy(pipeline, X_sim, y_sim, clinics_start_df, threshold=0.5, weights=None):
    clinics_df = clinics_start_df.copy().reset_index(drop=True)
    results = []
    for i, (idx, row) in enumerate(X_sim.iterrows()):
        x_in = pd.DataFrame([row.values], columns=X_sim.columns)
        proba = pipeline.predict_proba(x_in)[0, 1]
        refer = int(proba >= threshold)
        chosen_clinic = None
        if refer == 1:
            clinic_id = find_best_clinic(row, clinics_df, weights=weights)
            if clinic_id is not None:
                chosen_clinic = clinic_id
                cidx = clinics_df[clinics_df['clinic_id'] == clinic_id].index[0]
                clinics_df.at[cidx, 'current_load'] += 1
        results.append({'index': idx, 'true': int(y_sim.loc[idx]), 'pred': refer, 'proba': float(proba), 'clinic_id': chosen_clinic})
    res_df = pd.DataFrame(results)
    metrics = {'accuracy': accuracy_score(res_df['true'], res_df['pred']),
               'precision': precision_recall_fscore_support(res_df['true'], res_df['pred'], average='binary', zero_division=0)[0],
               'recall': precision_recall_fscore_support(res_df['true'], res_df['pred'], average='binary', zero_division=0)[1],
               'f1': precision_recall_fscore_support(res_df['true'], res_df['pred'], average='binary', zero_division=0)[2],
               'referrals_made': res_df['pred'].sum(),
               'successful_referrals': res_df['clinic_id'].notnull().sum()}
    return metrics, res_df, clinics_df
st.title("🏥 Advanced Patient Referral System — Sophisticated Policy Mode")
col1, col2 = st.columns((2, 1))
with col2:
    st.sidebar.header("Model & Policy Controls")
    if st.sidebar.button("Retrain model"):
        train_sophisticated_model(retrain=True)
        st.success("Model retrained.")
    threshold = st.sidebar.slider("Referral probability threshold", 0.0, 1.0, 0.5, 0.01)
    st.sidebar.markdown("Adjust the decision threshold used to decide referrals.")
    st.sidebar.markdown("Routing weights (higher availability favours capacity; distance penalizes farther clinics).")
    avail_w = st.sidebar.slider("Availability weight", 0.0, 3.0, 1.2, 0.1)
    dist_w = st.sidebar.slider("Distance weight", 0.0, 3.0, 0.8, 0.1)
    spec_w = st.sidebar.slider("Specialty match weight", 0.0, 3.0, 1.0, 0.1)
    routing_weights = {'availability': avail_w, 'distance': dist_w, 'specialty': spec_w}
with col1:
    st.subheader("Patient Intake (live)")
    with st.form("patient_live_form"):
        p_age = st.slider("Age", 1, 100, 50)
        p_vitals = st.slider("Vitals (1=critical,10=stable)", 1, 10, 7)
        p_comorbid = st.number_input("Comorbidities", 0, 10, 1)
        p_severity = st.selectbox("Severity", options=['low', 'medium', 'high'])
        p_condition = st.selectbox("Condition", options=['flu', 'fracture', 'cardiac', 'respiratory', 'other'])
        p_hosp_load = st.slider("Hospital load (%)", 0, 100, 60)
        p_lat = st.number_input("Patient latitude", 40.02, 40.12, 40.05, format="%.5f")
        p_lon = st.number_input("Patient longitude", -73.98, -74.12, -74.02, format="%.5f")
        if st.form_submit_button("Assess and Route"):
            input_df = pd.DataFrame([{'age': p_age, 'vitals_score': p_vitals, 'comorbidities': p_comorbid,
                                      'hospital_load': p_hosp_load, 'patient_lat': p_lat, 'patient_lon': p_lon,
                                      'severity': p_severity, 'condition': p_condition}])
            proba = st.session_state.model.predict_proba(input_df)[0,1]
            recommend = proba >= threshold
            st.markdown(f"Predicted referral probability: **{proba:.2%}**")
            if recommend:
                st.warning("Model recommends referral.")
                clinic_id = find_best_clinic(input_df.iloc[0], st.session_state.clinics_df, weights=routing_weights)
                if clinic_id is not None:
                    cinfo = st.session_state.clinics_df[st.session_state.clinics_df['clinic_id'] == clinic_id].iloc[0]
                    st.success(f"Refer to {cinfo['clinic_name']} (specialty: {cinfo['specialty']}, distance ~ {haversine_km(p_lat,p_lon,cinfo['lat'],cinfo['lon']):.1f} km, availability {int(cinfo['capacity']-cinfo['current_load'])})")
                    idx = st.session_state.clinics_df[st.session_state.clinics_df['clinic_id'] == clinic_id].index[0]
                    st.session_state.clinics_df.at[idx, 'current_load'] += 1
                    persist_clinics(st.session_state.clinics_df)
                else:
                    st.error("No suitable clinic available. Consider on-site treatment or temporary overflow solutions.")
            else:
                st.info("Model does not recommend referral.")
                if p_hosp_load >= 92:
                    st.warning("Hospital is critically overloaded — consider referral despite model output.")
                st.write("Action: treat at hospital / monitor.")
with st.sidebar:
    st.header("Model diagnostics")
    ma = st.session_state.model_artifact
    st.metric("AUC", f"{ma['auc']:.3f}")
    st.metric("Brier score", f"{ma['brier']:.3f}")
    st.subheader("Classification report (threshold 0.5)")
    st.dataframe(ma['class_report_df'].style.format("{:.2f}"))
    st.subheader("Permutation importance (top features)")
    top_imp = ma['importances_df'].head(8).set_index('feature')
    st.bar_chart(top_imp['importance_mean'])
st.markdown("---")
st.subheader("Live Clinics")
st.dataframe(st.session_state.clinics_df, use_container_width=True)
st.caption("Clinic load persists to a cloud Google Sheet.")
st.markdown("---")
st.subheader("Policy Simulation / Stress Test")
sim_col1, sim_col2 = st.columns((1, 2))
with sim_col1:
    n_sim = st.number_input("Number of test rows to simulate (from built test set)", 50, 2000, 500, 50)
    if st.button("Run simulation with current threshold & routing weights"):
        ma = st.session_state.model_artifact
        X_test = ma['X_test'].reset_index(drop=True)
        y_test = ma['y_test'].reset_index(drop=True)
        X_sub, y_sub = X_test.head(n_sim), y_test.head(n_sim)
        metrics, res_df, clinics_after = simulate_policy(st.session_state.model, X_sub, y_sub, st.session_state.clinics_df, threshold, routing_weights)
        st.write("Simulation metrics:", metrics)
        st.dataframe(res_df.head(200))
        st.dataframe(clinics_after)
        st.info("Simulation used a snapshot of current clinic loads; persistent loads were not changed.")
with sim_col2:
    st.subheader("Operational recommendations")
    st.markdown("- Use the threshold slider to tune sensitivity (lower threshold => more referrals).\n"
                "- Increase availability weight to prefer clinics with spare capacity.\n"
                "- Increase specialty weight to enforce specialty matching.\n"
                "- Use the simulation to measure trade-offs (referrals vs successful routing).")
st.markdown("---")
if st.button("Export model & clinics snapshot"):
    export_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    joblib.dump({'model_artifact': st.session_state.model_artifact, 'clinics_snapshot': st.session_state.clinics_df},
                f"export_snapshot_{export_time}.joblib")
    st.success("Export saved.")