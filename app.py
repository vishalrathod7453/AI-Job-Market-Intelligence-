import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_lottie import st_lottie

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Job Market Analytics Suite",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DYNAMIC PATH RESOLUTION & MODEL LOADING
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

def find_model_file(candidates):
    """Locates model files reliably, ignoring spacing or case mismatches (e.g., '.pk l' vs '.pkl')."""
    for candidate in candidates:
        target_path = BASE_DIR / candidate
        if target_path.exists():
            return target_path
    
    # Fallback search across directory files
    for file_path in BASE_DIR.iterdir():
        clean_filename = file_path.name.replace(" ", "").lower()
        for candidate in candidates:
            clean_candidate = candidate.replace(" ", "").lower()
            if clean_candidate in clean_filename or clean_filename in clean_candidate:
                return file_path
    return None

@st.cache_resource
def load_models():
    model1_path = find_model_file([
        "Model_1 job Analytics.pk l", 
        "Model_1 job analytics.pkl", 
        "Model_1 job Analytics.pkl"
    ])
    model2_path = find_model_file([
        "Molde_2 job analytics.pkl", 
        "Model_2 job analytics.pkl"
    ])
    
    if not model1_path or not model2_path:
        missing = []
        if not model1_path: missing.append("Model_1 job Analytics.pk l")
        if not model2_path: missing.append("Molde_2 job analytics.pkl")
        raise FileNotFoundError(f"Could not locate required model file(s): {', '.join(missing)}")

    with open(model1_path, 'rb') as f1:
        model_1 = pickle.load(f1)
    with open(model2_path, 'rb') as f2:
        model_2 = pickle.load(f2)
        
    return model_1, model_2

def load_lottieurl(url: str):
    """Fetch Lottie animations with error handling."""
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# CUSTOM GLASSMORPHISM & ANIMATED STYLING (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Gradient Theme */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Animated Glowing Header */
    @keyframes pulse-glow {
        0% { text-shadow: 0 0 10px rgba(99, 102, 241, 0.5); }
        50% { text-shadow: 0 0 25px rgba(168, 85, 247, 0.8), 0 0 35px rgba(99, 102, 241, 0.6); }
        100% { text-shadow: 0 0 10px rgba(99, 102, 241, 0.5); }
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse-glow 3s infinite ease-in-out;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Animated Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(168, 85, 247, 0.4);
    }

    /* Output Metric Card */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%);
        border: 1px solid rgba(168, 85, 247, 0.5);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }

    .metric-value {
        font-size: 2.3rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Styled Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# Load Models
try:
    reg_model, clf_model = load_models()
except Exception as e:
    st.error(f"❌ Error Loading Models: {e}")
    st.info("Place both model `.pkl` files in the same folder as `app.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# CATEGORICAL ENCODING DICTIONARIES
# -----------------------------------------------------------------------------
JOB_TITLES = {"Data Scientist": 0, "Machine Learning Engineer": 1, "Data Engineer": 2, "Data Analyst": 3, "Software Engineer": 4}
COMPANY_SIZES = {"Small (1-50)": 0, "Medium (51-500)": 1, "Large (500+)": 2}
INDUSTRIES = {"Tech": 0, "Finance": 1, "Healthcare": 2, "E-commerce": 3, "Education": 4}
COUNTRIES = {"USA": 0, "UK": 1, "Canada": 2, "Germany": 3, "India": 4, "Remote/Other": 5}
REMOTE_TYPES = {"On-site": 0, "Hybrid": 1, "Full-Remote": 2}
EXP_LEVELS = {"Entry-Level": 0, "Mid-Level": 1, "Senior-Level": 2, "Executive": 3}
EDU_LEVELS = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}

# -----------------------------------------------------------------------------
# HEADER & HERO SECTION
# -----------------------------------------------------------------------------
col_header, col_anim = st.columns([3, 1])

with col_header:
    st.markdown('<h1 class="main-title">Job Market Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predict Compensation & Classify Hiring Urgency using Trained ML Models</p>', unsafe_allow_html=True)

with col_anim:
    lottie_analytics = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp156ysq.json")
    if lottie_analytics:
        st_lottie(lottie_analytics, height=110, key="hero_anim")

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Analytics Mode")
app_mode = st.sidebar.radio(
    "Choose Prediction Tool:",
    ["💰 Salary Predictor (Model 1)", "🔥 Hiring Urgency Classifier (Model 2)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Model Details")
st.sidebar.markdown("**Model 1:** Linear Regression (Salary Output)")
st.sidebar.markdown("**Model 2:** Logistic Regression (Urgency Class)")
st.sidebar.markdown("**Scikit-Learn:** 1.6.1")

# -----------------------------------------------------------------------------
# MODEL 1: SALARY PREDICTOR (LINEAR REGRESSION)
# -----------------------------------------------------------------------------
if app_mode == "💰 Salary Predictor (Model 1)":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Predict Job Salary Target")
    st.write("Fill in position details to generate estimated annual compensation.")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        job_title = st.selectbox("Job Title", list(JOB_TITLES.keys()), key="m1_title")
        company_size = st.selectbox("Company Size", list(COMPANY_SIZES.keys()), key="m1_size")
        company_industry = st.selectbox("Industry", list(INDUSTRIES.keys()), key="m1_ind")
        country = st.selectbox("Country", list(COUNTRIES.keys()), key="m1_country")
        
    with c2:
        remote_type = st.selectbox("Remote Type", list(REMOTE_TYPES.keys()), key="m1_remote")
        experience_level = st.selectbox("Experience Level", list(EXP_LEVELS.keys()), key="m1_exp")
        years_experience = st.slider("Years of Experience", 0, 25, 4, key="m1_yrs")
        education_level = st.selectbox("Education Level", list(EDU_LEVELS.keys()), key="m1_edu")
        
    with c3:
        hiring_urgency = st.selectbox("Hiring Urgency", ["Low", "Medium", "High"], key="m1_urgency")
        urgency_val = {"Low": 0, "Medium": 1, "High": 2}[hiring_urgency]
        job_openings = st.number_input("Open Positions", min_value=1, max_value=500, value=5, key="m1_openings")
        job_posting_month = st.slider("Posting Month", 1, 12, 6, key="m1_month")
        job_posting_year = st.selectbox("Posting Year", [2024, 2025, 2026], key="m1_year")

    st.write("---")
    st.markdown("**Technical Skills Required:**")
    sk1, sk2, sk3, sk4, sk5 = st.columns(5)
    skills_python = sk1.checkbox("Python", value=True, key="m1_py")
    skills_sql = sk2.checkbox("SQL", value=True, key="m1_sql")
    skills_ml = sk3.checkbox("Machine Learning", value=False, key="m1_ml")
    skills_deep_learning = sk4.checkbox("Deep Learning", value=False, key="m1_dl")
    skills_cloud = sk5.checkbox("Cloud Infrastructure", value=False, key="m1_cloud")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Calculate Salary Estimate"):
        # Create DataFrame with exact column order from Model 1 schema
        m1_inputs = pd.DataFrame([{
            'job_title': JOB_TITLES[job_title],
            'company_size': COMPANY_SIZES[company_size],
            'company_industry': INDUSTRIES[company_industry],
            'country': COUNTRIES[country],
            'remote_type': REMOTE_TYPES[remote_type],
            'experience_level': EXP_LEVELS[experience_level],
            'years_experience': years_experience,
            'education_level': EDU_LEVELS[education_level],
            'skills_python': int(skills_python),
            'skills_sql': int(skills_sql),
            'skills_ml': int(skills_ml),
            'skills_deep_learning': int(skills_deep_learning),
            'skills_cloud': int(skills_cloud),
            'job_posting_month': job_posting_month,
            'job_posting_year': job_posting_year,
            'hiring_urgency': urgency_val,
            'job_openings': job_openings
        }])

        predicted_salary = reg_model.predict(m1_inputs)[0]
        
        st.balloons()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Annual Salary</div>
            <div class="metric-value">${max(predicted_salary, 0.0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL 2: HIRING URGENCY CLASSIFIER (LOGISTIC REGRESSION)
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔥 Classify Hiring Urgency & Demand")
    st.write("Evaluate role features and target salary to predict hiring urgency class.")

    c1, c2, c3 = st.columns(3)

    with c1:
        job_title = st.selectbox("Job Title", list(JOB_TITLES.keys()), key="m2_title")
        company_size = st.selectbox("Company Size", list(COMPANY_SIZES.keys()), key="m2_size")
        company_industry = st.selectbox("Industry", list(INDUSTRIES.keys()), key="m2_ind")
        country = st.selectbox("Country", list(COUNTRIES.keys()), key="m2_country")

    with c2:
        remote_type = st.selectbox("Remote Type", list(REMOTE_TYPES.keys()), key="m2_remote")
        experience_level = st.selectbox("Experience Level", list(EXP_LEVELS.keys()), key="m2_exp")
        years_experience = st.slider("Years of Experience", 0, 25, 4, key="m2_yrs")
        education_level = st.selectbox("Education Level", list(EDU_LEVELS.keys()), key="m2_edu")

    with c3:
        salary = st.number_input("Offered Salary ($)", min_value=20000, max_value=500000, value=95000, step=5000, key="m2_sal")
        job_openings = st.number_input("Open Positions", min_value=1, max_value=500, value=10, key="m2_openings")
        job_posting_month = st.slider("Posting Month", 1, 12, 6, key="m2_month")
        job_posting_year = st.selectbox("Posting Year", [2024, 2025, 2026], key="m2_year")

    st.write("---")
    st.markdown("**Technical Skills Required:**")
    sk1, sk2, sk3, sk4, sk5 = st.columns(5)
    skills_python = sk1.checkbox("Python", value=True, key="m2_py")
    skills_sql = sk2.checkbox("SQL", value=True, key="m2_sql")
    skills_ml = sk3.checkbox("Machine Learning", value=True, key="m2_ml")
    skills_deep_learning = sk4.checkbox("Deep Learning", value=False, key="m2_dl")
    skills_cloud = sk5.checkbox("Cloud Infrastructure", value=True, key="m2_cloud")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Predict Urgency Category"):
        # Create DataFrame with exact column order from Model 2 schema
        m2_inputs = pd.DataFrame([{
            'job_title': JOB_TITLES[job_title],
            'company_size': COMPANY_SIZES[company_size],
            'company_industry': INDUSTRIES[company_industry],
            'country': COUNTRIES[country],
            'remote_type': REMOTE_TYPES[remote_type],
            'experience_level': EXP_LEVELS[experience_level],
            'years_experience': years_experience,
            'education_level': EDU_LEVELS[education_level],
            'skills_python': int(skills_python),
            'skills_sql': int(skills_sql),
            'skills_ml': int(skills_ml),
            'skills_deep_learning': int(skills_deep_learning),
            'skills_cloud': int(skills_cloud),
            'salary': salary,
            'job_posting_month': job_posting_month,
            'job_posting_year': job_posting_year,
            'job_openings': job_openings
        }])

        prediction = clf_model.predict(m2_inputs)[0]
        prediction_proba = clf_model.predict_proba(m2_inputs)[0] if hasattr(clf_model, "predict_proba") else None

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Urgency Category</div>
            <div class="metric-value">Class {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        if prediction_proba is not None:
            st.write("### Prediction Confidence")
            for idx, prob in enumerate(prediction_proba):
                st.progress(float(prob), text=f"Class {idx} Probability: {prob * 100:.1f}%")
