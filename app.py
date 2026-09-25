
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Job Analytics Suite",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM STYLING & ANIMATIONS (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Animated Title Keyframe */
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
        margin-bottom: 0px;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(168, 85, 247, 0.4);
    }

    /* Metric Display Box */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Styled Prediction Button */
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
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load pickled scikit-learn models."""
    with open('Model_1 job analytics.pkl', 'rb') as f1:
        model_1 = pickle.load(f1)
    with open('Molde_2 job analytics.pkl', 'rb') as f2:
        model_2 = pickle.load(f2)
    return model_1, model_2

def load_lottieurl(url: str):
    """Fetch Lottie animation JSON."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

try:
    reg_model, clf_model = load_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Ensure 'Model_1 job analytics.pkl' and 'Molde_2 job analytics.pkl' are placed in the app directory.")
    st.stop()

# -----------------------------------------------------------------------------
# CATEGORICAL MAPPINGS (MAPPED TO NUMERIC FOR SCIKIT-LEARN MODELS)
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
    st.markdown('<p class="sub-title">Predict Salaries & Classify Hiring Urgency powered by Machine Learning</p>', unsafe_allow_html=True)

with col_anim:
    lottie_analytics = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp156ysq.json")
    if lottie_analytics:
        st_lottie(lottie_analytics, height=120, key="analytics_hero")

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Navigation & Controls")
app_mode = st.sidebar.radio("Select Analytics Tool:", ["💰 Salary Predictor (Model 1)", "🔥 Hiring Urgency Classifier (Model 2)"])

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.write("**Model 1:** Linear Regression")
st.sidebar.write("**Model 2:** Logistic Regression")
st.sidebar.write("**Scikit-Learn Version:** 1.6.1")

# -----------------------------------------------------------------------------
# TAB 1: SALARY PREDICTOR (MODEL 1 - LINEAR REGRESSION)
# -----------------------------------------------------------------------------
if app_mode == "💰 Salary Predictor (Model 1)":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Estimate Target Job Salary")
    st.write("Fill in the job requirements to predict expected compensation.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_title = st.selectbox("Job Title", list(JOB_TITLES.keys()), key="m1_title")
        company_size = st.selectbox("Company Size", list(COMPANY_SIZES.keys()), key="m1_size")
        company_industry = st.selectbox("Industry", list(INDUSTRIES.keys()), key="m1_ind")
        country = st.selectbox("Country", list(COUNTRIES.keys()), key="m1_country")
        
    with col2:
        remote_type = st.selectbox("Remote Type", list(REMOTE_TYPES.keys()), key="m1_remote")
        experience_level = st.selectbox("Experience Level", list(EXP_LEVELS.keys()), key="m1_exp")
        years_experience = st.slider("Years of Experience", 0, 25, 3, key="m1_yrs")
        education_level = st.selectbox("Education Level", list(EDU_LEVELS.keys()), key="m1_edu")
        
    with col3:
        hiring_urgency = st.selectbox("Hiring Urgency", ["Low", "Medium", "High"], key="m1_urgency")
        urgency_val = {"Low": 0, "Medium": 1, "High": 2}[hiring_urgency]
        job_openings = st.number_input("Open Positions", min_value=1, max_value=500, value=5, key="m1_openings")
        job_posting_month = st.slider("Posting Month", 1, 12, 6, key="m1_month")
        job_posting_year = st.selectbox("Posting Year", [2024, 2025, 2026], key="m1_year")

    st.write("---")
    st.markdown("**Required Technical Skills:**")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    skills_python = sc1.checkbox("Python", value=True, key="m1_py")
    skills_sql = sc2.checkbox("SQL", value=True, key="m1_sql")
    skills_ml = sc3.checkbox("Machine Learning", value=False, key="m1_ml")
    skills_deep_learning = sc4.checkbox("Deep Learning", value=False, key="m1_dl")
    skills_cloud = sc5.checkbox("Cloud (AWS/GCP/Azure)", value=False, key="m1_cloud")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Calculate Estimated Salary"):
        # Features array corresponding to Model 1's feature ordering
        feature_vector = np.array([[
            JOB_TITLES[job_title],
            COMPANY_SIZES[company_size],
            INDUSTRIES[company_industry],
            COUNTRIES[country],
            REMOTE_TYPES[remote_type],
            EXP_LEVELS[experience_level],
            years_experience,
            EDU_LEVELS[education_level],
            int(skills_python),
            int(skills_sql),
            int(skills_ml),
            int(skills_deep_learning),
            int(skills_cloud),
            job_posting_month,
            job_posting_year,
            urgency_val,
            job_openings
        ]])

        predicted_salary = reg_model.predict(feature_vector)[0]
        
        st.balloons()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Annual Compensation</div>
            <div class="metric-value">${max(predicted_salary, 0.0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: HIRING URGENCY CLASSIFIER (MODEL 2 - LOGISTIC REGRESSION)
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔥 Classify Hiring Demand & Urgency")
    st.write("Determine the urgency/demand profile of a job opening.")

    col1, col2, col3 = st.columns(3)

    with col1:
        job_title = st.selectbox("Job Title", list(JOB_TITLES.keys()), key="m2_title")
        company_size = st.selectbox("Company Size", list(COMPANY_SIZES.keys()), key="m2_size")
        company_industry = st.selectbox("Industry", list(INDUSTRIES.keys()), key="m2_ind")
        country = st.selectbox("Country", list(COUNTRIES.keys()), key="m2_country")

    with col2:
        remote_type = st.selectbox("Remote Type", list(REMOTE_TYPES.keys()), key="m2_remote")
        experience_level = st.selectbox("Experience Level", list(EXP_LEVELS.keys()), key="m2_exp")
        years_experience = st.slider("Years of Experience", 0, 25, 3, key="m2_yrs")
        education_level = st.selectbox("Education Level", list(EDU_LEVELS.keys()), key="m2_edu")

    with col3:
        salary = st.number_input("Offered Salary ($)", min_value=20000, max_value=500000, value=95000, step=5000, key="m2_sal")
        job_openings = st.number_input("Open Positions", min_value=1, max_value=500, value=10, key="m2_openings")
        job_posting_month = st.slider("Posting Month", 1, 12, 6, key="m2_month")
        job_posting_year = st.selectbox("Posting Year", [2024, 2025, 2026], key="m2_year")

    st.write("---")
    st.markdown("**Required Technical Skills:**")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    skills_python = sc1.checkbox("Python", value=True, key="m2_py")
    skills_sql = sc2.checkbox("SQL", value=True, key="m2_sql")
    skills_ml = sc3.checkbox("Machine Learning", value=True, key="m2_ml")
    skills_deep_learning = sc4.checkbox("Deep Learning", value=False, key="m2_dl")
    skills_cloud = sc5.checkbox("Cloud (AWS/GCP/Azure)", value=True, key="m2_cloud")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Predict Hiring Demand Class"):
        # Features array corresponding to Model 2's feature ordering
        feature_vector = np.array([[
            JOB_TITLES[job_title],
            COMPANY_SIZES[company_size],
            INDUSTRIES[company_industry],
            COUNTRIES[country],
            REMOTE_TYPES[remote_type],
            EXP_LEVELS[experience_level],
            years_experience,
            EDU_LEVELS[education_level],
            int(skills_python),
            int(skills_sql),
            int(skills_ml),
            int(skills_deep_learning),
            int(skills_cloud),
            salary,
            job_posting_month,
            job_posting_year,
            job_openings
        ]])

        prediction = clf_model.predict(feature_vector)[0]
        prediction_proba = clf_model.predict_proba(feature_vector)[0] if hasattr(clf_model, "predict_proba") else None

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Classification Output</div>
            <div class="metric-value">Class {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        if prediction_proba is not None:
            st.write("### Prediction Confidence")
            for cls_idx, prob in enumerate(prediction_proba):
                st.progress(float(prob), text=f"Class {cls_idx} Probability: {prob * 100:.1f}%")
