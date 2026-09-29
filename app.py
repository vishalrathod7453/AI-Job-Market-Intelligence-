import os
import glob
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Job Analytics Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM GLASSMORPHISM & ANIMATED CSS
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #0f0c20, #15102a, #1a1b35, #0b132b);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e2e8f0;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 198, 255, 0.25);
        border-color: rgba(0, 198, 255, 0.4);
    }

    .title-text {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #7f00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulseText 3s infinite alternate;
        text-align: center;
        margin-bottom: 5px;
    }

    @keyframes pulseText {
        0% { filter: drop-shadow(0 0 2px rgba(0, 198, 255, 0.2)); }
        100% { filter: drop-shadow(0 0 12px rgba(127, 0, 255, 0.6)); }
    }

    .metric-box {
        text-align: center;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 12px; }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: #ffffff !important;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ROBUST MODEL LOADER WITH VALIDATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(search_str):
    files = glob.glob(os.path.join(BASE_DIR, "*.pkl"))
    for f in files:
        if search_str.lower() in os.path.basename(f).lower():
            return f
    return None

@st.cache_resource
def load_models_from_disk():
    m1_path = get_model_path("Model_1") or os.path.join(BASE_DIR, "Model_1 Job Analytics.pkl")
    m2_path = get_model_path("Model_2") or os.path.join(BASE_DIR, "Model_2 job analytics.pkl")
    
    m1, m2 = None, None
    if os.path.exists(m1_path):
        try:
            with open(m1_path, "rb") as f1:
                m1 = pickle.load(f1)
        except Exception:
            pass
            
    if os.path.exists(m2_path):
        try:
            with open(m2_path, "rb") as f2:
                m2 = pickle.load(f2)
        except Exception:
            pass
            
    return m1, m2

model_1, model_2 = load_models_from_disk()

# Sidebar fallback upload options with feature validation
if model_1 is None or model_2 is None:
    st.sidebar.warning("⚠️ Model file(s) missing from working directory.")
    if model_1 is None:
        uploaded_m1 = st.sidebar.file_uploader("Upload Model 1 (.pkl) [LinearRegression]", type=["pkl"], key="up_m1")
        if uploaded_m1 is not None:
            obj = pickle.load(uploaded_m1)
            if hasattr(obj, "n_features_in_") and obj.n_features_in_ == 17:
                model_1 = obj
                st.sidebar.success("Model 1 Loaded Successfully!")
            else:
                st.sidebar.error("Invalid Model 1 file. Expected LinearRegression model with 17 features.")

    if model_2 is None:
        uploaded_m2 = st.sidebar.file_uploader("Upload Model 2 (.pkl) [LogisticRegression]", type=["pkl"], key="up_m2")
        if uploaded_m2 is not None:
            obj = pickle.load(uploaded_m2)
            if hasattr(obj, "n_features_in_") and obj.n_features_in_ == 18:
                model_2 = obj
                st.sidebar.success("Model 2 Loaded Successfully!")
            else:
                st.sidebar.error("Invalid Model 2 file. Expected LogisticRegression model with 18 features.")

# Categorical Mappings
JOB_TITLES = ["Data Analyst", "Data Scientist", "Machine Learning Engineer", "Data Engineer", "AI Architect"]
COMPANY_SIZES = ["Small (1-50)", "Medium (51-500)", "Large (500+)"]
INDUSTRIES = ["Technology", "Finance", "Healthcare", "E-commerce", "Consulting"]
COUNTRIES = ["United States", "India", "United Kingdom", "Canada", "Germany"]
REMOTE_TYPES = ["On-site", "Hybrid", "Remote"]
EXPERIENCE_LEVELS = ["Entry-level", "Mid-level", "Senior-level", "Executive"]
EDUCATION_LEVELS = ["Bachelor's", "Master's", "PhD", "Other"]

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<h1 class="title-text">💼 AI Job Market & Salary Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem;">Predict job compensation and classification using Machine Learning</p>', unsafe_allow_html=True)
st.write("---")

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Candidate & Job Inputs")

job_title = st.sidebar.selectbox("Job Title", JOB_TITLES)
company_size = st.sidebar.selectbox("Company Size", COMPANY_SIZES)
company_industry = st.sidebar.selectbox("Industry", INDUSTRIES)
country = st.sidebar.selectbox("Country", COUNTRIES)
remote_type = st.sidebar.selectbox("Remote Type", REMOTE_TYPES)
experience_level = st.sidebar.selectbox("Experience Level", EXPERIENCE_LEVELS)
years_exp = st.sidebar.slider("Years of Experience", 0, 20, 3)
education_level = st.sidebar.selectbox("Education Level", EDUCATION_LEVELS)

st.sidebar.markdown("### 🛠 Technical Skills")
skill_python = st.sidebar.checkbox("Python", value=True)
skill_sql = st.sidebar.checkbox("SQL", value=True)
skill_ml = st.sidebar.checkbox("Machine Learning")
skill_dl = st.sidebar.checkbox("Deep Learning")
skill_cloud = st.sidebar.checkbox("Cloud Platforms (AWS/GCP/Azure)")

st.sidebar.markdown("### 📅 Posting & Demand Info")
posting_month = st.sidebar.slider("Posting Month", 1, 12, 6)
posting_year = st.sidebar.slider("Posting Year", 2020, 2026, 2025)
hiring_urgency = st.sidebar.select_slider("Hiring Urgency Rating", options=[1, 2, 3, 4, 5], value=3)
job_openings = st.sidebar.number_input("Open Positions Available", min_value=1, max_value=500, value=5)

# Construct Model 1 Feature Vector (17 features)
m1_features = np.array([[
    JOB_TITLES.index(job_title),
    COMPANY_SIZES.index(company_size),
    INDUSTRIES.index(company_industry),
    COUNTRIES.index(country),
    REMOTE_TYPES.index(remote_type),
    EXPERIENCE_LEVELS.index(experience_level),
    years_exp,
    EDUCATION_LEVELS.index(education_level),
    int(skill_python),
    int(skill_sql),
    int(skill_ml),
    int(skill_dl),
    int(skill_cloud),
    posting_month,
    posting_year,
    hiring_urgency,
    job_openings
]], dtype=object)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["💵 Model 1: Salary Regression", "📊 Model 2: Job Classification"])

# TAB 1: MODEL 1
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 Estimated Annual Salary Prediction")
    st.write("Model 1 uses **Linear Regression** to estimate base compensation.")
    
    if st.button("🚀 Predict Salary (Model 1)", key="btn_model_1"):
        if model_1 is not None:
            try:
                pred = model_1.predict(m1_features)
                # Safely extract scalar value from array
                pred_val = float(np.ravel(pred)[0])
                pred_val = abs(pred_val)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-value">${pred_val:,.2f}</div>
                            <div class="metric-label">Estimated Base Salary</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-value">${(pred_val / 12):,.2f}</div>
                            <div class="metric-label">Monthly Estimate</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-value">{years_exp} Yrs</div>
                            <div class="metric-label">Experience Tier</div>
                        </div>
                    ''', unsafe_allow_html=True)

                st.write("")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_val,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Salary Market Tier", 'font': {'size': 20, 'color': '#ffffff'}},
                    gauge={
                        'axis': {'range': [None, 250000], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#00c6ff"},
                        'bgcolor': "rgba(15, 23, 42, 0.8)",
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 80000], 'color': 'rgba(239, 68, 68, 0.3)'},
                            {'range': [80000, 150000], 'color': 'rgba(234, 179, 8, 0.3)'},
                            {'range': [150000, 250000], 'color': 'rgba(34, 197, 94, 0.3)'}
                        ],
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as err:
                st.error(f"Error executing Model 1 prediction: {err}")
        else:
            st.error("Please upload 'Model_1 Job Analytics.pkl' under Upload Model 1 (.pkl) in the sidebar.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: MODEL 2
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 Job Categorization & Classification")
    st.write("Model 2 uses **Logistic Regression** to analyze market positioning using profile metadata and target salary.")

    custom_salary = st.number_input(
        "Target Annual Salary ($) for Classification Input",
        min_value=10000,
        max_value=500000,
        value=110000,
        step=5000
    )

    # Insert salary at index 13 to create Model 2's 18-feature vector
    m2_features = np.insert(m1_features, 13, custom_salary).reshape(1, -1)

    if st.button("⚡ Classify Job Profile (Model 2)", key="btn_model_2"):
        if model_2 is not None:
            try:
                class_pred = model_2.predict(m2_features)
                class_val = np.ravel(class_pred)[0]
                class_probs = model_2.predict_proba(m2_features)[0]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-label">Predicted Class</div>
                            <div class="metric-value" style="color: #7f00ff;">Tier {class_val}</div>
                            <p style="color: #94a3b8; margin-top: 10px; font-size: 0.85rem;">
                                Classified using parameters & target salary of ${custom_salary:,.0f}
                            </p>
                        </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    classes = [f"Tier {c}" for c in model_2.classes_]
                    fig_bar = go.Figure(go.Bar(
                        x=classes,
                        y=class_probs,
                        marker=dict(color=class_probs, colorscale='Viridis')
                    ))
                    fig_bar.update_layout(
                        title="Class Probability Distribution",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#e2e8f0"},
                        xaxis=dict(title="Category Tiers"),
                        yaxis=dict(title="Probability", range=[0, 1])
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as err:
                st.error(f"Error executing Model 2 classification: {err}")
        else:
            st.error("Please upload 'Model_2 job analytics.pkl' under Upload Model 2 (.pkl) in the sidebar.")
            
    st.markdown('</div>', unsafe_allow_html=True)
