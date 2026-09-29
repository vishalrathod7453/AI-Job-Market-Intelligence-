import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ANIMATED DARK GLASSMORPHISM THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Job Analytics Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Gradient Background Animation */
    .stApp {
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

    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Animated Glowing Header Title */
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

    /* Metric Display Box */
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

    /* Streamlit Navigation Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
        font-weight: 600;
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
# 2. DYNAMIC CSV PIPELINE & MODEL TRAINING FUNCTION
# -----------------------------------------------------------------------------
@st.cache_data
def process_data_and_train_models(df):
    """Clean dataset, encode categorical features, and train Linear & Logistic Regression models."""
    df_clean = df.copy()
    
    # Handle Missing Values
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')

    # Label Encoders dictionary
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le

    # Dynamic target detection logic
    salary_col = None
    for col in df_clean.columns:
        if 'salary' in col.lower() or 'compensation' in col.lower() or 'pay' in col.lower():
            salary_col = col
            break

    if salary_col is None:
        salary_col = num_cols[0] if len(num_cols) > 0 else df_clean.columns[-1]

    # Create synthetic tier classification if target tier class is missing
    tier_col = None
    for col in df_clean.columns:
        if 'tier' in col.lower() or 'class' in col.lower() or 'category' in col.lower() or 'level' in col.lower():
            if col != salary_col:
                tier_col = col
                break

    if tier_col is None:
        # Create binned salary tiers (e.g., 0: Entry, 1: Mid, 2: High)
        df_clean['job_tier'] = pd.qcut(df_clean[salary_col], q=3, labels=[0, 1, 2]).astype(int)
        tier_col = 'job_tier'

    # Features for Model 1 (Linear Regression - Salary continuous prediction)
    X_m1 = df_clean.drop(columns=[salary_col, tier_col], errors='ignore')
    y_m1 = df_clean[salary_col]

    m1 = LinearRegression()
    m1.fit(X_m1, y_m1)
    
    # Calculate M1 accuracy metric
    y_m1_pred = m1.predict(X_m1)
    r2_m1 = r2_score(y_m1, y_m1_pred)

    # Features for Model 2 (Logistic Regression - Classification)
    X_m2 = df_clean.drop(columns=[tier_col], errors='ignore')
    y_m2 = df_clean[tier_col]

    m2 = LogisticRegression(max_iter=1000)
    m2.fit(X_m2, y_m2)
    
    y_m2_pred = m2.predict(X_m2)
    acc_m2 = accuracy_score(y_m2, y_m2_pred)

    return df_clean, encoders, m1, m2, salary_col, tier_col, list(X_m1.columns), list(X_m2.columns), r2_m1, acc_m2

# -----------------------------------------------------------------------------
# 3. SIDEBAR DATASET UPLOADER & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("📁 Upload & Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Job Dataset (.csv)", type=["csv"])

# Default Options Fallback
JOB_TITLES = ["Data Analyst", "Data Scientist", "Machine Learning Engineer", "Data Engineer", "AI Architect"]
COMPANY_SIZES = ["Small (1-50)", "Medium (51-500)", "Large (500+)"]
INDUSTRIES = ["Technology", "Finance", "Healthcare", "E-commerce", "Consulting"]
COUNTRIES = ["United States", "India", "United Kingdom", "Canada", "Germany"]
REMOTE_TYPES = ["On-site", "Hybrid", "Remote"]
EXPERIENCE_LEVELS = ["Entry-level", "Mid-level", "Senior-level", "Executive"]
EDUCATION_LEVELS = ["Bachelor's", "Master's", "PhD", "Other"]

st.sidebar.markdown("---")
st.sidebar.title("📌 Candidate Inputs")

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

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD DISPLAY
# -----------------------------------------------------------------------------
st.markdown('<h1 class="title-text">💼 AI Job Market & Salary Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem;">Predict compensation and classify job profiles from custom CSV datasets using Machine Learning</p>', unsafe_allow_html=True)
st.write("---")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        processed_df, encoders, model_1, model_2, target_salary, target_tier, m1_cols, m2_cols, r2_m1, acc_m2 = process_data_and_train_models(raw_df)
        
        st.sidebar.success(f"Dataset Loaded! ({len(raw_df)} rows)")
        st.sidebar.info(f"Target Salary Column: `{target_salary}`")
        
        tab_data, tab1, tab2 = st.tabs(["📋 Dataset Overview", "💵 Model 1: Salary Regression", "📊 Model 2: Job Classification"])

        # -----------------------------------------------------------------------------
        # TAB DATASET OVERVIEW
        # -----------------------------------------------------------------------------
        with tab_data:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Dataset Statistics & Overview")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Sample Records", f"{len(raw_df):,}")
            c2.metric("Total Features", f"{len(raw_df.columns)}")
            c3.metric("Model 1 Fit (R² Score)", f"{r2_m1:.2f}")
            c4.metric("Model 2 Accuracy", f"{acc_m2*100:.1f}%")
            
            st.write("### Raw Dataset Preview")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # CONSTRUCT INPUT VECTOR FOR INFERENCE
        # -----------------------------------------------------------------------------
        input_dict = {
            'job_title': JOB_TITLES.index(job_title),
            'company_size': COMPANY_SIZES.index(company_size),
            'company_industry': INDUSTRIES.index(company_industry),
            'country': COUNTRIES.index(country),
            'remote_type': REMOTE_TYPES.index(remote_type),
            'experience_level': EXPERIENCE_LEVELS.index(experience_level),
            'years_experience': years_exp,
            'education_level': EDUCATION_LEVELS.index(education_level),
            'skills_python': int(skill_python),
            'skills_sql': int(skill_sql),
            'skills_ml': int(skill_ml),
            'skills_deep_learning': int(skill_dl),
            'skills_cloud': int(skill_cloud),
            'job_posting_month': posting_month,
            'job_posting_year': posting_year,
            'hiring_urgency': hiring_urgency,
            'job_openings': job_openings
        }

        # Vector matching trained dataset feature count dynamically
        m1_vector = []
        for col in m1_cols:
            if col in input_dict:
                m1_vector.append(input_dict[col])
            else:
                m1_vector.append(0)
                
        m1_features = np.array([m1_vector])

        # -----------------------------------------------------------------------------
        # TAB 1: MODEL 1 (Linear Regression)
        # -----------------------------------------------------------------------------
        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔮 Estimated Annual Salary Prediction")
            st.write("Model 1 uses **Linear Regression** trained on the uploaded CSV dataset.")

            if st.button("🚀 Predict Salary (Model 1)", key="btn_model_1"):
                try:
                    pred_res = model_1.predict(m1_features)
                    pred_val = abs(float(np.ravel(pred_res)[0]))

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
                    # Plotly Gauge Chart
                    max_gauge = max(250000.0, pred_val * 1.3)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_val,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Salary Market Tier", 'font': {'size': 20, 'color': '#ffffff'}},
                        gauge={
                            'axis': {'range': [None, max_gauge], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': "#00c6ff"},
                            'bgcolor': "rgba(15, 23, 42, 0.8)",
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, max_gauge*0.35], 'color': 'rgba(239, 68, 68, 0.3)'},
                                {'range': [max_gauge*0.35, max_gauge*0.7], 'color': 'rgba(234, 179, 8, 0.3)'},
                                {'range': [max_gauge*0.7, max_gauge], 'color': 'rgba(34, 197, 94, 0.3)'}
                            ],
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as err:
                    st.error(f"Error executing Model 1 prediction: {err}")

            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # TAB 2: MODEL 2 (Logistic Regression)
        # -----------------------------------------------------------------------------
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 Job Categorization & Classification")
            st.write("Model 2 uses **Logistic Regression** to analyze market positioning using metadata and target salary.")

            custom_salary = st.number_input(
                "Target Annual Salary ($) for Classification Input",
                min_value=10000,
                max_value=500000,
                value=110000,
                step=5000
            )

            m2_vector = []
            for col in m2_cols:
                if col == target_salary or 'salary' in col.lower():
                    m2_vector.append(custom_salary)
                elif col in input_dict:
                    m2_vector.append(input_dict[col])
                else:
                    m2_vector.append(0)

            m2_features = np.array([m2_vector])

            if st.button("⚡ Classify Job Profile (Model 2)", key="btn_model_2"):
                try:
                    class_pred = model_2.predict(m2_features)
                    class_val = int(np.ravel(class_pred)[0])
                    class_probs = model_2.predict_proba(m2_features)[0]

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f'''
                            <div class="metric-box">
                                <div class="metric-label">Predicted Class</div>
                                <div class="metric-value" style="color: #7f00ff;">Tier {class_val}</div>
                                <p style="color: #94a3b8; margin-top: 10px; font-size: 0.85rem;">
                                    Classified using parameters & target salary of ${custom_salary:,-0f}
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

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to process CSV file: {e}")
else:
    st.info("👈 Please upload your **Job Analytics Dataset (.csv)** in the sidebar to dynamically train models and execute salary predictions.")
