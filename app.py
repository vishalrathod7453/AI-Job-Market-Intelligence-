import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Job Analytics Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

.hero {
    text-align:center;
    padding:20px;
    border-radius:20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    animation: fadeIn 2s;
}

@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}

.metric-card{
    padding:15px;
    border-radius:15px;
    background:#1e293b;
    color:white;
    text-align:center;
}

.stButton>button{
    width:100%;
    border-radius:12px;
    height:3em;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
salary_model = joblib.load("Model_1 Job Analytics.pkl")
hiring_model = joblib.load("Model_2 job analytics.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<div class='hero'>
<h1>🚀 AI Job Market Analytics Dashboard</h1>
<h4>Salary Prediction + Hiring Urgency Prediction</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4149/4149647.png",
    width=120
)

st.sidebar.title("⚙ Job Information")

# ---------------- INPUTS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    job_title = st.number_input("Job Title Encoded", 0, 100)
    company_size = st.number_input("Company Size", 0, 100)
    company_industry = st.number_input("Industry Encoded", 0, 100)
    country = st.number_input("Country Encoded", 0, 100)
    remote_type = st.number_input("Remote Type", 0, 10)

with col2:
    experience_level = st.number_input("Experience Level", 0, 10)
    years_experience = st.number_input("Years Experience", 0, 50)
    education_level = st.number_input("Education Level", 0, 10)

    skills_python = st.selectbox("Python Skill", [0,1])
    skills_sql = st.selectbox("SQL Skill", [0,1])

with col3:
    skills_ml = st.selectbox("Machine Learning", [0,1])
    skills_deep_learning = st.selectbox("Deep Learning", [0,1])
    skills_cloud = st.selectbox("Cloud", [0,1])

    job_posting_month = st.slider("Month",1,12,1)
    job_posting_year = st.slider("Year",2020,2030,2026)

job_openings = st.slider("Job Openings",1,100,10)

# ---------------- PREDICT SALARY ----------------

if st.button("💰 Predict Salary"):

    salary_input = pd.DataFrame([[
        job_title,
        company_size,
        company_industry,
        country,
        remote_type,
        experience_level,
        years_experience,
        education_level,
        skills_python,
        skills_sql,
        skills_ml,
        skills_deep_learning,
        skills_cloud,
        job_posting_month,
        job_posting_year,
        1,
        job_openings
    ]], columns=[
        'job_title',
        'company_size',
        'company_industry',
        'country',
        'remote_type',
        'experience_level',
        'years_experience',
        'education_level',
        'skills_python',
        'skills_sql',
        'skills_ml',
        'skills_deep_learning',
        'skills_cloud',
        'job_posting_month',
        'job_posting_year',
        'hiring_urgency',
        'job_openings'
    ])

    salary = salary_model.predict(salary_input)[0][0]

    st.success(f"💰 Predicted Salary : ${salary:,.2f}")

# ---------------- PREDICT HIRING URGENCY ----------------

if st.button("🔥 Predict Hiring Urgency"):

    salary_value = st.number_input(
        "Enter Salary for Hiring Prediction",
        value=50000
    )

    urgency_input = pd.DataFrame([[
        job_title,
        company_size,
        company_industry,
        country,
        remote_type,
        experience_level,
        years_experience,
        education_level,
        skills_python,
        skills_sql,
        skills_ml,
        skills_deep_learning,
        skills_cloud,
        salary_value,
        job_posting_month,
        job_posting_year,
        job_openings
    ]], columns=[
        'job_title',
        'company_size',
        'company_industry',
        'country',
        'remote_type',
        'experience_level',
        'years_experience',
        'education_level',
        'skills_python',
        'skills_sql',
        'skills_ml',
        'skills_deep_learning',
        'skills_cloud',
        'salary',
        'job_posting_month',
        'job_posting_year',
        'job_openings'
    ])

    urgency = hiring_model.predict(urgency_input)[0]

    label_map = {
        0: "🟢 Low",
        1: "🟡 Medium",
        2: "🔴 High"
    }

    st.success(
        f"Hiring Urgency : {label_map.get(urgency,'Unknown')}"
    )
