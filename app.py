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
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error

# Optional PDF parsing import
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & TABLEAU-INSPIRED GLASSMORPHISM THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tableau AI Job & Salary Analytics Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Tableau Dark/Slate Palette Animation */
    .stApp {
        background: linear-gradient(-45deg, #0b132b, #1c2541, #1e293b, #0f172a);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Tableau Dashboard Card Layout */
    .tableau-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    .tableau-header {
        border-bottom: 2px solid #38bdf8;
        padding-bottom: 8px;
        margin-bottom: 16px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Animated Glowing Title */
    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Metric Visual Boxes */
    .metric-box {
        text-align: center;
        background: rgba(15, 23, 42, 0.8);
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-label {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Condition Status Badges */
    .badge-pass {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .badge-warn {
        background-color: rgba(234, 179, 8, 0.2);
        color: #fde047;
        border: 1px solid #eab308;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .badge-fail {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Navigation Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 8px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #4338ca 100%);
        color: #ffffff !important;
        border: none;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PDF PARSER & SAMPLE DATASET GENERATOR
# -----------------------------------------------------------------------------
def extract_pdf_text(uploaded_pdf):
    """Extract text content from uploaded PDF file."""
    if uploaded_pdf is None:
        return None
    try:
        if HAS_PYPDF:
            reader = pypdf.PdfReader(uploaded_pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text if text.strip() else "PDF contains no extractable text."
        else:
            return "PyPDF library is not installed in the environment. Using fallback text reader."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 300
    job_titles = ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'AI Research Scientist', 'BI Developer']
    industries = ['Technology', 'Finance', 'Healthcare', 'E-commerce', 'Consulting']
    exp_levels = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive']
    education = ['Bachelors', 'Masters', 'PhD']
    
    titles = np.random.choice(job_titles, n)
    inds = np.random.choice(industries, n)
    exps = np.random.choice(exp_levels, n)
    eds = np.random.choice(education, n)
    years_exp = np.random.randint(0, 20, n)
    
    # Base salary generation
    base_salary = 52000 + (years_exp * 6800) + np.random.normal(0, 11000, n)
    base_salary = np.clip(base_salary, 35000, 280000)
    
    df = pd.DataFrame({
        'Job Title': titles,
        'Industry': inds,
        'Experience Level': exps,
        'Education': eds,
        'Years of Experience': years_exp,
        'Python Skill': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'SQL Skill': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'ML Skill': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'Salary USD': np.round(base_salary, 2)
    })
    
    # Missing values & duplicates injection for testing cleaning pipeline
    df.loc[np.random.choice(n, 12), 'Years of Experience'] = np.nan
    df.loc[np.random.choice(n, 8), 'Industry'] = np.nan
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df

# -----------------------------------------------------------------------------
# 3. ADVANCED DATA CLEANING & MODEL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def process_data_pipeline(df):
    df_clean = df.copy()
    df_clean.columns = [c.strip().lower().replace(' ', '_') for c in df_clean.columns]

    initial_rows = len(df_clean)
    initial_nulls = int(df_clean.isnull().sum().sum())
    duplicate_rows = int(df_clean.duplicated().sum())
    
    # Remove duplicate records
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Detect salary target column
    salary_col = None
    for col in df_clean.columns:
        if any(kw in col for kw in ['salary', 'compensation', 'pay', 'usd']):
            salary_col = col
            break
            
    if salary_col is None:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        salary_col = num_cols[0] if len(num_cols) > 0 else df_clean.columns[-1]

    # Detect industry column
    industry_col = None
    for col in df_clean.columns:
        if any(kw in col for kw in ['industry', 'sector', 'domain', 'field']):
            industry_col = col
            break

    # Missing values imputation
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in cat_cols:
        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
        df_clean[col] = df_clean[col].fillna(mode_val)

    final_nulls = int(df_clean.isnull().sum().sum())

    # Create dynamic quantile salary tiers (Tier 0, Tier 1, Tier 2)
    tier_col = 'job_tier'
    if 'tier' in df_clean.columns:
        tier_col = 'tier'
    elif 'category' in df_clean.columns:
        tier_col = 'category'
    else:
        df_clean['job_tier'] = pd.qcut(df_clean[salary_col], q=3, labels=[0, 1, 2]).astype(int)

    # Encode categorical features
    encoders = {}
    options = {}
    df_encoded = df_clean.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        options[col] = list(le.classes_)

    # Model 1: Salary Regression
    X_m1 = df_encoded.drop(columns=[salary_col, tier_col], errors='ignore')
    y_m1 = df_encoded[salary_col]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X_m1, y_m1, test_size=0.2, random_state=42)
    m1 = LinearRegression()
    m1.fit(X1_train, y1_train)
    
    y1_pred = m1.predict(X1_test)
    r2_m1 = float(r2_score(y1_test, y1_pred))
    mae_m1 = float(mean_absolute_error(y1_test, y1_pred))
    rmse_m1 = float(np.sqrt(mean_squared_error(y1_test, y1_pred)))

    # Model 2: Job Tier Classification
    X_m2 = df_encoded.drop(columns=[tier_col], errors='ignore')
    y_m2 = df_encoded[tier_col]

    X2_train, X2_test, y2_train, y2_test = train_test_split(X_m2, y_m2, test_size=0.2, random_state=42)
    m2 = LogisticRegression(max_iter=1000)
    m2.fit(X2_train, y2_train)
    
    y2_pred = m2.predict(X2_test)
    acc_m2 = float(accuracy_score(y2_test, y2_pred))

    cleaning_stats = {
        'initial_rows': initial_rows,
        'clean_rows': len(df_clean),
        'duplicates_removed': duplicate_rows,
        'initial_nulls': initial_nulls,
        'final_nulls': final_nulls,
        'salary_col': salary_col,
        'industry_col': industry_col,
        'mae_m1': mae_m1,
        'rmse_m1': rmse_m1
    }

    return (
        df_clean, df_encoded, encoders, options, m1, m2, 
        salary_col, tier_col, industry_col, 
        list(X_m1.columns), list(X_m2.columns), 
        r2_m1, acc_m2, cleaning_stats
    )

# -----------------------------------------------------------------------------
# 4. CONDITION VERIFICATION ENGINE
# -----------------------------------------------------------------------------
def verify_dataset_conditions(df, salary_col, clean_stats, r2_score_val, acc_score_val, pdf_text_present):
    checks = []
    
    # Condition 1: Minimal Sample Size
    has_min_rows = len(df) >= 50
    checks.append({
        "rule": "Dataset Sample Size (>= 50 rows)",
        "status": "PASS" if has_min_rows else "FAIL",
        "detail": f"Dataset contains {len(df)} records."
    })
    
    # Condition 2: Target Salary Identification
    has_salary = salary_col is not None
    checks.append({
        "rule": "Target Compensation Column Present",
        "status": "PASS" if has_salary else "FAIL",
        "detail": f"Detected salary column: '{salary_col}'."
    })

    # Condition 3: Missing Value Resolution
    no_nulls = clean_stats['final_nulls'] == 0
    checks.append({
        "rule": "Missing Value Zero Imputation Check",
        "status": "PASS" if no_nulls else "WARN",
        "detail": f"Initial nulls: {clean_stats['initial_nulls']} ➔ Remaining nulls: {clean_stats['final_nulls']}."
    })

    # Condition 4: Regression Model Fit (R²)
    r2_pass = r2_score_val >= 0.50
    checks.append({
        "rule": "Regression Model Fit (R² Score >= 0.50)",
        "status": "PASS" if r2_pass else "WARN",
        "detail": f"Achieved R² score = {r2_score_val:.3f}."
    })

    # Condition 5: Classifier Model Accuracy
    acc_pass = acc_score_val >= 0.60
    checks.append({
        "rule": "Classifier Model Accuracy (>= 60%)",
        "status": "PASS" if acc_pass else "WARN",
        "detail": f"Achieved classification accuracy = {acc_score_val * 100:.1f}%."
    })

    # Condition 6: Problem Statement File Attachment
    checks.append({
        "rule": "Problem Statement Attachment (.pdf)",
        "status": "PASS" if pdf_text_present else "INFO",
        "detail": "PDF problem statement loaded and parsed." if pdf_text_present else "Using default built-in problem statement specification."
    })

    return checks

# -----------------------------------------------------------------------------
# 5. SIDEBAR INGESTION & CONTROLS
# -----------------------------------------------------------------------------
st.markdown('<h1 class="title-text">📊 Tableau AI Job & Salary Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.05rem; margin-bottom: 25px;">Interactive Business Intelligence, Data Quality Verification & Predictive Modeling</p>', unsafe_allow_html=True)

st.sidebar.title("📂 Data & Document Ingestion")
st.sidebar.caption("Upload your dataset CSV and Problem Statement PDF here:")

# CSV & PDF File Uploaders side-by-side in sidebar
uploaded_file = st.sidebar.file_uploader("Upload Job Dataset (.csv)", type=["csv"], key="csv_uploader")
uploaded_pdf = st.sidebar.file_uploader("Upload Problem Statement (.pdf)", type=["pdf"], key="pdf_uploader")

# Parse PDF text if available
pdf_content = extract_pdf_text(uploaded_pdf) if uploaded_pdf is not None else None

# Load CSV dataset or fallback
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset CSV loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        raw_df = generate_sample_data()
else:
    st.sidebar.info("💡 Using built-in sample AI job dataset. Upload custom files above.")
    raw_df = generate_sample_data()

# Process Data Pipeline
(
    df_clean, df_encoded, encoders, options, model_1, model_2, 
    target_salary, target_tier, industry_col, m1_cols, m2_cols, 
    r2_m1, acc_m2, clean_stats
) = process_data_pipeline(raw_df)

# Run Condition Verification
verification_results = verify_dataset_conditions(
    df_clean, target_salary, clean_stats, r2_m1, acc_m2, pdf_content is not None
)

st.sidebar.write("---")
st.sidebar.title("📌 Candidate Profile Inputs")

# Dynamic sidebar inputs matching dataframe schema
input_data = {}
for col in m1_cols:
    col_label = col.replace('_', ' ').title()
    if col in options:
        selected_val = st.sidebar.selectbox(col_label, options[col], key=f"sb_{col}")
        input_data[col] = encoders[col].transform([selected_val])[0]
    elif any(kw in col for kw in ['year', 'exp', 'experience']):
        input_data[col] = st.sidebar.slider(col_label, 0, 25, 4, key=f"sl_{col}")
    elif 'month' in col:
        input_data[col] = st.sidebar.slider(col_label, 1, 12, 6, key=f"sl_{col}")
    elif any(kw in col for kw in ['urgency', 'rating', 'level']):
        input_data[col] = st.sidebar.slider(col_label, 1, 5, 3, key=f"sl_{col}")
    elif any(kw in col for kw in ['opening', 'count', 'positions']):
        input_data[col] = st.sidebar.number_input(col_label, min_value=1, max_value=500, value=5, key=f"num_{col}")
    elif col.startswith('skill') or df_clean[col].nunique() <= 2:
        input_data[col] = int(st.sidebar.checkbox(col_label, value=True, key=f"cb_{col}"))
    else:
        input_data[col] = st.sidebar.number_input(col_label, value=float(df_clean[col].median()), key=f"num_{col}")

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION TABS
# -----------------------------------------------------------------------------
tab_tableaudash, tab_problem, tab_clean, tab1, tab2 = st.tabs([
    "📊 Tableau BI Dashboard",
    "📑 Problem Statement & Verification",
    "🧼 Data Pipeline & Quality", 
    "💵 Model 1: Salary Regression", 
    "🎯 Model 2: Job Classification"
])

# -----------------------------------------------------------------------------
# TAB 1: TABLEAU-STYLE INTERACTIVE BI DASHBOARD
# -----------------------------------------------------------------------------
with tab_tableaudash:
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.markdown('<div class="tableau-header"><h3>📈 Executive Compensation Analytics Dashboard</h3><span>Theme: Tableau Slate</span></div>', unsafe_allow_html=True)

    # Top KPI Cards Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    avg_annual = float(df_clean[target_salary].mean())
    median_annual = float(df_clean[target_salary].median())
    max_annual = float(df_clean[target_salary].max())
    total_records = len(df_clean)

    kpi1.markdown(f'''<div class="metric-box"><div class="metric-value">${avg_annual:,.0f}</div><div class="metric-label">Average Compensation</div></div>''', unsafe_allow_html=True)
    kpi2.markdown(f'''<div class="metric-box"><div class="metric-value">${median_annual:,.0f}</div><div class="metric-label">Median Compensation</div></div>''', unsafe_allow_html=True)
    kpi3.markdown(f'''<div class="metric-box"><div class="metric-value">${max_annual:,.0f}</div><div class="metric-label">Peak Compensation</div></div>''', unsafe_allow_html=True)
    kpi4.markdown(f'''<div class="metric-box"><div class="metric-value">{total_records:,}</div><div class="metric-label">Cleaned Records</div></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tableau Multi-Chart Grid
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
        st.write("#### 🏢 Industry Sector Salary Breakdown")
        if industry_col and industry_col in df_clean.columns:
            ind_agg = df_clean.groupby(industry_col)[target_salary].mean().reset_index()
            fig_ind = px.bar(
                ind_agg, 
                x=industry_col, 
                y=target_salary, 
                color=target_salary,
                color_continuous_scale=px.colors.sequential.Tealgrn,
                text_auto='.2s',
                labels={target_salary: 'Avg Salary ($)'}
            )
            fig_ind.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=320)
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.info("No explicit industry column detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
        st.write("#### 📈 Experience vs Compensation Trajectory")
        exp_col = [c for c in df_clean.columns if 'exp' in c or 'year' in c]
        if exp_col:
            fig_scat = px.scatter(
                df_clean, 
                x=exp_col[0], 
                y=target_salary, 
                color='job_tier' if 'job_tier' in df_clean.columns else None,
                trendline="ols",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={exp_col[0]: 'Years of Experience', target_salary: 'Salary ($)'}
            )
            fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=320)
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("No experience column found for trajectory plotting.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Distribution and Boxplot
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
        st.write("#### 📊 Salary Distribution (Histogram & Density)")
        fig_hist = px.histogram(
            df_clean, 
            x=target_salary, 
            nbins=25, 
            color_discrete_sequence=['#38bdf8'],
            marginal="box"
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=320)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart4:
        st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
        st.write("#### 🍩 Market Salary Tier Share")
        tier_counts = df_clean[target_tier].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        tier_counts['Tier'] = tier_counts['Tier'].apply(lambda x: f"Tier {x}")
        fig_pie = px.pie(
            tier_counts, 
            names='Tier', 
            values='Count', 
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Tableau10
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: PROBLEM STATEMENT & CONDITION VERIFICATION
# -----------------------------------------------------------------------------
with tab_problem:
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.subheader("📋 Dataset & Model Condition Verification Matrix")
    st.write("Automated compliance audit checking dataset integrity, PDF attachment, and model performance criteria:")

    # Display Condition Checks Table
    for check in verification_results:
        c1, c2, c3 = st.columns([3, 1, 4])
        with c1:
            st.write(f"**{check['rule']}**")
        with c2:
            if check['status'] == 'PASS':
                st.markdown('<span class="badge-pass">✔ PASS</span>', unsafe_allow_html=True)
            elif check['status'] == 'WARN':
                st.markdown('<span class="badge-warn">⚠️️ WARN</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-fail">✖ FAIL / INFO</span>', unsafe_allow_html=True)
        with c3:
            st.write(check['detail'])
        st.markdown('<hr style="margin: 8px 0; border-color: rgba(255,255,255,0.05);"/>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Display Problem Statement PDF Content or Specification
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.subheader("📑 Problem Statement Specification")

    if pdf_content:
        st.success("📄 Extracted content from uploaded Problem Statement PDF:")
        st.text_area("PDF Document Content", value=pdf_content, height=280)
    else:
        st.info("ℹ️ No PDF uploaded. Showing standard project problem statement formulation:")
        st.markdown("""
        > **Strategic Objective:** Develop an end-to-end data processing and predictive machine learning system for AI and technology talent compensation analysis.
        > 
        > **Core Deliverables:**
        > 1. **Data Cleaning Pipeline:** Deduplication, standardizing attributes, and median/mode imputation for missing data.
        > 2. **Salary Regression Engine (Model 1):** Train a linear regression model to accurately forecast baseline annual compensation.
        > 3. **Job Category Classifier (Model 2):** Classify candidate profiles into standardized quantile salary tiers using Logistic Regression.
        > 4. **Tableau Business Intelligence Dashboard:** Provide real-time interactive visual analytics, condition validation, and market insights.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: DATA CLEANING & QUALITY OVERVIEW
# -----------------------------------------------------------------------------
with tab_clean:
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.subheader("🧼 Data Pipeline & Cleaning Summary")
    
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Initial / Clean Rows", f"{clean_stats['initial_rows']:,} / {clean_stats['clean_rows']:,}")
    q2.metric("Duplicates Removed", f"{clean_stats['duplicates_removed']:,}")
    q3.metric("Nulls Imputed", f"{clean_stats['initial_nulls']:,} ➔ {clean_stats['final_nulls']}")
    q4.metric("Target Salary Column", f"`{target_salary}`")
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Performance Cards
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.subheader("🎯 Machine Learning Model Scorecard")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{r2_m1:.3f}</div><div class="metric-label">Model 1 R² Score</div></div>''', unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">${clean_stats['mae_m1']:,.0f}</div><div class="metric-label">Model 1 MAE</div></div>''', unsafe_allow_html=True)

    with m_col3:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">{acc_m2 * 100:.1f}%</div><div class="metric-label">Model 2 Accuracy</div></div>''', unsafe_allow_html=True)

    with m_col4:
        st.markdown(f'''<div class="metric-box"><div class="metric-value">Tier 0 / 1 / 2</div><div class="metric-label">Quantile Categories</div></div>''', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Preview Tables
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("### 🔝 Top 5 Highest Compensation Profiles")
        top_5 = df_clean.sort_values(by=target_salary, ascending=False).head(5)
        st.dataframe(top_5, use_container_width=True)

    with col_b:
        st.write("### 🔻 Bottom 5 Lowest Compensation Profiles")
        bottom_5 = df_clean.sort_values(by=target_salary, ascending=True).head(5)
        st.dataframe(bottom_5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: MODEL 1 - SALARY REGRESSION ENGINE
# -----------------------------------------------------------------------------
m1_vector = np.array([[input_data[col] for col in m1_cols]])

with tab1:
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.subheader("🔮 Salary Prediction Engine (Linear Regression)")
    st.write("Predict estimated base annual and monthly compensation using candidate profile features configured in the sidebar.")

    if st.button("🚀 Execute Salary Prediction", key="btn_m1"):
        try:
            pred_res = model_1.predict(m1_vector)
            pred_val = abs(float(np.ravel(pred_res)[0]))
            monthly_val = pred_val / 12.0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''<div class="metric-box"><div class="metric-value">${pred_val:,.2f}</div><div class="metric-label">Predicted Annual Salary</div></div>''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''<div class="metric-box"><div class="metric-value">${monthly_val:,.2f}</div><div class="metric-label">Predicted Monthly Salary</div></div>''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''<div class="metric-box"><div class="metric-value">R² {r2_m1:.2f}</div><div class="metric-label">Model Fit Accuracy</div></div>''', unsafe_allow_html=True)

            st.write("")
            max_gauge = float(max(250000.0, pred_val * 1.3))
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Salary Tier Gauge ($)", 'font': {'size': 18, 'color': '#ffffff'}},
                gauge={
                    'axis': {'range': [None, max_gauge], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "rgba(15, 23, 42, 0.8)",
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, max_gauge * 0.33], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [max_gauge * 0.33, max_gauge * 0.66], 'color': 'rgba(234, 179, 8, 0.3)'},
                        {'range': [max_gauge * 0.66, max_gauge], 'color': 'rgba(34, 197, 94, 0.3)'}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as err:
            st.error(f"Error executing Model 1 prediction: {err}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 5: MODEL 2 - JOB CLASSIFICATION ENGINE
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="tableau-card">', unsafe_allow_html=True)
    st.subheader("🎯 Job Categorization & Classification Engine")
    st.write("Model 2 uses **Logistic Regression** to analyze market positioning across salary tiers.")

    custom_salary = st.number_input(
        "Target Annual Salary ($) for Classification Input",
        min_value=10000,
        max_value=1000000,
        value=110000,
        step=5000
    )

    m2_input_dict = input_data.copy()
    m2_input_dict[target_salary] = custom_salary
    m2_vector = np.array([[m2_input_dict[col] for col in m2_cols]])

    if st.button("⚡ Classify Job Profile (Model 2)", key="btn_m2"):
        try:
            class_pred = model_2.predict(m2_vector)
            class_val = int(np.ravel(class_pred)[0])
            class_probs = model_2.predict_proba(m2_vector)[0]
            classes = [f"Tier {c}" for c in model_2.classes_]

            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown(f'''
                    <div class="metric-box">
                        <div class="metric-label">Predicted Category</div>
                        <div class="metric-value" style="color: #c084fc;">Tier {class_val}</div>
                        <p style="color: #94a3b8; margin-top: 10px; font-size: 0.85rem;">
                            Classified at salary: <b>${custom_salary:,.0f}</b>
                        </p>
                        <p style="color: #38bdf8; font-size: 0.9rem;">Model Accuracy: {acc_m2*100:.1f}%</p>
                    </div>
                ''', unsafe_allow_html=True)

            with col_right:
                fig_bar = go.Figure(go.Bar(
                    x=classes,
                    y=class_probs,
                    marker=dict(color=class_probs, colorscale='Viridis'),
                    text=[f"{p*100:.1f}%" for p in class_probs],
                    textposition='auto'
                ))
                fig_bar.update_layout(
                    title="<b>Class Probability Distribution (Bar Chart)</b>",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#e2e8f0"},
                    yaxis=dict(range=[0, 1]),
                    height=280
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.write("---")
            v1, v2 = st.columns(2)

            with v1:
                fig_line = px.line(
                    x=classes, 
                    y=class_probs, 
                    markers=True,
                    title="<b>Probability Trajectory Line</b>",
                    labels={'x': 'Category Tier', 'y': 'Probability'}
                )
                fig_line.update_traces(line_color='#38bdf8', line_width=3, marker_size=10)
                fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=280)
                st.plotly_chart(fig_line, use_container_width=True)

            with v2:
                tier_counts = df_clean[target_tier].value_counts().reset_index()
                tier_counts.columns = ['Tier', 'Count']
                tier_counts['Tier'] = tier_counts['Tier'].apply(lambda x: f"Tier {x}")

                fig_pie = px.pie(
                    tier_counts, 
                    names='Tier', 
                    values='Count', 
                    hole=0.4,
                    title="<b>Overall Dataset Category Share</b>",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=280)
                st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as err:
            st.error(f"Error executing Model 2 classification: {err}")

    st.markdown('</div>', unsafe_allow_html=True)
