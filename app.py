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
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ANIMATED GLASSMORPHISM THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Job Analytics & Salary Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
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
# 2. ADVANCED DATA CLEANING & MODEL TRAINING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def process_data_pipeline(df):
    raw_copy = df.copy()
    
    # Standardize column names
    df_clean = df.copy()
    df_clean.columns = [c.strip().lower().replace(' ', '_') for c in df_clean.columns]

    # Data Quality Stats
    initial_rows = len(df_clean)
    initial_nulls = df_clean.isnull().sum().sum()
    duplicate_rows = df_clean.duplicated().sum()
    
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

    # Clean missing values
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in cat_cols:
        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
        df_clean[col] = df_clean[col].fillna(mode_val)

    final_nulls = df_clean.isnull().sum().sum()

    # Create dynamic quantile salary tiers (Tier 0: Entry, Tier 1: Mid, Tier 2: Senior/High)
    tier_col = 'job_tier'
    if 'tier' in df_clean.columns:
        tier_col = 'tier'
    elif 'category' in df_clean.columns:
        tier_col = 'category'
    else:
        df_clean['job_tier'] = pd.qcut(df_clean[salary_col], q=3, labels=[0, 1, 2]).astype(int)

    # Encode categorical features for modeling
    encoders = {}
    options = {}
    df_encoded = df_clean.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        options[col] = list(le.classes_)

    # Model 1: Salary Regression (Linear Regression)
    X_m1 = df_encoded.drop(columns=[salary_col, tier_col], errors='ignore')
    y_m1 = df_encoded[salary_col]

    m1 = LinearRegression()
    m1.fit(X_m1, y_m1)
    r2_m1 = r2_score(y_m1, m1.predict(X_m1))

    # Model 2: Job Classification (Logistic Regression)
    X_m2 = df_encoded.drop(columns=[tier_col], errors='ignore')
    y_m2 = df_encoded[tier_col]

    m2 = LogisticRegression(max_iter=1000)
    m2.fit(X_m2, y_m2)
    acc_m2 = accuracy_score(y_m2, m2.predict(X_m2))

    cleaning_stats = {
        'initial_rows': initial_rows,
        'clean_rows': len(df_clean),
        'duplicates_removed': duplicate_rows,
        'initial_nulls': initial_nulls,
        'final_nulls': final_nulls,
        'salary_col': salary_col,
        'industry_col': industry_col
    }

    return df_clean, df_encoded, encoders, options, m1, m2, salary_col, tier_col, industry_col, list(X_m1.columns), list(X_m2.columns), r2_m1, acc_m2, cleaning_stats

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & CSV FILE UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.title("📁 Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Job Dataset (.csv)", type=["csv"])

st.markdown('<h1 class="title-text">💼 AI Job Market & Salary Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem;">Clean data, analyze compensation trends, predict salaries, and classify job profiles</p>', unsafe_allow_html=True)
st.write("---")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        df_clean, df_encoded, encoders, options, model_1, model_2, target_salary, target_tier, industry_col, m1_cols, m2_cols, r2_m1, acc_m2, clean_stats = process_data_pipeline(raw_df)

        st.sidebar.success(f"Loaded: {clean_stats['clean_rows']:,} records")
        st.sidebar.title("📌 Candidate Profile Inputs")

        # Dynamic sidebar controls matching CSV schema
        input_data = {}
        for col in m1_cols:
            if col in options:
                selected_val = st.sidebar.selectbox(col.replace('_', ' ').title(), options[col], key=f"sb_{col}")
                input_data[col] = encoders[col].transform([selected_val])[0]
            elif any(kw in col for kw in ['year', 'exp', 'experience']):
                input_data[col] = st.sidebar.slider(col.replace('_', ' ').title(), 0, 25, 3, key=f"sl_{col}")
            elif 'month' in col:
                input_data[col] = st.sidebar.slider(col.replace('_', ' ').title(), 1, 12, 6, key=f"sl_{col}")
            elif any(kw in col for kw in ['urgency', 'rating', 'level']):
                input_data[col] = st.sidebar.slider(col.replace('_', ' ').title(), 1, 5, 3, key=f"sl_{col}")
            elif any(kw in col for kw in ['opening', 'count', 'positions']):
                input_data[col] = st.sidebar.number_input(col.replace('_', ' ').title(), min_value=1, max_value=500, value=5, key=f"num_{col}")
            elif col.startswith('skill') or df_clean[col].nunique() <= 2:
                input_data[col] = int(st.sidebar.checkbox(col.replace('_', ' ').title(), value=True, key=f"cb_{col}"))
            else:
                input_data[col] = st.sidebar.number_input(col.replace('_', ' ').title(), value=float(df_clean[col].median()), key=f"num_{col}")

        # Main Navigation Tabs
        tab_clean, tab1, tab2 = st.tabs([
            "🧼 Data Cleaning & Overview", 
            "💵 Model 1: Salary Regression", 
            "📊 Model 2: Job Classification"
        ])

        # -----------------------------------------------------------------------------
        # TAB 0: DATA CLEANING & STATISTICAL OVERVIEW
        # -----------------------------------------------------------------------------
        with tab_clean:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🧹 Data Cleaning Summary & Quality Metrics")
            
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Initial / Clean Rows", f"{clean_stats['initial_rows']:,} / {clean_stats['clean_rows']:,}")
            q2.metric("Duplicates Removed", f"{clean_stats['duplicates_removed']:,}")
            q3.metric("Nulls Imputed", f"{clean_stats['initial_nulls']:,} ➔ {clean_stats['final_nulls']}")
            q4.metric("Target Salary Column", f"`{target_salary}`")
            st.markdown('</div>', unsafe_allow_html=True)

            # Salary Key Performance Indicators
            avg_annual = df_clean[target_salary].mean()
            avg_monthly = avg_annual / 12
            highest_row = df_clean.loc[df_clean[target_salary].idxmax()]
            lowest_row = df_clean.loc[df_clean[target_salary].idxmin()]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("💰 Salary Analytics & Industry Benchmarks")
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Average Annual Salary", f"${avg_annual:,.2f}")
            s2.metric("Average Monthly Salary", f"${avg_monthly:,.2f}")
            s3.metric("Highest Compensation", f"${highest_row[target_salary]:,.2f}")
            s4.metric("Lowest Compensation", f"${lowest_row[target_salary]:,.2f}")

            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("### 🔝 Top 5 Highest Salary Profiles")
                top_5 = df_clean.sort_values(by=target_salary, ascending=False).head(5)
                st.dataframe(top_5, use_container_width=True)

            with col_b:
                st.write("### 🔻 Bottom 5 Lowest Salary Profiles")
                bottom_5 = df_clean.sort_values(by=target_salary, ascending=True).head(5)
                st.dataframe(bottom_5, use_container_width=True)

            # Industry Value Analysis
            if industry_col and industry_col in df_clean.columns:
                st.write("### 🏢 Industry Salary Benchmarks (Most Valuable Industry)")
                ind_df = df_clean.groupby(industry_col)[target_salary].agg(['mean', 'max', 'count']).reset_index()
                ind_df.columns = [industry_col.title(), 'Average Salary', 'Max Salary', 'Job Count']
                ind_df = ind_df.sort_values(by='Average Salary', ascending=False)
                
                fig_ind = px.bar(
                    ind_df, 
                    x=industry_col.title(), 
                    y='Average Salary', 
                    color='Average Salary',
                    color_continuous_scale='Viridis',
                    title="Average Annual Salary by Industry"
                )
                fig_ind.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                st.plotly_chart(fig_ind, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # TAB 1: MODEL 1 - SALARY REGRESSION
        # -----------------------------------------------------------------------------
        m1_vector = np.array([[input_data[col] for col in m1_cols]])

        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔮 Salary Prediction Engine (Linear Regression)")
            st.write("Predict estimated base annual and monthly compensation using candidate profile features.")

            if st.button("🚀 Execute Salary Prediction", key="btn_m1"):
                try:
                    pred_res = model_1.predict(m1_vector)
                    pred_val = abs(float(np.ravel(pred_res)[0]))
                    monthly_val = pred_val / 12

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'''
                            <div class="metric-box">
                                <div class="metric-value">${pred_val:,.2f}</div>
                                <div class="metric-label">Predicted Annual Salary</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'''
                            <div class="metric-box">
                                <div class="metric-value">${monthly_val:,.2f}</div>
                                <div class="metric-label">Predicted Monthly Salary</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'''
                            <div class="metric-box">
                                <div class="metric-value">R² {r2_m1:.2f}</div>
                                <div class="metric-label">Model Accuracy Score</div>
                            </div>
                        ''', unsafe_allow_html=True)

                    st.write("")
                    max_gauge = max(250000.0, pred_val * 1.3)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_val,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Market Salary Tier Gauge", 'font': {'size': 20, 'color': '#ffffff'}},
                        gauge={
                            'axis': {'range': [None, max_gauge], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': "#00c6ff"},
                            'bgcolor': "rgba(15, 23, 42, 0.8)",
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, max_gauge * 0.33], 'color': 'rgba(239, 68, 68, 0.3)'},
                                {'range': [max_gauge * 0.33, max_gauge * 0.66], 'color': 'rgba(234, 179, 8, 0.3)'},
                                {'range': [max_gauge * 0.66, max_gauge], 'color': 'rgba(34, 197, 94, 0.3)'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                    st.plotly_chart(fig_gauge, use_container_width=True)

                except Exception as err:
                    st.error(f"Error executing Model 1 prediction: {err}")

            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------------------------------------------------------
        # TAB 2: MODEL 2 - JOB CLASSIFICATION & VISUALIZATIONS
        # -----------------------------------------------------------------------------
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 Job Categorization & Classification Engine")
            st.write("Model 2 uses **Logistic Regression** to analyze market positioning across salary tiers.")

            custom_salary = st.number_input(
                "Target Annual Salary ($) for Classification Input",
                min_value=10000,
                max_value=1000000,
                value=110000,
                step=5000
            )

            # Construct Model 2 feature vector incorporating target salary
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
                        # Fixed format specifier: ,.0f
                        st.markdown(f'''
                            <div class="metric-box">
                                <div class="metric-label">Predicted Category</div>
                                <div class="metric-value" style="color: #7f00ff;">Tier {class_val}</div>
                                <p style="color: #94a3b8; margin-top: 10px; font-size: 0.85rem;">
                                    Classified at target salary of <b>${custom_salary:,.0f}</b>
                                </p>
                                <p style="color: #00f2fe; font-size: 0.9rem;">Model Accuracy: {acc_m2*100:.1f}%</p>
                            </div>
                        ''', unsafe_allow_html=True)

                    with col_right:
                        # 1. BAR CHART: Class Probability
                        fig_bar = go.Figure(go.Bar(
                            x=classes,
                            y=class_probs,
                            marker=dict(color=class_probs, colorscale='Plasma'),
                            text=[f"{p*100:.1f}%" for p in class_probs],
                            textposition='auto'
                        ))
                        fig_bar.update_layout(
                            title="<b>1. Class Probability Distribution (Bar Chart)</b>",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#e2e8f0"},
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    st.write("---")
                    v1, v2 = st.columns(2)

                    with v1:
                        # 2. LINE CHART: Probability Trajectory
                        fig_line = px.line(
                            x=classes, 
                            y=class_probs, 
                            markers=True,
                            title="<b>2. Class Probability Trend (Line Chart)</b>",
                            labels={'x': 'Category Tier', 'y': 'Probability'}
                        )
                        fig_line.update_traces(line_color='#00c6ff', line_width=3, marker_size=10)
                        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                        st.plotly_chart(fig_line, use_container_width=True)

                    with v2:
                        # 3. PIE CHART: Overall Dataset Tier Breakdown
                        tier_counts = df_clean[target_tier].value_counts().reset_index()
                        tier_counts.columns = ['Tier', 'Count']
                        tier_counts['Tier'] = tier_counts['Tier'].apply(lambda x: f"Tier {x}")

                        fig_pie = px.pie(
                            tier_counts, 
                            names='Tier', 
                            values='Count', 
                            hole=0.4,
                            title="<b>3. Overall Dataset Class Distribution (Pie Chart)</b>",
                            color_discrete_sequence=px.colors.sequential.RdBu
                        )
                        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                        st.plotly_chart(fig_pie, use_container_width=True)

                except Exception as err:
                    st.error(f"Error executing Model 2 classification: {err}")

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing CSV dataset: {e}")
else:
    st.info("👈 Please upload your dataset CSV in the sidebar (e.g., `AI Job ...et EDA.csv`) to begin.")
