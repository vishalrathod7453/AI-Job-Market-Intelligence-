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
from sklearn.metrics import r2_score, accuracy_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & GLASSMORPHISM THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Job Analytics Portal",
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
# 2. DATA PROCESSING & MODEL TRAINING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def process_and_train(df):
    df_proc = df.copy()
    
    # Clean column names
    df_proc.columns = [c.strip().lower().replace(' ', '_') for c in df_proc.columns]

    # Identify continuous salary target column
    salary_col = None
    for col in df_proc.columns:
        if any(keyword in col for keyword in ['salary', 'compensation', 'pay', 'usd']):
            salary_col = col
            break
            
    if salary_col is None:
        num_cols = df_proc.select_dtypes(include=[np.number]).columns
        salary_col = num_cols[0] if len(num_cols) > 0 else df_proc.columns[-1]

    # Fill missing numeric values
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    # Fill missing categorical values
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_proc[col] = df_proc[col].fillna('Unknown')

    # Fit LabelEncoders and retain mapping options
    encoders = {}
    options = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        encoders[col] = le
        options[col] = list(le.classes_)

    # Define classification target tier (quantile binning into 3 tiers if not present)
    tier_col = 'job_tier'
    if 'tier' in df_proc.columns:
        tier_col = 'tier'
    elif 'category' in df_proc.columns:
        tier_col = 'category'
    else:
        df_proc['job_tier'] = pd.qcut(df_proc[salary_col], q=3, labels=[0, 1, 2]).astype(int)

    # Model 1: Salary Regression
    X_m1 = df_proc.drop(columns=[salary_col, tier_col], errors='ignore')
    y_m1 = df_proc[salary_col]

    m1 = LinearRegression()
    m1.fit(X_m1, y_m1)
    r2_m1 = r2_score(y_m1, m1.predict(X_m1))

    # Model 2: Logistic Regression Classification
    X_m2 = df_proc.drop(columns=[tier_col], errors='ignore')
    y_m2 = df_proc[tier_col]

    m2 = LogisticRegression(max_iter=1000)
    m2.fit(X_m2, y_m2)
    acc_m2 = accuracy_score(y_m2, m2.predict(X_m2))

    return df_proc, encoders, options, m1, m2, salary_col, tier_col, list(X_m1.columns), list(X_m2.columns), r2_m1, acc_m2

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("📁 Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Job Dataset (.csv)", type=["csv"])

st.markdown('<h1 class="title-text">💼 AI Job Market & Salary Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem;">Predict compensation and classify job profiles from custom CSV datasets using Machine Learning</p>', unsafe_allow_html=True)
st.write("---")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        df_proc, encoders, options, model_1, model_2, target_salary, target_tier, m1_cols, m2_cols, r2_m1, acc_m2 = process_and_train(raw_df)
        
        st.sidebar.success(f"Loaded: {len(raw_df):,} records")
        st.sidebar.title("📌 Candidate Inputs")

        # Dynamic Sidebar Selectboxes based on uploaded CSV values
        input_data = {}
        for col in m1_cols:
            if col in options:
                selected_val = st.sidebar.selectbox(col.replace('_', ' ').title(), options[col], key=f"sb_{col}")
                input_data[col] = encoders[col].transform([selected_val])[0]
            elif 'year' in col or 'exp' in col:
                input_data[col] = st.sidebar.slider(col.replace('_', ' ').title(), 0, 20, 3, key=f"sl_{col}")
            elif 'month' in col:
                input_data[col] = st.sidebar.slider(col.replace('_', ' ').title(), 1, 12, 6, key=f"sl_{col}")
            elif 'urgency' in col or 'rating' in col:
                input_data[col] = st.sidebar.slider(col.replace('_', ' ').title(), 1, 5, 3, key=f"sl_{col}")
            elif 'opening' in col or 'count' in col:
                input_data[col] = st.sidebar.number_input(col.replace('_', ' ').title(), min_value=1, max_value=500, value=5, key=f"num_{col}")
            elif col.startswith('skill') or df_proc[col].nunique() <= 2:
                input_data[col] = int(st.sidebar.checkbox(col.replace('_', ' ').title(), value=True, key=f"cb_{col}"))
            else:
                input_data[col] = st.sidebar.number_input(col.replace('_', ' ').title(), value=float(df_proc[col].median()), key=f"num_{col}")

        tab_data, tab1, tab2 = st.tabs(["📋 Dataset Overview", "💵 Model 1: Salary Regression", "📊 Model 2: Job Classification"])

        # TAB 0: DATASET OVERVIEW
        with tab_data:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Dataset Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{len(raw_df):,}")
            c2.metric("Total Features", f"{len(raw_df.columns)}")
            c3.metric("Model 1 Fit (R²)", f"{r2_m1:.2f}")
            c4.metric("Model 2 Accuracy", f"{acc_m2*100:.1f}%")
            
            st.write("### Dataset Preview")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Build feature vectors
        m1_vector = np.array([[input_data[col] for col in m1_cols]])

        # TAB 1: MODEL 1
        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔮 Estimated Annual Salary Prediction")
            st.write("Model 1 uses **Linear Regression** trained on the uploaded dataset.")

            if st.button("🚀 Predict Salary (Model 1)", key="btn_m1"):
                try:
                    pred = model_1.predict(m1_vector)
                    pred_val = abs(float(np.ravel(pred)[0]))

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
                                <div class="metric-value">R² {r2_m1:.2f}</div>
                                <div class="metric-label">Model Confidence</div>
                            </div>
                        ''', unsafe_allow_html=True)

                    st.write("")
                    max_g = max(250000.0, pred_val * 1.25)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_val,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Salary Market Indicator", 'font': {'size': 20, 'color': '#ffffff'}},
                        gauge={
                            'axis': {'range': [None, max_g], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': "#00c6ff"},
                            'bgcolor': "rgba(15, 23, 42, 0.8)",
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, max_g * 0.33], 'color': 'rgba(239, 68, 68, 0.3)'},
                                {'range': [max_g * 0.33, max_g * 0.66], 'color': 'rgba(234, 179, 8, 0.3)'},
                                {'range': [max_g * 0.66, max_g], 'color': 'rgba(34, 197, 94, 0.3)'}
                            ],
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"})
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as err:
                    st.error(f"Error executing Model 1 prediction: {err}")

            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 2: MODEL 2
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 Job Categorization & Classification")
            st.write("Model 2 uses **Logistic Regression** to analyze market tier positioning.")

            custom_salary = st.number_input(
                "Target Annual Salary ($) for Classification Input",
                min_value=10000,
                max_value=1000000,
                value=110000,
                step=5000
            )

            # Construct M2 Vector (includes target salary input)
            m2_input_dict = input_data.copy()
            m2_input_dict[target_salary] = custom_salary
            m2_vector = np.array([[m2_input_dict[col] for col in m2_cols]])

            if st.button("⚡ Classify Job Profile (Model 2)", key="btn_m2"):
                try:
                    class_pred = model_2.predict(m2_vector)
                    class_val = int(np.ravel(class_pred)[0])
                    class_probs = model_2.predict_proba(m2_vector)[0]

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        # Fixed string formatting specifier (,.0f)
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

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to process CSV dataset: {e}")
else:
    st.info("👈 Upload your dataset CSV (e.g., `AI Job ...et EDA.csv`) in the sidebar to train models and execute analytics.")
