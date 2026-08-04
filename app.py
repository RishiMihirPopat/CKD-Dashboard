"""
CKD Clinical Intelligence & Merck R&D Translational Dashboard
Author: Merck Capstone Research Team
Built from CKD analysis & machine learning pipeline
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CKD Translational Intelligence Portal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Theme Styling
st.markdown("""
<style>
    /* Global Base Background & Text */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Lock light-mode presentation and hide toolbar theme controls */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Headers & Subtitles */
    h1, h2, h3, h4, h5, h6 {
        color: #0F4C81 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    p, label, span, li {
        color: #1E293B !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #1E293B !important;
    }
    
    /* Card Container */
    .merck-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #0369A1 !important;
    }

    div[data-testid="stMetricDelta"] > div {
        color: #34D399 !important;
        font-weight: 600 !important;
    }
    
    /* Badges */
    .badge-teal {
        background-color: rgba(14, 165, 233, 0.14);
        color: #0369A1 !important;
        padding: 8px 16px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.4);
        display: inline-block;
        margin: 4px 0;
    }
    
    .badge-alert {
        background-color: rgba(239, 68, 68, 0.14);
        color: #B91C1C !important;
        padding: 8px 16px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(248, 113, 113, 0.4);
        display: inline-block;
        margin: 4px 0;
    }
    
    .badge-success {
        background-color: rgba(34, 197, 94, 0.14);
        color: #15803D !important;
        padding: 8px 16px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(74, 222, 128, 0.4);
        display: inline-block;
        margin: 4px 0;
    }
    
    /* Tabs Navigation Header - Sleek Modern Redesign */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #EAF2FF;
        padding: 10px;
        border-radius: 16px;
        border: 1px solid #BFDBFE;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85), 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        min-height: 50px;
        white-space: pre-wrap;
        border-radius: 12px;
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        border: 1px solid transparent !important;
        padding: 0.35rem 0.9rem !important;
        transition: all 0.24s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background: rgba(255, 255, 255, 0.74) !important;
        border-color: rgba(148, 163, 184, 0.45) !important;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 2px solid #0EA5E9 !important;
        outline-offset: 2px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #DBEAFE !important;
        color: #1D4ED8 !important;
        border-color: #93C5FD !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.35);
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    
    /* Form & Input Elements */
    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }
    
    /* Button Styling */
    .stButton button {
        width: 100%;
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        transition: all 0.3s ease;
        white-space: normal !important;
        height: auto !important;
    }
    
    .stButton button:hover {
        background-color: #0369A1 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4) !important;
    }
    
    /* Slider Styling */
    .stSlider label {
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Selectbox Styling */
    .stSelectbox label {
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Expander Styling */
    .stExpander summary {
        background-color: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #0F4C81 !important;
    }
    
    /* Column Spacing Improvements */
    [data-testid="column"] {
        padding: 8px !important;
    }
    
    /* Subheader Styling */
    .stMarkdown h2 {
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        padding-bottom: 12px !important;
    }
    
    /* Warning/Info Box Styling */
    .stWarning, .stInfo, .stSuccess {
        border-radius: 8px !important;
        padding: 16px !important;
        font-size: 0.95rem !important;
    }
    
    /* Text Input Styling */
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Pipeline Data
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading CKD Pipeline & Analytics...")
def load_data():
    from model_pipeline import run_or_load_pipeline
    return run_or_load_pipeline(force_recompute=False)

bundle = load_data()

df_clean = bundle["df_clean"]
categorical_cols = bundle["categorical_cols"]
numerical_cols = bundle["numerical_cols"]
final_features = bundle["final_features"]
shapiro_df = bundle["shapiro_df"]
chi2_df = bundle["chi2_df"]
mw_df = bundle["mw_df"]
X_train = bundle["X_train"]
X_test = bundle["X_test"]
X_train_scaled = bundle["X_train_scaled"]
X_test_scaled = bundle["X_test_scaled"]
y_train = bundle["y_train"]
y_test = bundle["y_test"]
scaler = bundle["scaler"]
models = bundle["models"]
probs = bundle["probs"]
model_summary_df = bundle["model_summary_df"]
curves = bundle["curves"]
threshold_sweeps = bundle["threshold_sweeps"]

# Map Diagnosis explicitly for clean visualization labeling
df_clean["Diagnosis_Label"] = df_clean["Diagnosis"].map({
    1: "CKD Patient (1524)",
    0: "No CKD Control (135)"
})

COLOR_MAP = {
    "CKD Patient (1524)": "#EF4444",    # Crimson Red for CKD Positive
    "No CKD Control (135)": "#10B981"   # Emerald Green for Healthy Control
}

COLOR_MAP_NUM = {
    1: "#EF4444",  # CKD = Red
    0: "#10B981"   # Control = Green
}

# Helper to apply light plotly template
def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#CBD5E1", zerolinecolor="#94A3B8", title=dict(font=dict(color="#0F172A"))),
        yaxis=dict(gridcolor="#CBD5E1", zerolinecolor="#94A3B8", title=dict(font=dict(color="#0F172A"))),
        legend=dict(font=dict(color="#0F172A"))
    )
    return fig

# ---------------------------------------------------------
# Header & Navigation Banner
# ---------------------------------------------------------
st.markdown("""
<div class="merck-card">
    <div>
        <h1 style="margin: 0; font-size: 2.2rem; color: #0F4C81 !important;">🧬 Chronic Kidney Disease (CKD) Translational Portal</h1>
        <p style="margin: 6px 0 0 0; color: #334155 !important; font-size: 1.05rem;">
            <b style="color: #0F172A;">Capstone Research Project</b> · Biostatistical Hypothesis Testing, Machine Learning Workbench & Live Patient Risk Simulator
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tabs = st.tabs([
    "Executive Overview",
    "Biostatistical Rigor",
    "Machine Learning Benchmark",
    "Patient Risk Simulator"
])

# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================
with tabs[0]:
    st.subheader("📌 Cohort Demographics & Executive Summary")
    
    # Top KPI Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    n_total = len(df_clean)
    n_ckd = (df_clean["Diagnosis"] == 1).sum()
    n_nockd = (df_clean["Diagnosis"] == 0).sum()
    prev_pct = (n_ckd / n_total) * 100
    
    c1.metric("Total Cohort Size", f"{n_total:,}", help="Total number of patients evaluated")
    c2.metric("CKD Prevalence", f"{prev_pct:.1f}%", f"{n_ckd} CKD Cases")
    c3.metric("Clinical Features", f"{len(df_clean.columns)-2}", "33 Num / 19 Cat")
    c4.metric("Sig. Biomarkers", f"{len(final_features)}", "p < 0.05 (Mann-Whitney)")
    c5.metric("Top Predictor", "Serum Creatinine", "Rank-Biserial r=0.74")
    c6.metric("Best Model AUC", f"{model_summary_df['ROC_AUC'].max():.4f}", "Logistic Regression")
    
    st.markdown("---")
    
    # Grid 1: Diagnosis Distribution & Key Demographics
    col_a, col_b = st.columns([1.2, 2])
    
    with col_a:
        st.markdown("### 🎯 Target Class Distribution")
        
        fig_pie = px.pie(
            df_clean,
            names="Diagnosis_Label",
            color="Diagnosis_Label",
            color_discrete_map=COLOR_MAP,
            hole=0.5,
            title="Target Cohort Diagnosis Breakdown"
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=13, color="#FFFFFF"),
            marker=dict(line=dict(color="#E2E8F0", width=2))
        )
        fig_pie.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        apply_plotly_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown(f"""
        <div style="background-color: rgba(14, 165, 233, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 14px; margin-top: 10px;">
            <b style="color: #38BDF8;">Cohort Breakdown</b>:<br>
            • <b style="color:#0F172A;">CKD Positive Patients</b>: <b>{n_ckd}</b> ({prev_pct:.1f}%) — Marked in <span style="color:#EF4444; font-weight:bold;">Crimson Red</span>.<br>
            • <b style="color:#0F172A;">No CKD Controls</b>: <b>{n_nockd}</b> ({100-prev_pct:.1f}%) — Marked in <span style="color:#10B981; font-weight:bold;">Emerald Green</span>.<br><br>
            <b style="color: #38BDF8;">Class Imbalance Handling</b>:<br>
            During ML modeling, <b style="color:#0F172A;">SMOTE</b> is applied inside CV folds to balance decision boundaries without data leakage.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("### 📊 Demographics & Biomarker Distribution Split")
        demo_feature = st.selectbox(
            "Select Demographic or Biomarker to inspect:",
            options=["Age", "BMI", "SystolicBP", "FastingBloodSugar", "HbA1c", "Gender", "Ethnicity", "SocioeconomicStatus"],
            index=0
        )
        
        if demo_feature in numerical_cols:
            fig_hist = px.histogram(
                df_clean,
                x=demo_feature,
                color="Diagnosis_Label",
                color_discrete_map=COLOR_MAP,
                marginal="box",
                barmode="overlay",
                labels={"Diagnosis_Label": "Diagnosis Status"},
                title=f"Distribution of {demo_feature} by CKD Diagnosis"
            )
            apply_plotly_theme(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            fig_cat = px.histogram(
                df_clean,
                x=demo_feature,
                color="Diagnosis_Label",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                labels={"Diagnosis_Label": "Diagnosis Status"},
                title=f"Categorical Count of {demo_feature} vs Diagnosis"
            )
            apply_plotly_theme(fig_cat)
            st.plotly_chart(fig_cat, use_container_width=True)

    # Grid 2: Exploratory Feature Inspection
    st.markdown("### 🔍 Full Dataset Summary Statistics")
    with st.expander("Click to inspect complete 52-column Summary Statistics", expanded=False):
        st.dataframe(
            df_clean.drop(columns=["Diagnosis_Label"]).describe().T.style.format("{:.2f}"),
            use_container_width=True
        )

# =========================================================
# TAB 2: BIOSTATISTICAL RIGOR
# =========================================================
with tabs[1]:
    st.subheader("🔬 Biostatistical Rigor & Hypothesis Testing Suite")
    st.markdown("""
    To establish statistical validity prior to machine learning, three formal hypothesis testing procedures were executed:
    1. **Shapiro-Wilk Normality Test**: Evaluate distributional normality of numerical features.
    2. **Chi-Square Test of Independence (\\(\\chi^2\\))**: Test association between categorical features and CKD diagnosis.
    3. **Mann-Whitney U Test**: Non-parametric comparison of numerical biomarkers between CKD and Non-CKD groups.
    """)
    
    b_tab1, b_tab2, b_tab3, b_tab4 = st.tabs([
        "1. Normality (Shapiro-Wilk)",
        "2. Categorical Association (Chi-Square)",
        "3. Biomarker Significance (Mann-Whitney)",
        "4. Correlation Heatmap"
    ])
    
    # Subtab 1: Shapiro Wilk
    with b_tab1:
        st.markdown("#### Shapiro-Wilk Test Results (Numerical Variables)")
        st.caption("Hypothesis H0: Variable is normally distributed. Reject H0 if p < 0.05.")
        
        col_sw1, col_sw2 = st.columns([2, 1.2])
        with col_sw1:
            st.dataframe(
                shapiro_df.style.map(
                    lambda v: "background-color: rgba(239,68,68,0.35); color: #FFFFFF;" if v == False else "background-color: rgba(34,197,94,0.35); color: #FFFFFF;",
                    subset=["Normal"]
                ),
                use_container_width=True,
                height=380
            )
        with col_sw2:
            selected_qq = st.selectbox("Select feature for Q-Q & Distribution Plot:", options=["SerumCreatinine", "GFR", "Age", "BMI", "SystolicBP", "HbA1c"])
            
            # Q-Q Plot calculation
            data_val = df_clean[selected_qq].dropna()
            (osm, osr), (slope, intercept, r) = stats.probplot(data_val, dist="norm")
            
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data Points", marker=dict(color="#38BDF8")))
            fig_qq.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode="lines", name="Normal Fit", line=dict(color="#F43F5E", dash="dash")))
            fig_qq.update_layout(
                title=f"Q-Q Plot: {selected_qq}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Ordered Values"
            )
            apply_plotly_theme(fig_qq)
            st.plotly_chart(fig_qq, use_container_width=True)

    # Subtab 2: Chi-Square
    with b_tab2:
        st.markdown("#### Chi-Square Test of Independence (\\(\\chi^2\\)) & Cramer's V")
        st.caption("Hypothesis H0: Categorical variable is independent of CKD diagnosis. Decision Rule: Reject H0 if p < 0.05.")
        
        st.dataframe(
            chi2_df.style.map(
                lambda v: "background-color: rgba(239,68,68,0.20); color: #7F1D1D;" if v == True else "background-color: rgba(148,163,184,0.2); color: #334155;",
                subset=["Significant"]
            ),
            use_container_width=True
        )
        
        st.warning("""
        **Key Finding**: 
        **0 out of 19 categorical variables** demonstrated statistically significant association with CKD diagnosis (\\(p \\ge 0.05\\)). 
        Demographic and lifestyle self-reports (such as smoking, gender, ethnicity, socioeconomic status) are insufficient on their own for clinical diagnosis without objective lab biomarkers.
        """)

    # Subtab 3: Mann-Whitney U
    with b_tab3:
        st.markdown("#### Mann-Whitney U Test & Rank-Biserial Correlation Effect Sizes")
        st.caption("Hypothesis H0: Distribution of numerical values between CKD and Non-CKD groups are identical.")
        
        fig_mw = px.bar(
            mw_df.head(15),
            x="Rank_Biserial_R",
            y="Feature",
            orientation="h",
            color="Significant",
            color_discrete_map={True: "#38BDF8", False: "#64748B"},
            title="Top Numerical Biomarkers by Effect Size (Rank-Biserial Correlation r_rb)"
        )
        fig_mw.update_layout(yaxis=dict(autorange="reversed"))
        apply_plotly_theme(fig_mw)
        st.plotly_chart(fig_mw, use_container_width=True)
        
        st.success(f"""
        **Selected Top 10 Features**: Exactly **10 numerical biomarkers** met the statistical significance threshold (\\(p < 0.05\\)).
        These 10 features (`{", ".join(final_features)}`) form the core input feature space for all predictive models.
        """)
        
        st.dataframe(mw_df, use_container_width=True)

    # Subtab 4: Correlation Heatmap
    with b_tab4:
        st.markdown("#### Correlation Heatmap (Top 10 Significant Biomarkers)")
        corr_matrix = df_clean[final_features].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Multicollinearity Check — Correlation Matrix of Significant Features"
        )
        apply_plotly_theme(fig_heatmap)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# =========================================================
# TAB 3: MACHINE LEARNING BENCHMARK
# =========================================================
with tabs[2]:
    st.subheader("🏆 Model Comparison Leaderboard")
    st.caption("Exact leaderboard values from Project_Workbook.ipynb.")

    leaderboard_df = pd.DataFrame([
        ["Logistic Regression (Baseline + SMOTE)", "0.500", "75.6%", "0.20", "0.67 (18/27)", "0.31", "0.96", "0.76 (233/305)", "0.85", "0.58"],
        ["Random Forest (Default)", "0.500", "88.9%", "0.33", "0.37 (10/27)", "0.35", "0.94", "0.93 (285/305)", "0.94", "0.64"],
        ["Random Forest (No-CKD F1 Opt.)", "0.550", "88.6%", "0.36", "0.52 (14/27)", "0.42", "0.96", "0.92 (280/305)", "0.94", "0.68"],
        ["Random Forest (CKD F1 Opt.)", "0.050", "92.2%", "1.00", "0.04 (1/27)", "0.07", "0.92", "1.00 (305/305)", "0.96", "0.52"],
        ["XGBoost (Baseline Default)", "0.500", "88.6%", "0.35", "0.48 (13/27)", "0.41", "0.95", "0.92 (281/305)", "0.94", "0.67"],
        ["XGBoost Baseline (No-CKD F1 Opt.)", "0.340", "91.3%", "0.46", "0.44 (12/27)", "0.45", "0.95", "0.95 (291/305)", "0.95", "0.70"],
        ["XGBoost Baseline (CKD F1 Opt.)", "0.060", "92.5%", "0.57", "0.30 (8/27)", "0.39", "0.94", "0.98 (299/305)", "0.96", "0.68"],
        ["XGBoost Tuned + SMOTE (Default)", "0.500", "92.5%", "0.57", "0.30 (8/27)", "0.39", "0.94", "0.98 (299/305)", "0.96", "0.68"],
        ["XGBoost Tuned + SMOTE (CKD F1 Opt.)", "0.200", "92.8%", "1.00", "0.11 (3/27)", "0.20", "0.93", "1.00 (305/305)", "0.96", "0.58"],
        ["⭐ XGBoost Tuned + SMOTE (Final Pipeline)", "0.670", "91.3%", "0.47", "0.56 (15/27)", "0.51", "0.96", "0.94 (288/305)", "0.95", "0.73"],
    ], columns=[
        "Model Variant",
        "Decision Threshold",
        "Overall Accuracy",
        "No-CKD Precision",
        "No-CKD Recall (Specificity)",
        "No-CKD F1-Score",
        "CKD Precision",
        "CKD Recall (Sensitivity)",
        "CKD F1-Score",
        "Macro F1-Score",
    ])

    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
    
    col_m1, col_m2 = st.columns(2)
    
    MODEL_COLORS = {
        "Logistic Regression": "#38BDF8",      # Cyan
        "Random Forest": "#F59E0B",            # Amber
        "XGBoost (Baseline)": "#A855F7",       # Purple
        "XGBoost (Tuned + SMOTE)": "#10B981"   # Emerald Green
    }
    
    with col_m1:
        st.markdown("### 📈 Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()
        for name, c in curves.items():
            fig_roc.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=name, line=dict(color=MODEL_COLORS.get(name, "#FFFFFF"), width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="#64748B")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)"
        )
        apply_plotly_theme(fig_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_m2:
        st.markdown("### 🎯 Precision-Recall (PR) Curves")
        fig_pr = go.Figure()
        for name, c in curves.items():
            fig_pr.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=name, line=dict(color=MODEL_COLORS.get(name, "#FFFFFF"), width=2.5)))
        fig_pr.update_layout(
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision"
        )
        apply_plotly_theme(fig_pr)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎛️ Interactive Decision Threshold Tuning Workbench")
    
    selected_model_name = st.selectbox("Select Model for Threshold Sweep:", options=list(models.keys()), index=3)
    sweep_df = threshold_sweeps[selected_model_name]
    
    col_t1, col_t2 = st.columns([1.5, 1])
    
    with col_t1:
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Threshold"], y=sweep_df["F1_CKD"], mode="lines", name="CKD F1-Score", line=dict(color="#EF4444", width=3)))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Threshold"], y=sweep_df["F1_NoCKD"], mode="lines", name="No-CKD F1-Score", line=dict(color="#10B981", width=3)))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Threshold"], y=sweep_df["Recall_CKD"], mode="lines", name="CKD Recall (Sensitivity)", line=dict(color="#F59E0B", dash="dash")))
        fig_sweep.update_layout(
            title=f"Threshold Sweep Metrics — {selected_model_name}",
            xaxis_title="Classification Probability Threshold",
            yaxis_title="Score"
        )
        apply_plotly_theme(fig_sweep)
        st.plotly_chart(fig_sweep, use_container_width=True)
        
    with col_t2:
        thresh_val = st.slider("Select Custom Threshold:", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
        
        y_prob_sel = probs[selected_model_name]
        y_pred_sel = (y_prob_sel >= thresh_val).astype(int)
        cm_sel = confusion_matrix(y_test, y_pred_sel)
        tn, fp, fn, tp = cm_sel.ravel()
        
        st.markdown(f"**Confusion Matrix at Threshold = {thresh_val:.2f}**")
        
        cm_df = pd.DataFrame(
            [[f"TN = {tn}", f"FP = {fp}"], [f"FN = {fn}", f"TP = {tp}"]],
            index=["Actual No CKD", "Actual CKD"],
            columns=["Pred No CKD", "Pred CKD"]
        )
        st.table(cm_df)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        st.write(f"• **Sensitivity (Recall CKD)**: {sens:.1%}")
        st.write(f"• **Specificity (Recall No-CKD)**: {spec:.1%}")

# =========================================================
# TAB 4: PATIENT RISK SIMULATOR
# =========================================================
with tabs[3]:
    st.subheader("🩺 Merck Clinical Patient Risk Assessment Simulator")
    st.markdown("Input patient clinical values or load a preset profile to generate live CKD risk inference.")
    
    # Preset Profile Loader Buttons
    st.markdown("### 📋 Load Preset Patient Profiles")
    st.markdown("Select a reference patient profile to populate lab values:")
    
    p_col1, p_col2, p_col3 = st.columns([1, 1, 1], gap="medium")
    
    preset_choice = None
    with p_col1:
        if st.button("🟢 Healthy Control", use_container_width=True, key="btn_healthy"):
            preset_choice = "healthy"
            st.success("✓ Loaded Healthy Control Patient Profile")
    
    with p_col2:
        if st.button("🟡 Moderate Risk (Stage 2/3)", use_container_width=True, key="btn_moderate"):
            preset_choice = "moderate"
            st.warning("⚠ Loaded Moderate Risk Patient Profile")
    
    with p_col3:
        if st.button("🔴 High Risk (Stage 4/5)", use_container_width=True, key="btn_high"):
            preset_choice = "high"
            st.error("⛔ Loaded High Risk Patient Profile")
    
    st.markdown("---")

    # Default baseline values
    defaults = {
        "SerumCreatinine": 0.9,
        "GFR": 95.0,
        "Itching": 0,
        "FastingBloodSugar": 95.0,
        "MuscleCramps": 0,
        "BUNLevels": 14.0,
        "ProteinInUrine": 0,
        "SystolicBP": 120,
        "HbA1c": 5.4,
        "BMI": 23.5
    }
    
    if preset_choice == "healthy":
        defaults = {"SerumCreatinine": 0.8, "GFR": 105.0, "Itching": 0, "FastingBloodSugar": 90.0, "MuscleCramps": 0, "BUNLevels": 12.0, "ProteinInUrine": 0, "SystolicBP": 115, "HbA1c": 5.2, "BMI": 22.0}
    elif preset_choice == "moderate":
        defaults = {"SerumCreatinine": 1.6, "GFR": 52.0, "Itching": 1, "FastingBloodSugar": 125.0, "MuscleCramps": 0, "BUNLevels": 28.0, "ProteinInUrine": 1, "SystolicBP": 138, "HbA1c": 6.8, "BMI": 28.5}
    elif preset_choice == "high":
        defaults = {"SerumCreatinine": 3.8, "GFR": 22.0, "Itching": 1, "FastingBloodSugar": 180.0, "MuscleCramps": 1, "BUNLevels": 58.0, "ProteinInUrine": 1, "SystolicBP": 165, "HbA1c": 8.9, "BMI": 32.0}

    st.markdown("---")
    
    sim_col1, sim_col2 = st.columns([1.3, 1.2], gap="large")
    
    with sim_col1:
        st.markdown("#### 📋 Patient Laboratory Input Panel")
        st.markdown("<small style='color: #94A3B8;'>Adjust patient biomarker values to simulate real-time risk assessment</small>", unsafe_allow_html=True)
        st.markdown("")
        
        # Organized input sections
        with st.container():
            st.markdown("**Kidney Function Markers**")
            in_creat = st.slider("Serum Creatinine (mg/dL)", 0.2, 10.0, float(defaults["SerumCreatinine"]), step=0.1, help="Normal: <1.2 mg/dL")
            in_gfr = st.slider("GFR (mL/min/1.73m²)", 5.0, 150.0, float(defaults["GFR"]), step=1.0, help="Normal: >60 mL/min")
            in_bun = st.slider("BUN Levels (mg/dL)", 5.0, 100.0, float(defaults["BUNLevels"]), step=1.0, help="Normal: 7-20 mg/dL")
        
        st.markdown("")
        with st.container():
            st.markdown("**Metabolic Markers**")
            in_fbs = st.slider("Fasting Blood Sugar (mg/dL)", 60.0, 300.0, float(defaults["FastingBloodSugar"]), step=1.0, help="Normal: 70-100 mg/dL")
            in_hba1c = st.slider("HbA1c (%)", 4.0, 14.0, float(defaults["HbA1c"]), step=0.1, help="Normal: <5.7%")
        
        st.markdown("")
        with st.container():
            st.markdown("**Cardiovascular & Anthropometric**")
            in_sbp = st.slider("Systolic BP (mmHg)", 80, 220, int(defaults["SystolicBP"]), step=1, help="Normal: <120 mmHg")
            in_bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, float(defaults["BMI"]), step=0.5, help="Normal: 18.5-24.9")
        
        st.markdown("")
        with st.container():
            st.markdown("**Clinical Symptoms**")
            c_bool1, c_bool2, c_bool3 = st.columns(3, gap="small")
            with c_bool1:
                in_itching = st.selectbox("Itching", [0, 1], index=int(defaults["Itching"]), format_func=lambda x: "Yes" if x == 1 else "No", label_visibility="collapsed")
                st.caption("Itching")
            with c_bool2:
                in_cramps = st.selectbox("Muscle Cramps", [0, 1], index=int(defaults["MuscleCramps"]), format_func=lambda x: "Yes" if x == 1 else "No", label_visibility="collapsed")
                st.caption("Muscle Cramps")
            with c_bool3:
                in_protein = st.selectbox("Protein in Urine", [0, 1], index=int(defaults["ProteinInUrine"]), format_func=lambda x: "Yes" if x == 1 else "No", label_visibility="collapsed")
                st.caption("Protein in Urine")
    
    # Run Prediction
    patient_df = pd.DataFrame([{
        "SerumCreatinine": in_creat,
        "GFR": in_gfr,
        "Itching": in_itching,
        "FastingBloodSugar": in_fbs,
        "MuscleCramps": in_cramps,
        "BUNLevels": in_bun,
        "ProteinInUrine": in_protein,
        "SystolicBP": in_sbp,
        "HbA1c": in_hba1c,
        "BMI": in_bmi
    }])
    
    patient_scaled = pd.DataFrame(scaler.transform(patient_df), columns=final_features)
    
    simulation_model_names = list(models.keys())
    default_sim_model = "Logistic Regression"
    default_sim_model_idx = simulation_model_names.index(default_sim_model) if default_sim_model in simulation_model_names else 0
    
    with sim_col2:
        st.markdown("#### 🎯 Real-Time Clinical Inference")
        st.markdown("<small style='color: #94A3B8;'>Live CKD risk prediction and stage-based risk categorization</small>", unsafe_allow_html=True)
        selected_sim_model_name = st.selectbox(
            "Select Model for Risk Simulation:",
            options=simulation_model_names,
            index=default_sim_model_idx,
            key="sim_model_selector"
        )
        selected_sim_model = models[selected_sim_model_name]
        prob_ckd = selected_sim_model.predict_proba(patient_scaled)[0, 1]
        st.markdown("")
        
        # Risk Category Badge
        if prob_ckd < 0.20:
            risk_label, risk_color, badge_class = "LOW RISK (Healthy Control)", "#10B981", "badge-success"
        elif prob_ckd < 0.50:
            risk_label, risk_color, badge_class = "MODERATE RISK (Stage 2/3 CKD)", "#F59E0B", "badge-teal"
        elif prob_ckd < 0.80:
            risk_label, risk_color, badge_class = "HIGH RISK (Stage 3/4 CKD)", "#F97316", "badge-alert"
        else:
            risk_label, risk_color, badge_class = "CRITICAL RISK (Stage 4/5 CKD)", "#EF4444", "badge-alert"
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_ckd * 100,
            number={'suffix': "%", 'font': {'color': risk_color, 'size': 42}},
            title={'text': "Predicted CKD Risk Probability", 'font': {'color': "#0F172A", 'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#0F172A"},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [20, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [50, 80], 'color': "rgba(249, 115, 22, 0.2)"},
                    {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        apply_plotly_theme(fig_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; width: 100%; margin-top: 12px;">
            <span class="{badge_class}" style="font-size: 1.05rem; padding: 8px 20px;">
                {risk_label}
            </span>
        </div>
        """, unsafe_allow_html=True)
