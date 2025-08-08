# pages/2_🔧_Tools.py

# ==============================================================================
# LIBRARIES & IMPORTS (Copy all from your original script)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from scipy import stats
from scipy.stats import beta, norm, t
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import shap

# ==============================================================================
# APP CONFIGURATION (Optional, but good practice)
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="V&V Tools",
    page_icon="🔬"
)

# ==============================================================================
# CSS STYLES (Copy from original script)
# ==============================================================================
st.markdown("""
<style>
    /* Same CSS as your original script */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #333;
    }
    .main .block-container {
        padding: 2rem 5rem;
        max-width: 1600px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #F0F2F6; border-radius: 4px 4px 0px 0px;
        padding: 0px 24px; border-bottom: 2px solid transparent; transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; font-weight: 600; border-bottom: 2px solid #0068C9;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E0E0E0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); padding: 15px 20px; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ICONS DICTIONARY (Copy from original script)
# ==============================================================================
ICONS = {
    "Confidence Interval Concept": "arrows-angle-expand", "Core Validation Parameters": "clipboard2-check",
    "Gage R&R / VCA": "rulers", "LOD & LOQ": "search", "Linearity & Range": "graph-up",
    "Non-Linear Regression (4PL/5PL)": "bezier2", "ROC Curve Analysis": "bullseye",
    "Equivalence Testing (TOST)": "arrows-collapse", "Assay Robustness (DOE)": "shield-check",
    "Method Comparison": "people-fill", "Process Stability (SPC)": "activity",
    "Process Capability (Cpk)": "gem", "Tolerance Intervals": "distribute-vertical",
    "Pass/Fail Analysis": "toggles", "Bayesian Inference": "moon-stars-fill",
    "Run Validation (Westgard)": "check2-circle", "Multivariate SPC": "grid-3x3-gap-fill",
    "Small Shift Detection": "graph-up-arrow", "Time Series Analysis": "clock-history",
    "Stability Analysis (Shelf-Life)": "calendar2-week", "Reliability / Survival Analysis": "hourglass-split",
    "Multivariate Analysis (MVA)": "motherboard", "Clustering (Unsupervised)": "bounding-box-circles",
    "Predictive QC (Classification)": "cpu-fill", "Anomaly Detection": "eye-fill",
    "Explainable AI (XAI)": "lightbulb-fill", "Advanced AI Concepts": "robot",
    "Causal Inference": "diagram-3-fill",
}

# ==============================================================================
# ALL HELPER & PLOTTING FUNCTIONS (Copy all `plot_...` functions here)
# ==============================================================================
# @st.cache_data
# def plot_ci_concept(n=30):
#   ...
# (Paste ALL plot functions from your original script here, from plot_ci_concept to plot_causal_inference)
#
# <--- PASTE ALL 21 PLOTTING FUNCTIONS HERE --->
#

# ==============================================================================
# ALL UI RENDERING FUNCTIONS (Copy all `render_...` functions here)
# ==============================================================================
# def render_ci_concept():
#   ...
# (Paste ALL render functions from your original script here, from render_ci_concept to render_causal_inference)
#
# <--- PASTE ALL 21+ RENDERING FUNCTIONS HERE --->
#


# ==============================================================================
# MAIN APP LOGIC FOR TOOLS PAGE
# ==============================================================================

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    st.markdown("Select a method to explore.")

    all_options = {
        "ACT I: FOUNDATION & CHARACTERIZATION": [
            "Confidence Interval Concept", "Core Validation Parameters", "Gage R&R / VCA", "LOD & LOQ", 
            "Linearity & Range", "Non-Linear Regression (4PL/5PL)", "ROC Curve Analysis", 
            "Equivalence Testing (TOST)", "Assay Robustness (DOE)", "Causal Inference"
        ],
        "ACT II: TRANSFER & STABILITY": [
            "Process Stability (SPC)", "Process Capability (Cpk)", "Tolerance Intervals", 
            "Method Comparison", "Pass/Fail Analysis", "Bayesian Inference"
        ],
        "ACT III: LIFECYCLE & PREDICTIVE MGMT": [
            "Run Validation (Westgard)", "Multivariate SPC", "Small Shift Detection", "Time Series Analysis",
            "Stability Analysis (Shelf-Life)", "Reliability / Survival Analysis", "Multivariate Analysis (MVA)",
            "Clustering (Unsupervised)", "Predictive QC (Classification)", "Anomaly Detection",
            "Explainable AI (XAI)", "Advanced AI Concepts"
        ]
    }
    
    options = [item for sublist in all_options.values() for item in sublist]
    icons = [ICONS.get(opt, "question-circle") for opt in options]

    try:
        default_idx = options.index(st.session_state.get('method_key', options[0]))
    except ValueError:
        default_idx = 0

    selected_option = option_menu(
        menu_title=None, options=options, icons=icons,
        menu_icon="cast", default_index=default_idx,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0068C9"},
        }
    )
    st.session_state.method_key = selected_option

# --- Main Content Area Dispatcher ---
if 'method_key' not in st.session_state:
    st.session_state.method_key = "Confidence Interval Concept"

method_key = st.session_state.method_key
st.header(f"🔧 {method_key}")

PAGE_DISPATCHER = {
    # Act I
    "Confidence Interval Concept": render_ci_concept,
    "Core Validation Parameters": render_core_validation_params,
    "Gage R&R / VCA": render_gage_rr,
    "LOD & LOQ": render_lod_loq,
    "Linearity & Range": render_linearity,
    "Non-Linear Regression (4PL/5PL)": render_4pl_regression,
    "ROC Curve Analysis": render_roc_curve,
    "Equivalence Testing (TOST)": render_tost,
    "Assay Robustness (DOE)": render_advanced_doe,
    "Causal Inference": render_causal_inference,
    # Act II
    "Process Stability (SPC)": render_spc_charts,
    "Process Capability (Cpk)": render_capability,
    "Tolerance Intervals": render_tolerance_intervals,
    "Method Comparison": render_method_comparison,
    "Pass/Fail Analysis": render_pass_fail,
    "Bayesian Inference": render_bayesian,
    # Act III
    "Run Validation (Westgard)": render_multi_rule,
    "Multivariate SPC": render_multivariate_spc,
    "Small Shift Detection": render_ewma_cusum,
    "Time Series Analysis": render_time_series_analysis,
    "Stability Analysis (Shelf-Life)": render_stability_analysis,
    "Reliability / Survival Analysis": render_survival_analysis,
    "Multivariate Analysis (MVA)": render_mva_pls,
    "Clustering (Unsupervised)": render_clustering,
    "Predictive QC (Classification)": render_classification_models,
    "Anomaly Detection": render_anomaly_detection,
    "Explainable AI (XAI)": render_xai_shap,
    "Advanced AI Concepts": render_advanced_ai_concepts,
}

if method_key in PAGE_DISPATCHER:
    PAGE_DISPATCHER[method_key]()
else:
    st.error("Selected module not found.")
