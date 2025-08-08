# app.py (The new, unified, single-page application)

# ==============================================================================
# LIBRARIES & IMPORTS (All imports are here)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Biotech V&V Analytics Toolkit",
    page_icon="🔬"
)

# ==============================================================================
# CSS STYLES
# ==============================================================================
st.markdown("""
<style>
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
# ALL HELPER & PLOTTING FUNCTIONS
# ==============================================================================
#
# <--- PASTE ALL 21+ PLOTTING FUNCTIONS HERE --->
# (The code is very long, so it's omitted for brevity, but you must paste
#  all your `plot_...` functions here, from `plot_v_model` to `plot_causal_inference`)
#

# ==============================================================================
# ALL UI RENDERING FUNCTIONS
# ==============================================================================

def render_introduction_content():
    """Renders the three introductory sections as a single page."""
    st.title("🛠️ Biotech V&V Analytics Toolkit")
    st.markdown("### An Interactive Guide to Assay Validation, Tech Transfer, and Lifecycle Management")
    st.markdown("Welcome! This toolkit is a collection of interactive modules designed to explore the statistical and machine learning methods that form the backbone of a robust V&V, technology transfer, and process monitoring plan.")
    st.info("#### 👈 Select a tool from the sidebar to explore an interactive module.")
    
    st.header("📖 The Scientist's/Engineer's Journey: A Three-Act Story")
    st.markdown("""The journey from a novel idea to a robust, routine process can be viewed as a three-act story, with each act presenting unique analytical challenges. The toolkit is structured to follow that narrative.""")
    act1, act2, act3 = st.columns(3)
    with act1: 
        st.subheader("Act I: Foundation & Characterization")
        st.markdown("Before a method or process can be trusted, its fundamental capabilities, limitations, and sensitivities must be deeply understood. This is the act of building a solid, data-driven foundation.")
    with act2: 
        st.subheader("Act II: Transfer & Stability")
        st.markdown("Here, the method faces its crucible. It must prove its performance in a new environment—a new lab, a new scale, a new team. This act is about demonstrating stability and equivalence.")
    with act3: 
        st.subheader("Act III: The Guardian (Lifecycle Management)")
        st.markdown("Once live, the journey isn't over. This final act is about continuous guardianship: monitoring process health, detecting subtle drifts, and using advanced analytics to predict and prevent future failures.")
    
    st.divider()

    st.header("🚀 The V&V Model: A Strategic Framework")
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework for ensuring a system meets its intended purpose, from initial requirements to final deployment.")
    # st.plotly_chart(plot_v_model(), use_container_width=True) # Assuming plot_v_model is defined
    
    st.divider()
    
    st.header("📈 Project Workflow")
    st.markdown("This timeline organizes the entire toolkit by its application in a typical project lifecycle. Tools are grouped by the project phase where they provide the most value, and are ordered chronologically within each phase.")
    # st.plotly_chart(plot_act_grouped_timeline(), use_container_width=True) # Assuming plot_act_grouped_timeline is defined

#
# <--- PASTE ALL 21+ TOOL-SPECIFIC RENDERING FUNCTIONS HERE --->
# (The code is very long, so it's omitted for brevity, but you must paste
#  all your `render_...` functions here, from `render_ci_concept` to `render_causal_inference`)
#

# ==============================================================================
# MAIN APP LOGIC AND LAYOUT
# ==============================================================================

# --- Initialize Session State ---
# The default view will be the V&V Framework
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'V&V Strategic Framework'

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    
    all_options = {
        "PROJECT FRAMEWORK": [
            "The Scientist's Journey",
            "V&V Strategic Framework",
            "Project Workflow Timeline"
        ],
        "ACT I: FOUNDATION & CHARACTERIZATION": ["Confidence Interval Concept", "Core Validation Parameters", "Gage R&R / VCA", "LOD & LOQ", "Linearity & Range", "Non-Linear Regression (4PL/5PL)", "ROC Curve Analysis", "Equivalence Testing (TOST)", "Assay Robustness (DOE)", "Causal Inference"],
        "ACT II: TRANSFER & STABILITY": ["Process Stability (SPC)", "Process Capability (Cpk)", "Tolerance Intervals", "Method Comparison", "Pass/Fail Analysis", "Bayesian Inference"],
        "ACT III: LIFECYCLE & PREDICTIVE MGMT": ["Run Validation (Westgard)", "Multivariate SPC", "Small Shift Detection", "Time Series Analysis", "Stability Analysis (Shelf-Life)", "Reliability / Survival Analysis", "Multivariate Analysis (MVA)", "Clustering (Unsupervised)", "Predictive QC (Classification)", "Anomaly Detection", "Explainable AI (XAI)", "Advanced AI Concepts"]
    }

    # Create a button for each tool, which updates the session state on click
    for act_title, act_tools in all_options.items():
        st.subheader(act_title)
        for tool in act_tools:
            if st.button(tool, key=tool, use_container_width=True):
                st.session_state.current_view = tool
                st.rerun()

# --- Main Content Area Dispatcher ---
# This logic checks the session state and decides what to render.
view = st.session_state.current_view

# The list of introductory "tools" that all point to the same rendering function
INTRO_VIEWS = ["The Scientist's Journey", "V&V Strategic Framework", "Project Workflow Timeline"]

if view in INTRO_VIEWS:
    render_introduction_content()
else:
    # Render the selected tool
    st.header(f"🔧 {view}")

    PAGE_DISPATCHER = {
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
        "Process Stability (SPC)": render_spc_charts,
        "Process Capability (Cpk)": render_capability,
        "Tolerance Intervals": render_tolerance_intervals,
        "Method Comparison": render_method_comparison,
        "Pass/Fail Analysis": render_pass_fail,
        "Bayesian Inference": render_bayesian,
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

    if view in PAGE_DISPATCHER:
        PAGE_DISPATCHER[view]()
    else:
        st.error("Error: Could not find the selected tool to render.")
        st.session_state.current_view = 'V&V Strategic Framework'
        st.rerun()
