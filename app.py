# ==============================================================================
# LIBRARIES & IMPORTS
# ==============================================================================
import streamlit as st
from streamlit_option_menu import option_menu

# Import all plotting and UI rendering functions from their respective files
from plotting import *
from ui_rendering import *

# ==============================================================================
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Biotech V&V Analytics Toolkit",
    page_icon="🔬"
)

# Centralized CSS for a consistent look and feel
st.markdown("""
<style>
    /* Base Font & Colors */
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
    .st-emotion-cache-16txtl3 { padding: 2rem 1.5rem; }
    .section-header {
        font-weight: 600; color: #0068C9; padding-bottom: 4px;
        border-bottom: 1px solid #E0E0E0; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA DICTIONARIES FOR NAVIGATION
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

# ==============================================================================
# HEADER AND INTRODUCTORY TABS
# ==============================================================================
st.title("🛠️ Biotech V&V Analytics Toolkit")
st.markdown("### An Interactive Guide to Assay Validation, Tech Transfer, and Lifecycle Management")
st.markdown("Welcome! This toolkit is a collection of interactive modules designed to explore the statistical and machine learning methods that form the backbone of a robust V&V, technology transfer, and process monitoring plan.")

tab_intro, tab_timeline, tab_journey = st.tabs(["🚀 The V&V Framework", "📈 Project Timeline", "📖 The Scientist's Journey"])
with tab_intro:
    st.markdown('<h4 class="section-header">The V&V Model: A Strategic Framework</h4>', unsafe_allow_html=True)
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework for ensuring a system meets its intended purpose, from initial requirements to final deployment.")
    st.plotly_chart(plot_v_model(), use_container_width=True)

with tab_timeline:
    st.markdown('<h4 class="section-header">A Typical Project Workflow</h4>', unsafe_allow_html=True)
    st.markdown("This timeline organizes the entire toolkit by its application in a typical project lifecycle. Tools are grouped by the project phase where they provide the most value, and are ordered chronologically within each phase.")
    st.plotly_chart(plot_act_grouped_timeline(), use_container_width=True)

with tab_journey:
    st.header("The Scientist's/Engineer's Journey: A Three-Act Story")
    st.markdown("""The journey from a novel idea to a robust, routine process can be viewed as a three-act story, with each act presenting unique analytical challenges. This toolkit is structured to follow that narrative.""")
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

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    st.markdown("Select a method to explore.")

    all_options_structure = {
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
    
    options = [item for sublist in all_options_structure.values() for item in sublist]
    icons = [ICONS.get(opt, "question-circle") for opt in options]

    try:
        default_idx = options.index(st.session_state.get('method_key', options[0]))
    except ValueError:
        default_idx = 0

    selected_option = option_menu(
        menu_title=None, options=options, icons=icons, menu_icon="cast", default_index=default_idx,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0068C9"},
        }
    )
    st.session_state.method_key = selected_option

# ==============================================================================
# MAIN CONTENT AREA DISPATCHER
# ==============================================================================
method_key = st.session_state.method_key
st.header(f"🔧 {method_key}")

if method_key in PAGE_DISPATCHER:
    PAGE_DISPATCHER[method_key]()
else:
    st.error("Selected module not found. Please select an option from the sidebar.")
    st.session_state.method_key = "Confidence Interval Concept"
    st.rerun()
