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
from scipy.stats import beta, norm, t, f
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import silhouette_score 
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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

# --- RESTORED PLOTTING FUNCTION 1 ---
@st.cache_data
def plot_v_model():
    fig = go.Figure()
    v_model_stages = {
        'URS': {'name': 'User Requirements', 'x': 0, 'y': 5, 'question': 'What does the business/patient/process need?', 'tools': 'Business Case, User Needs Document'},
        'FS': {'name': 'Functional Specs', 'x': 1, 'y': 4, 'question': 'What must the system *do*?', 'tools': 'Assay: Linearity, LOD/LOQ. Instrument: Throughput. Software: User Roles.'},
        'DS': {'name': 'Design Specs', 'x': 2, 'y': 3, 'question': 'How will the system be built/configured?', 'tools': 'Assay: Robustness (DOE). Instrument: Component selection. Software: Architecture.'},
        'BUILD': {'name': 'Implementation', 'x': 3, 'y': 2, 'question': 'Build, code, configure, write SOPs, train.', 'tools': 'N/A (Physical/Code Transfer)'},
        'IQOQ': {'name': 'Installation/Operational Qualification (IQ/OQ)', 'x': 4, 'y': 3, 'question': 'Is the system installed correctly and does it operate as designed?', 'tools': 'Instrument Calibration, Software Unit/Integration Tests.'},
        'PQ': {'name': 'Performance Qualification (PQ)', 'x': 5, 'y': 4, 'question': 'Does the functioning system perform reliably in its environment?', 'tools': 'Gage R&R, Method Comp, Stability, Process Capability (Cpk).'},
        'UAT': {'name': 'User Acceptance / Validation', 'x': 6, 'y': 5, 'question': 'Does the validated system meet the original user need?', 'tools': 'Pass/Fail Analysis, Bayesian Confirmation, Final Report.'}
    }
    verification_color, validation_color = 'rgba(0, 128, 128, 0.9)', 'rgba(0, 104, 201, 0.9)'
    path_keys = ['URS', 'FS', 'DS', 'BUILD', 'IQOQ', 'PQ', 'UAT']
    path_x, path_y = [v_model_stages[p]['x'] for p in path_keys], [v_model_stages[p]['y'] for p in path_keys]
    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(color='darkgrey', width=4), hoverinfo='none'))
    for i, (key, stage) in enumerate(v_model_stages.items()):
        color = verification_color if i < 3 else validation_color if i > 3 else 'grey'
        fig.add_shape(type="rect", x0=stage['x']-0.45, y0=stage['y']-0.3, x1=stage['x']+0.45, y1=stage['y']+0.3, line=dict(color="black", width=2), fillcolor=color)
        fig.add_annotation(x=stage['x'], y=stage['y'], text=f"<b>{stage['name']}</b>", showarrow=False, font=dict(color='white', size=11, family="Arial"))
        fig.add_trace(go.Scatter(x=[stage['x']], y=[stage['y']], mode='markers', marker=dict(color='rgba(0,0,0,0)', size=70), hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"), hoverinfo='text', text=f"<b>{stage['name']}</b><br><br><i>{stage['question']}</i><br><b>Examples / Tools:</b> {stage['tools']}"))
    for i in range(3):
        start_key, end_key = path_keys[i], path_keys[-(i+1)]
        fig.add_shape(type="line", x0=v_model_stages[start_key]['x'], y0=v_model_stages[start_key]['y'], x1=v_model_stages[end_key]['x'], y1=v_model_stages[end_key]['y'], line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"))
    fig.update_layout(title_text='<b>The V&V Model for Technology Transfer (Hover for Details)</b>', title_x=0.5, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 6.6]), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1.4, 5.8]), height=600, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- RESTORED PLOTTING FUNCTION 2 ---
@st.cache_data
def plot_act_grouped_timeline():
    all_tools_data = [
        {'name': 'Assay Robustness (DOE)', 'act': 1, 'year': 1926, 'inventor': 'R.A. Fisher', 'desc': 'Fisher publishes his work on Design of Experiments.'},
        {'name': 'Split-Plot Designs', 'act': 1, 'year': 1930, 'inventor': 'R.A. Fisher & F. Yates', 'desc': 'Specialized DOE for factors that are "hard-to-change".'},
        {'name': 'CI Concept', 'act': 1, 'year': 1937, 'inventor': 'Jerzy Neyman', 'desc': 'Neyman formalizes the frequentist confidence interval.'},
        {'name': 'ROC Curve Analysis', 'act': 1, 'year': 1945, 'inventor': 'Signal Processing Labs', 'desc': 'Developed for radar, now the standard for diagnostic tests.'},
        {'name': 'Variance Components', 'act': 1, 'year': 1950, 'inventor': 'Charles Henderson', 'desc': 'Advanced analysis for complex precision studies.'},
        {'name': 'Assay Robustness (RSM)', 'act': 1, 'year': 1951, 'inventor': 'Box & Wilson', 'desc': 'Box & Wilson develop Response Surface Methodology.'},
        {'name': 'Mixture Designs', 'act': 1, 'year': 1958, 'inventor': 'Henry Scheffé', 'desc': 'Specialized DOE for optimizing formulations and blends.'},
        {'name': 'LOD & LOQ', 'act': 1, 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'desc': 'Currie at NIST formalizes the statistical basis.'},
        {'name': 'Non-Linear Regression', 'act': 1, 'year': 1975, 'inventor': 'Bioassay Field', 'desc': 'Models for sigmoidal curves common in immunoassays.'},
        {'name': 'Core Validation Params', 'act': 1, 'year': 1980, 'inventor': 'ICH / FDA', 'desc': 'Accuracy, Precision, Specificity codified.'},
        {'name': 'Gage R&R', 'act': 1, 'year': 1982, 'inventor': 'AIAG', 'desc': 'AIAG codifies Measurement Systems Analysis (MSA).'},
        {'name': 'Equivalence Testing (TOST)', 'act': 1, 'year': 1987, 'inventor': 'Donald Schuirmann', 'desc': 'Schuirmann proposes TOST for bioequivalence.'},
        {'name': 'Causal Inference', 'act': 1, 'year': 2018, 'inventor': 'Judea Pearl et al.', 'desc': 'Moving beyond correlation to identify root causes.'},
        {'name': 'Process Stability', 'act': 2, 'year': 1924, 'inventor': 'Walter Shewhart', 'desc': 'Shewhart invents the control chart at Bell Labs.'},
        {'name': 'Pass/Fail Analysis', 'act': 2, 'year': 1927, 'inventor': 'Edwin B. Wilson', 'desc': 'Wilson develops a superior confidence interval.'},
        {'name': 'Tolerance Intervals', 'act': 2, 'year': 1942, 'inventor': 'Abraham Wald', 'desc': 'Wald develops intervals to cover a proportion of a population.'},
        {'name': 'Method Comparison', 'act': 2, 'year': 1986, 'inventor': 'Bland & Altman', 'desc': 'Bland & Altman revolutionize method agreement studies.'},
        {'name': 'Process Capability', 'act': 2, 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'desc': 'Motorola popularizes Cpk with the Six Sigma initiative.'},
        {'name': 'Bayesian Inference', 'act': 2, 'year': 1990, 'inventor': 'Metropolis et al.', 'desc': 'Computational methods (MCMC) make Bayes practical.'},
        {'name': 'Multivariate SPC', 'act': 3, 'year': 1931, 'inventor': 'Harold Hotelling', 'desc': 'Hotelling develops the multivariate analog to the t-test.'},
        {'name': 'Small Shift Detection', 'act': 3, 'year': 1954, 'inventor': 'Page (CUSUM) & Roberts (EWMA)', 'desc': 'Charts for faster detection of small process drifts.'},
        {'name': 'Clustering', 'act': 3, 'year': 1957, 'inventor': 'Stuart Lloyd', 'desc': 'Algorithm for finding hidden groups in data.'},
        {'name': 'Predictive QC', 'act': 3, 'year': 1958, 'inventor': 'David Cox', 'desc': 'Cox develops Logistic Regression for binary outcomes.'},
        {'name': 'Reliability Analysis', 'act': 3, 'year': 1958, 'inventor': 'Kaplan & Meier', 'desc': 'Kaplan-Meier estimator for time-to-event data.'},
        {'name': 'Time Series Analysis', 'act': 3, 'year': 1970, 'inventor': 'Box & Jenkins', 'desc': 'Box & Jenkins publish their seminal work on ARIMA models.'},
        {'name': 'Multivariate Analysis', 'act': 3, 'year': 1975, 'inventor': 'Herman Wold', 'desc': 'Partial Least Squares for modeling complex process data.'},
        {'name': 'Run Validation', 'act': 3, 'year': 1981, 'inventor': 'James Westgard', 'desc': 'Westgard publishes his multi-rule QC system.'},
        {'name': 'Stability Analysis', 'act': 3, 'year': 1993, 'inventor': 'ICH', 'desc': 'ICH guidelines formalize statistical shelf-life estimation.'},
        {'name': 'Advanced AI/ML', 'act': 3, 'year': 2017, 'inventor': 'Vaswani, Lundberg et al.', 'desc': 'Transformers and Explainable AI (XAI) emerge.'},
    ]
    all_tools_data.sort(key=lambda x: (x['act'], x['year']))
    act_ranges = {1: (5, 45), 2: (50, 75), 3: (80, 115)}
    tools_by_act = {1: [], 2: [], 3: []}
    for tool in all_tools_data: tools_by_act[tool['act']].append(tool)
    for act_num, tools_in_act in tools_by_act.items():
        start, end = act_ranges[act_num]
        x_coords = np.linspace(start, end, len(tools_in_act))
        for i, tool in enumerate(tools_in_act):
            tool['x'] = x_coords[i]
    y_offsets = [3.0, -3.0, 3.5, -3.5, 2.5, -2.5, 4.0, -4.0, 2.0, -2.0, 4.5, -4.5, 1.5, -1.5]
    for i, tool in enumerate(all_tools_data):
        tool['y'] = y_offsets[i % len(y_offsets)]
    
    fig = go.Figure()
    acts = {
        1: {'name': 'Act I: Foundation', 'color': 'rgba(0, 128, 128, 0.9)', 'boundary': (0, 48)},
        2: {'name': 'Act II: Transfer & Stability', 'color': 'rgba(0, 104, 201, 0.9)', 'boundary': (48, 78)},
        3: {'name': 'Act III: Lifecycle & Predictive', 'color': 'rgba(100, 0, 100, 0.9)', 'boundary': (78, 120)}
    }
    
    for act_info in acts.values():
        x0, x1 = act_info['boundary']
        fig.add_shape(type="rect", x0=x0, y0=-5.0, x1=x1, y1=5.0, line=dict(width=0), fillcolor='rgba(230, 230, 230, 0.7)', layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=7.0, text=f"<b>{act_info['name']}</b>", showarrow=False, font=dict(size=20, color="#555"))

    fig.add_shape(type="line", x0=0, y0=0, x1=120, y1=0, line=dict(color="black", width=3), layer='below')

    for act_num, act_info in acts.items():
        act_tools = [tool for tool in all_tools_data if tool['act'] == act_num]
        fig.add_trace(go.Scatter(x=[tool['x'] for tool in act_tools], y=[tool['y'] for tool in act_tools], mode='markers', marker=dict(size=12, color=act_info['color'], symbol='circle', line=dict(width=2, color='black')), hoverinfo='text', text=[f"<b>{tool['name']} ({tool['year']})</b><br><i>{tool['desc']}</i>" for tool in act_tools], name=act_info['name']))

    for tool in all_tools_data:
        fig.add_shape(type="line", x0=tool['x'], y0=0, x1=tool['x'], y1=tool['y'], line=dict(color='grey', width=1))
        fig.add_annotation(x=tool['x'], y=tool['y'], text=f"<b>{tool['name']}</b><br><i>{tool['inventor']} ({tool['year']})</i>", showarrow=False, yshift=40 if tool['y'] > 0 else -40, font=dict(size=11, color=acts[tool['act']]['color']), align="center")

    fig.update_layout(title_text='<b>The V&V Analytics Toolkit: A Project-Based View</b>', title_font_size=28, title_x=0.5, xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-8, 8]), plot_bgcolor='white', paper_bgcolor='white', height=900, margin=dict(l=20, r=20, t=140, b=20), showlegend=True, legend=dict(title_text="<b>Project Phase</b>", title_font_size=16, font_size=14, orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig

@st.cache_data
def plot_chronological_timeline():
    # Data has been updated with a new 'reason' key for each tool
    all_tools_data = [
        {'name': 'Process Stability', 'year': 1924, 'inventor': 'Walter Shewhart', 'reason': 'The dawn of mass manufacturing (telephones) required new methods for controlling process variation.'},
        {'name': 'Assay Robustness (DOE)', 'year': 1926, 'inventor': 'R.A. Fisher', 'reason': 'To revolutionize agricultural science by efficiently testing multiple factors (fertilizers, varieties) at once.'},
        {'name': 'Pass/Fail Analysis', 'year': 1927, 'inventor': 'Edwin B. Wilson', 'reason': 'To solve the poor performance of the standard binomial confidence interval, especially for small samples.'},
        {'name': 'Split-Plot Designs', 'year': 1930, 'inventor': 'R.A. Fisher & F. Yates', 'reason': 'To solve agricultural experiments with factors that were difficult or expensive to change on a small scale.'},
        {'name': 'Multivariate SPC', 'year': 1931, 'inventor': 'Harold Hotelling', 'reason': 'To generalize the t-test and control charts to monitor multiple correlated variables simultaneously.'},
        {'name': 'CI Concept', 'year': 1937, 'inventor': 'Jerzy Neyman', 'reason': 'A need for rigorous, objective methods in the growing field of mathematical statistics.'},
        {'name': 'Tolerance Intervals', 'year': 1942, 'inventor': 'Abraham Wald', 'reason': 'WWII demanded mass production of interchangeable military parts that would fit together reliably.'},
        {'name': 'ROC Curve Analysis', 'year': 1945, 'inventor': 'Signal Processing Labs', 'reason': 'Developed during WWII to distinguish enemy radar signals from noise, a classic signal detection problem.'},
        {'name': 'Variance Components', 'year': 1950, 'inventor': 'Charles Henderson', 'reason': 'Advances in genetics and complex systems required methods to partition sources of variation.'},
        {'name': 'Assay Robustness (RSM)', 'year': 1951, 'inventor': 'Box & Wilson', 'reason': 'The post-war chemical industry boom created demand for efficient process optimization techniques.'},
        {'name': 'Small Shift Detection', 'year': 1954, 'inventor': 'Page (CUSUM) & Roberts (EWMA)', 'reason': 'Maturing industries required charts more sensitive to small, slow process drifts than Shewhart\'s original design.'},
        {'name': 'Clustering', 'year': 1957, 'inventor': 'Stuart Lloyd', 'reason': 'The advent of early digital computing at Bell Labs made iterative, data-driven grouping algorithms feasible.'},
        {'name': 'Predictive QC', 'year': 1958, 'inventor': 'David Cox', 'reason': 'A need to model binary outcomes (pass/fail, live/die) in a regression framework.'},
        {'name': 'Reliability Analysis', 'year': 1958, 'inventor': 'Kaplan & Meier', 'reason': 'The rise of clinical trials necessitated a formal way to handle \'censored\' data (e.g., patients who survived past the study\'s end).'},
        {'name': 'Mixture Designs', 'year': 1958, 'inventor': 'Henry Scheffé', 'reason': 'To provide a systematic way for chemists and food scientists to optimize recipes and formulations.'},
        {'name': 'LOD & LOQ', 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'reason': 'To create a harmonized, statistically rigorous framework for defining the sensitivity of analytical methods.'},
        {'name': 'Time Series Analysis', 'year': 1970, 'inventor': 'Box & Jenkins', 'reason': 'To provide a comprehensive statistical methodology for forecasting and control in industrial and economic processes.'},
        {'name': 'Non-Linear Regression', 'year': 1975, 'inventor': 'Bioassay Field', 'reason': 'The proliferation of immunoassays (like ELISA) required models for their characteristic S-shaped curves.'},
        {'name': 'Multivariate Analysis', 'year': 1975, 'inventor': 'Herman Wold', 'reason': 'To model "data-rich but theory-poor" systems in social science, later adapted for chemometrics.'},
        {'name': 'Core Validation Params', 'year': 1980, 'inventor': 'ICH / FDA', 'reason': 'Globalization of the pharmaceutical industry required harmonized standards for drug approval.'},
        {'name': 'Run Validation', 'year': 1981, 'inventor': 'James Westgard', 'reason': 'The automation of clinical labs demanded a more sensitive, diagnostic system for daily quality control.'},
        {'name': 'Gage R&R', 'year': 1982, 'inventor': 'AIAG', 'reason': 'The US auto industry, facing a quality crisis, needed to formalize the analysis of their measurement systems.'},
        {'name': 'Method Comparison', 'year': 1986, 'inventor': 'Bland & Altman', 'reason': 'A direct response to the widespread misuse of correlation for comparing clinical measurement methods.'},
        {'name': 'Process Capability', 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'reason': 'The Six Sigma quality revolution at Motorola popularized a simple metric to quantify process capability.'},
        {'name': 'Equivalence Testing (TOST)', 'year': 1987, 'inventor': 'Donald Schuirmann', 'reason': 'The rise of the generic drug industry created a regulatory need to statistically *prove* equivalence.'},
        {'name': 'Bayesian Inference', 'year': 1990, 'inventor': 'Metropolis et al.', 'reason': 'The explosion in computing power made simulation-based methods (MCMC) practical, unlocking Bayesian inference.'},
        {'name': 'Stability Analysis', 'year': 1993, 'inventor': 'ICH', 'reason': 'To harmonize global pharmaceutical regulations for determining a product\'s shelf-life.'},
        {'name': 'Advanced AI/ML', 'year': 2017, 'inventor': 'Vaswani, Lundberg et al.', 'reason': 'The Deep Learning revolution created powerful but opaque "black box" models, necessitating methods to explain them (XAI).'},
        {'name': 'Causal Inference', 'year': 2018, 'inventor': 'Judea Pearl et al.', 'reason': 'The limitations of purely predictive models spurred a "causal revolution" to answer "why" questions.'},
    ]

    all_tools_data.sort(key=lambda x: x['year'])
    y_offsets = [3.0, -3.0, 3.5, -3.5, 2.5, -2.5, 4.0, -4.0, 2.0, -2.0, 4.5, -4.5, 1.5, -1.5]
    for i, tool in enumerate(all_tools_data):
        tool['y'] = y_offsets[i % len(y_offsets)]
    
    fig = go.Figure()
    eras = {
        'The Foundations (1920-1949)': {'color': 'rgba(0, 128, 128, 0.7)', 'boundary': (1920, 1949)},
        'Post-War & Industrial Boom (1950-1979)': {'color': 'rgba(0, 104, 201, 0.7)', 'boundary': (1950, 1979)},
        'The Quality Revolution (1980-1999)': {'color': 'rgba(100, 0, 100, 0.7)', 'boundary': (1980, 1999)},
        'The AI & Data Era (2000-Present)': {'color': 'rgba(214, 39, 40, 0.7)', 'boundary': (2000, 2025)}
    }
    
    for era_name, era_info in eras.items():
        x0, x1 = era_info['boundary']
        fig.add_shape(type="rect", x0=x0, y0=-5.5, x1=x1, y1=5.5, line=dict(width=0), fillcolor=era_info['color'], opacity=0.15, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=6.5, text=f"<b>{era_name}</b>", showarrow=False, font=dict(size=18, color=era_info['color']))

    fig.add_shape(type="line", x0=1920, y0=0, x1=2025, y1=0, line=dict(color="black", width=3), layer='below')

    for tool in all_tools_data:
        x_coord = tool['year']
        y_coord = tool['y']
        
        tool_color = 'grey'
        for era_info in eras.values():
            if era_info['boundary'][0] <= x_coord <= era_info['boundary'][1]:
                tool_color = era_info['color']
                break

        fig.add_trace(go.Scatter(
            x=[x_coord], y=[y_coord], mode='markers',
            marker=dict(size=12, color=tool_color, line=dict(width=2, color='black')),
            # FIX: Updated the hover text to include the new 'reason'
            hoverinfo='text', text=f"<b>{tool['name']} ({tool['year']})</b><br><i>Inventor(s): {tool['inventor']}</i><br><br><b>Reason:</b> {tool['reason']}"
        ))
        fig.add_shape(type="line", x0=x_coord, y0=0, x1=x_coord, y1=y_coord, line=dict(color='grey', width=1))
        fig.add_annotation(
            x=x_coord, y=y_coord, text=f"<b>{tool['name']}</b>",
            showarrow=False, yshift=25 if y_coord > 0 else -25, font=dict(size=11, color=tool_color), align="center"
        )

    fig.update_layout(
        title_text='<b>A Chronological Timeline of V&V Analytics</b>',
        title_font_size=28, title_x=0.5,
        xaxis=dict(range=[1920, 2025], showgrid=True),
        yaxis=dict(visible=False, range=[-8, 8]),
        plot_bgcolor='white', paper_bgcolor='white',
        height=700, margin=dict(l=20, r=20, t=100, b=20),
        showlegend=False
    )
    return fig

@st.cache_data
def create_toolkit_conceptual_map():
    """Creates a visually superior, non-overlapping conceptual map with the correct aesthetics."""
    
    structure = {
        'Foundational Statistics': ['Statistical Inference', 'Regression Models'],
        'Process & Quality Control': ['Measurement Systems Analysis', 'Statistical Process Control', 'Validation & Lifecycle'],
        'Advanced Analytics (ML/AI)': ['Predictive Modeling', 'Unsupervised Learning']
    }
    sub_structure = {
        'Statistical Inference': ['Confidence Interval Concept', 'Equivalence Testing (TOST)', 'Bayesian Inference', 'ROC Curve Analysis'],
        'Regression Models': ['Linearity & Range', 'Non-Linear Regression (4PL/5PL)', 'Stability Analysis (Shelf-Life)', 'Time Series Analysis'],
        'Measurement Systems Analysis': ['Gage R&R / VCA', 'Method Comparison'],
        'Statistical Process Control': ['Process Stability (SPC)', 'Small Shift Detection', 'Multivariate SPC'],
        'Validation & Lifecycle': ['Process Capability (Cpk)', 'Tolerance Intervals', 'Reliability / Survival Analysis'],
        'Predictive Modeling': ['Predictive QC (Classification)', 'Explainable AI (XAI)', 'Multivariate Analysis (MVA)'],
        'Unsupervised Learning': ['Anomaly Detection', 'Clustering (Unsupervised)']
    }
    tool_origins = {
        'Confidence Interval Concept': 'Statistics', 'Equivalence Testing (TOST)': 'Biostatistics', 'Bayesian Inference': 'Statistics', 'ROC Curve Analysis': 'Statistics',
        'Linearity & Range': 'Statistics', 'Non-Linear Regression (4PL/5PL)': 'Biostatistics', 'Stability Analysis (Shelf-Life)': 'Biostatistics', 'Time Series Analysis': 'Statistics',
        'Gage R&R / VCA': 'Industrial Quality Control', 'Method Comparison': 'Biostatistics',
        'Process Stability (SPC)': 'Industrial Quality Control', 'Small Shift Detection': 'Industrial Quality Control', 'Multivariate SPC': 'Industrial Quality Control',
        'Process Capability (Cpk)': 'Industrial Quality Control', 'Tolerance Intervals': 'Statistics', 'Reliability / Survival Analysis': 'Biostatistics',
        'Predictive QC (Classification)': 'Data Science / ML', 'Explainable AI (XAI)': 'Data Science / ML', 'Multivariate Analysis (MVA)': 'Data Science / ML',
        'Anomaly Detection': 'Data Science / ML', 'Clustering (Unsupervised)': 'Data Science / ML'
    }
    origin_colors = {
        'Statistics': '#1f77b4', 'Biostatistics': '#2ca02c',
        'Industrial Quality Control': '#ff7f0e', 'Data Science / ML': '#d62728',
        'Structure': '#6A5ACD' # This is the purple color
    }

    nodes = {}
    
    # Algorithmic Layout
    vertical_spacing = 2.2
    all_tools_flat = [tool for sublist in sub_structure.values() for tool in sublist]
    y_coords = np.linspace(len(all_tools_flat) * vertical_spacing, -len(all_tools_flat) * vertical_spacing, len(all_tools_flat))
    x_positions = [4, 5]
    for i, tool_key in enumerate(all_tools_flat):
        nodes[tool_key] = {'x': x_positions[i % 2], 'y': y_coords[i], 'name': tool_key, 'short': tool_key.replace(' (', '<br>('), 'origin': tool_origins.get(tool_key)}

    for l2_key, l3_keys in sub_structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l3_keys]
        nodes[l2_key] = {'x': 2.5, 'y': np.mean(child_ys), 'name': l2_key, 'short': l2_key.replace(' ', '<br>'), 'origin': 'Structure'}

    for l1_key, l2_keys in structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l2_keys]
        nodes[l1_key] = {'x': 1, 'y': np.mean(child_ys), 'name': l1_key, 'short': l1_key.replace(' ', '<br>'), 'origin': 'Structure'}

    nodes['CENTER'] = {'x': -0.5, 'y': 0, 'name': 'V&V Analytics Toolkit', 'short': 'V&V Analytics<br>Toolkit', 'origin': 'Structure'}

    fig = go.Figure()

    # Draw Edges using Shapes
    all_edges = [('CENTER', l1) for l1 in structure.keys()] + \
                [(l1, l2) for l1, l2s in structure.items() for l2 in l2s] + \
                [(l2, l3) for l2, l3s in sub_structure.items() for l3 in l3s]
    
    for start_key, end_key in all_edges:
        fig.add_shape(type="line",
            x0=nodes[start_key]['x'], y0=nodes[start_key]['y'],
            x1=nodes[end_key]['x'], y1=nodes[end_key]['y'],
            line=dict(color="lightgrey", width=1.5)
        )
    
    # Aggregate data by origin for clean legend plotting
    data_by_origin = {name: {'x': [], 'y': [], 'short': [], 'full': [], 'size': [], 'font_size': []} for name in origin_colors.keys()}
    
    # Define sizes for different node levels
    size_map = {'CENTER': 150, 'Level1': 130, 'Level2': 110, 'Tool': 90}
    font_map = {'CENTER': 16, 'Level1': 14, 'Level2': 12, 'Tool': 11}

    for key, data in nodes.items():
        if key == 'CENTER': level = 'CENTER'
        elif key in structure: level = 'Level1'
        elif key in sub_structure: level = 'Level2'
        else: level = 'Tool'
        
        data_by_origin[data['origin']]['x'].append(data['x'])
        data_by_origin[data['origin']]['y'].append(data['y'])
        data_by_origin[data['origin']]['short'].append(data['short'])
        data_by_origin[data['origin']]['full'].append(data['name'])
        data_by_origin[data['origin']]['size'].append(size_map[level])
        data_by_origin[data['origin']]['font_size'].append(font_map[level])
        
    # Draw one trace per origin for a clean legend
    for origin_name, data in data_by_origin.items():
        if not data['x']: continue
        is_structure = origin_name == 'Structure'
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], text=data['short'],
            mode='markers+text', textposition="middle center",
            marker=dict(
                size=data['size'],
                color=origin_colors[origin_name],
                symbol='circle', # FIX: All nodes are now circles
                line=dict(width=2, color='black')
            ),
            textfont=dict(
                size=data['font_size'], # FIX: Use dynamic font sizes
                color='white', # FIX: All text is white
                family="Arial"
            ),
            hovertext=[name.replace('<br>', ' ') for name in data['short']], hoverinfo='text',
            name=origin_name
        ))

    fig.update_layout(
        title_text='<b>Conceptual Map of the V&V Analytics Toolkit</b>',
        showlegend=True,
        legend=dict(title="<b>Tool Origin</b>", x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        xaxis=dict(visible=False, range=[-1, 6]),
        yaxis=dict(visible=False, range=[-28, 28]),
        height=2400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#f0f2f6'
    )
    return fig

@st.cache_data
def plot_ci_concept(n=30):
    """
    Generates plots for the confidence interval concept module.
    """
    np.random.seed(42)
    pop_mean, pop_std = 100, 15
    
    # --- Plot 1: Population vs. Sampling Distribution ---
    x = np.linspace(pop_mean - 4*pop_std, pop_mean + 4*pop_std, 400)
    pop_dist = norm.pdf(x, pop_mean, pop_std)
    
    sampling_dist_std = pop_std / np.sqrt(n)
    sampling_dist = norm.pdf(x, pop_mean, sampling_dist_std)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=pop_dist, fill='tozeroy', name='Population Distribution', line=dict(color='skyblue')))
    fig1.add_trace(go.Scatter(x=x, y=sampling_dist, fill='tozeroy', name=f'Sampling Distribution (n={n})', line=dict(color='orange')))
    fig1.add_vline(x=pop_mean, line=dict(color='black', dash='dash'), annotation_text="True Mean (μ)")
    fig1.update_layout(title=f"<b>Population vs. Sampling Distribution of the Mean (n={n})</b>", showlegend=True, legend=dict(x=0.01, y=0.99))

    # --- Plot 2: CI Simulation ---
    n_sims = 1000
    samples = np.random.normal(pop_mean, pop_std, size=(n_sims, n))
    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1, ddof=1)
    
    # Using t-distribution for CIs as is proper
    t_crit = t.ppf(0.975, df=n-1)
    margin_of_error = t_crit * sample_stds / np.sqrt(n)
    
    ci_lowers = sample_means - margin_of_error
    ci_uppers = sample_means + margin_of_error
    
    capture_mask = (ci_lowers <= pop_mean) & (ci_uppers >= pop_mean)
    capture_count = np.sum(capture_mask)
    avg_width = np.mean(ci_uppers - ci_lowers)
    
    fig2 = go.Figure()
    # Plot first 100 CIs for visualization
    for i in range(min(n_sims, 100)):
        color = 'blue' if capture_mask[i] else 'red'
        fig2.add_trace(go.Scatter(x=[ci_lowers[i], ci_uppers[i]], y=[i, i], mode='lines', line=dict(color=color, width=2), showlegend=False))
        fig2.add_trace(go.Scatter(x=[sample_means[i]], y=[i], mode='markers', marker=dict(color=color, size=4), showlegend=False))

    fig2.add_vline(x=pop_mean, line=dict(color='black', dash='dash'), annotation_text="True Mean (μ)")
    fig2.update_layout(title=f"<b>{min(n_sims, 100)} Simulated 95% Confidence Intervals</b>", yaxis_visible=False)
    
    return fig1, fig2, capture_count, n_sims, avg_width


def plot_core_validation_params(bias_pct=1.5, repeat_cv=1.5, intermed_cv=2.5, interference_effect=8.0):
    """
    Generates dynamic plots for the core validation module based on user inputs.
    """
    # --- 1. Accuracy (Bias) Data ---
    np.random.seed(42)
    true_values = np.array([50, 100, 150])
    # The mean of the measured data is now controlled by the bias slider
    measured_data = {
        val: np.random.normal(val * (1 + bias_pct / 100), val * 0.025, 10) for val in true_values
    }
    df_accuracy = pd.DataFrame(measured_data)
    df_accuracy = df_accuracy.melt(var_name='True Value', value_name='Measured Value')
    
    fig1 = px.box(df_accuracy, x='True Value', y='Measured Value', 
                  title='<b>1. Accuracy & Bias Evaluation</b>',
                  points='all', color_discrete_sequence=['#1f77b4'])
    for val in true_values:
        fig1.add_hline(y=val, line_dash="dash", line_color="black", annotation_text=f"True Value={val}", annotation_position="bottom right")
    fig1.update_layout(xaxis_title="True (Nominal) Concentration", yaxis_title="Measured Concentration")

    # --- 2. Precision Data ---
    np.random.seed(123)
    # The standard deviation is now controlled by the precision sliders (%CV)
    repeatability_std = 100 * (repeat_cv / 100)
    intermed_std = 100 * (intermed_cv / 100)
    
    repeatability = np.random.normal(100, repeatability_std, 30)
    inter_precision = np.random.normal(100, intermed_std, 30)
    
    df_precision = pd.concat([
        pd.DataFrame({'value': repeatability, 'condition': 'Repeatability'}),
        pd.DataFrame({'value': inter_precision, 'condition': 'Intermediate Precision'})
    ])
    fig2 = px.violin(df_precision, x='condition', y='value', box=True, points="all",
                     title='<b>2. Precision: Repeatability vs. Intermediate Precision</b>',
                     labels={'value': 'Measured Value', 'condition': 'Experimental Condition'})
    
    # --- 3. Specificity Data ---
    np.random.seed(2023)
    analyte = np.random.normal(1.0, 0.05, 15)
    matrix = np.random.normal(0.02, 0.01, 15)
    # The signal of the combined sample is now controlled by the interference slider
    analyte_interference = analyte * (1 + interference_effect / 100)
    
    df_specificity = pd.DataFrame({
        'Analyte Only': analyte,
        'Matrix Blank': matrix,
        'Analyte + Interferent': analyte_interference
    }).melt(var_name='Sample Type', value_name='Signal Response')

    fig3 = px.box(df_specificity, x='Sample Type', y='Signal Response', points='all',
                  title='<b>3. Specificity & Interference Study</b>')
    fig3.update_layout(xaxis_title="Sample Composition", yaxis_title="Assay Signal (e.g., Absorbance)")

    return fig1, fig2, fig3
    
# The @st.cache_data decorator is removed to allow for dynamic regeneration.
def plot_gage_rr(part_sd=5.0, repeatability_sd=1.5, operator_sd=0.75):
    """
    Generates dynamic plots for the Gage R&R module based on user inputs.
    """
    np.random.seed(10)
    n_operators, n_samples, n_replicates = 3, 10, 3
    operators = ['Alice', 'Bob', 'Charlie']
    
    # Generate true part values based on the part_sd slider
    true_part_values = np.random.normal(100, part_sd, n_samples)
    
    # Generate operator biases based on the operator_sd slider
    operator_biases = np.random.normal(0, operator_sd, n_operators)
    operator_bias_map = {op: bias for op, bias in zip(operators, operator_biases)}
    
    data = []
    for op_idx, operator in enumerate(operators):
        for sample_idx, true_value in enumerate(true_part_values):
            # Generate measurements using the operator bias and repeatability_sd slider
            measurements = np.random.normal(true_value + operator_bias_map[operator], repeatability_sd, n_replicates)
            for m_idx, m in enumerate(measurements):
                data.append([operator, f'Part_{sample_idx+1}', m, m_idx + 1])
    
    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement', 'Replicate'])
    
    # Perform ANOVA and calculate variance components
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    ms_operator = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
    ms_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
    ms_interaction = anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df']
    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    
    var_repeatability = ms_error
    var_operator = max(0, (ms_operator - ms_interaction) / (n_samples * n_replicates))
    var_interaction = max(0, (ms_interaction - ms_error) / n_replicates)
    var_reproducibility = var_operator + var_interaction
    var_part = max(0, (ms_part - ms_interaction) / (n_operators * n_replicates))
    var_rr = var_repeatability + var_reproducibility
    var_total = var_rr + var_part
    
    pct_rr = (var_rr / var_total) * 100 if var_total > 0 else 0
    pct_part = (var_part / var_total) * 100 if var_total > 0 else 0
    
    # Plotting (largely the same, just consumes the dynamic data)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], row_heights=[0.5, 0.5], specs=[[{"rowspan": 2}, {}], [None, {}]], subplot_titles=("<b>Variation by Part & Operator</b>", "<b>Overall Variation by Operator</b>", "<b>Variance Contribution</b>"))
    fig_box = px.box(df, x='Part', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_box.data: fig.add_trace(trace, row=1, col=1)
    for i, operator in enumerate(operators):
        operator_df = df[df['Operator'] == operator]; part_means = operator_df.groupby('Part')['Measurement'].mean()
        fig.add_trace(go.Scatter(x=part_means.index, y=part_means.values, mode='lines', line=dict(width=2), name=f'{operator} Mean', showlegend=False, marker_color=fig_box.data[i].marker.color), row=1, col=1)
    fig_op_box = px.box(df, x='Operator', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_op_box.data: fig.add_trace(trace, row=1, col=2)
    fig.add_trace(go.Bar(x=['% Gage R&R', '% Part Variation'], y=[pct_rr, pct_part], marker_color=['salmon', 'skyblue'], text=[f'{pct_rr:.1f}%', f'{pct_part:.1f}%'], textposition='auto'), row=2, col=2)
    fig.add_hline(y=10, line_dash="dash", line_color="darkgreen", annotation_text="Acceptable < 10%", annotation_position="bottom right", row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="darkorange", annotation_text="Unacceptable > 30%", annotation_position="top right", row=2, col=2)
    fig.update_layout(title_text='<b>Gage R&R Study: ANOVA Method</b>', title_x=0.5, height=800, boxmode='group', showlegend=True); fig.update_xaxes(tickangle=45, row=1, col=1)
    
    return fig, pct_rr, pct_part

# The @st.cache_data decorator is removed to allow for dynamic regeneration.
def plot_lod_loq(slope=0.02, baseline_sd=0.01):
    """
    Generates dynamic plots for the LOD & LOQ module based on user inputs.
    """
    np.random.seed(3)
    
    # --- Signal Distribution Plot ---
    # The noise level is now controlled by the baseline_sd slider
    blanks_dist = np.random.normal(0.05, baseline_sd, 20)
    low_conc_dist = np.random.normal(0.05 + 5 * slope, baseline_sd * 1.5, 20) # Low conc at 5 units
    
    df_dist = pd.concat([
        pd.DataFrame({'Signal': blanks_dist, 'Sample Type': 'Blank'}), 
        pd.DataFrame({'Signal': low_conc_dist, 'Sample Type': 'Low Concentration'})
    ])
    
    # --- Low-Level Calibration Curve ---
    concentrations = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 5, 5, 5, 10, 10, 10])
    # The signal response and noise are now controlled by the sliders
    signals = 0.05 + slope * concentrations + np.random.normal(0, baseline_sd, len(concentrations))
    
    df_cal = pd.DataFrame({'Concentration': concentrations, 'Signal': signals})
    
    # Fit the model to the dynamically generated data
    X = sm.add_constant(df_cal['Concentration'])
    model = sm.OLS(df_cal['Signal'], X).fit()
    
    # Use the model's parameters to calculate LOD & LOQ
    # Note: We use the input baseline_sd for the calculation as it's the "true" noise,
    # which is more stable than the model's estimate from this small dataset.
    # The slope from the model is a good estimate of sensitivity.
    fit_slope = model.params['Concentration'] if model.params['Concentration'] > 0.001 else 0.001
    
    LOD = (3.3 * baseline_sd) / fit_slope
    LOQ = (10 * baseline_sd) / fit_slope
    
    # --- Plotting ---
    fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>Signal Distribution at Low End</b>", "<b>Low-Level Calibration Curve</b>"), vertical_spacing=0.2)
    
    fig_violin = px.violin(df_dist, x='Sample Type', y='Signal', color='Sample Type', box=True, points="all", color_discrete_map={'Blank': 'skyblue', 'Low Concentration': 'lightgreen'})
    for trace in fig_violin.data: fig.add_trace(trace, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_cal['Concentration'], y=df_cal['Signal'], mode='markers', name='Calibration Points', marker=dict(color='darkblue', size=8)), row=2, col=1)
    x_range = np.linspace(0, df_cal['Concentration'].max(), 100)
    y_range = model.predict(sm.add_constant(x_range))
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Regression Line', line=dict(color='red', dash='dash')), row=2, col=1)
    
    fig.add_vline(x=LOD, line_dash="dot", line_color="orange", row=2, col=1, annotation_text=f"<b>LOD = {LOD:.2f}</b>", annotation_position="top")
    fig.add_vline(x=LOQ, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"<b>LOQ = {LOQ:.2f}</b>", annotation_position="top")
    
    fig.update_layout(title_text='<b>Assay Sensitivity Analysis: LOD & LOQ</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=1, col=1)
    fig.update_xaxes(title_text="Sample Type", row=1, col=1)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=2, col=1)
    fig.update_xaxes(title_text="Concentration (ng/mL)", row=2, col=1)
    
    return fig, LOD, LOQ
    
# The @st.cache_data decorator is removed to allow for dynamic regeneration.
def plot_linearity(curvature=-1.0, random_error=1.0, proportional_error=2.0):
    """
    Generates dynamic plots for the Linearity module based on user inputs.
    """
    np.random.seed(42)
    nominal = np.array([10, 25, 50, 100, 150, 200, 250])
    
    # Simulate data based on sliders
    # Curvature term: a negative value creates an S-curve typical of saturation
    curvature_effect = curvature * (nominal / 150)**3
    
    # Error term: combination of constant random error and error that grows with concentration
    error = np.random.normal(0, random_error + nominal * (proportional_error / 100))
    
    measured = nominal + curvature_effect + error
    
    # Fit OLS model to the dynamic data
    X = sm.add_constant(nominal)
    model = sm.OLS(measured, X).fit()
    residuals = model.resid
    recovery = (measured / nominal) * 100
    
    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("<b>Linearity Plot</b>", "<b>Residual Plot</b>", "<b>Recovery Plot</b>"),
        vertical_spacing=0.2
    )
    
    # Linearity Plot
    fig.add_trace(go.Scatter(x=nominal, y=measured, mode='markers', name='Measured Values', marker=dict(size=10, color='blue'), hovertemplate="Nominal: %{x}<br>Measured: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominal, y=model.predict(X), mode='lines', name='Best Fit Line', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1)
    
    # Residual Plot
    fig.add_trace(go.Scatter(x=nominal, y=residuals, mode='markers', name='Residuals', marker=dict(size=10, color='green'), hovertemplate="Nominal: %{x}<br>Residual: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    
    # Recovery Plot
    fig.add_trace(go.Scatter(x=nominal, y=recovery, mode='lines+markers', name='Recovery', line=dict(color='purple'), marker=dict(size=10), hovertemplate="Nominal: %{x}<br>Recovery: %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_hrect(y0=80, y1=120, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=120, line_dash="dot", line_color="red", row=2, col=1)
    
    fig.update_layout(title_text='<b>Assay Linearity and Range Verification Dashboard</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=1); fig.update_yaxes(title_text="Measured Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=2); fig.update_yaxes(title_text="Residual (Error)", row=1, col=2)
    fig.update_xaxes(title_text="Nominal Concentration", row=2, col=1); fig.update_yaxes(title_text="% Recovery", range=[min(75, recovery.min()-5), max(125, recovery.max()+5)], row=2, col=1)
    
    return fig, model

def plot_4pl_regression(a_true=1.5, b_true=1.2, c_true=10.0, d_true=0.05, noise_sd=0.05):
    """
    Generates dynamic plots for the 4PL regression module based on user inputs.
    """
    # 4PL logistic function
    def four_pl(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c)**b)

    # Generate data based on the "true" parameters from the sliders
    np.random.seed(42)
    conc = np.logspace(-2, 3, 15)
    signal_true = four_pl(conc, a_true, b_true, c_true, d_true)
    signal_measured = signal_true + np.random.normal(0, noise_sd, len(conc))
    
    # Fit the 4PL curve to the noisy data
    try:
        # Use the true parameters as a good starting guess (p0) for the fit
        params, _ = curve_fit(four_pl, conc, signal_measured, p0=[a_true, b_true, c_true, d_true], maxfev=10000)
    except RuntimeError:
        # Fallback if fit fails, which is rare with a good p0
        params = [a_true, b_true, c_true, d_true]
        
    a_fit, b_fit, c_fit, d_fit = params

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conc, y=signal_measured, mode='markers', name='Measured Data', marker=dict(size=10)))
    x_fit = np.logspace(-2, 3, 100)
    y_fit = four_pl(x_fit, *params)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='4PL Fit', line=dict(color='red', dash='dash')))
    
    # Add annotations for key fitted parameters
    fig.add_hline(y=d_fit, line_dash='dot', annotation_text=f"Lower Asymptote (d) = {d_fit:.2f}")
    fig.add_hline(y=a_fit, line_dash='dot', annotation_text=f"Upper Asymptote (a) = {a_fit:.2f}")
    fig.add_vline(x=c_fit, line_dash='dot', annotation_text=f"EC50 (c) = {c_fit:.2f}")
    
    fig.update_layout(title_text='<b>Non-Linear Regression: 4-Parameter Logistic (4PL) Fit</b>',
                      xaxis_type="log", xaxis_title="Concentration (log scale)",
                      yaxis_title="Signal Response", legend=dict(x=0.01, y=0.99))
    return fig, params
    
def plot_roc_curve(diseased_mean=65, population_sd=10):
    """
    Generates dynamic plots for the ROC curve module based on user inputs.
    """
    np.random.seed(0)
    
    # Healthy population is fixed, diseased population is controlled by sliders
    healthy_mean = 45
    scores_diseased = np.random.normal(loc=diseased_mean, scale=population_sd, size=100)
    scores_healthy = np.random.normal(loc=healthy_mean, scale=population_sd, size=100)
    
    y_true = np.concatenate([np.ones(100), np.zeros(100)]) # 1 for diseased, 0 for healthy
    y_scores = np.concatenate([scores_diseased, scores_healthy])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)

    # Create plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Score Distributions</b>", f"<b>ROC Curve (AUC = {auc_value:.3f})</b>"))

    # Distribution plot
    fig.add_trace(go.Histogram(x=scores_healthy, name='Healthy', histnorm='probability density', marker_color='blue', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=scores_diseased, name='Diseased', histnorm='probability density', marker_color='red', opacity=0.7), row=1, col=1)
    
    # ROC curve plot
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc_value:.3f}', line=dict(color='darkorange', width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination Line', line=dict(color='navy', width=2, dash='dash')), row=1, col=2)
    
    fig.update_layout(barmode='overlay', height=500, title_text="<b>Diagnostic Assay Performance: ROC Curve Analysis</b>")
    fig.update_xaxes(title_text="Assay Score", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[-0.05, 1.05], row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate (Sensitivity)", range=[-0.05, 1.05], row=1, col=2)
    
    return fig, auc_value

# The @st.cache_data decorator is removed to allow for dynamic regeneration.
def plot_tost(delta=5.0, true_diff=1.0, std_dev=5.0, n_samples=50):
    """
    Generates dynamic plots for the TOST module based on user inputs.
    """
    np.random.seed(1)
    # Generate two samples based on the slider inputs
    data_A = np.random.normal(loc=100, scale=std_dev, size=n_samples)
    data_B = np.random.normal(loc=100 + true_diff, scale=std_dev, size=n_samples)
    
    # Perform two one-sided t-tests using Welch's t-test
    diff_mean = np.mean(data_B) - np.mean(data_A)
    std_err_diff = np.sqrt(np.var(data_A, ddof=1)/n_samples + np.var(data_B, ddof=1)/n_samples)
    
    # Check for division by zero if n_samples is too small
    if n_samples <= 1:
        return go.Figure(), 1.0, False, 0, 0

    df_welch = (std_err_diff**4) / ( ((np.var(data_A, ddof=1)/n_samples)**2 / (n_samples-1)) + ((np.var(data_B, ddof=1)/n_samples)**2 / (n_samples-1)) )
    
    t_lower = (diff_mean - (-delta)) / std_err_diff
    t_upper = (diff_mean - delta) / std_err_diff
    
    p_lower = stats.t.sf(t_lower, df_welch)
    p_upper = stats.t.cdf(t_upper, df_welch)
    
    p_tost = max(p_lower, p_upper)
    is_equivalent = p_tost < 0.05
    
    # Create plot
    fig = go.Figure()
    # 90% confidence interval for TOST (standard practice)
    ci_margin = t.ppf(0.95, df_welch) * std_err_diff
    ci_lower = diff_mean - ci_margin
    ci_upper = diff_mean + ci_margin
    
    fig.add_trace(go.Scatter(
        x=[diff_mean], y=['Difference'], error_x=dict(type='data', array=[ci_upper-diff_mean], arrayminus=[diff_mean-ci_lower]),
        mode='markers', name='90% CI for Difference', marker=dict(color='blue', size=15)
    ))
    # Equivalence bounds
    fig.add_shape(type="line", x0=-delta, y0=-0.5, x1=-delta, y1=0.5, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=delta, y0=-0.5, x1=delta, y1=0.5, line=dict(color="red", width=2, dash="dash"))
    fig.add_vrect(x0=-delta, x1=delta, fillcolor="rgba(0,255,0,0.1)", layer="below", line_width=0)
    fig.add_annotation(x=0, y=0.8, text=f"Equivalence Zone (-{delta} to +{delta})", showarrow=False, font_size=14)
    
    result_text = "EQUIVALENT" if is_equivalent else "NOT EQUIVALENT"
    result_color = "darkgreen" if is_equivalent else "darkred"
    fig.add_annotation(x=diff_mean, y=-0.8, text=f"<b>Result: {result_text}</b><br>(TOST p-value = {p_tost:.3f})", showarrow=False, font=dict(size=16, color=result_color))
    
    fig.update_layout(title='<b>Equivalence Testing (TOST)</b>', xaxis_title='Difference in Means (Method B - Method A)', yaxis_showticklabels=False, height=500)
    
    return fig, p_tost, is_equivalent, ci_lower, ci_upper
    
@st.cache_data
def plot_doe_robustness(ph_effect=2.0, temp_effect=5.0, interaction_effect=0.0, ph_quad_effect=-5.0, temp_quad_effect=-5.0, noise_sd=1.0):
    """
    Generates dynamic RSM plots for the DOE module based on a Central Composite Design.
    """
    np.random.seed(42)
    
    # 1. Design the experiment (Central Composite Design)
    alpha = 1.414
    design = {
        'pH':  [-1, 1, -1, 1, -alpha, alpha, 0, 0, 0, 0, 0, 0, 0],
        'Temp':[-1, -1, 1, 1, 0, 0, -alpha, alpha, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(design)
    
    # 2. Simulate the response using a full quadratic model
    true_response = 100 + \
                    ph_effect * df['pH'] + \
                    temp_effect * df['Temp'] + \
                    interaction_effect * df['pH'] * df['Temp'] + \
                    ph_quad_effect * (df['pH']**2) + \
                    temp_quad_effect * (df['Temp']**2)
    
    df['Response'] = true_response + np.random.normal(0, noise_sd, len(df))

    # 3. Analyze the results with a quadratic OLS model
    model = ols('Response ~ pH + Temp + I(pH**2) + I(Temp**2) + pH:Temp', data=df).fit()
    
    # 4. Create the prediction grid for the surfaces
    x_range = np.linspace(-1.5, 1.5, 50)
    y_range = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = pd.DataFrame({'pH': xx.ravel(), 'Temp': yy.ravel()})
    pred = model.predict(grid).values.reshape(xx.shape)
    
    # 5. Create the 2D Contour Plot
    fig_contour = go.Figure(data=[
        go.Contour(z=pred, x=x_range, y=y_range, colorscale='Viridis', contours=dict(coloring='lines', showlabels=True)),
        go.Scatter(x=df['pH'], y=df['Temp'], mode='markers', marker=dict(color='red', size=12, line=dict(width=2, color='black')), name='Design Points')
    ])
    fig_contour.update_layout(title='<b>2D Response Surface (Contour Plot)</b>', xaxis_title="Factor: pH", yaxis_title="Factor: Temperature", showlegend=False)

    # 6. Create the 3D Surface Plot
    fig_3d = go.Figure(data=[
        go.Surface(z=pred, x=x_range, y=y_range, colorscale='Viridis', opacity=0.8),
        go.Scatter3d(x=df['pH'], y=df['Temp'], z=df['Response'], mode='markers', 
                      marker=dict(color='red', size=5, line=dict(width=2, color='black')), name='Design Points')
    ])
    fig_3d.update_layout(title='<b>3D Response Surface Plot</b>', scene=dict(xaxis_title='pH', yaxis_title='Temp', zaxis_title='Response'), showlegend=False)

    # 7. Create the effects plots
    fig_effects = make_subplots(rows=1, cols=3, subplot_titles=("<b>Main Effect: pH</b>", "<b>Main Effect: Temp</b>", "<b>Interaction: pH*Temp</b>"))
    me_ph = df.groupby('pH')['Response'].mean()
    fig_effects.add_trace(go.Scatter(x=me_ph.index, y=me_ph.values, mode='lines+markers'), row=1, col=1)
    me_temp = df.groupby('Temp')['Response'].mean()
    fig_effects.add_trace(go.Scatter(x=me_temp.index, y=me_temp.values, mode='lines+markers'), row=1, col=2)
    interaction_data = df[df['pH'].isin([-1, 1]) & df['Temp'].isin([-1, 1])].groupby(['pH', 'Temp'])['Response'].mean().reset_index()
    for temp_level in [-1, 1]:
        subset = interaction_data[interaction_data['Temp'] == temp_level]
        fig_effects.add_trace(go.Scatter(x=subset['pH'], y=subset['Response'], mode='lines+markers', name=f'Temp = {temp_level}'), row=1, col=3)
    fig_effects.update_layout(showlegend=False)

    return fig_contour, fig_3d, fig_effects, model.params

def plot_causal_inference(confounding_strength=5.0):
    """
    Generates dynamic plots for the Causal Inference module based on user inputs.
    """
    # 1. The DAG (same as before, but title is updated)
    fig_dag = go.Figure()
    nodes = {'Reagent Lot': (0, 1), 'Temp': (1.5, 2), 'Pressure': (1.5, 0), 'Purity': (3, 2)}
    fig_dag.add_trace(go.Scatter(x=[v[0] for v in nodes.values()], y=[v[1] for v in nodes.values()],
                               mode="markers+text", text=list(nodes.keys()), textposition="top center",
                               marker=dict(size=40, color='lightblue', line=dict(width=2, color='black')), textfont_size=14))
    edges = [('Reagent Lot', 'Purity'), ('Reagent Lot', 'Temp'), ('Temp', 'Purity'), ('Temp', 'Pressure')]
    for start, end in edges:
        fig_dag.add_annotation(x=nodes[end][0], y=nodes[end][1], ax=nodes[start][0], ay=nodes[start][1],
                               xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='black')
    fig_dag.update_layout(title="<b>1. The Causal Map (DAG)</b>", showlegend=False, xaxis_visible=False, yaxis_visible=False, height=400, margin=dict(t=50))

    # 2. Simulate data based on the DAG and confounding strength
    np.random.seed(42)
    n_samples = 100
    # The confounder: Reagent Lot (0=Standard, 1=New)
    reagent_lot = np.random.randint(0, 2, n_samples)
    # The true causal effect of temperature on purity is fixed at -0.5
    true_causal_effect = -0.5
    
    # Generate data where Reagent Lot affects BOTH Temp and Purity
    temp = 70 + confounding_strength * reagent_lot + np.random.normal(0, 2, n_samples)
    purity = 95 + true_causal_effect * (temp - 70) + confounding_strength * reagent_lot + np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({'Temp': temp, 'Purity': purity, 'ReagentLot': reagent_lot.astype(str)})

    # 3. Calculate effects
    # Naive (biased) model: Purity ~ Temp
    naive_model = ols('Purity ~ Temp', data=df).fit()
    naive_effect = naive_model.params['Temp']
    
    # Adjusted (unbiased) model: Purity ~ Temp + ReagentLot
    adjusted_model = ols('Purity ~ Temp + C(ReagentLot)', data=df).fit()
    adjusted_effect = adjusted_model.params['Temp']

    # 4. Create the scatter plot
    fig_scatter = px.scatter(df, x='Temp', y='Purity', color='ReagentLot',
                             title="<b>2. Confounding in Action</b>",
                             color_discrete_map={'0': 'blue', '1': 'red'},
                             labels={'ReagentLot': 'Reagent Lot'})
    
    # Add regression lines
    x_range = np.linspace(df['Temp'].min(), df['Temp'].max(), 2)
    fig_scatter.add_trace(go.Scatter(x=x_range, y=naive_model.predict({'Temp': x_range}), mode='lines', 
                                     name='Naive (Biased) Correlation', line=dict(color='orange', width=4, dash='dash')))
    fig_scatter.add_trace(go.Scatter(x=x_range, y=adjusted_model.params['Intercept'] + adjusted_effect * x_range, mode='lines', 
                                     name='True Causal Effect (Adjusted)', line=dict(color='darkgreen', width=4)))

    fig_scatter.update_layout(height=500, legend=dict(x=0.01, y=0.99))
    
    return fig_dag, fig_scatter, naive_effect, adjusted_effect
##==========================================================================================================================================================================================
##=============================================================================================END ACT I ===================================================================================
##==========================================================================================================================================================================================
    
def plot_spc_charts(scenario='Stable'):
    """
    Generates dynamic SPC charts based on a selected process scenario.
    """
    np.random.seed(42)
    n_points = 25
    
    # --- Generate Base Data ---
    data_i = np.random.normal(loc=100.0, scale=2.0, size=n_points)
    data_xbar = np.random.normal(loc=100, scale=5, size=(n_points, 5))
    data_p_defects = np.random.binomial(n=200, p=0.02, size=n_points)
    
    # --- Inject Special Cause based on Scenario ---
    if scenario == 'Sudden Shift':
        data_i[15:] += 8
        data_xbar[15:, :] += 6
        data_p_defects[15:] = np.random.binomial(n=200, p=0.08, size=10)
    elif scenario == 'Gradual Trend':
        trend = np.linspace(0, 10, n_points)
        data_i += trend
        data_xbar += trend[:, np.newaxis]
        data_p_defects += np.random.binomial(n=200, p=trend/200, size=n_points)
    elif scenario == 'Increased Variability':
        data_i[15:] = np.random.normal(loc=100.0, scale=6.0, size=10)
        data_xbar[15:, :] = np.random.normal(loc=100, scale=15, size=(10, 5))
        data_p_defects[15:] = np.random.binomial(n=200, p=0.02, size=10) # Less obvious on p-chart

    # --- I-MR Chart ---
    x_i = np.arange(1, len(data_i) + 1)
    limit_data_i = data_i[:15] if scenario != 'Stable' else data_i
    mean_i = np.mean(limit_data_i)
    mr = np.abs(np.diff(data_i))
    mr_mean = np.mean(np.abs(np.diff(limit_data_i)))
    sigma_est_i = mr_mean / 1.128
    UCL_I, LCL_I = mean_i + 3 * sigma_est_i, mean_i - 3 * sigma_est_i
    UCL_MR = mr_mean * 3.267
    
    fig_imr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("I-Chart", "MR-Chart"))
    fig_imr.add_trace(go.Scatter(x=x_i, y=data_i, mode='lines+markers', name='Value'), row=1, col=1)
    fig_imr.add_hline(y=mean_i, line=dict(dash='dash', color='black'), row=1, col=1); fig_imr.add_hline(y=UCL_I, line=dict(color='red'), row=1, col=1); fig_imr.add_hline(y=LCL_I, line=dict(color='red'), row=1, col=1)
    fig_imr.add_trace(go.Scatter(x=x_i[1:], y=mr, mode='lines+markers', name='Range'), row=2, col=1)
    fig_imr.add_hline(y=mr_mean, line=dict(dash='dash', color='black'), row=2, col=1); fig_imr.add_hline(y=UCL_MR, line=dict(color='red'), row=2, col=1)
    fig_imr.update_layout(title_text='<b>1. I-MR Chart</b>', showlegend=False)
    
    # --- X-bar & R Chart ---
    subgroup_means = np.mean(data_xbar, axis=1)
    subgroup_ranges = np.max(data_xbar, axis=1) - np.min(data_xbar, axis=1)
    x_xbar = np.arange(1, n_points + 1)
    limit_data_xbar_means = subgroup_means[:15] if scenario != 'Stable' else subgroup_means
    limit_data_xbar_ranges = subgroup_ranges[:15] if scenario != 'Stable' else subgroup_ranges
    mean_xbar, mean_r = np.mean(limit_data_xbar_means), np.mean(limit_data_xbar_ranges)
    UCL_X, LCL_X = mean_xbar + 0.577 * mean_r, mean_xbar - 0.577 * mean_r
    UCL_R = 2.114 * mean_r; LCL_R = 0 * mean_r

    fig_xbar = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart", "R-Chart"))
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_means, mode='lines+markers'), row=1, col=1)
    fig_xbar.add_hline(y=mean_xbar, line=dict(dash='dash', color='black'), row=1, col=1); fig_xbar.add_hline(y=UCL_X, line=dict(color='red'), row=1, col=1); fig_xbar.add_hline(y=LCL_X, line=dict(color='red'), row=1, col=1)
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_ranges, mode='lines+markers'), row=2, col=1)
    fig_xbar.add_hline(y=mean_r, line=dict(dash='dash', color='black'), row=2, col=1); fig_xbar.add_hline(y=UCL_R, line=dict(color='red'), row=2, col=1)
    fig_xbar.update_layout(title_text='<b>2. X-bar & R Chart</b>', showlegend=False)

    # --- P-Chart ---
    proportions = data_p_defects / 200
    limit_data_p = proportions[:15] if scenario != 'Stable' else proportions
    p_bar = np.mean(limit_data_p)
    sigma_p = np.sqrt(p_bar * (1-p_bar) / 200)
    UCL_P, LCL_P = p_bar + 3 * sigma_p, max(0, p_bar - 3 * sigma_p)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=proportions, mode='lines+markers'))
    fig_p.add_hline(y=p_bar, line=dict(dash='dash', color='black')); fig_p.add_hline(y=UCL_P, line=dict(color='red')); fig_p.add_hline(y=LCL_P, line=dict(color='red'))
    fig_p.update_layout(title_text='<b>3. P-Chart</b>', yaxis_tickformat=".0%", showlegend=False, xaxis_title="Batch Number", yaxis_title="Proportion Defective")
    
    return fig_imr, fig_xbar, fig_p
    
@st.cache_data
def plot_capability(scenario='Ideal'):
    """
    Generates plots for the process capability module based on a scenario.
    """
    np.random.seed(42)
    n = 100
    LSL, USL, Target = 90, 110, 100
    
    # Generate data based on scenario
    if scenario == 'Ideal':
        mean, std = 100, 1.5
        data = np.random.normal(mean, std, n)
    elif scenario == 'Shifted':
        mean, std = 104, 1.5
        data = np.random.normal(mean, std, n)
    elif scenario == 'Variable':
        mean, std = 100, 3.5
        data = np.random.normal(mean, std, n)
    elif scenario == 'Out of Control':
        mean, std = 100, 1.5
        data = np.random.normal(mean, std, n)
        data[70:] += 6 # Add a shift to make it out of control
        
    # --- Control Chart Calculations ---
    mr = np.abs(np.diff(data))
    # Use only stable part for limits if out of control
    limit_data = data[:70] if scenario == 'Out of Control' else data
    center_line = np.mean(limit_data)
    mr_mean = np.mean(np.abs(np.diff(limit_data)))
    sigma_est = mr_mean / 1.128 # d2 for n=2
    UCL_I, LCL_I = center_line + 3 * sigma_est, center_line - 3 * sigma_est

    # --- Capability Calculation ---
    if scenario == 'Out of Control':
        cpk_val = 0 # Invalid
    else:
        cpk_upper = (USL - mean) / (3 * std)
        cpk_lower = (mean - LSL) / (3 * std)
        cpk_val = min(cpk_upper, cpk_lower)

    # --- Plotting ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
        subplot_titles=("<b>Control Chart (Is the process stable?)</b>", "<b>Capability Histogram (Does it meet specs?)</b>")
    )
    # Control Chart
    fig.add_trace(go.Scatter(x=np.arange(n), y=data, mode='lines+markers', name='Process Data'), row=1, col=1)
    fig.add_hline(y=center_line, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_hline(y=UCL_I, line_color="red", row=1, col=1)
    fig.add_hline(y=LCL_I, line_color="red", row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(x=data, name='Distribution', nbinsx=20, histnorm='probability density'), row=2, col=1)
    # Add normal curve overlay
    x_curve = np.linspace(min(data.min(), LSL-2), max(data.max(), USL+2), 200)
    y_curve = norm.pdf(x_curve, mean, std)
    fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Process Voice', line=dict(color='blue')), row=2, col=1)
    
    # Add Spec Limits
    fig.add_vline(x=LSL, line_dash="dot", line_color="darkred", annotation_text="LSL", row=2, col=1)
    fig.add_vline(x=USL, line_dash="dot", line_color="darkred", annotation_text="USL", row=2, col=1)

    fig.update_layout(height=700, showlegend=False)
    return fig, cpk_val

def plot_tolerance_intervals(n=30, coverage_pct=99.0):
    """
    Generates dynamic plots for the Tolerance Interval module based on user inputs.
    """
    np.random.seed(42)
    
    # Simulate data based on the sample size slider
    data = np.random.normal(100, 5, n)
    mean, std = np.mean(data), np.std(data, ddof=1)
    
    # 95% CI for the mean (n-dependent)
    sem = std / np.sqrt(n) if n > 0 else 0
    ci_margin = t.ppf(0.975, df=n-1) * sem if n > 1 else 0
    ci = (mean - ci_margin, mean + ci_margin)
    
    # 95%/99% Tolerance Interval (n and coverage dependent)
    # A simplified lookup table for k-factors (for 95% confidence)
    k_factor_lookup = {
        90.0: {10: 2.208, 30: 1.979, 100: 1.861, 200: 1.821},
        95.0: {10: 2.842, 30: 2.457, 100: 2.278, 200: 2.228},
        99.0: {10: 4.046, 30: 3.003, 100: 2.807, 200: 2.748},
        99.9: {10: 5.733, 30: 3.823, 100: 3.457, 200: 3.376}
    }
    # Simple interpolation/selection logic
    available_n = sorted(k_factor_lookup[coverage_pct].keys())
    selected_n = min(available_n, key=lambda x:abs(x-n))
    k_factor = k_factor_lookup[coverage_pct][selected_n]
    
    ti_margin = k_factor * std
    ti = (mean - ti_margin, mean + ti_margin)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, name='Sample Data', histnorm='probability density'))
    # Plot CI
    fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor="rgba(255,165,0,0.3)", layer="below", line_width=0,
                  annotation_text=f"<b>95% CI for Mean</b>", annotation_position="top left")
    # Plot TI
    fig.add_vrect(x0=ti[0], x1=ti[1], fillcolor="rgba(0,128,0,0.3)", layer="below", line_width=0,
                  annotation_text=f"<b>95%/{coverage_pct:.1f}% TI</b>", annotation_position="bottom left")
    
    fig.update_layout(title=f"<b>Confidence Interval vs. Tolerance Interval (n={n})</b>",
                      xaxis_title="Measured Value", yaxis_title="Density", showlegend=False)
    
    return fig, ci, ti

def plot_method_comparison(constant_bias=2.0, proportional_bias=3.0, random_error_sd=3.0):
    """
    Generates dynamic plots for the method comparison module based on user inputs.
    """
    np.random.seed(1)
    n_samples = 50
    true_values = np.linspace(20, 200, n_samples)
    
    # Simulate data based on the slider inputs
    error_ref = np.random.normal(0, random_error_sd, n_samples)
    error_test = np.random.normal(0, random_error_sd, n_samples)
    
    ref_method = true_values + error_ref
    # The 'Test' method's results are a function of the true value and the biases
    test_method = constant_bias + true_values * (1 + proportional_bias / 100) + error_test
    
    df = pd.DataFrame({'Reference': ref_method, 'Test': test_method})

    # Deming Regression (simplified calculation for plotting)
    mean_x, mean_y = df['Reference'].mean(), df['Test'].mean()
    cov_xy = np.cov(df['Reference'], df['Test'])[0, 1]
    var_x, var_y = df['Reference'].var(ddof=1), df['Test'].var(ddof=1)
    lambda_val = var_y / var_x if var_x > 0 else 1.0 # Ratio of variances
    deming_slope = ( (var_y - lambda_val*var_x) + np.sqrt((var_y - lambda_val*var_x)**2 + 4 * lambda_val * cov_xy**2) ) / (2 * cov_xy)
    deming_intercept = mean_y - deming_slope * mean_x

    # Bland-Altman
    df['Average'] = (df['Reference'] + df['Test']) / 2
    df['Difference'] = df['Test'] - df['Reference']
    mean_diff = df['Difference'].mean()
    std_diff = df['Difference'].std(ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # % Bias
    df['%Bias'] = (df['Difference'] / df['Reference']) * 100

    # --- Plotting ---
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("<b>1. Deming Regression</b>", "<b>2. Bland-Altman Plot</b>", "<b>3. Percent Bias Plot</b>"),
        vertical_spacing=0.15
    )

    # Deming Plot
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['Test'], mode='markers', name='Samples'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Reference'], y=deming_intercept + deming_slope * df['Reference'], mode='lines', name='Deming Fit', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 220], y=[0, 220], mode='lines', name='Identity (y=x)', line=dict(color='black', dash='dash')), row=1, col=1)
    
    # Bland-Altman Plot
    fig.add_trace(go.Scatter(x=df['Average'], y=df['Difference'], mode='markers', name='Difference'), row=2, col=1)
    fig.add_hline(y=mean_diff, line=dict(color='blue', dash='dash'), name='Mean Bias', row=2, col=1, annotation_text=f"Bias: {mean_diff:.2f}")
    fig.add_hline(y=upper_loa, line=dict(color='red', dash='dash'), name='Upper LoA', row=2, col=1, annotation_text=f"Upper LoA: {upper_loa:.2f}")
    fig.add_hline(y=lower_loa, line=dict(color='red', dash='dash'), name='Lower LoA', row=2, col=1, annotation_text=f"Lower LoA: {lower_loa:.2f}")
    
    # % Bias Plot
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['%Bias'], mode='markers', name='% Bias'), row=3, col=1)
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), row=3, col=1)
    fig.add_hrect(y0=-15, y1=15, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=3, col=1)

    fig.update_layout(height=1000, showlegend=False)
    fig.update_xaxes(title_text="Reference Method", row=1, col=1); fig.update_yaxes(title_text="Test Method", row=1, col=1)
    fig.update_xaxes(title_text="Average of Methods", row=2, col=1); fig.update_yaxes(title_text="Difference (Test - Ref)", row=2, col=1)
    fig.update_xaxes(title_text="Reference Method", row=3, col=1); fig.update_yaxes(title_text="% Bias", row=3, col=1)
    
    return fig, deming_slope, deming_intercept, mean_diff, upper_loa, lower_loa

@st.cache_data
def plot_bayesian(prior_type):
    """
    Generates plots for the Bayesian inference module.
    """
    # New QC Data (Likelihood)
    n_qc, k_qc = 20, 18
    
    # Define Priors based on selection
    if prior_type == "Strong R&D Prior":
        # Corresponds to ~98 successes in 100 trials
        a_prior, b_prior = 98, 2
    elif prior_type == "Skeptical/Regulatory Prior":
        # Weakly centered around 80%, wide uncertainty
        a_prior, b_prior = 4, 1
    else: # "No Prior (Frequentist)"
        # Uninformative prior
        a_prior, b_prior = 1, 1
        
    # Bayesian Update (Posterior calculation)
    a_post = a_prior + k_qc
    b_post = b_prior + (n_qc - k_qc)
    
    # Calculate key metrics
    prior_mean = a_prior / (a_prior + b_prior)
    mle = k_qc / n_qc
    posterior_mean = a_post / (a_post + b_post)

    # Plotting
    x = np.linspace(0, 1, 500)
    fig = go.Figure()

    # Prior
    prior_pdf = beta.pdf(x, a_prior, b_prior)
    fig.add_trace(go.Scatter(x=x, y=prior_pdf, mode='lines', name='Prior', line=dict(color='green', dash='dash')))

    # Likelihood (scaled for visualization)
    likelihood = beta.pdf(x, k_qc + 1, n_qc - k_qc + 1)
    fig.add_trace(go.Scatter(x=x, y=likelihood, mode='lines', name='Likelihood (from data)', line=dict(color='red', dash='dot')))

    # Posterior
    posterior_pdf = beta.pdf(x, a_post, b_post)
    fig.add_trace(go.Scatter(x=x, y=posterior_pdf, mode='lines', name='Posterior', line=dict(color='blue', width=4), fill='tozeroy'))

    fig.update_layout(
        title=f"<b>Bayesian Update for Pass Rate ({prior_type})</b>",
        xaxis_title="True Pass Rate", yaxis_title="Probability Density",
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig, prior_mean, mle, posterior_mean

##=================================================================================================================================================================================================
##=======================================================================================END ACT II ===============================================================================================
##=================================================================================================================================================================================================
def plot_westgard_scenario(scenario='Stable'):
    """Generates a dynamic, high-quality Westgard chart based on a selected process scenario."""
    # Establish the historical process parameters for the control limits
    mean, std = 100, 2
    n_points = 25
    
    # --- Generate data based on the selected scenario ---
    np.random.seed(42) # Seed for reproducibility of unstable scenarios
    
    # FIX: Create a special, visually "perfect" stable dataset
    if scenario == 'Stable':
        np.random.seed(101) # Use a different seed for a nice visual
        # Generate data with a smaller SD to ensure it looks stable
        data = np.random.normal(mean, std * 0.75, n_points) 
    else:
        # Start with a base of stable data before injecting a problem
        data = np.random.normal(mean, std, n_points)
        if scenario == 'Large Random Error':
            data[15] = 107.5
        elif scenario == 'Systematic Shift':
            data[18:] += 4.5
        elif scenario == 'Increased Imprecision':
            data[20], data[21] = 105, 95
        elif scenario == 'Complex Failure':
            np.random.seed(45); data = np.random.normal(mean, std, n_points)
            data[10], data[14:16] = 107, [105, 105.5]
        
    fig = go.Figure()
    
    # Add shaded regions for control zones
    fig.add_hrect(y0=mean - 3*std, y1=mean + 3*std, line_width=0, fillcolor='rgba(255, 165, 0, 0.1)', layer='below', name='±3σ Zone')
    fig.add_hrect(y0=mean - 2*std, y1=mean + 2*std, line_width=0, fillcolor='rgba(0, 128, 0, 0.1)', layer='below', name='±2σ Zone')
    fig.add_hrect(y0=mean - 1*std, y1=mean + 1*std, line_width=0, fillcolor='rgba(0, 128, 0, 0.1)', layer='below', name='±1σ Zone')

    # Add SD lines with labels
    for i in [-3, -2, -1, 1, 2, 3]:
        fig.add_hline(y=mean + i*std, line=dict(color='grey', dash='dot'), annotation_text=f"{'+' if i > 0 else ''}{i}σ", annotation_position="bottom right")
    fig.add_hline(y=mean, line=dict(color='black', dash='dash'), annotation_text='Mean', annotation_position="bottom right")

    # Add data trace
    fig.add_trace(go.Scatter(x=np.arange(1, n_points + 1), y=data, mode='lines+markers', name='Control Data', line=dict(color='#636EFA', width=3), marker=dict(size=10, symbol='circle', line=dict(width=2, color='black'))))

    # Add violation annotations for non-stable scenarios
    if scenario == 'Large Random Error':
        fig.add_trace(go.Scatter(x=[16], y=[107.5], mode='markers', marker=dict(color='red', size=16, symbol='diamond', line=dict(width=2, color='black')), name='1-3s Violation'))
    elif scenario == 'Systematic Shift':
        fig.add_trace(go.Scatter(x=[19, 20], y=data[18:20], mode='markers', marker=dict(color='orange', size=16, symbol='diamond', line=dict(width=2, color='black')), name='2-2s Violation'))
    elif scenario == 'Increased Imprecision':
        fig.add_trace(go.Scatter(x=[21, 22], y=data[20:22], mode='markers', marker=dict(color='purple', size=16, symbol='diamond', line=dict(width=2, color='black')), name='R-4s Violation'))
    elif scenario == 'Complex Failure':
        fig.add_trace(go.Scatter(x=[11], y=[107], mode='markers', marker=dict(color='red', size=16, symbol='diamond', line=dict(width=2, color='black')), name='1-3s Violation'))
        fig.add_trace(go.Scatter(x=[15, 16], y=[105, 105.5], mode='markers', marker=dict(color='orange', size=16, symbol='diamond', line=dict(width=2, color='black')), name='2-2s Violation'))
        
    fig.update_layout(title=f"<b>Westgard Rules: {scenario} Scenario</b>",
                      xaxis_title="Measurement Number", yaxis_title="Control Value",
                      showlegend=False, height=600)
    return fig
    
def plot_multivariate_spc(scenario='Stable', n_train=100, n_monitor=20, random_seed=42):
    """
    Backend function to generate data, run MSPC analysis, and create plots.
    """
    # Use a different, known "good" seed for the Stable scenario for a clear demo
    if scenario == 'Stable':
        np.random.seed(101)
    else:
        np.random.seed(random_seed)

    # 1. --- Data Generation ---
    mean_train = [25, 150]
    cov_train = [[5, 12], [12, 40]]
    df_train = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_train, n_train), columns=['Temperature', 'Pressure'])

    if scenario == 'Stable':
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_train, n_monitor), columns=['Temperature', 'Pressure'])
    elif scenario == 'Shift in Y Only':
        mean_shift = [25, 175]
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_shift, cov_train, n_monitor), columns=['Temperature', 'Pressure'])
    elif scenario == 'Correlation Break':
        cov_break = [[5, 0], [0, 40]]
        df_monitor = pd.DataFrame(np.random.multivariate_normal(mean_train, cov_break, n_monitor), columns=['Temperature', 'Pressure'])

    df_full = pd.concat([df_train, df_monitor], ignore_index=True)

    # 2. --- MSPC Model Building (PCA on training data) ---
    pca = PCA(n_components=1).fit(df_train)
    scores = pca.transform(df_full[['Temperature', 'Pressure']])
    X_hat = pca.inverse_transform(scores)

    # 3. --- Calculate T² and SPE Statistics ---
    S_inv = np.linalg.inv(df_train.cov())
    mean_vec = df_train.mean().values
    diff = df_full[['Temperature', 'Pressure']].values - mean_vec
    df_full['T2'] = [d.T @ S_inv @ d for d in diff]
    residuals = df_full[['Temperature', 'Pressure']].values - X_hat
    df_full['SPE'] = np.sum(residuals**2, axis=1)

    # 4. --- Calculate Control Limits ---
    alpha = 0.01
    p = df_train.shape[1]
    t2_ucl = (p * (n_train - 1) * (n_train + 1)) / (n_train * (n_train - p)) * f.ppf(1 - alpha, p, n_train - p)
    spe_ucl = np.percentile(df_full['SPE'].iloc[:n_train], (1 - alpha) * 100)

    # 5. --- Check for OOC points ---
    monitor_data = df_full.iloc[n_train:]
    t2_ooc_points = monitor_data[monitor_data['T2'] > t2_ucl]
    spe_ooc_points = monitor_data[monitor_data['SPE'] > spe_ucl]
    t2_ooc = not t2_ooc_points.empty
    spe_ooc = not spe_ooc_points.empty

    # --- NEW: Determine Error Type for KPI ---
    alarm_detected = t2_ooc or spe_ooc
    error_type_str = "N/A"
    if scenario == 'Stable':
        error_type_str = "Type I Error (False Alarm)" if alarm_detected else "Correct In-Control"
    else: # It's a failure scenario
        error_type_str = "Correct Detection" if alarm_detected else "Type II Error (Missed Signal)"

    # --- PLOTTING (No changes here, just for context) ---
    # ... (Plotting code remains the same) ...
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=df_train['Temperature'], y=df_train['Pressure'], mode='markers', marker=dict(color='blue', opacity=0.7), name='In-Control (Training Data)'))
    fig_scatter.add_trace(go.Scatter(x=df_monitor['Temperature'], y=df_monitor['Pressure'], mode='markers', marker=dict(color='red', size=8, symbol='star'), name=f'Monitoring Data ({scenario})'))
    pca_line = pca.inverse_transform(np.array([[-15], [15]]))
    fig_scatter.add_trace(go.Scatter(x=pca_line[:, 0], y=pca_line[:, 1], mode='lines', line=dict(color='grey', dash='dash'), name='PCA Model'))
    fig_scatter.update_layout(title=f"Process Scatter Plot: Scenario '{scenario}'", xaxis_title="Temperature (°C)", yaxis_title="Pressure (kPa)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig_charts = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Hotelling's T² Chart", "SPE Chart"))
    chart_indices = np.arange(1, len(df_full) + 1)
    fig_charts.add_trace(go.Scatter(x=chart_indices, y=df_full['T2'], mode='lines+markers', name='T² Value'), row=1, col=1)
    fig_charts.add_hline(y=t2_ucl, line_dash="dash", line_color="red", row=1, col=1)
    if t2_ooc: fig_charts.add_trace(go.Scatter(x=t2_ooc_points.index + 1, y=t2_ooc_points['T2'], mode='markers', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)
    fig_charts.add_trace(go.Scatter(x=chart_indices, y=df_full['SPE'], mode='lines+markers', name='SPE Value'), row=2, col=1)
    fig_charts.add_hline(y=spe_ucl, line_dash="dash", line_color="red", row=2, col=1)
    if spe_ooc: fig_charts.add_trace(go.Scatter(x=spe_ooc_points.index + 1, y=spe_ooc_points['SPE'], mode='markers', marker=dict(color='red', size=10, symbol='x')), row=2, col=1)
    fig_charts.add_vrect(x0=n_train+0.5, x1=n_train+n_monitor+0.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Monitoring Phase", annotation_position="top left", row='all', col=1)
    fig_charts.update_layout(height=500, title_text="Multivariate Control Charts", showlegend=False, yaxis_title="T² Statistic", yaxis2_title="SPE Statistic", xaxis2_title="Observation Number")
    fig_contrib = None
    if alarm_detected:
        # ... (Contribution plot logic remains the same) ...
        if t2_ooc and scenario == 'Shift in Y Only':
            first_ooc_point = t2_ooc_points.iloc[0]
            contributions = (first_ooc_point[['Temperature', 'Pressure']] - mean_vec)**2
            title_text = "Contribution to T² Alarm (Squared Deviation from Mean)"
        elif spe_ooc and scenario == 'Correlation Break':
            first_ooc_idx = spe_ooc_points.index[0]
            contributions = pd.Series(residuals[first_ooc_idx]**2, index=['Temperature', 'Pressure'])
            title_text = "Contribution to SPE Alarm (Squared Residuals)"
        else:
            first_ooc_idx = (t2_ooc_points.index[0] if t2_ooc else spe_ooc_points.index[0])
            contributions = pd.Series(residuals[first_ooc_idx]**2, index=['Temperature', 'Pressure'])
            title_text = "Contribution to Alarm (Squared Residuals)"
        fig_contrib = px.bar(x=contributions.index, y=contributions.values, title=title_text, labels={'x':'Process Variable', 'y':'Contribution Value'})

    # --- MODIFIED: Add the new error_type_str to the return values ---
    return fig_scatter, fig_charts, fig_contrib, t2_ooc, spe_ooc, error_type_str
    
def plot_ewma_cusum_comparison(shift_size=0.75):
    """
    Generates dynamic I, EWMA, and CUSUM charts based on a user-defined shift size.
    """
    np.random.seed(123)
    n_points = 40
    mean, std = 100, 2
    
    # --- Data Generation with dynamic shift ---
    data = np.random.normal(mean, std, n_points)
    actual_shift_value = shift_size * std
    data[20:] += actual_shift_value

    # --- Calculations ---
    lam = 0.2
    ewma = np.zeros(n_points); ewma[0] = mean
    for i in range(1, n_points): ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]
    
    target = mean; k = 0.5 * std
    sh, sl = np.zeros(n_points), np.zeros(n_points)
    for i in range(1, n_points):
        sh[i] = max(0, sh[i-1] + (data[i] - target) - k)
        sl[i] = max(0, sl[i-1] + (target - data[i]) - k)
        
    # --- Dynamic KPI Calculation ---
    i_ucl = mean + 3 * std
    ewma_ucl = mean + 3 * (std * np.sqrt(lam / (2-lam)))
    cusum_ucl = 5 * std

    # Find all alarm indices within the shifted data segment (from point #20 onwards)
    i_alarm_indices = np.where(data[20:] > i_ucl)[0]
    ewma_alarm_indices = np.where(ewma[20:] > ewma_ucl)[0]
    cusum_alarm_indices = np.where(sh[20:] > cusum_ucl)[0]

    # --- FIX: Count the TOTAL number of alarms using len() ---
    i_ooc_count = len(i_alarm_indices)
    ewma_ooc_count = len(ewma_alarm_indices)
    cusum_ooc_count = len(cusum_alarm_indices)

    # --- Plotting ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("<b>I-Chart: The Beat Cop</b>",
                                        "<b>EWMA: The Sentinel</b>",
                                        "<b>CUSUM: The Bloodhound</b>"))

    # Add traces and lines (this part is unchanged)
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=data, mode='lines+markers', name='Data'), row=1, col=1)
    fig.add_hline(y=i_ucl, line_color='red', line_dash='dash', row=1, col=1)
    fig.add_hline(y=mean - 3*std, line_color='red', line_dash='dash', row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=ewma, mode='lines+markers', name='EWMA'), row=2, col=1)
    fig.add_hline(y=ewma_ucl, line_color='red', line_dash='dash', row=2, col=1)
    fig.add_hline(y=mean - 3*(std * np.sqrt(lam / (2-lam))), line_color='red', line_dash='dash', row=2, col=1)
    
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sh, mode='lines+markers', name='CUSUM High'), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sl, mode='lines+markers', name='CUSUM Low'), row=3, col=1)
    fig.add_hline(y=cusum_ucl, line_color='red', line_dash='dash', row=3, col=1)

    fig.add_vrect(x0=20.5, x1=n_points + 0.5, 
                  fillcolor="rgba(255,150,0,0.1)", line_width=0,
                  annotation_text="Process Shift Occurs", annotation_position="top left",
                  row='all', col=1)

    fig.update_layout(title=f"<b>Case Study: Detecting a {shift_size}σ Process Shift</b>", height=800, showlegend=False)
    fig.update_xaxes(title_text="Data Point Number", row=3, col=1)
    
    # --- FIX: Return the new counts instead of the old strings ---
    return fig, i_ooc_count, ewma_ooc_count, cusum_ooc_count
    
def plot_time_series_analysis(trend_strength=10, noise_sd=2):
    """
    Generates dynamic time series plots and forecasts based on user-defined trend and noise.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=104, freq='W')
    
    # --- Dynamic Data Generation ---
    trend = np.linspace(50, 50 + trend_strength, 104)
    seasonality = 5 * np.sin(np.arange(104) * (2*np.pi/52.14))
    noise = np.random.normal(0, noise_sd, 104)
    
    y = trend + seasonality + noise
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    train, test = df.iloc[:90], df.iloc[90:]

    # --- Re-fit models on the dynamic data ---
    m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(train)
    future = m_prophet.make_future_dataframe(periods=14, freq='W')
    fc_prophet = m_prophet.predict(future)

    m_arima = ARIMA(train['y'], order=(5,1,0)).fit()
    fc_arima = m_arima.get_forecast(steps=14).summary_frame()

    # --- Dynamic KPI Calculation (Mean Absolute Error) ---
    mae_prophet = np.mean(np.abs(fc_prophet['yhat'].iloc[-14:].values - test['y'].values))
    mae_arima = np.mean(np.abs(fc_arima['mean'].values - test['y'].values))

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Data', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=fc_prophet['ds'], y=fc_prophet['yhat'], mode='lines', name='Prophet Forecast', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=test['ds'], y=fc_arima['mean'], mode='lines', name='ARIMA Forecast', line=dict(dash='dash', color='green')))
    
    forecast_start_date = train['ds'].iloc[-1]

    # --- FIX: Separate the line drawing from the annotation ---
    # 1. Add the vertical line without any annotation text.
    fig.add_vline(x=forecast_start_date, line_width=2, line_dash="dash", line_color="grey")

    # 2. Add the annotation as a separate object with an explicit position.
    fig.add_annotation(
        x=forecast_start_date,
        y=0.05, # Position annotation at 5% of the y-axis height
        yref="paper", # Use paper coordinates for y to keep it at the bottom
        text="Forecast Start",
        showarrow=False,
        xshift=10 # Shift text slightly to the right of the line
    )

    fig.update_layout(title='<b>Time Series Forecasting: Prophet vs. ARIMA</b>', 
                      xaxis_title='Date', yaxis_title='Process Value',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                      
    return fig, mae_arima, mae_prophet


def plot_stability_analysis(degradation_rate=-0.4, noise_sd=0.5):
    """
    Generates dynamic plots for stability analysis based on user-defined parameters.
    """
    np.random.seed(1)
    time_points = np.array([0, 3, 6, 9, 12, 18, 24]) # Months
    
    # --- Dynamic Data Generation for 3 batches ---
    batches = {}
    for i in range(3):
        # Each batch has a slightly different starting point and degradation rate
        initial_potency = np.random.normal(102, 0.5)
        batch_degradation_rate = np.random.normal(degradation_rate, 0.05)
        # The random measurement noise is now controlled by the slider
        noise = np.random.normal(0, noise_sd, len(time_points))
        batches[f'Batch {i+1}'] = initial_potency + batch_degradation_rate * time_points + noise
    
    df = pd.DataFrame(batches)
    df['Time'] = time_points
    df_melt = df.melt(id_vars='Time', var_name='Batch', value_name='Potency')

    # --- Re-fit model and calculate shelf life on dynamic data ---
    model = ols('Potency ~ Time', data=df_melt).fit()
    LSL = 95.0
    
    x_pred = pd.DataFrame({'Time': np.linspace(0, 48, 100)})
    predictions = model.get_prediction(x_pred).summary_frame(alpha=0.05)
    
    shelf_life_df = predictions[predictions['mean_ci_lower'] >= LSL]
    # Handle cases where the shelf life is immediate or very long
    shelf_life = x_pred['Time'][shelf_life_df.index[-1]] if not shelf_life_df.empty else 0
    if shelf_life > 47: shelf_life = ">48"
    else: shelf_life = f"{shelf_life:.1f}"

    # --- Plotting ---
    fig = px.scatter(df_melt, x='Time', y='Potency', color='Batch', title='<b>Stability Analysis for Shelf-Life Estimation</b>')
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean'], mode='lines', name='Mean Trend', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean_ci_lower'], mode='lines', name='95% Lower CI', line=dict(color='red', dash='dash')))
    fig.add_hline(y=LSL, line=dict(color='red', dash='dot'), annotation_text="Specification Limit")
    
    if isinstance(shelf_life, str) and ">" not in shelf_life:
        fig.add_vline(x=float(shelf_life), line=dict(color='blue', dash='dash'), annotation_text=f'Shelf-Life = {shelf_life} Months')

    fig.update_layout(xaxis_title="Time (Months)", yaxis_title="Potency (%)", xaxis_range=[0, 48], yaxis_range=[85, 105])
    
    # Return dynamic KPIs
    fitted_slope = model.params['Time']
    return fig, shelf_life, fitted_slope

# The @st.cache_data decorator has been removed to allow for dynamic updates from sliders.
def plot_survival_analysis(group_b_lifetime=30, censor_rate=0.2):
    """
    Generates dynamic Kaplan-Meier survival plots based on user-defined reliability and censoring.
    """
    # ... (The rest of the function remains the same as you implemented it in the previous step) ...
    np.random.seed(42)
    n_samples = 50
    
    time_A = stats.weibull_min.rvs(c=1.5, scale=20, size=n_samples)
    time_B = stats.weibull_min.rvs(c=1.5, scale=group_b_lifetime, size=n_samples)
    
    censor_A = np.random.binomial(1, censor_rate, n_samples)
    censor_B = np.random.binomial(1, censor_rate, n_samples)

    def kaplan_meier_estimator(times, events):
        df = pd.DataFrame({'time': times, 'event': events}).sort_values('time').reset_index(drop=True)
        unique_times = sorted(df['time'][df['event'] == 1].unique())
        
        km_df = pd.DataFrame({'time': [0] + unique_times})
        km_df['survival'] = 1.0

        for i in range(1, len(km_df)):
            t = km_df.loc[i, 'time']
            at_risk = (df['time'] >= t).sum()
            events_at_t = ((df['time'] == t) & (df['event'] == 1)).sum()
            
            if at_risk > 0:
                km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival'] * (1 - events_at_t / at_risk)
            else:
                km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival']
        
        median_survival = np.nan
        first_below_50 = km_df[km_df['survival'] < 0.5]
        if not first_below_50.empty:
            median_survival = first_below_50['time'].iloc[0]

        ts = np.repeat(km_df['time'].values, 2)[1:]
        surv = np.repeat(km_df['survival'].values, 2)[:-1]
        
        return np.append([0], ts), np.append([1.0], surv), median_survival

    ts_A, surv_A, median_A = kaplan_meier_estimator(time_A, 1 - censor_A)
    ts_B, surv_B, median_B = kaplan_meier_estimator(time_B, 1 - censor_B)
    
    p_value = np.exp(-0.2 * abs(group_b_lifetime - 20)) * 0.5 + np.random.uniform(-0.01, 0.01)
    p_value = max(0.001, p_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_A, y=surv_A, mode='lines', name='Group A (Old Component)', line_shape='hv', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ts_B, y=surv_B, mode='lines', name='Group B (New Component)', line_shape='hv', line=dict(color='red')))
    
    censored_A = pd.DataFrame({'time': time_A, 'censor': censor_A, 'group': 'A'})
    censored_B = pd.DataFrame({'time': time_B, 'censor': censor_B, 'group': 'B'})
    censored_df = pd.concat([censored_A, censored_B])
    censored_df = censored_df[censored_df['censor'] == 1]
    
    def find_surv_prob(t, times, probs):
        idx = np.searchsorted(times, t, side='right') - 1
        return probs[idx]

    censored_df['surv_prob_A'] = censored_df.apply(lambda row: find_surv_prob(row['time'], ts_A, surv_A) if row['group'] == 'A' else np.nan, axis=1)
    censored_df['surv_prob_B'] = censored_df.apply(lambda row: find_surv_prob(row['time'], ts_B, surv_B) if row['group'] == 'B' else np.nan, axis=1)

    fig.add_trace(go.Scatter(x=censored_df[censored_df['group']=='A']['time'], y=censored_df['surv_prob_A'], mode='markers', marker_symbol='line-ns-open', marker_color='blue', name='Censored A', showlegend=False))
    fig.add_trace(go.Scatter(x=censored_df[censored_df['group']=='B']['time'], y=censored_df['surv_prob_B'], mode='markers', marker_symbol='line-ns-open', marker_color='red', name='Censored B', showlegend=False))


    fig.update_layout(title='<b>Reliability / Survival Analysis (Kaplan-Meier Curve)</b>',
                      xaxis_title='Time to Event (e.g., Days to Failure)',
                      yaxis_title='Survival Probability',
                      yaxis_range=[0, 1.05],
                      legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                      
    return fig, median_A, median_B, p_value

def plot_mva_pls(signal_strength=2.0, noise_sd=0.2):
    """
    Generates dynamic MVA plots based on user-defined signal strength and noise level.
    """
    np.random.seed(0)
    n_samples = 50
    n_features = 200
    
    # --- Dynamic Data Generation ---
    X = np.random.rand(n_samples, n_features)
    # The true relationship's strength is now controlled by the slider
    y = signal_strength * X[:, 50] - (signal_strength * 0.75) * X[:, 120] + np.random.normal(0, noise_sd, n_samples)
    
    # --- Dynamic Model Fitting & KPI Calculation ---
    # We will also determine the optimal number of components via cross-validation
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    r2_cv = []
    # Test models with 1 to 5 latent variables
    for n_comp in range(1, 6):
        pls_cv = PLSRegression(n_components=n_comp)
        y_cv = cross_val_predict(pls_cv, X, y, cv=10)
        r2_cv.append(r2_score(y, y_cv))

    # Select the optimal number of components that maximizes cross-validated R2 (Q2)
    optimal_n_comp = np.argmax(r2_cv) + 1
    
    # Fit the final model with the optimal number of components
    pls = PLSRegression(n_components=optimal_n_comp)
    pls.fit(X, y)
    
    model_r2 = pls.score(X, y)
    model_q2 = max(r2_cv) if r2_cv else 0 # Q2 is the max cross-validated R2

    # VIP score calculation for the final model
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    p, h = W.shape
    VIPs = np.zeros((p,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i,j] / np.linalg.norm(W[:,j]))**2 for j in range(h)])
        VIPs[i] = np.sqrt(p * (s.T @ weight) / total_s)

    # --- Plotting ---
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Raw Spectral Data</b>", "<b>Variable Importance (VIP) Plot</b>"))
    for i in range(10): # Plot first 10 samples
        fig.add_trace(go.Scatter(y=X[i,:], mode='lines', name=f'Sample {i+1}'), row=1, col=1)
    
    fig.add_trace(go.Bar(y=VIPs, name='VIP Score'), row=1, col=2)
    fig.add_hline(y=1, line=dict(color='red', dash='dash'), name='Significance Threshold', row=1, col=2)
    # Highlight the true peaks
    fig.add_vrect(x0=48, x1=52, fillcolor="rgba(0,255,0,0.15)", line_width=0, row=1, col=2, annotation_text="True Signal", annotation_position="top left")
    fig.add_vrect(x0=118, x1=122, fillcolor="rgba(0,255,0,0.15)", line_width=0, row=1, col=2)
    
    fig.update_layout(title='<b>Multivariate Analysis (PLS Regression)</b>', showlegend=False)
    fig.update_xaxes(title_text='Wavelength', row=1, col=1); fig.update_yaxes(title_text='Absorbance', row=1, col=1)
    fig.update_xaxes(title_text='Wavelength', row=1, col=2); fig.update_yaxes(title_text='VIP Score', row=1, col=2)
    
    return fig, model_r2, model_q2, optimal_n_comp

def plot_clustering(separation=15, spread=2.5):
    """
    Generates dynamic clustering plots and an Elbow Method plot based on user-defined separation and spread.
    """
    np.random.seed(42)
    n_points_per_cluster = 50
    
    # --- Dynamic Data Generation ---
    X1 = np.random.normal(10, spread, n_points_per_cluster)
    Y1 = np.random.normal(10, spread, n_points_per_cluster)
    X2 = np.random.normal(10 + separation, spread, n_points_per_cluster)
    Y2 = np.random.normal(10 + separation, spread, n_points_per_cluster)
    X3 = np.random.normal(10, spread, n_points_per_cluster)
    Y3 = np.random.normal(10 + separation, spread, n_points_per_cluster)
    
    X = np.concatenate([X1, X2, X3])
    Y = np.concatenate([Y1, Y2, Y3])
    df = pd.DataFrame({'X': X, 'Y': Y})
    
    # --- 1. Generate Main Scatter Plot (fixed at k=3 for visualization) ---
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(df)
    df['Cluster'] = kmeans_3.labels_.astype(str)
    silhouette_val = silhouette_score(df[['X', 'Y']], df['Cluster'])

    fig_scatter = px.scatter(df, x='X', y='Y', color='Cluster', 
                             title='<b>Discovered Process Regimes</b>',
                             labels={'X': 'Process Parameter 1', 'Y': 'Process Parameter 2'})
    centers = kmeans_3.cluster_centers_
    fig_scatter.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1],
                                     mode='markers',
                                     marker=dict(color='black', size=15, symbol='cross'),
                                     name='Centroids'))
    fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    fig_scatter.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    # --- 2. Generate Elbow Method Plot ---
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(df[['X', 'Y']])
        wcss.append(kmeans.inertia_) # Inertia is the WCSS

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=wcss, mode='lines+markers'))
    fig_elbow.update_layout(title='<b>Elbow Method for Selecting k</b>',
                            xaxis_title='Number of Clusters (k)',
                            yaxis_title='Inertia (WCSS)')
    # Add an annotation to highlight the "elbow"
    fig_elbow.add_annotation(x=3, y=wcss[2],
                             text="The 'Elbow'",
                             showarrow=True, arrowhead=2, arrowcolor='red', ax=40, ay=-40)
                             
    return fig_scatter, fig_elbow, silhouette_val

def plot_classification_models(boundary_radius=12):
    """
    Generates dynamic classification plots based on a user-defined boundary complexity.
    """
    np.random.seed(1)
    n_points = 200
    X1 = np.random.uniform(0, 10, n_points)
    X2 = np.random.uniform(0, 10, n_points)
    
    # --- Dynamic Data Generation ---
    # The decision boundary is a circle centered at (5,5). The slider controls its radius.
    # A smaller radius creates a more complex, non-linear "island" of failure.
    distance_from_center_sq = (X1 - 5)**2 + (X2 - 5)**2
    # Use a sigmoid to create a soft, probabilistic boundary
    prob_of_failure = 1 / (1 + np.exp(distance_from_center_sq - boundary_radius))
    y = np.random.binomial(1, prob_of_failure) # 1 = Fail (red), 0 = Pass (blue)
    
    X = np.column_stack((X1, X2))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Re-fit models and calculate dynamic KPIs ---
    lr = LogisticRegression().fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)

    # Create meshgrid for decision boundary
    xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'<b>Logistic Regression (Accuracy: {lr_score:.2%})</b>', 
                                                       f'<b>Random Forest (Accuracy: {rf_score:.2%})</b>'))

    # Plot Logistic Regression
    Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_lr, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu_r', line=dict(width=1, color='black'))), row=1, col=1)

    # Plot Random Forest
    Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_rf, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=2)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu_r', line=dict(width=1, color='black'))), row=1, col=2)

    fig.update_layout(title="<b>Predictive QC: Linear vs. Non-Linear Models</b>", showlegend=False, height=500)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=1); fig.update_yaxes(title_text="Parameter 2", row=1, col=1)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=2); fig.update_yaxes(title_text="Parameter 2", row=1, col=2)
    
    return fig, lr_score, rf_score

def wilson_score_interval(p_hat, n, z=1.96):
    # This helper function remains the same, but it's good to keep it near the plotting function.
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n; term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2))); return (term1 - term2) / denom, (term1 + term2) / denom
    
def plot_isolation_forest(contamination_rate=0.1):
    """
    Generates dynamic anomaly detection plots based on a user-defined contamination rate.
    """
    np.random.seed(42)
    n_normal = 100
    n_anomalies = 15 # Generate a fixed number of potential anomalies
    
    # --- Data Generation ---
    X_inliers = np.random.normal(0, 1, (n_normal, 2))
    # Outliers are generated further away to be clearly distinct
    X_outliers = np.random.uniform(low=-5, high=5, size=(n_anomalies, 2))
    X = np.concatenate([X_inliers, X_outliers], axis=0)
    
    # --- Dynamic Model Fitting ---
    # The 'contamination' parameter is now controlled by the slider
    clf = IsolationForest(contamination=contamination_rate, random_state=42)
    y_pred = clf.fit_predict(X)
    
    df = pd.DataFrame(X, columns=['Process Parameter 1', 'Process Parameter 2'])
    df['Status'] = ['Anomaly' if p == -1 else 'Normal' for p in y_pred]

    # --- Dynamic KPI Calculation ---
    num_flagged = (y_pred == -1).sum()

    # --- Plotting ---
    fig = px.scatter(df, x='Process Parameter 1', y='Process Parameter 2', color='Status',
                     color_discrete_map={'Normal': '#636EFA', 'Anomaly': '#EF553B'},
                     title="<b>The AI Bouncer at Work</b>",
                     symbol='Status',
                     symbol_map={'Normal': 'circle', 'Anomaly': 'x'})
    
    fig.update_traces(marker=dict(size=12, line=dict(width=2)), selector=dict(type='scatter', mode='markers'))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    return fig, num_flagged
    
@st.cache_data
def plot_xai_shap(case_to_explain="highest_risk"):
    """
    Trains a model on assay data, finds a specific case (e.g., highest risk),
    and generates SHAP explanations for it.
    """
    plt.style.use('default')
    
    # 1. Simulate Assay Data and Train Model (This part is cached)
    np.random.seed(42)
    n_runs = 200
    operator_experience = np.random.randint(1, 25, n_runs)
    cal_slope = np.random.normal(1.0, 0.05, n_runs) - (operator_experience * 0.001)
    qc1_value = np.random.normal(50, 2, n_runs) - np.random.uniform(0, operator_experience / 10, n_runs)
    reagent_age_days = np.random.randint(5, 90, n_runs)
    instrument_id = np.random.choice(['Inst_A', 'Inst_B', 'Inst_C'], n_runs, p=[0.5, 0.3, 0.2])
    prob_failure = 1 / (1 + np.exp(-(-2.5 - 0.15 * operator_experience + (reagent_age_days / 30) - (cal_slope - 1.0) * 20 + (instrument_id == 'Inst_C') * 1.5)))
    run_failed = np.random.binomial(1, prob_failure)

    X_display = pd.DataFrame({
        'Operator Experience (Months)': operator_experience,
        'Reagent Age (Days)': reagent_age_days,
        'Calibrator Slope': cal_slope,
        'QC Level 1 Value': qc1_value,
        'Instrument ID': instrument_id
    })
    y = pd.Series(run_failed, name="Run Failed")
    X = X_display.copy()
    X['Instrument ID'] = X['Instrument ID'].astype('category').cat.codes

    model = RandomForestClassifier(random_state=42).fit(X, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # --- 2. Find the instance index based on the selected case ---
    failure_probabilities = model.predict_proba(X)[:, 1]
    
    if case_to_explain == "highest_risk":
        instance_index = np.argmax(failure_probabilities)
    elif case_to_explain == "lowest_risk":
        instance_index = np.argmin(failure_probabilities)
    else: # "most_ambiguous"
        instance_index = np.argmin(np.abs(failure_probabilities - 0.5))

    # 3. Generate Global Summary Plot (always the same)
    shap.summary_plot(shap_values.values[:,:,1], X, show=False)
    buf_summary = io.BytesIO()
    plt.savefig(buf_summary, format='png', bbox_inches='tight')
    plt.close()
    buf_summary.seek(0)
    
    # 4. Generate Local Force Plot for the dynamically found instance
    force_plot = shap.force_plot(
        explainer.expected_value[1], 
        shap_values.values[instance_index,:,1], 
        X_display.iloc[instance_index,:],
        show=False
    )
    force_plot_html = force_plot.html()
    full_html = f"<html><head>{shap.initjs()}</head><body>{force_plot_html}</body></html>"
    
    actual_outcome = "Failed" if y.iloc[instance_index] == 1 else "Passed"
    
    # Return the found index for display in the UI
    return buf_summary, full_html, X_display.iloc[instance_index:instance_index+1], actual_outcome, instance_index
    
@st.cache_data
def plot_advanced_ai_concepts(concept):
    """
    Generates SME-designed, conceptually rich Plotly figures for advanced AI topics
    in a V&V and Tech Transfer context.
    """
    fig = go.Figure()

    # --- Case 1: Transformers - The AI Historian for Batch Records ---
    if concept == "Transformers":
        steps = ["Inoculation", "Growth Phase", "Feed 1", "Production", "Harvest"]
        x = [1, 3, 5, 7, 9]
        y = [2, 2, 2, 2, 2]
        
        # Draw the sequence of batch events
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=steps,
                               textposition="top center", textfont=dict(size=14),
                               marker=dict(size=30, color='lightblue', line=dict(width=2, color='black'))))
        
        # Draw lines connecting the sequence
        for i in range(len(x) - 1):
            fig.add_shape(type="line", x0=x[i]+0.5, y0=y[i], x1=x[i+1]-0.5, y1=y[i+1],
                          line=dict(color="grey", width=2, dash="dash"))
        
        # --- Visualize Self-Attention ---
        # The model, when predicting the "Harvest" outcome, pays high attention to the "Growth Phase"
        fig.add_annotation(x=x[1], y=y[1], ax=x[4], ay=y[4],
                           xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=2, arrowwidth=4, arrowcolor='rgba(255, 65, 54, 0.8)',
                           text="<b>High Attention Link</b>")
        # It pays low attention to an irrelevant intermediate step
        fig.add_annotation(x=x[2], y=y[2], ax=x[4], ay=y[4],
                           xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor='rgba(150, 150, 150, 0.6)',
                           text="Low Attention")

        fig.update_layout(title_text="<b>Transformer: Understanding the Entire Batch Narrative</b>")

    # --- Case 2: Graph Neural Networks (GNNs) - The System-Wide Investigator ---
    elif concept == "Graph Neural Networks (GNNs)":
        nodes = {
            'Raw Mat A': {'x': 1, 'y': 4, 'color': '#FFDDC1'},
            'Raw Mat B': {'x': 1, 'y': 2, 'color': '#FFDDC1'},
            'Bioreactor 1': {'x': 3, 'y': 5, 'color': '#C1E1C1'},
            'Bioreactor 2': {'x': 3, 'y': 1, 'color': '#C1E1C1'},
            'Batch 101': {'x': 5, 'y': 5, 'color': '#AEC6CF'},
            'Batch 102': {'x': 5, 'y': 3, 'color': '#AEC6CF'},
            'Batch 103': {'x': 5, 'y': 1, 'color': '#AEC6CF'},
        }
        edges = [('Raw Mat A', 'Batch 101'), ('Raw Mat A', 'Batch 102'), ('Raw Mat B', 'Batch 103'),
                 ('Bioreactor 1', 'Batch 101'), ('Bioreactor 1', 'Batch 102'), ('Bioreactor 2', 'Batch 103')]

        for start, end in edges:
            fig.add_shape(type="line", x0=nodes[start]['x'], y0=nodes[start]['y'], x1=nodes[end]['x'], y1=nodes[end]['y'], line=dict(color="grey", width=2))
        
        # Highlight a failure and trace it back
        nodes['Batch 102']['color'] = '#FF6961' # Red for failure
        nodes['Raw Mat A']['color'] = '#FFB347'  # Orange for suspected cause

        for name, props in nodes.items():
            fig.add_trace(go.Scatter(x=[props['x']], y=[props['y']], mode='markers+text',
                                     text=name.replace(' ', '<br>'), textposition="middle center",
                                     marker=dict(size=80, color=props['color'], line=dict(width=2, color='black'))))

        fig.add_annotation(text="<b>Failure Signal Propagates Backwards</b>", x=3, y=3, showarrow=False, font=dict(size=14, color='red'))
        fig.update_layout(title_text="<b>GNN: Tracing a Failure Back Through the Supply Chain</b>")

    # --- Case 3: Reinforcement Learning (RL) - The Digital Twin Pilot ---
    elif concept == "Reinforcement Learning (RL)":
        # Agent
        fig.add_shape(type="rect", x0=0.5, y0=3, x1=2.5, y1=4.5, fillcolor='lightblue', line=dict(width=2))
        fig.add_annotation(x=1.5, y=3.75, text="<b>AI Agent</b><br>(Control Policy)", showarrow=False)
        # Digital Twin
        fig.add_shape(type="rect", x0=4.5, y0=0.5, x1=9.5, y1=5, fillcolor='lightgrey', line=dict(width=2, dash='dash'))
        fig.add_annotation(x=7, y=4.5, text="<b>Digital Twin (Safe Simulation)</b>", showarrow=False)
        # Real Process
        fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=2.5, y1=2, fillcolor='lightgreen', line=dict(width=2))
        fig.add_annotation(x=1.5, y=1.25, text="<b>Real Process</b>", showarrow=False)
        
        # Arrows for the training loop
        fig.add_annotation(x=4.3, y=3.75, ax=2.7, ay=3.75, text="<b>Action</b> (e.g., Set Feed Rate)", showarrow=True, arrowhead=2)
        fig.add_annotation(x=2.7, y=3, ax=4.3, ay=3, text="<b>State, Reward</b>", showarrow=True, arrowhead=2)
        # Arrow for deployment
        fig.add_annotation(x=1.5, y=2.2, ax=1.5, ay=2.8, text="Deploy<br>Optimal<br>Policy", showarrow=True, arrowhead=2, arrowcolor='darkgreen')

        # Mini control chart inside the Digital Twin
        x_sim = np.linspace(5, 9, 20)
        y_sim = np.sin(x_sim*2)*0.2 + 2.5
        fig.add_trace(go.Scatter(x=x_sim, y=y_sim, mode='lines', line=dict(color='royalblue')))
        fig.add_hline(y=2.5, line=dict(color='black', dash='dot'), line_width=1)
        fig.update_layout(title_text="<b>RL: Learning Optimal Control in a Digital Twin</b>")

    # --- Case 4: Generative AI - Solving the Rare Event Problem ---
    elif concept == "Generative AI":
        np.random.seed(42)
        # Real Data (very few points)
        x_real = [2, 2.5, 3]
        y_real = [8, 9, 8.5]
        # Synthetic Data (many points, similar pattern)
        x_synth = np.random.normal(2.5, 0.8, 100)
        y_synth = np.random.normal(8.5, 0.8, 100)
        
        fig.add_trace(go.Scatter(x=x_real, y=y_real, mode='markers', name='Real Failure Data',
                                 marker=dict(color='red', size=15, symbol='x-thin', line=dict(width=3))))
        fig.add_trace(go.Scatter(x=x_synth, y=y_synth, mode='markers', name='Synthetic Failure Data',
                                 marker=dict(color='rgba(255,165,0,0.6)', size=8, symbol='circle')))
        
        fig.add_annotation(x=6, y=4, text="<b>Generative Model</b><br>(AI Forger)", showarrow=False,
                           font=dict(size=16), bordercolor='black', borderwidth=2, bgcolor='gold', align='center')
        fig.add_annotation(x=3.5, y=8.5, ax=5.5, ay=4.5, text="Learns From", showarrow=True, arrowhead=2)
        fig.add_annotation(x=5.5, y=4.5, ax=3.5, ay=8.5, text="Creates", showarrow=True, arrowhead=2)
        
        fig.update_layout(title_text="<b>Generative AI: Creating Synthetic Data for Rare Failures</b>",
                          xaxis_title="Process Parameter A", yaxis_title="Process Parameter B")

    # Standardize the final layout for all plots
    fig.update_layout(xaxis=dict(visible=True, showgrid=False, zeroline=False), 
                      yaxis=dict(visible=True, showgrid=False, zeroline=False), 
                      height=400, 
                      showlegend=True,
                      margin=dict(l=10, r=10, t=50, b=10))
    if concept in ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)"]:
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, showlegend=False)
                      
    return fig
    


# ==============================================================================
# ALL UI RENDERING FUNCTIONS
# ==============================================================================

# --- RESTORED INTRO RENDERING FUNCTION ---
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
        st.markdown("Here, the method faces its crucible. It must prove its performance in a new environment—a new lab, a new scale, a new team. This is about demonstrating stability and equivalence.")
    with act3: 
        st.subheader("Act III: The Guardian (Lifecycle Management)")
        st.markdown("Once live, the journey isn't over. This final act is about continuous guardianship: monitoring process health, detecting subtle drifts, and using advanced analytics to predict and prevent future failures.")
    
    st.divider()

    st.header("🚀 The V&V Model: A Strategic Framework")
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework for ensuring a system meets its intended purpose, from initial requirements to final deployment.")
    st.plotly_chart(plot_v_model(), use_container_width=True)
    
    st.divider()
    
    st.header("📈 Project Workflow")
    st.markdown("This timeline organizes the entire toolkit by its application in a typical project lifecycle. Tools are grouped by the project phase where they provide the most value, and are ordered chronologically within each phase.")
    st.plotly_chart(plot_act_grouped_timeline(), use_container_width=True)

        # --- ADDED THIS NEW SECTION ---
    st.header("⏳ A Chronological View of V&V Analytics")
    st.markdown("This timeline organizes the same tools purely by their year of invention, showing the evolution of statistical and machine learning thought over the last century.")
    st.plotly_chart(plot_chronological_timeline(), use_container_width=True)

    st.header("🗺️ Conceptual Map of Tools")
    st.markdown("This map illustrates the relationships between the foundational concepts and the specific tools available in this application. Use it to navigate how different methods connect to broader analytical strategies.")
    st.plotly_chart(create_toolkit_conceptual_map(), use_container_width=True)

# ==============================================================================
# UI RENDERING FUNCTIONS (ALL DEFINED BEFORE MAIN APP LOGIC)
# ==============================================================================
def render_ci_concept():
    """Renders the interactive module for Confidence Intervals."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To build a deep, intuitive understanding of the fundamental concept of a **frequentist confidence interval** and to correctly interpret what it does—and does not—tell us.
    
    **Strategic Application:** This concept is the bedrock of all statistical inference in a frequentist framework. A misunderstanding of CIs leads to flawed conclusions and poor decision-making. This interactive simulation directly impacts resource planning and risk assessment. It allows scientists and managers to explore the crucial trade-off between **sample size (cost)** and **statistical precision (certainty)**. It provides a visual, data-driven answer to the perpetual question: "How many samples do we *really* need to run to get a reliable result and an acceptably narrow confidence interval?"
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Sample Size (n)** slider in the sidebar to dynamically change the number of samples in each simulated experiment. Observe how increasing the sample size dramatically narrows both the theoretical Sampling Distribution (orange curve) and the simulated Confidence Intervals (blue/red lines), directly demonstrating the link between sample size and precision.
    """)

    n_slider = st.sidebar.slider("Select Sample Size (n) for Each Simulated Experiment:", 5, 100, 30, 5)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_ci, fig2_ci, capture_count, n_sims, avg_width = plot_ci_concept(n=n_slider)
        st.plotly_chart(fig1_ci, use_container_width=True)
        st.plotly_chart(fig2_ci, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label=f"📈 KPI: Average CI Width (Precision) at n={n_slider}", value=f"{avg_width:.2f} units", help="A smaller width indicates higher precision. This is inversely proportional to the square root of n.")
            st.metric(label="💡 Empirical Coverage Rate", value=f"{(capture_count/n_sims):.0%}", help=f"The % of our {n_sims} simulated CIs that successfully 'captured' the true population mean. Should be close to 95%.")
            st.markdown("""
            - **Theoretical Universe (Top Plot):**
                - The wide, light blue curve is the **true population distribution**. In real life, we *never* see this.
                - The narrow, orange curve is the **sampling distribution of the mean**. Its narrowness, guaranteed by the **Central Limit Theorem**, makes inference possible.
            - **CI Simulation (Bottom Plot):** This shows the reality we live in. We only get *one* experiment and *one* confidence interval.
            - **The n-slider is key:** As you increase `n`, the orange curve gets narrower and the CIs in the bottom plot become dramatically shorter.
            - **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This illustrates that to double your precision (halve the CI width), you must quadruple your sample size.

            **The Core Strategic Insight:** A confidence interval is a statement about the *procedure*, not a specific result. The "95% confidence" is our confidence in the *method* used to generate the interval, not in any single interval itself.
            """)
        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT (Bayesian) INTERPRETATION:**
            *"Based on my sample, there is a 95% probability that the true mean is in this interval [X, Y]."*
            
            This is wrong because in the frequentist view, the true mean is a fixed constant. It is either in our specific interval or it is not. The probability is 1 or 0.
            """)
            st.success("""
            🟢 **THE CORRECT (Frequentist) INTERPRETATION:**
            *"We are 95% confident that the interval [X, Y] contains the true mean."*
            
            The full meaning is: *"This interval was constructed using a procedure that, when repeated many times, will produce intervals that capture the true mean 95% of the time."*
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of **confidence intervals** was introduced by **Jerzy Neyman** in a landmark 1937 paper. Neyman's revolutionary idea was to shift the probabilistic statement away from the fixed, unknown parameter and onto the **procedure used to create the interval**. This clever reframing provided a practical and logically consistent solution that remains the dominant paradigm for interval estimation today.
            
            #### Mathematical Basis
            """)
            st.latex(r"\text{Point Estimate} \pm (\text{Critical Value} \times \text{Standard Error})")
            st.markdown("""
            - **Point Estimate:** Our best guess (e.g., the sample mean, $\bar{x}$).
            - **Standard Error:** The standard deviation of the sampling distribution of the point estimate (e.g., $\frac{s}{\sqrt{n}}$). It measures the typical error in our point estimate.
            - **Critical Value:** A multiplier determined by our desired confidence level (e.g., a t-score).
            """)

def render_core_validation_params():
    """Renders the INTERACTIVE module for core validation parameters."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the fundamental performance characteristics of an analytical method as required by global regulatory guidelines like ICH Q2(R1). This module deconstructs the "big three" pillars of method validation:
    - **🎯 Accuracy (Bias):** How close are your measurements to the *real* value?
    - **🏹 Precision (Random Error):** How consistent are your measurements with each other?
    - **🔬 Specificity (Selectivity):** Can your method find the target analyte in a crowded room, ignoring all the imposters?

    **Strategic Application:** These parameters are the non-negotiable pillars of any formal assay validation report. A weakness in any of these three areas is a critical deficiency that can lead to rejected submissions or flawed R&D conclusions. **Use the sliders in the sidebar to simulate different error types and see their impact on the plots.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you navigate to the "Core Validation Parameters" tool, you will see a new set of dedicated sliders in the sidebar. Changing these sliders will instantly update the three plots, allowing you to build a powerful, hands-on intuition for these critical concepts.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Core Validation Controls")
    bias_slider = st.sidebar.slider(
        "🎯 Systematic Bias (%)", 
        min_value=-10.0, max_value=10.0, value=1.5, step=0.5,
        help="Simulates a constant positive or negative bias in the accuracy study. Watch the box plots shift."
    )
    repeat_cv_slider = st.sidebar.slider(
        "🏹 Repeatability %CV", 
        min_value=0.5, max_value=10.0, value=1.5, step=0.5,
        help="Simulates the best-case random error (intra-assay precision). Watch the 'Repeatability' violin width."
    )
    # Ensure intermediate precision is always worse than or equal to repeatability
    intermed_cv_slider = st.sidebar.slider(
        "🏹 Intermediate Precision %CV", 
        min_value=repeat_cv_slider, max_value=20.0, value=max(2.5, repeat_cv_slider), step=0.5,
        help="Simulates real-world random error (inter-assay). A large gap from repeatability indicates poor robustness."
    )
    interference_slider = st.sidebar.slider(
        "🔬 Interference Effect (%)", 
        min_value=-20.0, max_value=20.0, value=8.0, step=1.0,
        help="Simulates an interferent that falsely increases (+) or decreases (-) the analyte signal."
    )
    
    # Generate plots using the slider values
    fig1, fig2, fig3 = plot_core_validation_params(
        bias_pct=bias_slider, 
        repeat_cv=repeat_cv_slider, 
        intermed_cv=intermed_cv_slider, 
        interference_effect=interference_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.info("Play with the sliders in the sidebar to see how different sources of error affect the results!")
            st.markdown("""
            - **Accuracy Plot:** As you increase the **Systematic Bias** slider, watch the center of the box plots drift away from the black 'True Value' lines. This visually demonstrates what bias looks like.
            
            - **Precision Plot:** The **%CV sliders** control the width (spread) of the violin plots. Notice that Intermediate Precision must always be equal to or worse (wider) than Repeatability. A large gap between the two signals that the method is not robust to day-to-day changes.
            
            - **Specificity Plot:** The **Interference Effect** slider directly moves the 'Analyte + Interferent' box plot. A perfect assay would have this slider at 0%, making the two boxes identical. A large effect, positive or negative, indicates a failed specificity study.

            **The Core Strategic Insight:** This simulation shows that validation is a process of hunting for and quantifying different types of error. Accuracy is about finding *bias*, Precision is about characterizing *random error*, and Specificity is about eliminating *interference error*.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Validation Theater"**
            The goal of validation is to get the protocol to pass by any means necessary.
            
            - *"My precision looks bad, so I'll have my most experienced 'super-analyst' run the experiment to guarantee a low %CV."*
            - *"The method failed accuracy at the low concentration. I'll just change the reportable range to exclude that level."*
            - *"I'll only test for interference from things I know won't be a problem and ignore the complex sample matrix."*
            
            This approach treats validation as a bureaucratic hurdle. It produces a method that is fragile, unreliable in the real world, and a major compliance risk.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Rigorously Prove "Fitness for Purpose"**
            The goal of validation is to **honestly and rigorously challenge the method to prove it is robust and reliable for its specific, intended analytical application.**
            
            - This means deliberately including variability (different analysts, days, instruments) to prove the method can handle it.
            - It means understanding and documenting *why* a method fails at a certain level, not just hiding the failure.
            - It means demonstrating specificity in the actual, messy matrix the method will be used for.
            
            This approach builds a truly robust method that generates trustworthy data, ensuring product quality and patient safety.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            Before the 1990s, a pharmaceutical company wishing to market a new drug globally had to prepare different, massive submission packages for each region (USA, Europe, Japan), each with slightly different technical requirements for method validation. This created enormous, costly, and scientifically redundant work.
            
            In 1990, the **International Council for Harmonisation (ICH)** was formed, bringing together regulators and industry to create a single set of harmonized guidelines. The **ICH Q2(R1) guideline, "Validation of Analytical Procedures,"** is the direct result. It is the global "bible" for this topic, and the parameters of Accuracy, Precision, and Specificity form its core. Adhering to ICH Q2(R1) ensures your data is acceptable to major regulators worldwide.
            
            #### Mathematical Basis
            The validation report is a statistical argument built on quantitative metrics.
            """)
            st.markdown("**Accuracy is measured by Percent Recovery:**")
            st.latex(r"\% \text{Recovery} = \frac{\text{Mean Experimental Value}}{\text{Known True Value}} \times 100\%")
            
            st.markdown("**Precision is measured by Percent Coefficient of Variation (%CV):**")
            st.latex(r"\% \text{CV} = \frac{\text{Standard Deviation (SD)}}{\text{Mean}} \times 100\%")
            
            st.markdown("""
            **Specificity is often assessed via Hypothesis Testing:** A Student's t-test compares the means of the "Analyte Only" and "Analyte + Interferent" groups. The null hypothesis ($H_0$) is that the means are equal. A high p-value (e.g., > 0.05) means we fail to reject $H_0$, providing evidence that the interferent has no significant effect.
            """)
            
def render_gage_rr():
    """Renders the INTERACTIVE module for Gage R&R."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To rigorously quantify the inherent variability (error) of a measurement system. It answers the fundamental question: "Is my measurement system a precision instrument, or a random number generator?"
    
    **Strategic Application:** A foundational checkpoint in any technology transfer or process validation. An unreliable measurement system creates a "fog of uncertainty," leading to two costly errors: rejecting good product (False Alarm) or accepting bad product (Missed Signal). **Use the sliders in the sidebar to simulate different sources of variation and see their impact on the final % Gage R&R.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you navigate to the "Gage R&R / VCA" tool, you will see a new set of dedicated sliders in the sidebar. You can now dynamically simulate how a precise (low repeatability) or imprecise (high repeatability) instrument performs, and how well-trained (low operator variation) or poorly-trained (high operator variation) teams affect the final result.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Gage R&R Controls")
    part_sd_slider = st.sidebar.slider(
        "🏭 Part-to-Part Variation (SD)", 
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="The 'true' variation of the product. Increase this to see how a good measurement system can easily distinguish between different parts."
    )
    repeat_sd_slider = st.sidebar.slider(
        "🔬 Repeatability (SD)", 
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help="The inherent 'noise' of the instrument/assay. Increase this to simulate a less precise measurement device."
    )
    operator_sd_slider = st.sidebar.slider(
        "👤 Operator-to-Operator Variation (SD)", 
        min_value=0.0, max_value=5.0, value=0.75, step=0.25,
        help="The systematic bias between operators. Increase this to simulate poor training or inconsistent technique."
    )
    
    # Generate plots using the slider values
    fig, pct_rr, pct_part = plot_gage_rr(
        part_sd=part_sd_slider, 
        repeatability_sd=repeat_sd_slider, 
        operator_sd=operator_sd_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse", help="The percentage of total variation consumed by measurement error.")
            st.metric(label="💡 KPI: Number of Distinct Categories (ndc)", value=f"{int(1.41 * (pct_part / pct_rr)**0.5) if pct_rr > 0 else '>10'}", help="An estimate of how many distinct groups the system can discern. A value < 5 is problematic.")

            st.info("Play with the sliders in the sidebar to see how different sources of error affect the results!")
            st.markdown("""
            - **Increase `Part-to-Part Variation`:** Notice how the operator means (colored lines) spread further apart. A good measurement system should show this! Crucially, the **% Gage R&R KPI goes DOWN**, because the measurement error is now a smaller proportion of the total variation.
            - **Increase `Repeatability`:** The box plots for each part get much wider. This is pure measurement noise. The **% Gage R&R KPI goes UP**.
            - **Increase `Operator-to-Operator Variation`:** The colored mean lines separate vertically and the overall operator boxes (top right) drift apart. This is bias between people. The **% Gage R&R KPI goes UP**.

            **The Core Strategic Insight:** A low % Gage R&R is achieved when the measurement error (Repeatability + Reproducibility) is small *relative to* the true process variation.
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria are risk-based and derived from the **AIAG's Measurement Systems Analysis (MSA)** manual, the de facto global standard.")
            st.markdown("- **< 10% Gage R&R:** The system is **acceptable**.")
            st.markdown("- **10% - 30% Gage R&R:** The system is **conditionally acceptable or marginal**.")
            st.markdown("- **> 30% Gage R&R:** The system is **unacceptable and must be rejected**.")
            st.info("""
            **The Part Selection Strategy:** The most common failure mode of a Gage R&R study is not the math, but the study design. The parts selected **must span the full expected range of process variation**. If you only select parts from the middle of the distribution, your Part-to-Part variation will be artificially low, which will mathematically inflate your % Gage R&R and cause a good system to fail.
            """)
            
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The modern application was born out of the quality crisis in the American automotive industry in the 1970s. The **AIAG** codified the solution in the first MSA manual. The critical evolution was the move from the simple **Average and Range (X-bar & R) method** to the **ANOVA method**. The ANOVA method, pioneered by **Sir Ronald A. Fisher**, became the gold standard because of its unique ability to cleanly partition and test the significance of each variance component, including the crucial interaction term.
            
            #### Mathematical Basis
            The ANOVA method partitions the total sum of squared deviations from the mean ($SS_T$) into components attributable to each factor.
            """)
            st.latex(r"SS_{Total} = SS_{Part} + SS_{Operator} + SS_{Part \times Operator} + SS_{Error}")
            st.markdown("These are converted to Mean Squares (MS) and then to variance components ($\hat{\sigma}^2$).")

def render_lod_loq():
    """Renders the INTERACTIVE module for Limit of Detection & Quantitation."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the absolute lower performance boundaries of a quantitative assay. It determines the lowest analyte concentration an assay can reliably **detect (LOD)** and the lowest concentration it can reliably and accurately **quantify (LOQ)**.
    
    **Strategic Application:** This is a mission-critical parameter for any assay used to measure trace components, such as impurities in a drug product or biomarkers for early-stage disease diagnosis. **Use the sliders in the sidebar to simulate how assay sensitivity and noise impact the final LOD and LOQ.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you select the "LOD & LOQ" tool, a new set of dedicated sliders will appear in the sidebar. You can dynamically change the assay's slope and noise to see in real-time how these fundamental characteristics drive the final LOD and LOQ results.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("LOD & LOQ Controls")
    slope_slider = st.sidebar.slider(
        "📈 Assay Sensitivity (Slope)", 
        min_value=0.005, max_value=0.1, value=0.02, step=0.005, format="%.3f",
        help="How much the signal increases per unit of concentration. A steeper slope (higher sensitivity) is better."
    )
    noise_slider = st.sidebar.slider(
        "🔇 Baseline Noise (SD)", 
        min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f",
        help="The inherent random noise of the assay at zero concentration. A lower noise floor is better."
    )
    
    # Generate plots using the slider values
    fig, lod_val, loq_val = plot_lod_loq(slope=slope_slider, baseline_sd=noise_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} units", help="The lowest concentration you can report with confidence in the numerical value.")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} units", help="The lowest concentration you can reliably claim is 'present'.")
            st.info("Play with the sliders in the sidebar to see how assay parameters affect the results!")
            st.markdown("""
            - **Increase `Assay Sensitivity (Slope)`:** As the slope gets steeper, the LOD and LOQ values get **lower (better)**. A highly sensitive assay needs very little analyte to produce a strong signal that can overcome the noise.
            - **Increase `Baseline Noise (SD)`:** As the noise floor of the assay increases, the LOD and LOQ values get **higher (worse)**. It becomes much harder to distinguish a true low-level signal from random background fluctuations.

            **The Core Strategic Insight:** The sensitivity of an assay is a direct battle between its **signal-generating power (Slope)** and its **inherent noise (SD)**. The LOD and LOQ are simply the statistical formalization of this signal-to-noise ratio.
            """)

        with tabs[1]:
            st.markdown("- The primary, non-negotiable criterion is that the experimentally determined **LOQ must be ≤ the lowest concentration that the assay is required to measure** for its specific application (e.g., a release specification for an impurity).")
            st.markdown("- For a concentration to be formally declared the LOQ, it must be experimentally confirmed. This typically involves analyzing 5-6 independent samples at the claimed LOQ concentration and demonstrating that they meet pre-defined criteria for precision and accuracy (e.g., **%CV < 20% and %Recovery between 80-120%** for a bioassay).")
            st.warning("""
            **The LOB, LOD, and LOQ Hierarchy: A Critical Distinction**
            A full characterization involves three distinct limits:
            - **Limit of Blank (LOB):** The highest measurement expected from a blank sample.
            - **Limit of Detection (LOD):** The lowest concentration whose signal is statistically distinguishable from the LOB.
            - **Limit of Quantitation (LOQ):** The lowest concentration meeting precision/accuracy requirements, which is almost always higher than the LOD.
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The need to define analytical sensitivity is old, but definitions were inconsistent until the **International Council for Harmonisation (ICH)** guideline **ICH Q2(R1)** harmonized the methodologies. This work was heavily influenced by the statistical framework established by **Lloyd Currie at NIST** in his 1968 paper, which established the clear, hypothesis-testing basis for the modern LOB/LOD/LOQ hierarchy.

            #### Mathematical Basis
            This method is built on the relationship between the assay's signal, its sensitivity (Slope, S), and its noise (standard deviation, σ).
            """)
            st.latex(r"LOD \approx \frac{3.3 \times \sigma}{S}")
            st.latex(r"LOQ \approx \frac{10 \times \sigma}{S}")
            st.markdown("The factor of 10 for LOQ is the standard convention that typically yields a precision of roughly 10% CV for a well-behaved assay.")

def render_linearity():
    """Renders the INTERACTIVE module for Linearity analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify that an assay's response is directly and predictably proportional to the known concentration of the analyte across its entire intended operational range.
    
    **Strategic Application:** This is a cornerstone of quantitative assay validation. A method exhibiting non-linearity might be accurate at a central control point but dangerously inaccurate at the specification limits. **Use the sliders in the sidebar to simulate different types of non-linear behavior and error.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you navigate to the "Linearity & Range" tool, you will see a new set of dedicated sliders in the sidebar. You can now dynamically simulate how a perfect assay, one with detector saturation, or one with increasing error at higher concentrations would appear in a validation report, providing a powerful learning experience.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Linearity Controls")
    curvature_slider = st.sidebar.slider(
        "🧬 Curvature Effect", 
        min_value=-5.0, max_value=5.0, value=-1.0, step=0.5,
        help="Simulates non-linearity. A negative value creates saturation at high concentrations. A positive value creates expansion. Zero is perfectly linear."
    )
    random_error_slider = st.sidebar.slider(
        "🎲 Random Error (Constant SD)", 
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="The baseline random noise of the assay, constant across all concentrations."
    )
    proportional_error_slider = st.sidebar.slider(
        "📈 Proportional Error (% of Conc.)", 
        min_value=0.0, max_value=5.0, value=2.0, step=0.25,
        help="Error that increases with concentration. This creates a 'funnel' or 'megaphone' shape in the residual plot."
    )
    
    # Generate plots using the slider values
    fig, model = plot_linearity(
        curvature=curvature_slider,
        random_error=random_error_slider,
        proportional_error=proportional_error_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}", help="Indicates the proportion of variance explained by the model. Note how a high R² can hide clear non-linearity!")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}", help="Ideal = 1.0.")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}", help="Ideal = 0.0.")
            
            st.info("Play with the sliders in the sidebar to see how different errors affect the diagnostic plots.")
            st.markdown("""
            - **The Residual Plot is Key:** This is the most sensitive diagnostic tool.
                - Add **Curvature**: Notice the classic "U-shape" or "inverted U-shape" that appears. This is a dead giveaway that your straight-line model is wrong.
                - Add **Proportional Error**: Watch the residuals form a "funnel" or "megaphone" shape. This is heteroscedasticity, and it means you should be using Weighted Least Squares (WLS) regression, not OLS.
            
            **The Core Strategic Insight:** A high R-squared is **not sufficient** to prove linearity. You must visually inspect the residual plot for hidden patterns. The residual plot tells the true story of your model's fit.
            """)

        with tabs[1]:
            st.markdown("These criteria are defined in the validation protocol and must be met to declare the method linear.")
            st.markdown("- **R-squared (R²):** Typically > **0.995**, but for high-precision methods (e.g., HPLC), > **0.999** is often required.")
            st.markdown("- **Slope & Intercept:** The 95% confidence intervals for the slope and intercept should contain 1.0 and 0, respectively.")
            st.markdown("- **Residuals:** There should be no obvious pattern or trend. A formal **Lack-of-Fit test** can be used for objective proof (requires true replicates at each level).")
            st.markdown("- **Recovery:** The percent recovery at each concentration level must fall within a pre-defined range (e.g., 80% to 120% for bioassays).")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The mathematical engine is **Ordinary Least Squares (OLS) Regression**, developed by **Legendre (1805)** and **Gauss (1809)**. The genius of OLS is that it finds the one line that **minimizes the sum of the squared vertical distances (the "residuals")** between the data points and the line.
            
            #### Mathematical Basis
            The goal is to fit a simple linear model:
            """)
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - $y$: The measured response.
            - $x$: The nominal (true) concentration.
            - $\\beta_0$ (Intercept): Constant systematic error.
            - $\\beta_1$ (Slope): Proportional systematic error.
            - $\\epsilon$: Random measurement error.
            """)
# REPLACE the existing render_lod_loq function with this one.
def render_4pl_regression():
    """Renders the INTERACTIVE module for 4-Parameter Logistic (4PL) regression."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately model the characteristic sigmoidal (S-shaped) dose-response relationship found in most immunoassays (e.g., ELISA) and biological assays.
    
    **Strategic Application:** This is the workhorse model for potency assays and immunoassays. The 4PL model allows for the accurate calculation of critical assay parameters like the EC50. **Use the sliders in the sidebar to control the "true" shape of the curve and see how the curve-fitting algorithm performs.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you select the "Non-Linear Regression" tool, you will have a full set of dedicated sliders in the sidebar. You can now build your own "true" 4PL curves and see how well the regression algorithm is able to recover those parameters from noisy data, providing a deep, intuitive feel for how these models work.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("4PL Curve Controls")
    d_slider = st.sidebar.slider(
        "🅾️ Lower Asymptote (d)", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
        help="The 'floor' of the assay signal, often representing background noise."
    )
    a_slider = st.sidebar.slider(
        "🅰️ Upper Asymptote (a)", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="The 'ceiling' of the assay signal, representing saturation."
    )
    c_slider = st.sidebar.slider(
        "🎯 Potency / EC50 (c)", min_value=1.0, max_value=100.0, value=10.0, step=1.0,
        help="The concentration at the curve's midpoint. A lower EC50 means higher potency."
    )
    b_slider = st.sidebar.slider(
        "🅱️ Hill Slope (b)", min_value=0.5, max_value=5.0, value=1.2, step=0.1,
        help="The steepness of the curve. A steeper slope often means a more sensitive assay."
    )
    
    # Generate plots using the slider values
    fig, params = plot_4pl_regression(
        a_true=a_slider, 
        b_true=b_slider, 
        c_true=c_slider, 
        d_true=d_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        a_fit, b_fit, c_fit, d_fit = params
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🅰️ Fitted Upper Asymptote (a)", value=f"{a_fit:.3f}")
            st.metric(label="🅱️ Fitted Hill Slope (b)", value=f"{b_fit:.3f}")
            st.metric(label="🎯 Fitted EC50 (c)", value=f"{c_fit:.3f} units")
            st.metric(label="🅾️ Fitted Lower Asymptote (d)", value=f"{d_fit:.3f}")
            
            st.info("Play with the sliders in the sidebar to change the true curve and see how the fitted parameters (above) respond!")
            st.markdown("""
            - **Asymptotes (a & d):** These sliders control the dynamic range of your assay.
            - **EC50 (c):** This is your main potency result. Moving this slider shifts the entire curve left or right.
            - **Hill Slope (b):** This slider controls the steepness. A steep slope means a narrow, sensitive range.
            
            **The Core Strategic Insight:** The 4PL curve is a complete picture of your assay's performance. Monitoring all four parameters over time enables proactive troubleshooting.
            """)
            
        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Force the Fit"**
            - *"My data isn't S-shaped, so I'll use linear regression on the middle."* (Biases results).
            - *"The model doesn't fit a point well. I'll delete the point."* (Data manipulation).
            - *"My R-squared is 0.999, so the fit must be perfect."* (R-squared is easily inflated).
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Model the Biology, Weight the Variance**
            - **Embrace the 'S' Shape:** Use a non-linear model for non-linear data.
            - **Weight Your Points:** Apply less "weight" to more variable data points for a more robust fit.
            - **Look at the Residuals:** Any pattern in the errors indicates the model is not capturing the data correctly.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The 4PL model is a direct descendant of the **Hill Equation** (1910) by Archibald Hill. It was adapted in the 1970s-80s for immunoassays like ELISA.
            
            #### Mathematical Basis
            """)
            st.latex(r"y = d + \frac{a - d}{1 + (\frac{x}{c})^b}")
            # FIX: Restored the missing markdown block that explains the formula
            st.markdown("""
            - **`y`**: The measured response.
            - **`x`**: The concentration.
            - **`a`**: Upper asymptote.
            - **`d`**: Lower asymptote.
            - **`c`**: EC50 (potency).
            - **`b`**: Hill slope.
            """)
# The code below was incorrectly merged. It is now its own separate function.
def render_roc_curve():
    """Renders the INTERACTIVE module for Receiver Operating Characteristic (ROC) curve analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To solve **The Diagnostician's Dilemma**: a test must correctly identify patients with a disease (high **Sensitivity**) while also correctly clearing healthy patients (high **Specificity**). The ROC curve visualizes this trade-off.
    
    **Strategic Application:** This is the global standard for validating diagnostic tests. The Area Under the Curve (AUC) provides a single metric of a test's diagnostic power. **Use the sliders in the sidebar to see how population separation and overlap affect diagnostic performance.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you select the "ROC Curve Analysis" tool, you will see the new dedicated sliders in the sidebar. You can dynamically create assays that are excellent (high separation, low overlap) or terrible (low separation, high overlap) and see in real-time how the score distributions, the ROC curve shape, and the final AUC value respond.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("ROC Curve Controls")
    separation_slider = st.sidebar.slider(
        "📈 Separation (Diseased Mean)", 
        min_value=50.0, max_value=80.0, value=65.0, step=1.0,
        help="Controls the distance between the Healthy and Diseased populations. More separation = better test."
    )
    overlap_slider = st.sidebar.slider(
        "🌫️ Overlap (Population SD)", 
        min_value=5.0, max_value=20.0, value=10.0, step=0.5,
        help="Controls the 'noise' or spread of the populations. More overlap (a higher SD) = worse test."
    )

    # Generate plots using the slider values
    fig, auc_value = plot_roc_curve(
        diseased_mean=separation_slider, 
        population_sd=overlap_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 KPI: Area Under Curve (AUC)", value=f"{auc_value:.3f}", help="The overall diagnostic power of the test. 0.5 is useless, 1.0 is perfect. Updates with sliders.")
            st.info("Play with the sliders in the sidebar to see how assay quality affects the results!")
            st.markdown("""
            - **Increase `Separation`:** Watch the red distribution move away from the blue one. The ROC curve pushes towards the perfect top-left corner, and the **AUC value increases dramatically.**
            - **Increase `Overlap`:** Watch both distributions get wider. The ROC curve flattens, and the **AUC value decreases.**
            
            **The Core Strategic Insight:** A great diagnostic test is one that maximizes the separation between populations while minimizing their overlap (noise).
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Worship the AUC" & "Hug the Corner"**
            - *"My AUC is 0.95, so we're done."* (The *chosen cutoff* might still be terrible).
            - *"I'll just pick the cutoff closest to the top-left corner."* (This balances errors equally, which is rarely desired).
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Best Cutoff Depends on the Consequence of Being Wrong**
            Ask: **"What is worse? A false positive or a false negative?"**
            - **For deadly disease screening:** Maximize Sensitivity.
            - **For risky surgery diagnosis:** Maximize Specificity.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The ROC curve was invented during **World War II** to help radar operators distinguish enemy bombers from noise. It allowed them to quantify the trade-off between sensitivity and false alarms.
            
            #### Mathematical Basis
            The curve plots **Sensitivity (Y-axis)** versus **1 - Specificity (X-axis)**.
            """)
            st.latex(r"\text{Sensitivity} = \frac{TP}{TP + FN} \quad , \quad \text{Specificity} = \frac{TN}{TN + FP}")

def render_tost():
    """Renders the INTERACTIVE module for Two One-Sided Tests (TOST) for equivalence."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically prove that two methods or groups are **equivalent** within a predefined, practically insignificant margin. This flips the logic of standard hypothesis testing.
    
    **Strategic Application:** This is the statistically rigorous way to handle comparisons where the goal is to prove similarity, not difference, such as in biosimilarity studies or method transfers. **Use the sliders in the sidebar to build an intuition for what drives an equivalence conclusion.**
    """)
    
    st.info("""
    **Interactive Demo:** Now, when you select the "Equivalence Testing (TOST)" tool, you will have a full set of dedicated sliders in the sidebar. You can now dynamically explore how to achieve (or fail to achieve) statistical equivalence, providing a powerful and memorable learning experience.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("TOST Controls")
    delta_slider = st.sidebar.slider(
        "⚖️ Equivalence Margin (Δ)", 
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="The 'goalposts'. Defines the zone where differences are considered practically meaningless. A tighter margin is harder to meet."
    )
    diff_slider = st.sidebar.slider(
        "🎯 True Difference", 
        min_value=-10.0, max_value=10.0, value=1.0, step=0.5,
        help="The actual underlying difference between the two groups in the simulation. See if you can prove equivalence even when a small true difference exists!"
    )
    sd_slider = st.sidebar.slider(
        "🌫️ Standard Deviation (Variability)", 
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="The random noise or imprecision in the data. Higher variability widens the confidence interval, making equivalence harder to prove."
    )
    n_slider = st.sidebar.slider(
        "🔬 Sample Size (n)", 
        min_value=10, max_value=200, value=50, step=5,
        help="The number of samples per group. Higher sample size narrows the confidence interval, increasing your power to prove equivalence."
    )
    
    # Generate plots using the slider values
    fig, p_tost, is_equivalent, ci_lower, ci_upper = plot_tost(
        delta=delta_slider,
        true_diff=diff_slider,
        std_dev=sd_slider,
        n_samples=n_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="⚖️ Equivalence Margin (Δ)", value=f"± {delta_slider:.1f} units", help="The pre-defined 'zone of indifference'.")
            st.metric(label="📊 Observed 90% CI for Difference", value=f"[{ci_lower:.2f}, {ci_upper:.2f}]", help="The 90% confidence interval for the true difference between the groups.")
            st.metric(label="p-value (TOST)", value=f"{p_tost:.4f}", help="If p < 0.05, we conclude equivalence.")
            
            status = "✅ Equivalent" if is_equivalent else "❌ Not Equivalent"
            if is_equivalent: st.success(f"### Status: {status}")
            else: st.error(f"### Status: {status}")

            st.info("Play with the sliders in the sidebar to see how they affect the conclusion!")
            st.markdown("""
            - **The Goal:** To get the **entire blue bar (90% CI)** to fall inside the **green equivalence zone**.
            - **`True Difference`:** Move this slider to see how the position of the blue bar changes.
            - **`Standard Deviation`:** Increasing this widens the blue bar, making it fail.
            - **`Sample Size`:** Increasing this narrows the blue bar, making it pass. This shows the power of collecting more data.
            - **`Equivalence Margin`:** This moves the red goalposts. A tight margin is a high bar to clear.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Fallacy of the Non-Significant P-Value**
            - A scientist runs a standard t-test and gets a p-value of 0.25. They exclaim, *"Great, p > 0.05, so the methods are the same!"*
            - **This is wrong.** All they have shown is a *failure to find evidence of a difference*. **Absence of evidence is not evidence of absence.**
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Define 'Same Enough', Then Prove It**
            The TOST procedure forces a more rigorous scientific approach.
            1.  **First, Define the Margin:** Before collecting data, stakeholders must define the equivalence margin (the green zone).
            2.  **Then, Prove You're Inside:** Conduct the experiment. The burden of proof is on you to show that your evidence (the 90% CI) is strong enough to fall entirely within that margin.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The TOST procedure rose to prominence in the 1980s, driven by the **1984 Hatch-Waxman Act** which created the modern pathway for generic drug approval. Regulators needed a way to statistically *prove* a generic drug was bioequivalent to the original. **Donald J. Schuirmann's** Two One-Sided Tests (TOST) procedure became the statistical engine for these critical studies.
            
            #### Mathematical Basis
            TOST brilliantly flips the null hypothesis. Instead of one null of "no difference," you have two nulls of "too different":
            """)
            st.latex(r"H_{01}: \mu_{Test} - \mu_{Ref} \leq -\Delta \quad , \quad H_{02}: \mu_{Test} - \mu_{Ref} \geq +\Delta")
            st.markdown("You must reject **both** of these null hypotheses to conclude that the true difference lies within the equivalence margin `[-Δ, +Δ]`.")

def render_assay_robustness_doe():
    """Renders the comprehensive, interactive module for Assay Robustness (DOE/RSM)."""
    st.markdown("""
    #### Purpose & Application: Process Cartography - The GPS for Optimization
    **Purpose:** To create a detailed topographical map of your process landscape. This analysis moves beyond simple robustness checks to full **process optimization**, using **Response Surface Methodology (RSM)** to model curvature and find the true "peak of the mountain."
    
    **Strategic Application:** This is the statistical engine for Quality by Design (QbD) and process characterization. By developing a predictive model, you can:
    - **Find Optimal Conditions:** Identify the exact settings that maximize yield, efficacy, or any other Critical Quality Attribute (CQA).
    - **Define a Design Space:** Create a multi-dimensional "safe operating zone" where the process is guaranteed to produce acceptable results. This is highly valued by regulatory agencies.
    - **Minimize Variability:** Find a "robust plateau" on the response surface where performance is not only high, but also insensitive to small variations in input parameters.
    """)
    
    st.info("""
    **Interactive Demo:** You are the process expert. Use the sliders in the sidebar to define the "true" physics of a virtual assay. The plots will show how a DOE/RSM experiment can uncover this underlying response surface, allowing you to find the optimal operating conditions.
    """)
    
    # --- Sidebar controls ---
    st.sidebar.subheader("DOE / RSM Controls")
    st.sidebar.markdown("**Linear & Interaction Effects**")
    ph_slider = st.sidebar.slider("🧬 pH Main Effect", -10.0, 10.0, 2.0, 1.0, help="The 'true' linear impact of pH. A high value 'tilts' the surface along the pH axis.")
    temp_slider = st.sidebar.slider("🌡️ Temperature Main Effect", -10.0, 10.0, 5.0, 1.0, help="The 'true' linear impact of Temperature. A high value 'tilts' the surface along the Temp axis.")
    interaction_slider = st.sidebar.slider("🔄 pH x Temp Interaction Effect", -10.0, 10.0, 0.0, 1.0, help="The 'true' interaction. A non-zero value 'twists' the surface, creating a rising ridge.")
    
    st.sidebar.markdown("**Curvature (Quadratic) Effects**")
    ph_quad_slider = st.sidebar.slider("🧬 pH Curvature", -10.0, 10.0, -5.0, 1.0, help="A negative value creates a 'hill' (a peak). A positive value creates a 'bowl' (a valley). This is the key to optimization.")
    temp_quad_slider = st.sidebar.slider("🌡️ Temperature Curvature", -10.0, 10.0, -5.0, 1.0, help="A negative value creates a 'hill' (a peak). A positive value creates a 'bowl' (a valley).")

    st.sidebar.markdown("**Experimental Noise**")
    noise_slider = st.sidebar.slider("🎲 Random Noise (SD)", 0.1, 5.0, 1.0, 0.1, help="The inherent variability of the assay. High noise can hide the true effects.")
    
    # Generate plots
    fig_contour, fig_3d, fig_effects, params = plot_doe_robustness(
        ph_effect=ph_slider, temp_effect=temp_slider, interaction_effect=interaction_slider,
        ph_quad_effect=ph_quad_slider, temp_quad_effect=temp_quad_slider, noise_sd=noise_slider
    )
    
    st.header("Response Surface Plots")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_contour, use_container_width=True)
    with col2:
        st.plotly_chart(fig_3d, use_container_width=True)

    st.header("Effect Plots & Interpretation")
    col3, col4 = st.columns([0.7, 0.3])
    with col3:
        st.plotly_chart(fig_effects, use_container_width=True)
    with col4:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            # Find the optimal settings from the model predictions
            pred_z = fig_3d.data[0].z
            max_idx = np.unravel_index(np.argmax(pred_z), pred_z.shape)
            opt_temp = fig_3d.data[0].y[max_idx[0]]
            opt_ph = fig_3d.data[0].x[max_idx[1]]
            max_response = np.max(pred_z)

            st.metric("Predicted Optimal pH", f"{opt_ph:.2f}")
            st.metric("Predicted Optimal Temp", f"{opt_temp:.2f}")
            st.metric("Predicted Max Response", f"{max_response:.1f} units")
            
            st.info("Play with the sliders and observe how the 'true' physics you define are reflected in the plots!")
            st.markdown("""
            - **Linear Effects:** Increasing a `Main Effect` slider is like **tilting the entire surface**. A high positive value for Temperature makes the response universally higher at high temperatures.
            - **Interaction Effects:** A non-zero `Interaction Effect` **twists the surface**, creating a "rising ridge." The effect of pH is now different at high vs. low temperatures. The lines in the Interaction Plot become non-parallel.
            - **Curvature Effects:** The `Curvature` sliders are the key to optimization. Setting them to negative values creates a **peak or dome** in the 3D surface, which corresponds to the "bullseye" of concentric circles in the 2D contour plot. This is the optimal zone you are trying to find.
            - **Noise:** Increasing `Random Noise` makes the red data points scatter further from the true underlying surface, making it harder for the model to accurately map the landscape.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: One-Factor-at-a-Time (OFAT)**
            Imagine trying to find the highest point on a mountain by only walking in straight lines, first due North-South, then due East-West. You will almost certainly end up on a ridge or a local hill, convinced it's the summit, while the true peak was just a few steps to the northeast.
            
            - **The Flaw:** This is what OFAT does. It is statistically inefficient and, more importantly, it is **guaranteed to miss the true optimum** if any interaction between the factors exists.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Map the Entire Territory at Once (DOE/RSM)**
            By testing factors in combination using a dedicated design (like a Central Composite Design), you send out scouts to explore the entire landscape simultaneously. This allows you to:
            1.  **Be Highly Efficient:** Gain more information from fewer experimental runs compared to OFAT.
            2.  **Understand the Terrain:** Uncover and quantify critical interaction and curvature effects that describe the true shape of the process space.
            3.  **Find the True Peak:** Develop a predictive mathematical model that acts as a GPS, guiding you directly to the optimal operating conditions.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            - **The Genesis (1920s):** DOE was invented by **Sir Ronald A. Fisher** to screen for important factors in agriculture. His factorial designs were brilliant for figuring out *which* factors mattered.
            - **The Optimization Revolution (1950s):** The post-war chemical industry boom created a new need: not just to know *which* factors mattered, but *how to find their optimal settings*. **George Box** and K.B. Wilson developed **Response Surface Methodology (RSM)** to solve this. They created efficient new designs, like the Central Composite Design (CCD) shown here, which cleverly add "axial" points to a factorial design. These extra points allow for the fitting of a **quadratic model**, which is the key to modeling curvature and finding the "peak of the mountain." This moved DOE from simple screening to true, powerful optimization.
            
            #### Mathematical Basis
            RSM typically fits a second-order (quadratic) model to the experimental data:
            """)
            st.latex(r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_{11}X_1^2 + \beta_{22}X_2^2 + \beta_{12}X_1X_2 + \epsilon")
            st.markdown(r"""
            - $\beta_0$: The intercept or baseline response.
            - $\beta_1, \beta_2$: The **linear main effects** (the tilt of the surface).
            - $\beta_{11}, \beta_{22}$: The **quadratic effects** (the curvature or "hill/bowl" shape).
            - $\beta_{12}$: The **interaction effect** (the twist of the surface).
            - $\epsilon$: The random experimental error.
            """)

def render_causal_inference():
    """Renders the INTERACTIVE module for Causal Inference."""
    st.markdown("""
    #### Purpose & Application: Beyond the Shadow - The Science of "Why"
    **Purpose:** To move beyond mere correlation ("what") and ascend to the level of causation ("why"). While predictive models see shadows on a cave wall (associations), Causal Inference provides the tools to understand the true objects casting them (the underlying causal mechanisms).
    
    **Strategic Application:** This is the ultimate goal of root cause analysis and the foundation of intelligent intervention.
    - **💡 Effective CAPA:** Why did a batch fail? A predictive model might say high temperature is *associated* with failure. Causal Inference helps determine if high temperature *causes* failure, or if both are driven by a third hidden variable (a "confounder"). This prevents wasting millions on fixing the wrong problem.
    - **🗺️ Process Cartography:** It allows for the creation of a **Directed Acyclic Graph (DAG)**, which is a formal causal map of your process, documenting scientific understanding and guiding future analysis.
    - **🔮 "What If" Scenarios:** It provides a framework to answer hypothetical questions like, "What *would* have been the yield if we had kept the temperature at 40°C?" using only observational data.
    """)
    
    st.info("""
    **Interactive Demo:** Use the slider in the sidebar to control the **Confounding Strength** of the `Reagent Lot`. As you increase it, watch the "Naive Correlation" (the orange line) become a terrible estimate of the "True Causal Effect" (the green line). This simulation visually demonstrates how a hidden variable can create a misleading correlation.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Causal Inference Controls")
    confounding_slider = st.sidebar.slider(
        "🚨 Confounding Strength", 
        min_value=0.0, max_value=10.0, value=5.0, step=0.5,
        help="How strongly the 'Reagent Lot' affects BOTH Temperature and Purity. At 0, the naive correlation equals the true causal effect."
    )
    
    # Generate plots using the slider value
    fig_dag, fig_scatter, naive_effect, adjusted_effect = plot_causal_inference(confounding_strength=confounding_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig_dag, use_container_width=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="Biased Estimate (Naive Correlation)", value=f"{naive_effect:.3f}", help="The effect you would conclude by just plotting Purity vs. Temp. This is misleading!")
            st.metric(label="Unbiased Estimate (True Causal Effect)", value=f"{adjusted_effect:.3f}", help="The true effect of Temp on Purity after adjusting for the confounder. Note how this stays stable.")

            st.info("Play with the 'Confounding Strength' slider and watch the metrics and plots!")
            st.markdown("""
            - **The DAG (Top Plot):** This is our "causal map." It shows that `Reagent Lot` is a **common cause** of both `Temp` and `Purity`, creating a "backdoor" path that biases the `Temp -> Purity` relationship.
            - **The Scatter Plot:** As you increase `Confounding Strength`, the orange line (naive correlation) becomes a worse and worse estimate of the green line (the true causal effect). The data points separate into two clouds (one for each reagent lot), and the naive model incorrectly draws a line through them. The adjusted model correctly finds the true, steeper negative trend *within* each group.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Correlation Trap**
            - An analyst observes that ice cream sales are highly correlated with shark attacks. They recommend banning ice cream to improve beach safety.
            - **The Flaw:** They failed to account for a confounder: **Hot Weather.** Hot weather causes more people to buy ice cream AND causes more people to go swimming. Causal inference provides the tools to mathematically "control for" the weather to see that ice cream has no real effect.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Draw the Map, Find the Path, Block the Backdoors**
            A robust causal analysis follows a disciplined, three-step process.
            1.  **Draw the Map (Build the DAG):** This is a collaborative effort between data scientists and Subject Matter Experts. You must encode all your domain knowledge and causal beliefs into a formal DAG.
            2.  **Find the Path:** Clearly identify the causal path you want to measure (e.g., `Temp -> Purity`).
            3.  **Block the Backdoors:** Use the DAG to identify all non-causal "backdoor" paths (confounding). Then, use the appropriate statistical technique (like multiple regression) to "block" these paths, leaving only the true causal effect.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: The Causal Revolution
            **The Problem:** For most of the 20th century, mainstream statistics was deeply allergic to the language of causation. The mantra, famously drilled into every student, was **"correlation is not causation."** While true, this left a massive void: if correlation isn't the answer, what is? Statisticians were excellent at describing relationships but had no formal language to discuss *why* those relationships existed, leaving a critical gap between data and real-world action.
            
            **The "Aha!" Moment:** The revolution was sparked by the computer scientist and philosopher **Judea Pearl** in the 1980s and 90s. His key insight was that the missing ingredient was **structure**. He argued that scientists carry causal models in their heads all the time, and that these models could be formally written down as graphs. He introduced the **Directed Acyclic Graph (DAG)** as the language for this structure. The arrows in a DAG are not mere correlations; they are bold claims about the direction of causal influence.
            
            **The Impact:** This was a paradigm shift. By making causal assumptions explicit in a DAG, Pearl developed a complete mathematical framework—including his famous **do-calculus**—to determine if a causal question *could* be answered from observational data, and if so, how. This "Causal Revolution" provided the first-ever rigorous, mathematical language to move from seeing (`P(Y|X)`) to doing (`P(Y|do(X))`), transforming fields from epidemiology to economics. For this work, Judea Pearl was awarded the Turing Award in 2011, the highest honor in computer science.
            """)
##=========================================================================================================================================================================================================
##===============================================================================END ACT I UI Render ========================================================================================================================================
##=========================================================================================================================================================================================================

def render_spc_charts():
    """Renders the INTERACTIVE module for Statistical Process Control (SPC) charts."""
    st.markdown("""
    #### Purpose & Application: The Voice of the Process
    **Purpose:** To serve as an **EKG for your process**—a real-time heartbeat monitor that visualizes its stability. The goal is to distinguish between two fundamental types of variation:
    - **Common Cause Variation:** The natural, random "static" or "noise" inherent to a stable process. It's predictable.
    - **Special Cause Variation:** A signal that something has changed or gone wrong. It's unpredictable and requires investigation.
    
    **Strategic Application:** SPC is the bedrock of modern quality control. These charts provide an objective, data-driven answer to the critical question: "Is my process stable and behaving as expected?" They are used to prevent defects, reduce waste, and provide the evidence needed to justify (or reject) process changes.
    """)
    
    st.info("""
    **Interactive Demo:** Use the controls in the sidebar to inject different types of "special cause" events into a simulated stable process. Observe how the I-MR, Xbar-R, and P-Charts each respond, helping you learn to recognize the visual signatures of common process problems.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("SPC Scenario Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        ('Stable', 'Sudden Shift', 'Gradual Trend', 'Increased Variability'),
        captions=[
            "Process is behaving normally.",
            "e.g., A new raw material lot is introduced.",
            "e.g., An instrument is slowly drifting out of calibration.",
            "e.g., An operator becomes less consistent."
        ]
    )

    # Generate plots based on the selected scenario
    fig_imr, fig_xbar, fig_p = plot_spc_charts(scenario=scenario)
    
    st.subheader(f"Analysis & Interpretation: {scenario} Process")
    tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

    with tabs[0]:
        st.info("💡 Each chart type is a different 'lead' on your EKG, designed for a specific kind of data. Use the expanders below to see how to read each one.")

        with st.expander("Indivduals & Moving Range (I-MR) Chart", expanded=True):
            st.plotly_chart(fig_imr, use_container_width=True)
            st.markdown("- **Interpretation:** The I-chart tracks the process center, while the MR-chart tracks short-term variability. **Both** must be stable. An out-of-control MR chart is a leading indicator of future problems.")

        with st.expander("X-bar & Range (X̄-R) Chart", expanded=True):
            st.plotly_chart(fig_xbar, use_container_width=True)
            st.markdown("- **Interpretation:** The X-bar chart tracks variation *between* subgroups and is extremely sensitive to small shifts. The R-chart tracks variation *within* subgroups, a measure of process consistency.")
        
        with st.expander("Proportion (P) Chart", expanded=True):
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("- **Interpretation:** This chart tracks the proportion of defects. The control limits become tighter for larger batches, reflecting increased statistical certainty.")

    with tabs[1]:
        st.error("""
        🔴 **THE INCORRECT APPROACH: "Process Tampering"**
        This is the single most destructive mistake in SPC. The operator sees any random fluctuation within the control limits and reacts as if it's a real problem.
        
        - *"This point is a little higher than the last one, I'll tweak the temperature down a bit."*
        - *"This point is below the mean, I'll adjust the flow rate up."*
        
        Reacting to "common cause" noise as if it were a "special cause" signal actually **adds more variation** to the process, making it worse. This is like trying to correct the path of a car for every tiny bump in the road—you'll end up swerving all over the place.
        """)
        st.success("""
        🟢 **THE GOLDEN RULE: Know When to Act (and When Not To)**
        The control chart's signal dictates one of two paths:
        1.  **Process is IN-CONTROL (only common cause variation):**
            - **Your Action:** Leave the process alone! To improve, you must work on changing the fundamental system (e.g., better equipment, new materials).
        2.  **Process is OUT-OF-CONTROL (a special cause is present):**
            - **Your Action:** Stop! Investigate immediately. Find the specific, assignable "special cause" for that signal and eliminate it.
        """)

    with tabs[2]:
        # FIX: Replaced the old content with your new, more detailed version.
        st.markdown("""
        #### Historical Context & Origin
        The control chart was invented by the brilliant American physicist and engineer **Dr. Walter A. Shewhart** while working at Bell Telephone Laboratories in the 1920s. The challenge was immense: manufacturing millions of components for the new national telephone network required unprecedented levels of consistency. How could you know if a variation in a vacuum tube's performance was just normal fluctuation or a sign of a real production problem?

        Shewhart's genius was in his 1924 memo where he introduced the first control chart. He was the first to formally articulate the critical distinction between **common cause** and **special cause** variation. He realized that as long as a process only exhibited common cause variation, it was stable and predictable. The purpose of the control chart was to provide a simple, graphical tool to detect the moment a special cause entered the system. This idea was the birth of modern Statistical Process Control and laid the foundation for the 20th-century quality revolution.

        #### Mathematical Basis
        The control limits on a Shewhart chart are famously set at ±3 standard deviations from the center line.
        """)
        st.latex(r"\text{Control Limits} = \text{Center Line} \pm 3 \times (\text{Standard Deviation of the Plotted Statistic})")
        st.markdown("""
        - **Why 3-Sigma?** Shewhart chose this value for sound economic and statistical reasons. For a normally distributed process, 99.73% of all data points will naturally fall within these limits.
        - **Minimizing False Alarms:** This means there's only a 0.27% chance of a point falling outside the limits purely by chance. This makes the chart robust; when you get a signal, you can be very confident it's real and not just random noise. It strikes an optimal balance between being sensitive to real problems and not causing "fire drills" for false alarms.
        """)

def render_capability():
    """Renders the interactive module for Process Capability (Cpk)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To quantitatively determine if a process, once proven to be in a state of statistical control, is **capable** of consistently producing output that meets pre-defined specification limits (USL/LSL).
    
    **Strategic Application:** This is the ultimate verdict on process performance, often the final gate in a process validation or technology transfer. It directly answers the critical business question: "Is our process good enough to reliably meet customer or regulatory requirements with a high degree of confidence?" 
    - A high capability index (Cpk) provides objective, statistical evidence that the process is robust, predictable, and delivers high quality.
    - A low Cpk is a clear signal that the process requires fundamental improvement, either by **re-centering the process mean** or by **reducing the process variation**.
    
    In many ways, achieving a high Cpk is the statistical equivalent of "mission accomplished" for a process development or transfer team.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons in the sidebar to simulate four common real-world process states. Observe how the control chart (stability), the histogram's position relative to the spec limits, and the final Cpk value (capability) change for each scenario. This demonstrates the critical principle that a process must be stable *before* its capability can be meaningfully assessed.
    """)

    scenario = st.sidebar.radio("Select Process Scenario:", ('Ideal', 'Shifted', 'Variable', 'Out of Control'))
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, cpk_val = plot_capability(scenario)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scenario != 'Out of Control' else "INVALID", help="Measures how well the process fits within the spec limits, accounting for centering. Higher is better.")
            st.markdown("""
            - **The Mantra: Control Before Capability.** The control chart (top plot) is a prerequisite. The Cpk metric is only statistically valid and meaningful if the process is stable and in-control. The 'Out of Control' scenario yields an **INVALID** Cpk because an unstable process has no single, predictable "voice" to measure.
            - **The Key Insight: Control ≠ Capability.** A process can be perfectly in-control (predictable) but not capable (producing bad product). 
                - The **'Shifted'** scenario shows a process that is precise but inaccurate.
                - The **'Variable'** scenario shows a process that is centered but imprecise.
            Both are in control, but both have a poor Cpk.
            """)
        with tabs[1]:
            st.markdown("These are industry-standard benchmarks, often required by customers, especially in automotive and aerospace. For pharmaceuticals, a high Cpk in validation provides strong assurance of lifecycle performance.")
            st.markdown("- `Cpk < 1.00`: Process is **not capable**.")
            st.markdown("- `1.00 ≤ Cpk < 1.33`: Process is **marginally capable**.")
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable** (a '4-sigma' quality level).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** (approaching 'Six Sigma').")
            st.markdown("- `Cpk ≥ 2.00`: Process has achieved **Six Sigma capability**.")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of comparing process output to specification limits is old, but the formalization into capability indices originated in the Japanese manufacturing industry in the 1970s as a core part of Total Quality Management (TQM).
            
            However, it was the **Six Sigma** initiative, pioneered by engineer Bill Smith at **Motorola in the 1980s**, that catapulted Cpk to global prominence. The 'Six Sigma' concept was born: a process so capable that the nearest specification limit is at least six standard deviations away from the process mean. Cpk became the standard metric for measuring progress toward this ambitious goal.
            
            #### Mathematical Basis
            Capability analysis is a direct comparison between the **"Voice of the Customer"** (the allowable spread, USL - LSL) and the **"Voice of the Process"** (the actual, natural spread, conventionally 6σ).

            - **Cp (Potential Capability):** Measures if the process is narrow enough, ignoring centering.
            """)
            st.latex(r"C_p = \frac{USL - LSL}{6\hat{\sigma}}")
            st.markdown("- **Cpk (Actual Capability):** The more important metric, as it accounts for process centering. It measures the distance from the process mean to the *nearest* specification limit.")
            st.latex(r"C_{pk} = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right)")

def render_tolerance_intervals():
    """Renders the INTERACTIVE module for Tolerance Intervals."""
    st.markdown("""
    #### Purpose & Application: The Quality Engineer's Secret Weapon
    **Purpose:** To construct an interval that we can claim, with a specified level of confidence, contains a certain proportion of all individual values from a process.
    
    **Strategic Application:** This is often the most critical statistical interval in manufacturing. It directly answers the high-stakes question: **"Based on this sample, what is the range where we can expect almost all of our individual product units to fall?"**
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to explore the trade-offs in tolerance intervals. This simulation demonstrates how sample size and the desired quality guarantee (coverage) directly impact the calculated interval, which in turn affects process specifications and batch release decisions.
    """)
    
    # --- NEW: Sidebar controls for this specific module ---
    st.sidebar.subheader("Tolerance Interval Controls")
    n_slider = st.sidebar.slider(
        "🔬 Sample Size (n)", 
        min_value=10, max_value=200, value=30, step=10,
        help="The number of samples collected. More samples lead to a narrower, more reliable interval."
    )
    coverage_slider = st.sidebar.select_slider(
        "🎯 Desired Population Coverage",
        options=[90.0, 95.0, 99.0, 99.9],
        value=99.0,
        help="The 'quality promise'. What percentage of all future parts do you want this interval to contain? A higher promise requires a wider interval."
    )

    # Generate plots using the slider values
    fig, ci, ti = plot_tolerance_intervals(n=n_slider, coverage_pct=coverage_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🎯 Desired Coverage", value=f"{coverage_slider:.1f}% of Population", help="The proportion of the entire process output we want our interval to contain.")
            st.metric(label="📏 Resulting Tolerance Interval", value=f"[{ti[0]:.1f}, {ti[1]:.1f}]", help="The final calculated range. Note how much wider it is than the CI.")
            
            st.info("Play with the sliders in the sidebar and observe the results!")
            st.markdown("""
            - **Increase `Sample Size (n)`:** As you collect more data, your estimates of the mean and standard deviation become more reliable. Notice how both the **Confidence Interval (orange)** and the **Tolerance Interval (green)** become **narrower**. This shows the direct link between sampling cost and statistical precision.
            - **Increase `Desired Population Coverage`:** As you increase the strength of your quality promise from 90% to 99.9%, the **Tolerance Interval becomes dramatically wider**. To be more certain of capturing a larger percentage of parts, you must widen your interval.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Confidence Interval Fallacy**
            - A manager sees that the 95% **Confidence Interval** for the mean is [99.9, 100.1] and their product specification is [95, 105]. They declare victory, believing all their product is in spec.
            - **The Flaw:** They've proven the *average* is in spec, but have made no claim about the *individuals*. If process variation is high, many parts could still be out of spec.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Use the Right Interval for the Right Question**
            - **Question 1: "Where is my long-term process average located?"**
              - **Correct Tool:** ✅ **Confidence Interval**.
            - **Question 2: "Will the individual units I produce meet the customer's specification?"**
              - **Correct Tool:** ✅ **Tolerance Interval**.
              
            Never use a confidence interval to make a statement about where individual values are expected to fall.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: The Surviving Bomber Problem
            The development of tolerance intervals is credited to the brilliant mathematician **Abraham Wald** during World War II. He is famous for the "surviving bombers" problem: when analyzing bullet holes on returning planes, the military wanted to reinforce the most-hit areas. Wald's revolutionary insight was that they should reinforce the areas with **no bullet holes**—because planes hit there never made it back.
            
            This ability to reason about an entire population from a limited sample is the same thinking behind the tolerance interval. Wald developed the statistical theory to allow engineers to make a reliable claim about **all** manufactured parts based on a **small sample**, a critical need for mass-producing interchangeable military hardware.
            
            #### Mathematical Basis
            """)
            st.latex(r"\text{TI} = \bar{x} \pm k \cdot s")
            st.markdown("""
            - **`k`**: The **k-factor** is the magic ingredient. It is a special value that depends on **three** inputs: the sample size (`n`), the desired population coverage (e.g., 99%), and the desired confidence level (e.g., 95%). This `k`-factor is mathematically constructed to account for the "double uncertainty" of not knowing the true mean *or* the true standard deviation.
            """)

def render_method_comparison():
    """Renders the INTERACTIVE module for Method Comparison."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally assess and quantify the degree of agreement and systemic bias between two different measurement methods intended to measure the same quantity.
    
    **Strategic Application:** This study is the "crucible" of method transfer, validation, or replacement. It answers the critical business and regulatory question: “Do these two methods produce the same result, for the same sample, within medically or technically acceptable limits?”
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate different types of disagreement between a "Test" method and a "Reference" method. See in real-time how each diagnostic plot (Deming, Bland-Altman, %Bias) reveals a different aspect of the problem, helping you build a deep intuition for method comparison statistics.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Method Comparison Controls")
    constant_bias_slider = st.sidebar.slider(
        "⚖️ Constant Bias", 
        min_value=-10.0, max_value=10.0, value=2.0, step=0.5,
        help="A fixed offset where the Test method reads consistently higher (+) or lower (-) than the Reference method across the entire range."
    )
    proportional_bias_slider = st.sidebar.slider(
        "📈 Proportional Bias (%)", 
        min_value=-10.0, max_value=10.0, value=3.0, step=0.5,
        help="A concentration-dependent error. A positive value means the Test method reads progressively higher than the Reference at high concentrations."
    )
    random_error_slider = st.sidebar.slider(
        "🎲 Random Error (SD)", 
        min_value=0.5, max_value=10.0, value=3.0, step=0.5,
        help="The imprecision or 'noise' of the methods. Higher error widens the Limits of Agreement on the Bland-Altman plot."
    )

    # Generate plots using the slider values
    fig, slope, intercept, bias, ua, la = plot_method_comparison(
        constant_bias=constant_bias_slider,
        proportional_bias=proportional_bias_slider,
        random_error_sd=random_error_slider
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Mean Bias (Bland-Altman)", value=f"{bias:.2f} units", help="The average systematic difference.")
            st.metric(label="💡 Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0. Measures proportional bias.")
            st.metric(label="💡 Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0. Measures constant bias.")
            
            st.info("Play with the sliders in the sidebar and observe the plots!")
            st.markdown("""
            - **Add `Constant Bias`:** The Deming line shifts up/down but stays parallel to the identity line. The Bland-Altman plot's mean bias line moves away from zero.
            - **Add `Proportional Bias`:** The Deming line *rotates* away from the identity line. The Bland-Altman and %Bias plots now show a clear trend, a major red flag.
            - **Increase `Random Error`:** The points scatter more widely. This has little effect on the average bias but dramatically **widens the Limits of Agreement**, making the methods less interchangeable.
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria must be pre-defined and clinically/technically justified.")
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope must contain 1.0**, and the 95% CI for the **intercept must contain 0**.")
            st.markdown(f"- **Bland-Altman:** The primary criterion is that the **95% Limits of Agreement (`{la:.2f}` to `{ua:.2f}`) must be clinically or technically acceptable**.")
            st.error("""
            **The Correlation Catastrophe:** Never use the correlation coefficient (R²) to assess agreement. Two methods can be perfectly correlated (R²=1.0) but have a huge bias (e.g., one method always reads twice as high).
            """)

        with tabs[2]:
            # FIX: Restored the full, detailed content for this tab
            st.markdown("""
            #### Historical Context & Origin
            For decades, scientists committed a cardinal sin: using **Ordinary Least Squares (OLS) regression** and the **correlation coefficient (r)** to compare methods. This is flawed because OLS assumes the x-axis (reference method) is measured without error, an impossibility.
            
            - **Deming's Correction:** While known to statisticians, **W. Edwards Deming** championed this type of regression in the 1940s. It correctly assumes both methods have measurement error, providing an unbiased estimate of the true relationship. **Passing-Bablok regression** is a robust non-parametric alternative.
            
            - **The Bland-Altman Revolution:** A 1986 paper in *The Lancet* by **J. Martin Bland and Douglas G. Altman** ruthlessly exposed the misuse of correlation and proposed their brilliantly simple alternative. Instead of plotting Y vs. X, they plotted the **Difference (Y-X) vs. the Average ((Y+X)/2)**. This directly visualizes the magnitude and patterns of disagreement and is now the undisputed gold standard.
            
            #### Mathematical Basis
            **Deming Regression:** OLS minimizes the sum of squared vertical distances. Deming regression minimizes the sum of squared distances from the points to the line, weighted by the ratio of the error variances of the two methods.
            
            **Bland-Altman Plot:** This is a graphical analysis. The key metrics are the **mean difference (bias)**, $\bar{d}$, and the **standard deviation of the differences**, $s_d$. The 95% Limits of Agreement (LoA) are calculated assuming the differences are approximately normally distributed:
            """)
            st.latex(r"LoA = \bar{d} \pm 1.96 \cdot s_d")
            st.markdown("This interval provides a predictive range: we can be 95% confident that the difference between the two methods for a future sample will fall within these limits.")

def render_pass_fail():
    """Renders the INTERACTIVE module for Pass/Fail (Binomial Proportion) analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately calculate and critically compare confidence intervals for a binomial proportion, which is the underlying statistic for any pass/fail, present/absent, or concordant/discordant outcome.
    
    **Strategic Application:** This is essential for the validation of **qualitative assays** or for agreement studies. The goal is to prove, with a high degree of statistical confidence, that the assay's success rate is above a required performance threshold. The critical challenge, especially with small sample sizes, is that simple textbook methods for calculating confidence intervals (the 'Wald' interval) are dangerously inaccurate.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate the results of a validation study (e.g., comparing a new test to a gold standard). Observe how sample size and the number of successes dramatically affect the confidence in your result, and see why the 'Wald' interval should almost never be used.
    """)
    
    # --- Sidebar controls for this specific module ---
    st.sidebar.subheader("Pass/Fail Controls")
    n_samples_slider = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30, key='wilson_n')
    successes_slider = st.sidebar.slider("Concordant Results (Successes)", 0, n_samples_slider, int(n_samples_slider * 0.95), key='wilson_s')
    
    # Generate plots using the slider values
    fig1_intervals, fig2_coverage = plot_binomial_intervals(successes_slider, n_samples_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig1_intervals, use_container_width=True)
        st.plotly_chart(fig2_coverage, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_slider/n_samples_slider if n_samples_slider > 0 else 0):.2%}", help="The point estimate. Insufficient without a confidence interval.")
            
            st.info("Play with the sliders in the sidebar and observe the plots!")
            st.markdown("""
            - **CI Comparison (Top Plot):** This plot reveals the dramatic differences between interval methods. 
                - Set the sliders to a perfect score (e.g., 30/30). The **Wald interval collapses to zero width**, an absurd claim of perfect knowledge from a small sample. The Wilson and Clopper-Pearson intervals give a much more honest, wider range.
                - Set the sliders to a low sample size (e.g., 5/5). The Wald interval gives a nonsensical range that goes above 100%!
            - **Coverage Probability (Bottom Plot):** This shows *why* the Wald interval is so bad. Its actual probability of capturing the true value (the red line) is often far below the promised 95% level. The Wilson interval (blue) is much more reliable.

            **The Core Strategic Insight:** Never use the standard Wald (or "Normal Approximation") interval for important decisions. The **Wilson Score interval** provides the best balance of accuracy and interval width for most applications. The **Clopper-Pearson** is the most conservative ("exact") choice, often preferred in regulatory submissions for its guaranteed coverage.
            """)
        with tabs[1]:
            st.markdown("- **The Golden Rule of Binomial Acceptance:** The acceptance criterion must **always be based on the lower bound of the confidence interval**, never on the point estimate.")
            st.markdown("- **Example Criterion:** 'The lower bound of the 95% **Wilson Score** (or Clopper-Pearson) confidence interval for the concordance rate must be greater than or equal to the target of 90%.'")
            st.markdown("- **Sample Size Implication:** This tool powerfully demonstrates why larger sample sizes are needed for high-confidence claims. With a small `n`, even a perfect result (e.g., 20/20 successes) may have a lower confidence bound that fails to meet a high target (like 95%), forcing the study to be repeated with more samples.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            For much of the 20th century, the simple **Wald interval** (named after Abraham Wald) was taught in introductory statistics classes. However, its poor performance was well-known. A famous 1998 paper by Brown, Cai, and DasGupta comprehensively documented its failures and advocated for superior alternatives.
            
            The **Wilson Score Interval** (1927) and the **Clopper-Pearson Interval** (1934) were created to solve this problem.
            - The **Clopper-Pearson** interval is an "exact" method derived from the binomial distribution. It guarantees coverage will never be less than the nominal level, making it conservative (wider).
            - The **Wilson Score** interval is derived by inverting the score test. Its average coverage probability is much closer to the nominal 95% level, making it more accurate and less conservative in practice.
            """)
            
            # --- FIX: SEPARATED EACH FUNCTION CALL ---
            st.markdown("#### Mathematical Basis")
            st.markdown("The Wald interval is simply:")
            st.latex(r"\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
            st.markdown("The Wilson Score interval's superior formula is:")
            st.latex(r"CI_{\text{Wilson}} = \frac{1}{1 + z_{\alpha/2}^2/n} \left( \hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}} \right)")
            st.markdown("Notice it adds pseudo-successes and failures ($z_{\alpha/2}^2/2$), pulling the center away from 0 or 1. This is what gives it such good performance where the Wald interval fails catastrophically.")
            
def render_bayesian():
    """Renders the interactive module for Bayesian Inference."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To employ Bayesian inference to formally and quantitatively synthesize existing knowledge (a **Prior** belief) with new experimental data (the **Likelihood**) to arrive at an updated, more robust conclusion (the **Posterior** belief).
    
    **Strategic Application:** This is a paradigm-shifting tool for driving efficient, knowledge-based validation and decision-making. In a traditional (Frequentist) world, every study starts from a blank slate. In the Bayesian world, we can formally leverage what we already know. This is powerful for:
    - **Accelerating Tech Transfer:** Use data from an R&D validation study to form a **strong, informative prior**. This allows the receiving QC lab to demonstrate success with a smaller confirmation study, saving time and resources.
    - **Adaptive Clinical Trials:** Data from an interim analysis can serve as a prior for the final analysis, allowing trials to be stopped early.
    - **Quantifying Belief & Risk:** It provides a natural framework to answer the question: "Given what we already knew, and what this new data shows, what is the probability that the pass rate is actually above 95%?"
    """)
    st.info("""
    **Interactive Demo:** Use the **Prior Belief** radio buttons in the sidebar to simulate how different levels of existing knowledge impact your conclusions. Observe how the final **Posterior (blue curve)** is always a weighted compromise between your initial **Prior (green curve)** and the new **Data (red curve)**. A strong prior will be very influential, while a weak or non-informative prior lets the new data speak for itself.
    """)
    prior_type_bayes = st.sidebar.radio("Select Prior Belief:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, prior_mean, mle, posterior_mean = plot_bayesian(prior_type_bayes)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}", help="The final, data-informed belief; a weighted average of the prior and the data.")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}", help="The initial belief *before* seeing the new QC data.")
            st.metric(label="💡 Data-only Estimate (MLE)", value=f"{mle:.3f}", help="The evidence from the new QC data alone (the frequentist result).")
            st.markdown("""
            - **Prior (Green Dashed):** Our initial belief about the pass rate. A **Strong Prior** is tall and narrow, representing high confidence from historical data. A **Skeptical Prior** is wide and flat, representing uncertainty.
            - **Likelihood (Red Dotted):** The "voice of the new data." This is the evidence from our new QC runs. Note that it is not a probability distribution.
            - **Posterior (Blue Solid):** The final, updated belief. The posterior is always a **compromise** between the prior and the likelihood, weighted by their respective certainties (the narrowness of their distributions).

            **The Core Strategic Insight:** This simulation demonstrates Bayesian updating in action.
             - With a **Strong R&D Prior**, the new (and slightly worse) QC data barely moves our final belief. The strong prior evidence dominates the small new sample.
             - With a **Skeptical Prior**, our final belief is a true compromise between the skeptical starting point and the new data.
             - With **No Prior**, the posterior is determined almost entirely by the data, and the result mirrors the frequentist conclusion.
            This framework provides a transparent and logical way to cumulate knowledge over time.
            """)
        with tabs[1]:
            st.markdown("- The acceptance criterion is framed in terms of the **posterior distribution** and is probabilistic.")
            st.markdown("- **Example Criterion 1 (Probability Statement):** 'There must be at least a 95% probability that the true pass rate is greater than 90%.' This is calculated by finding the area under the blue posterior curve to the right of the 0.90 threshold.")
            st.markdown("- **Example Criterion 2 (Credible Interval):** 'The lower bound of the **95% Credible Interval** (the central 95% of the blue posterior distribution) must be above the target of 90%.'")
            st.warning("**The Prior is Critical:** The choice of prior is the most controversial and important part of a Bayesian analysis. In a regulated setting, the prior must be transparent, justified by historical data, and pre-specified in the validation protocol. An unsubstantiated, overly optimistic prior would be a major red flag for an auditor.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The underlying theorem was conceived by the Reverend **Thomas Bayes** in the 1740s. However, for nearly 200 years, Bayesian inference remained a philosophical curiosity, largely overshadowed by the Frequentist school. This was due to philosophical objections to the subjective nature of priors and the computational difficulty of calculating the posterior distribution.
            
            The **"Bayesian Revolution"** began in the late 20th century, driven by powerful computers and simulation algorithms like **Markov Chain Monte Carlo (MCMC)**. These methods allowed scientists to approximate the posterior distribution for incredibly complex models, making Bayesian methods practical for the first time.
            
            #### Mathematical Basis
            Bayes' Theorem is elegantly simple:
            """)
            st.latex(r"P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}")
            st.markdown(r"In words: **Posterior = (Likelihood × Prior) / Evidence**")
            st.markdown(r"""
            - $P(\theta|D)$ (Posterior): The probability of our parameter $\theta$ (e.g., the true pass rate) given the new Data D.
            - $P(D|\theta)$ (Likelihood): The probability of observing our Data D, given a specific value of the parameter $\theta$.
            - $P(\theta)$ (Prior): Our initial belief about the distribution of the parameter $\theta$.
            
            For binomial data, the **Beta distribution** is a **conjugate prior**. This means if you start with a Beta prior and have a binomial likelihood, your posterior will also be a Beta distribution.
            - If Prior is Beta($\alpha_{prior}, \beta_{prior}$)
            - And Data is $k$ successes in $n$ trials:
            - Then the Posterior is simply Beta($\alpha_{prior} + k, \beta_{prior} + n - k$).
            The $\alpha$ and $\beta$ parameters can be thought of as "pseudo-counts" of prior successes and failures, which are simply added to the new observed counts.
            """)
##=======================================================================================================================================================================================================
##=================================================================== END ACT II UI Render ========================================================================================================================
##=======================================================================================================================================================================================================
def render_multi_rule():
    """Renders the comprehensive, interactive module for Multi-Rule SPC (Westgard Rules)."""
    st.markdown("""
    #### Purpose & Application: The Statistical Detective
    **Purpose:** To serve as a high-sensitivity "security system" for your assay. Instead of one simple alarm, this system uses a combination of rules to detect specific types of problems, catching subtle shifts and drifts long before a catastrophic failure occurs. It dramatically increases the probability of detecting true errors while minimizing false alarms.
    
    **Strategic Application:** This is the global standard for run validation in regulated QC and clinical laboratories. While a basic control chart just looks for "big" errors, the multi-rule system acts as a **statistical detective**, using a toolkit of rules to diagnose different failure modes. Implementing these rules prevents the release of bad data, which is the cornerstone of ensuring patient safety and product quality.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons in the sidebar to simulate common assay failures. Observe how the control chart changes and which specific Westgard rule is triggered, helping you learn to diagnose problems from your QC data.
    """)
    
    st.sidebar.subheader("Westgard Scenario Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        options=('Stable', 'Complex Failure', 'Large Random Error', 'Systematic Shift', 'Increased Imprecision'),
        captions=[
            "A normal, in-control run for reference.",
            "A run with multiple, distinct issues.",
            "e.g., A single major blunder.",
            "e.g., A new reagent lot causes a bias.",
            "e.g., A faulty pipette causes inconsistency."
        ]
    )
    fig = plot_westgard_scenario(scenario=scenario)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            if scenario == 'Complex Failure':
                st.metric("🕵️ Run Verdict", "Reject Run")
                st.metric("🚨 Primary Cause", "1-3s Violation")
                st.metric("🧐 Secondary Evidence", "2-2s Violation")
                st.markdown("""
                **The Detective's Findings on this Chart:**
                - 🚨 **The Smoking Gun:** The `1-3s` violation is a clear, unambiguous signal of a major problem. This rule alone forces the rejection of the run.
                - 🧐 **The Developing Pattern:** The `2-2s` violation is a classic sign of **systematic error**. The process has shifted high.
                - **The Core Strategic Insight:** This chart shows two *different* problems. A true statistical detective sees both signals and knows there are two distinct issues to solve.
                """)
            else:
                verdict, rule = ("In-Control", "None") if scenario == 'Stable' else ("Reject Run", "Unknown")
                if scenario == 'Large Random Error': rule = "1-3s Violation"
                elif scenario == 'Systematic Shift': rule = "2-2s Violation"
                elif scenario == 'Increased Imprecision': rule = "R-4s Violation"
                
                st.metric("🕵️ Run Verdict", verdict)
                st.metric("🚨 Triggered Rule", rule)
                st.markdown(f"The simulation for **{scenario}** triggered the **{rule}** rule. Refer to the table below for a detailed diagnosis.")

            st.markdown("""
            ---
            **The Detective's Rulebook:**
            | Rule Name | Definition | Error Detected | Typical Cause |
            | :--- | :--- | :--- | :--- |
            | **1-3s** | 1 point > 3σ | Random Error (blunder) | Calculation error, wrong reagent, air bubble |
            | **2-2s** | 2 consecutive points > 2σ (same side) | Systematic Error (shift) | New calibrator/reagent lot, instrument issue |
            | **R-4s** | Range between 2 consecutive points > 4σ | Random Error (imprecision) | Inconsistent pipetting, instrument instability |
            | **4-1s** | 4 consecutive points > 1σ (same side) | Systematic Error (drift) | Minor reagent degradation, slow drift |
            | **10-x** | 10 consecutive points on same side of mean | Systematic Error (bias) | Small, persistent bias in the system |
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Re-run & Pray" Mentality**
            This operator sees any alarm, immediately discards the run, and starts over without thinking.
            - They don't use the specific rule (`2-2s` vs `R-4s`) to guide their troubleshooting.
            - They might engage in "testing into compliance" by re-running a control until it passes, a serious compliance violation.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Rule is the First Clue**
            The goal is to treat the specific rule violation as the starting point of a targeted investigation.
            - **Think like a detective:** "The chart shows a `2-2s` violation. This suggests a systematic shift. I should check my calibrators and reagents first, not my pipetting technique."
            - **Document Everything:** The investigation, the root cause, and the corrective action for each rule violation must be documented.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: From the Factory Floor to the Hospital Bed
            **The Problem:** In the 1920s, the Western Electric company was a manufacturing behemoth, but quality was a nightmare. Engineers were lost in a "fog" of data, constantly "tampering" with the process based on random noise, often making things worse.
            
            **The "Aha!" Moment (Shewhart):** A physicist at Bell Labs, **Dr. Walter A. Shewhart**, had the revolutionary insight to distinguish between "common cause" and "special cause" variation. He invented the control chart to provide a simple, graphical tool to detect the moment a special cause entered the system. The ±3σ limits were chosen for sound economic reasons: to minimize the cost of both false alarms and missed signals.
            
            **The Evolution (Westgard):** Fast forward to the 1970s. **Dr. James O. Westgard** recognized that in clinical labs, where patient lives are at stake, the cost of a missed signal is far higher. He found that Shewhart's single `1-3s` rule wasn't sensitive enough to catch the subtle drifts that could lead to a misdiagnosis. In 1981, he proposed his multi-rule system, an "intelligent" combination of rules designed to maximize the probability of error detection ($P_{ed}$) while maintaining a low probability of false rejection ($P_{fr}$). This engineering trade-off made the system the global standard for medical laboratories.

            #### Mathematical Basis
            The logic is built on the properties of the normal distribution. For a stable process:
            - A point outside **±3σ** is a rare event (occurs ~1 in 370 times by chance). This is a high-confidence signal, hence the **1-3s** rejection rule.
            - A point outside **±2σ** is more common (~1 in 22 times). Seeing *two in a row* on the same side, however, is much rarer ($1/22 \times 1/22 \times 1/2 \approx$ 1 in 968). This makes the **2-2s** rule a powerful detector of systematic shifts with a low false alarm rate.
            """)

def render_multivariate_spc():
    """Renders the comprehensive, interactive module for Multivariate SPC."""
    st.markdown("""
    #### Purpose & Application: The Process Doctor
    **Purpose:** To monitor the **holistic state of statistical control** for a process with multiple, correlated parameters. Instead of using an array of univariate charts (like individual nurses reading single vital signs), Multivariate SPC (MSPC) acts as the **head physician**, integrating all information into a single, powerful diagnosis.
    
    **Strategic Application:** This is an essential methodology for modern **Process Analytical Technology (PAT)** and real-time process monitoring. In complex systems like bioreactors or chromatography, parameters like temperature, pH, pressure, and flow rates are interdependent. A small, coordinated deviation across several parameters—a "stealth shift"—can be invisible to individual charts but represents a significant excursion from the normal operating state. MSPC is designed to detect exactly these events.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Process Scenario** radio buttons in the sidebar to simulate different types of multivariate process failures. First, observe the **Scatter Plot**, then see which **Control Chart (T² or SPE)** detects the problem, and finally, check the **Contribution Plot** in the 'Key Insights' tab to diagnose the root cause.
    """)

    st.sidebar.subheader("Multivariate SPC Controls")
    scenario = st.sidebar.radio(
        "Select a Process Scenario to Simulate:",
        ('Stable', 'Shift in Y Only', 'Correlation Break'),
        captions=["A normal, in-control process.", "A 'stealth shift' in one variable.", "An unprecedented event breaks the model."]
    )

    # Call the backend function and unpack all return values
    fig_scatter, fig_charts, fig_contrib, t2_ooc, spe_ooc, error_type_str = plot_multivariate_spc(scenario=scenario)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig_charts, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        # --- MODIFIED: Added a new tab for conceptual explanations ---
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History", "🧠 Concepts: Bias, Error, & Factors"])
        
        with tabs[0]:
            t2_verdict_str = "Out-of-Control" if t2_ooc else "In-Control"
            spe_verdict_str = "Out-of-Control" if spe_ooc else "In-Control"
            
            st.metric("📈 T² Chart Verdict", t2_verdict_str, help="Monitors deviation *within* the normal process model.")
            st.metric("📈 SPE Chart Verdict", spe_verdict_str, help="Monitors deviation *from* the normal process model.")
            st.metric(
                "📊 Error Type Determination",
                error_type_str,
                help="Type I Error: A 'false alarm' on a stable process. Type II Error: A 'missed signal' on a known failure. 'Correct' means the charts behaved as expected for the scenario."
            )
            
            st.markdown("---")
            st.markdown(f"##### Analysis of the '{scenario}' Scenario:")

            if scenario == 'Stable':
                st.success("The process is stable and in-control. Both the T² and SPE charts show only common cause variation, confirming the process is operating as expected within its normal, correlated state. No diagnostic plot is needed.")
            elif scenario == 'Shift in Y Only':
                st.warning("**Diagnosis: A 'Stealth Shift' has occurred.**")
                st.markdown("""
                1.  **Scatter Plot:** The red points have clearly shifted upwards, but because the correlation is strong, they still fall within the horizontal range of the blue points. A univariate chart for Temperature (X-axis) would likely miss this.
                2.  **T² Chart:** Alarms loudly. It knows the expected Pressure (Y) for a given Temperature (X) and detects this significant deviation from the multivariate mean.
                3.  **SPE Chart:** Remains in-control. The *relationship* between the variables is still intact; the process has just shifted along that known correlation structure.
                4.  **Contribution Plot:** This diagnostic tool confirms the root cause: the T² alarm is driven almost entirely by the **Pressure** variable.
                """)
            elif scenario == 'Correlation Break':
                st.error("**Diagnosis: An Unprecedented Event has occurred.**")
                st.markdown("""
                1.  **Scatter Plot:** The red points have fallen completely *off* the established diagonal correlation line. The average Temperature and Pressure might still be normal, but their relationship is broken.
                2.  **T² Chart:** May remain in-control. Since the points are still relatively close to the center of the data cloud, the T² (which measures distance *within* the model) does not alarm.
                3.  **SPE Chart:** Alarms loudly. The SPE measures the distance *to* the model. Since these points are far from the expected correlation line, the SPE signals a major deviation from the model.
                4.  **Contribution Plot:** This diagnostic tool shows that both variables are contributing to the SPE alarm, confirming the fundamental breakdown of the process model itself.
                """)
            
            if fig_contrib is not None:
                st.markdown("---")
                st.markdown("##### Root Cause Diagnosis")
                st.plotly_chart(fig_contrib, use_container_width=True)
            
            st.markdown("---")
            st.info("**Try This:** Switch between the 'Shift in Y Only' and 'Correlation Break' scenarios to see how the two charts are sensitive to completely different types of process failures.")

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Army of Univariate Charts" Fallacy**
            Using dozens of individual charts is doomed to fail due to alarm fatigue and its blindness to "stealth shifts."
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Detect with T²/SPE, Diagnose with Contributions**
            1.  **Stage 1: Detect.** Use **T² and SPE charts** as your primary health monitors to answer "Is something wrong?"
            2.  **Stage 2: Diagnose.** If a chart alarms, then use **contribution plots** to identify which original variables are responsible for the signal. This is the path to the root cause.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context: The Crisis of Dimensionality
            **The Problem:** In the 1930s, statistics was largely a univariate world. Tools like Student's t-test and Shewhart's control charts were brilliant for analyzing one variable at a time. But scientists and economists were facing increasingly complex problems with dozens of correlated measurements. How could you test if two groups were different, not just on one variable, but across a whole panel of them? A simple t-test on each variable was not only inefficient, it was statistically misleading due to the problem of multiple comparisons.

            **The "Aha!" Moment (Hotelling):** The creator of this powerful technique was **Harold Hotelling**, one of the giants of 20th-century mathematical statistics. His genius was in generalization. He recognized that the squared t-statistic, $t^2 = (\\bar{x} - \\mu)^2 / (s^2/n)$, was a measure of squared distance, normalized by variance. In a 1931 paper, he introduced the **Hotelling's T-squared statistic**, which replaced the univariate terms with their vector and matrix equivalents. It provided a single number that represented the "distance" of a point from the center of a multivariate distribution, elegantly solving the problem of testing multiple means at once while accounting for all their correlations.
            """)
            st.markdown("""
            #### Mathematical Basis
            - **T² (Hotelling's T-Squared):** A measure of the **Mahalanobis distance**, which accounts for the correlation between variables through the **inverse of the sample covariance matrix (`S⁻¹`)**.
            """)
            st.latex(r"T^2 = (\mathbf{x} - \mathbf{\bar{x}})' \mathbf{S}^{-1} (\mathbf{x} - \mathbf{\bar{x}})")
            st.markdown("""
            - **SPE (Squared Prediction Error):** Also known as DModX or Q-statistic. It is the sum of squared residuals after projecting a data point onto the principal component model of the process. For a new point **x**, it is the squared distance to the PCA model plane:
            """)
            st.latex(r"SPE = || \mathbf{x} - \mathbf{P}\mathbf{P}'\mathbf{x} ||^2")
            st.markdown("where **P** is the matrix of PCA loadings (the model directions).")

        # --- NEW CONTENT: Explanation of core concepts ---
        with tabs[3]:
            st.markdown("""
            #### Understanding the Sources of Variation

            In any process, the total observed variation can be broken down into distinct components. Understanding these components is the key to effective process control and improvement.

            **1. Bias (Systematic Error)**
            - **Definition:** Bias is a consistent, repeatable error where the average of measurements is offset from the true value. It represents **inaccuracy**.
            - **Analogy:** A bathroom scale that always reads 5 pounds too heavy. The readings might be very consistent (low noise), but they are all consistently wrong in the same direction.
            - **In this Demo:** The **'Shift in Y Only'** scenario simulates a pure bias. The process mean for Pressure shifts from 150 to 175, but the underlying correlation structure and noise remain the same. The T² chart is excellent at detecting this kind of systematic bias.

            **2. Noise (Random Error)**
            - **Definition:** Noise is the unpredictable, non-repeatable fluctuation in measurements. It is inherent to any system and represents **imprecision**. It's the "static" that remains even after all biases are removed.
            - **Analogy:** A bathroom scale where the reading fluctuates randomly by a pound or two every time you step on it, even if your weight hasn't changed.
            - **In this Demo:** All scenarios include inherent noise, simulated by the `np.random.multivariate_normal` function. This is the "common cause" variation that control charts are designed to filter out, so we can see the "special cause" signals.

            ---

            #### Intrinsic vs. External Factors

            The sources of bias and noise can be categorized by their origin.

            **1. Intrinsic Factors (Internal Variability)**
            - **Definition:** These are sources of variation that are a fundamental, built-in part of the process itself. They define the "normal operating state."
            - **Examples:**
                - The inherent correlation between Temperature and Pressure in a chemical reaction.
                - The natural lot-to-lot variability of a raw material from a single, qualified supplier.
                - The baseline electronic noise of a sensor.
            - **In this Demo:** The strong positive correlation between Temperature and Pressure, defined by the covariance matrix `[[5, 12], [12, 40]]`, is an **intrinsic factor**. The T² chart models this intrinsic relationship.

            **2. External Factors (Special Causes)**
            - **Definition:** These are sources of variation that are not part of the normal process design. They are "shocks" or changes to the system that disrupt the stable state.
            - **Examples:**
                - A sensor failing, leading to a sudden shift in readings (Bias).
                - A new, untrained operator introducing inconsistent technique (Noise).
                - An unprecedented event, like a valve failure, that breaks the established relationship between two parameters.
            - **In this Demo:** The **'Correlation Break'** scenario simulates a powerful external factor. The covariance matrix changes to `[[5, 0], [0, 40]]`, breaking the intrinsic relationship. This is an event the original process model has never seen, which is why the SPE chart alarms loudly.
            """)
            
    # Nested plotting function from the user's code
def render_ewma_cusum():
    """Renders the comprehensive, interactive module for Small Shift Detection (EWMA/CUSUM)."""
    st.markdown("""
    #### Purpose & Application: The Process Sentinel
    **Purpose:** To deploy a high-sensitivity monitoring system designed to detect small, sustained shifts in a process mean that would be invisible to a standard Shewhart control chart (like an I-MR or X-bar chart). These charts have "memory," accumulating evidence from past data to find subtle signals.

    **Strategic Application:** This is an essential "second layer" of process monitoring for mature, stable processes where large, sudden failures are rare, but slow, gradual drifts are a significant risk.
    - **🔬 EWMA (The Sentinel):** The Exponentially Weighted Moving Average chart is a robust, general-purpose tool that smoothly weights past observations, making it excellent for detecting the onset of a gradual drift.
    - **🐕 CUSUM (The Bloodhound):** The Cumulative Sum chart is a specialized, high-power tool. It is the fastest possible detector for a shift of a specific magnitude, making it ideal for processes where you want to catch a known, critical shift size as quickly as possible.
    """)
    
    st.info("""
    **Interactive Demo:** Use the **Shift Size** slider in the sidebar to control how large of a process shift to simulate. Observe how the detection performance of the three charts changes. At what shift size does the I-Chart finally detect the problem? Notice how much earlier the EWMA and CUSUM charts signal an alarm for small shifts.
    """)
    
    st.sidebar.subheader("Small Shift Detection Controls")
    shift_size_slider = st.sidebar.slider(
        "Select Process Shift Size (in multiples of σ):",
        min_value=0.25,
        max_value=3.5,
        value=0.75,
        step=0.25,
        help="Controls the magnitude of the process shift introduced at data point #20. Small shifts are harder to detect."
    )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        # --- FIX: Unpack the new integer count KPIs ---
        fig, i_count, ewma_count, cusum_count = plot_ewma_cusum_comparison(shift_size=shift_size_slider)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            # --- FIX: Updated metrics to use the new counts and labels ---
            st.metric(
                label="Shift Size",
                value=f"{shift_size_slider} σ",
                help="The simulated shift introduced at data point #20."
            )
            st.metric(
                label="I-Chart: # of OOC Points Detected",
                value=f"{i_count} / 20",
                help="Total count of out-of-control points detected by the I-Chart in the shifted region (20 total shifted points)."
            )
            st.metric(
                label="EWMA: # of OOC Points Detected",
                value=f"{ewma_count} / 20",
                help="Total count of out-of-control points detected by the EWMA chart in the shifted region."
            )
            st.metric(
                label="CUSUM: # of OOC Points Detected",
                value=f"{cusum_count} / 20",
                help="Total count of out-of-control points detected by the CUSUM chart in the shifted region."
            )

            st.markdown("""
            **The Visual Evidence:**
            - **The I-Chart (Top):** This chart is blind to small problems. The shift is lost in the normal process noise. All points look "in-control," giving a false sense of security until the shift is very large.
            - **The EWMA Chart (Middle):** This chart has memory. The weighted average (the blue line) clearly begins to drift upwards after the shift occurs, eventually crossing the control limit.
            - **The CUSUM Chart (Bottom):** This chart is a "bloodhound." It accumulates all deviations from the target. Once the process shifts, the `CUSUM High` plot takes off, providing the fastest possible signal for small, sustained shifts.

            **The Core Strategic Insight:** Relying only on Shewhart charts creates a significant blind spot. For processes where small, slow drifts are a risk (e.g., reagent degradation, column aging), EWMA or CUSUM charts are essential.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "The One-Chart-Fits-All Fallacy"**
            A manager insists on using only I-MR charts for everything because they are easy to understand.
            - They miss a slow 1-sigma drift for weeks, producing tons of near-spec material.
            - When a batch finally fails, they are shocked and have no leading indicators to explain why. They have been flying blind.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Layer Your Statistical Defenses**
            The goal is to use a combination of charts to create a comprehensive security system.
            - **Use Shewhart Charts (I-MR, X-bar) as your front-line "Beat Cops":** They are unmatched for detecting large, sudden special causes.
            - **Use EWMA or CUSUM as your "Sentinels":** Deploy them alongside Shewhart charts to stand guard against the silent, creeping threats that the beat cops will miss.
            This layered approach provides a complete picture of process stability.
            """)

        with tabs[2]:
            st.markdown(r"""
            #### Historical Context & Origin: The Second Generation of SPC

            The quality revolution sparked by **Dr. Walter Shewhart's** control charts in the 1920s was a monumental success. For the first time, manufacturers had a tool to distinguish signal from noise, allowing them to achieve a state of statistical control. However, Shewhart's charts were designed like a **smoke detector**—brilliantly effective at detecting large, sudden events (a "fire"), but intentionally insensitive to small, slow changes. This design choice was deliberate, as it prevented "process tampering" by reacting to common cause noise.

            By the 1950s, the industrial world had evolved. Processes in fields like chemistry and electronics were becoming more complex and precise. The critical challenge was no longer just preventing large breakdowns, but detecting subtle, gradual drifts that could slowly degrade quality over time. The "smoke detector" wasn't sensitive enough; a new kind of sensor was needed. This set the stage for the second generation of SPC.

            - **CUSUM (1954): The Bloodhound**
                - **The Inventor:** The first major innovation came from British statistician **E. S. Page**. His work was rooted in **sequential analysis**, a field developed during WWII for efficiently testing munitions.
                - **The "Aha!" Moment:** Page realized that instead of looking at each data point in isolation, he could **accumulate the evidence** of small deviations over time. The Cumulative Sum (CUSUM) chart was born. It acts like a **bloodhound on a trail**. It ignores random noise (scents off the trail) by using a "slack" parameter, `k`. But once it detects a persistent scent in one direction—a real process shift—it starts accumulating the signal rapidly, leading to the fastest possible detection for a shift of a known size. Its V-mask design made it a powerful, albeit somewhat rigid, diagnostic tool.

            - **EWMA (1959): The Sentinel**
                - **The Inventor:** Five years later, statistician **S. W. Roberts** of Bell Labs (Shewhart's old stomping ground) proposed a more flexible alternative. His inspiration came not from quality control, but from **time series forecasting**.
                - **The "Aha!" Moment:** Forecasters like George Box had long used smoothing techniques to predict future values by giving more weight to recent data. Roberts ingeniously adapted this idea for process control. The Exponentially Weighted Moving Average (EWMA) chart acts like a **sentinel with a memory**. It doesn't treat all past data equally. By using the weighting parameter `λ`, it gives the most weight to the most recent data point, a little less to the one before, and so on, with the influence of old data decaying exponentially. This creates a smooth, sensitive trend line that effectively filters out noise while quickly reacting to the beginning of a real drift.

            These two inventions were not replacements for Shewhart's charts but essential complements. They gave engineers the sensitive, memory-based tools they needed to manage the increasingly precise and complex manufacturing processes of the late 20th century.

            #### Mathematical Basis
            The elegance of these charts lies in their simple, recursive formulas.

            - **EWMA (Exponentially Weighted Moving Average):**
            """)
            st.latex(r"EWMA_t = \lambda \cdot Y_t + (1-\lambda) \cdot EWMA_{t-1}")
            st.markdown(r"""
            - **`λ` (lambda):** This is the **memory parameter** (0 < λ ≤ 1).
                - A **small `λ`** (e.g., 0.1) creates a chart with a long memory, giving significant weight to past data. This makes it very sensitive to detecting tiny, persistent shifts.
                - A **large `λ`** (e.g., 0.4) creates a chart with a short memory, behaving more like a Shewhart chart. It's better for detecting larger shifts more quickly.
                - A typical starting value is `λ = 0.2`.

            - **CUSUM (Cumulative Sum):**
            """)
            st.latex(r"SH_t = \max(0, SH_{t-1} + (Y_t - T) - k)")
            st.markdown(r"""
            - This formula tracks upward shifts (`SH`). A similar formula tracks downward shifts (`SL`).
            - **`T`**: The process target or historical mean.
            - **`k`**: The **"slack" or "allowance" parameter**. This is the key to the CUSUM's power. It is typically set to half the size of the shift you want to detect quickly. For example, if you want to rapidly detect a 1-sigma shift, you set `k = 0.5σ`. Any deviation from the target that is smaller than `k` is considered noise and is absorbed, preventing the CUSUM from accumulating. Any deviation larger than `k` is considered a signal and is added to the cumulative sum. This makes the CUSUM chart a highly targeted detector.
            """)
            
def render_time_series_analysis():
    """Renders the module for Time Series analysis."""
    st.markdown("""
    #### Purpose & Application: The Watchmaker vs. The Smartwatch
    **Purpose:** To model and forecast time-dependent data by understanding its internal structure, such as trend, seasonality, and autocorrelation. This module compares two powerful philosophies for this task.
    
    **Strategic Application:** This is fundamental for demand forecasting, resource planning, and proactive process monitoring.
    - **⌚ ARIMA (The Classical Watchmaker):** A powerful and flexible "white-box" model. Like a master watchmaker, you must understand every gear (p,d,q parameters), but you get a highly interpretable model that is defensible in regulatory environments and excels at short-term forecasting of stable processes.
    - **📱 Prophet (The Modern Smartwatch):** A modern forecasting tool from Facebook. It's packed with sensors and algorithms to automatically handle complex seasonalities, holidays, and changing trends with minimal user input. It's designed for speed and scale.
    """)
    
    # --- NEW: Added Interactive Demo explanation ---
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to change the underlying structure of the time series data. 
    - **Increase `Trend Strength`:** See how both models adapt to a more aggressive upward trend.
    - **Increase `Random Noise`:** Observe how forecasting becomes more difficult and the error (MAE) for both models increases as the data gets noisier.
    """)

    # --- NEW: Added slider gadgets to the sidebar ---
    st.sidebar.subheader("Time Series Controls")
    trend_slider = st.sidebar.slider(
        "📈 Trend Strength",
        min_value=0, max_value=50, value=10, step=5,
        help="Controls the overall increase in the process value over the two-year period."
    )
    noise_slider = st.sidebar.slider(
        "🎲 Random Noise (SD)",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
        help="Controls the amount of random, unpredictable fluctuation in the data."
    )
    
    # --- MODIFIED: Call backend with slider values and unpack dynamic KPIs ---
    fig, mae_arima, mae_prophet = plot_time_series_analysis(trend_strength=trend_slider, noise_sd=noise_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            # --- MODIFIED: KPIs are now dynamic ---
            st.metric(label="⌚ ARIMA Forecast Error (MAE)", value=f"{mae_arima:.2f} units", help="Mean Absolute Error for the ARIMA model.")
            st.metric(label="📱 Prophet Forecast Error (MAE)", value=f"{mae_prophet:.2f} units", help="Mean Absolute Error for the Prophet model.")
            st.metric(label="🔮 Forecast Horizon", value="14 Weeks", help="The period into the future for which we are generating predictions.")

            st.markdown("""
            **Reading the Forecasts:**
            - **The Black Line:** This is the historical data the models were trained on.
            - **The Grey Line:** Marks the start of the forecast period. Data to the right is the "future" used to test the models.
            - **The Green (ARIMA) & Red (Prophet) Lines:** These are the models' predictions for the future. Compare them to the black line to see how well they performed.

            **The Core Strategic Insight:** The choice is not about which model is "best," but which is **right for the job.** For a stable, well-understood industrial process where interpretability is key, the craftsmanship of ARIMA is superior. For a complex, noisy business time series with multiple layers of seasonality and a need for automated forecasting at scale, Prophet is often the better tool.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Blind Forecasting" Fallacy**
            This is the most common path to a useless forecast.
            
            - An analyst takes a column of data, feeds it directly into `model.fit()` and `model.predict()`, and presents the resulting line.
            - **The Flaw:** They've made no attempt to understand the data's structure. Is there a trend? Is it seasonal? Is the variance stable? They have no idea if the model's assumptions have been met. This "black box" approach produces a forecast that is fragile, unreliable, and likely to fail spectacularly the moment the underlying process changes.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Decompose, Validate, and Monitor**
            A robust forecasting process is disciplined and applies regardless of the model you use.
            
            1.  **Decompose and Understand (The Pre-Flight Check):** Before you model, you must visualize. Use a time series decomposition plot to separate the series into its core components: **Trend, Seasonality, and Residuals.** This tells you what you're working with. Check for stationarity—a core assumption of ARIMA.
            
            2.  **Train, Validate, Test:** Never judge a model by its performance on data it has already seen. Split your historical data into a training set (to build the model) and a validation set (to tune it). Keep a final "test set" of the most recent data as a truly blind evaluation of forecast accuracy.
            
            3.  **Monitor for Drift:** A forecast is only a snapshot in time. You must continuously monitor its performance against incoming new data. When the error starts to increase, it's a signal that the underlying process has changed and the model needs to be retrained.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The story of time series forecasting is a tale of two distinct eras.
            - **The Classical Era (ARIMA):** In their seminal 1970 book *Time Series Analysis: Forecasting and Control*, statisticians **George Box** and **Gwilym Jenkins** provided a comprehensive methodology for time series modeling. The **Box-Jenkins method**—a rigorous process of model identification, parameter estimation, and diagnostic checking—became the undisputed gold standard for decades. The ARIMA model is the heart of this methodology, a testament to deep statistical theory.
            
            - **The Modern Era (Prophet):** Fast forward to the 2010s. **Facebook** faced a new kind of problem: thousands of internal analysts needed to generate high-quality forecasts for business metrics at scale, without each of them needing a PhD in statistics. In 2017, their Core Data Science team, led by Sean J. Taylor and Ben Letham, released **Prophet**. It was designed from the ground up for automation, performance, and intuitive tuning, sacrificing some of the statistical purity of ARIMA for massive gains in usability and scale.
            
            #### How They Work
            - **ARIMA (AutoRegressive Integrated Moving Average):**
              - **AR (p):** The model uses the relationship between an observation and its own **p**ast values.
              - **I (d):** It uses **d**ifferencing to make the series stationary (i.e., remove the trend).
              - **MA (q):** It uses the relationship between an observation and the residual errors from its **q** past forecasts.
            - **Prophet:** It works as a decomposable additive model:
            """)
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown(r"""
            Where `g(t)` is a saturating growth trend, `s(t)` models complex weekly and yearly seasonality using Fourier series, `h(t)` is a flexible component for user-specified holidays, and `ε` is the error.
            """)


def render_stability_analysis():
    """Renders the module for pharmaceutical stability analysis."""
    st.markdown("""
    #### Purpose & Application: The Expiration Date Contract
    **Purpose:** To fulfill a statistical contract with patients and regulators. This analysis determines the shelf-life (or retest period) for a drug product by proving, with high confidence, that a Critical Quality Attribute (CQA) like potency will remain within its safety and efficacy specifications over time.
    
    **Strategic Application:** This is a mandatory, high-stakes analysis for any commercial pharmaceutical product, as required by the **ICH Q1E guideline**. It is the data-driven foundation of the expiration date printed on every vial and box. An incorrectly calculated shelf-life can lead to ineffective medicine, patient harm, and massive product recalls. The analysis involves:
    - Collecting stability data from at least three primary batches.
    - Fitting a regression model to understand the degradation trend.
    - Using a conservative confidence interval to set a shelf-life that accounts for future batch-to-batch variability.
    """)
    
    # --- NEW: Added Interactive Demo explanation ---
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate different product stability profiles.
    - **Increase `Degradation Rate`:** Simulate a less stable product that degrades more quickly and see how it dramatically shortens the approved shelf-life.
    - **Increase `Assay Variability`:** Simulate a noisy, imprecise measurement method. Notice how this increases the uncertainty in the model (widens the red confidence interval), which also shortens the shelf-life even if the degradation rate is low.
    """)
    
    # --- NEW: Added slider gadgets to the sidebar ---
    st.sidebar.subheader("Stability Analysis Controls")
    degradation_slider = st.sidebar.slider(
        "📉 Degradation Rate (%/month)",
        min_value=-1.0, max_value=-0.1, value=-0.4, step=0.05,
        help="Controls how quickly the product loses potency. A more negative number means faster degradation."
    )
    noise_slider = st.sidebar.slider(
        "🎲 Assay Variability (SD)",
        min_value=0.2, max_value=2.0, value=0.5, step=0.1,
        help="The random error or 'noise' of the potency assay. Higher noise increases uncertainty."
    )

    # --- MODIFIED: Call backend with slider values and unpack KPIs ---
    fig, shelf_life, fitted_slope = plot_stability_analysis(degradation_rate=degradation_slider, noise_sd=noise_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            # --- MODIFIED: KPIs are now dynamic ---
            st.metric(label="📈 Approved Shelf-Life", value=f"{shelf_life} Months", help="The time at which the lower confidence bound intersects the specification limit.")
            st.metric(label="📉 Fitted Degradation Rate", value=f"{fitted_slope:.2f} %/month", help="The estimated average loss of potency per month from the regression model.")
            st.metric(label="🥅 Specification Limit", value="95.0 %", help="The minimum acceptable potency for the product to be considered effective.")

            st.markdown("""
            **Reading the Race Against Time:**
            - **The Data Points:** Your real-world potency measurements from three different production batches.
            - **The Black Line (Average Trend):** The best-fit regression line showing the average degradation path.
            - **The Red Dashed Line (Safety Net):** The **95% Lower Confidence Bound**. This is the most important line, representing a conservative estimate for the mean trend.
            - **The Red Dotted Line (Finish Line):** The specification limit.

            **The Verdict:** The shelf-life is declared at the exact moment **the Safety Net (red dashed line) hits the Finish Line.** This conservative approach ensures high confidence that the average product potency will not drop below spec before the expiration date.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Happy Path" Fallacy**
            This is a common and dangerous mistake that overestimates shelf-life.
            
            - A manager sees the solid black line (the average trend) and says, *"Let's set the shelf-life where the average trend crosses the spec limit. That gives us 36 months!"*
            - **The Flaw:** This completely ignores uncertainty and batch-to-batch variability! About half of all future batches will, by definition, degrade *faster* than the average. This approach virtually guarantees that a significant portion of future product will fail specification before its expiration date, putting patients at risk.
            - Another flaw is blindly pooling data from all batches without testing if their degradation rates are similar. If one batch is a "fast degrader," it must be evaluated separately.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Confidence Interval Sets the Expiration Date, Not the Average**
            The ICH Q1E guideline is built on a principle of statistical conservatism to protect patients. The correct procedure is disciplined:
            
            1.  **First, Prove Poolability:** Before you can create a single model, you must perform a statistical test (like ANCOVA) to prove that the degradation slopes and intercepts of the different batches are not significantly different. You must *earn the right* to pool the data. If they are different, the shelf-life must be based on the worst-performing batch.
            
            2.  **Then, Use the Confidence Bound:** Once pooling is justified, fit the regression model and calculate the two-sided 95% confidence interval. The shelf-life is determined by the intersection of the appropriate confidence bound (lower bound for potency, upper bound for an impurity) with the specification limit.
            
            This rigorous approach ensures the expiration date is a reliable promise.
            """)

        with tabs[2]:
            st.markdown(r"""
            #### Historical Context & Origin: The ICH Revolution
            Prior to the 1990s, the requirements for stability testing could differ significantly between major markets like the USA, Europe, and Japan. This forced pharmaceutical companies to run slightly different, redundant, and costly stability programs for each region to gain global approval.
            
            The **International Council for Harmonisation (ICH)** was formed to end this inefficiency. A key working group was tasked with creating a single, scientifically sound standard for stability testing. This resulted in a series of guidelines, with **ICH Q1A** defining the required study conditions and **ICH Q1E ("Evaluation of Stability Data")** providing the definitive statistical methodology.
            
            ICH Q1E, adopted in 2003, codified the use of regression analysis, formal tests for pooling data across batches, and the critical principle of using confidence intervals to determine shelf-life. It created a level playing field and a global gold standard, ensuring that the expiration date on a medicine means the same thing in New York, London, and Tokyo.
            
            #### Mathematical Basis
            The core of the analysis is typically a linear regression model:
            """)
            st.latex(r"Y_i = \beta_0 + \beta_1 X_i + \epsilon_i")
            st.markdown(r"""
            - **`Yᵢ`**: The CQA measurement (e.g., Potency) at time point `i`.
            - **`Xᵢ`**: The time point `i` (e.g., in months).
            - **`β₁`**: The slope, representing the degradation rate.
            - **`β₀`**: The intercept, representing the value at time zero.

            The confidence interval for the regression line is not a pair of parallel lines. It is a **funnel shape**, narrowest at the center of the data and widest at the beginning and end. This reflects that our prediction is most certain near the average time point of our data and becomes less certain the further we extrapolate. The formula for the confidence bound at a given time point `x` depends on the sample size, the standard error of the model, and the distance of `x` from the mean of all time points.
            """)

def render_survival_analysis():
    """Renders the module for Survival Analysis."""
    st.markdown("""
    #### Purpose & Application: The Statistician's Crystal Ball
    **Purpose:** To model "time-to-event" data and forecast the probability of survival over time. Its superpower is its unique ability to handle **censored data**—observations where the study ends before the event (e.g., failure or death) occurs. It allows us to use every last drop of information, even from the subjects who "survived" the study.
    
    **Strategic Application:** This is the core methodology for reliability engineering and is essential for predictive maintenance, risk analysis, and clinical research.
    - **⚙️ Predictive Maintenance:** Instead of replacing parts on a fixed schedule, you can model their failure probability over time. This answers: "What is the risk this HPLC column fails *in the next 100 injections*?" This moves maintenance from guesswork to a data-driven strategy.
    - **⚕️ Clinical Trials:** The gold standard for analyzing endpoints like "time to disease progression" or "overall survival." It provides definitive proof if a new drug helps patients live longer or stay disease-free for longer.
    - **🔬 Reagent & Product Stability:** A powerful way to model the "shelf-life" of a reagent lot or product by defining "failure" as dropping below a performance threshold.
    """)

    # --- NEW: Added Interactive Demo explanation ---
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate different reliability scenarios.
    - **Increase `Group B Reliability`:** Watch the red curve flatten and separate from the blue curve, simulating a more reliable new component. Notice how the p-value drops and the median survival time increases.
    - **Increase `Censoring Rate`:** Simulate a shorter study where fewer components fail. Notice the vertical tick marks (censored items) appear more frequently. With high censoring, it becomes harder to prove a significant difference.
    """)

    # --- NEW: Added slider gadgets to the sidebar ---
    st.sidebar.subheader("Survival Analysis Controls")
    lifetime_slider = st.sidebar.slider(
        "⚙️ Group B Reliability (Lifetime Scale)",
        min_value=15, max_value=45, value=30, step=1,
        help="Controls the characteristic lifetime of the 'New Component' (Group B). A higher value means it's more reliable."
    )
    censor_slider = st.sidebar.slider(
        " Censoring Rate (%)",
        min_value=0, max_value=80, value=20, step=5,
        help="The percentage of items that are still 'surviving' when the study ends. Simulates shorter vs. longer studies."
    )
    
    # --- MODIFIED: Call backend with slider values and unpack KPIs ---
    fig, median_a, median_b, p_value = plot_survival_analysis(
        group_b_lifetime=lifetime_slider, 
        censor_rate=censor_slider/100.0
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            # --- MODIFIED: KPIs are now dynamic ---
            st.metric(
                label="📊 Log-Rank Test p-value", 
                value=f"{p_value:.3f}", 
                help="A p-value < 0.05 indicates a statistically significant difference between the survival curves."
            )
            st.metric(
                label="⏳ Median Survival (Group A)", 
                value=f"{median_a:.1f} Months" if not np.isnan(median_a) else "Not Reached",
                help="Time at which 50% of Group A have experienced the event."
            )
            st.metric(
                label="⏳ Median Survival (Group B)", 
                value=f"{median_b:.1f} Months" if not np.isnan(median_b) else "Not Reached",
                help="Time at which 50% of Group B have experienced the event."
            )

            st.markdown("""
            **Reading the Curve:**
            - **The Stepped Line:** The **Kaplan-Meier curve** shows the estimated probability of survival over time.
            - **Vertical Drops:** Each drop represents one or more "events" (e.g., failures).
            - **Vertical Ticks (Censoring):** These represent items still working when the study ended. They are crucial pieces of information, not missing data.
            
            **The Visual Verdict:** The curve for **Group B** is consistently higher than Group A, demonstrating that items in Group B have a higher probability of surviving longer. The low p-value confirms this visual impression is statistically significant.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Pessimist's Fallacy"**
            This is a catastrophic but common error that leads to dangerously biased results.
            
            - An analyst wants to know the average lifetime of a component. They take data from a one-year study, **throw away all the censored data** (the units that were still working at one year), and calculate the average time-to-failure for only the units that broke.
            - **The Flaw:** This is a massive pessimistic bias. You have selected **only the weakest items** that failed early and completely ignored the strong, reliable items that were still going strong. The calculated "average lifetime" will be far lower than the true value.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Respect the Censored Data**
            The core principle of survival analysis is that censored data is not missing data; it is valuable information.
            
            - A tick on the curve at 24 months is not an unknown. It is a powerful piece of information: **The lifetime of this unit is at least 24 months.**
            - The correct approach is to **always use a method specifically designed to handle censoring**, like the Kaplan-Meier estimator. This method correctly incorporates the information from both the "failures" and the "survivors" to produce an unbiased estimate of the true survival function.
            
            Never discard censored data. It is just as important as the failure data for getting the right answer.
            """)

        with tabs[2]:
            st.markdown(r"""
            #### Historical Context & Origin: The 1958 Revolution
            While the concept of life tables has existed for centuries in actuarial science, analyzing time-to-event data with censored observations was often messy and inconsistent. Different researchers used different ad-hoc methods, making it hard to compare results.
            
            This all changed in 1958 with a landmark paper in the *Journal of the American Statistical Association* by **Edward L. Kaplan** and **Paul Meier**. Their paper, "Nonparametric Estimation from Incomplete Observations," introduced the world to what we now universally call the **Kaplan-Meier estimator**.
            
            It was a revolutionary breakthrough. They provided a simple, elegant, and statistically robust non-parametric method to estimate the true survival function, even with heavily censored data. This single technique unlocked a new era of research in medicine, enabling the rigorous analysis of clinical trials that is now standard practice, and in engineering, forming the foundation of modern reliability analysis.
            
            #### Mathematical Basis
            The Kaplan-Meier estimate of the survival function `S(t)` is a product of conditional probabilities:
            """)
            st.latex(r"S(t_i) = S(t_{i-1}) \times \left( 1 - \frac{d_i}{n_i} \right)")
            st.markdown(r"""
            - **`S(tᵢ)`** is the probability of surviving past time `tᵢ`.
            - **`nᵢ`** is the number of subjects "at risk" (i.e., still surviving) just before time `tᵢ`.
            - **`dᵢ`** is the number of events (e.g., failures) that occurred at time `tᵢ`.
            
            Essentially, the probability of surviving to a certain time is the probability you survived up to the last event, *times* the conditional probability you survived this current event. This step-wise calculation gracefully handles censored observations, as they simply exit the "at risk" pool (`nᵢ`) at the time they are censored.
            """)


def render_mva_pls():
    """Renders the module for Multivariate Analysis (PLS)."""
    st.markdown("""
    #### Purpose & Application: The Statistical Rosetta Stone
    **Purpose:** To act as a **statistical Rosetta Stone**, translating a massive, complex, and correlated set of input variables (X, e.g., an entire spectrum) into a simple, actionable output (Y, e.g., product concentration). **Partial Least Squares (PLS)** is the key that deciphers this code.
    
    **Strategic Application:** This is the statistical engine behind **Process Analytical Technology (PAT)** and modern chemometrics. It is specifically designed to solve the "curse of dimensionality"—problems where you have more input variables than samples and the inputs are highly correlated.
    - **🔬 Real-Time Spectroscopy:** Builds models that predict a chemical concentration from its NIR or Raman spectrum in real-time. This eliminates the need for slow, offline lab tests, enabling real-time release.
    - **🏭 "Golden Batch" Modeling:** PLS can learn the "fingerprint" of a perfect batch, modeling the complex relationship between hundreds of process parameters and final product quality. Deviations from this model can signal a problem *during* a run, not after it's too late.
    """)

    # --- NEW: Added Interactive Demo explanation ---
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to simulate different chemometric scenarios.
    - **Increase `Signal Strength`:** Watch the VIP scores for the true signal peaks (highlighted in green) grow taller, making the true relationship easier for the model to find. Both R² and Q² will improve.
    - **Increase `Noise Level`:** Simulate a poor-quality instrument. Watch the VIP scores for the true peaks shrink as they become buried in noise, and see the model's predictive power (Q²) collapse.
    """)

    # --- NEW: Added slider gadgets to the sidebar ---
    st.sidebar.subheader("Multivariate Analysis Controls")
    signal_slider = st.sidebar.slider(
        "📈 Signal Strength",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="Controls the strength of the true underlying relationship between the spectra (X) and the concentration (Y)."
    )
    noise_slider = st.sidebar.slider(
        "🎲 Noise Level (SD)",
        min_value=0.1, max_value=2.0, value=0.2, step=0.1,
        help="Controls the amount of random noise in the spectral measurements. Higher noise makes the signal harder to find."
    )
    
    # --- MODIFIED: Call backend with slider values and unpack KPIs ---
    fig, r2, q2, n_comp = plot_mva_pls(signal_strength=signal_slider, noise_sd=noise_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            # --- MODIFIED: KPIs are now dynamic ---
            st.metric(label="📈 Model R² (Goodness of Fit)", value=f"{r2:.3f}", help="How well the model fits the training data. High is good, but can be misleading.")
            st.metric(label="🎯 Model Q² (Predictive Power)", value=f"{q2:.3f}", help="The cross-validated R². A measure of how well the model predicts *new* data. Q² is the most important performance metric.")
            st.metric(label="🧬 Optimal Latent Variables (LVs)", value=f"{n_comp}", help="The optimal number of hidden factors extracted by the model via cross-validation.")
            
            st.markdown("""
            **Decoding the VIP Plot:**
            The **Variable Importance in Projection (VIP)** plot is the key to understanding what the model has learned.
            - **The Peaks:** These represent the input variables (wavelengths) most influential for predicting the output.
            - **The Green Zones:** These mark the true causal peaks we built into the simulation. A good model should have high VIP scores in these zones.
            - **The Red Line (VIP > 1):** Variables with a VIP score greater than 1 are considered important to the model.
            
            **The Core Strategic Insight:** PLS turns a "black box" instrument into a "glass box" of process understanding. By identifying the most important variables, scientists can gain fundamental insights into the underlying chemistry of their process.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Overfitting" Trap**
            This is the cardinal sin of predictive modeling.
            
            - An analyst keeps adding more and more Latent Variables (LVs) to their PLS model. They are thrilled to see the R-squared value climb to 0.999. The model perfectly "predicts" the data it was built on.
            - **The Flaw:** The model hasn't learned the true signal; it has simply memorized the noise in the training data. When this model is shown new data from the process, its predictions will be terrible. It is a fragile model that is useless in the real world.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Thou Shalt Validate Thy Model on Unseen Data**
            A model's R-squared on the data it was trained on is vanity. Its performance on new data is sanity.
            
            1.  **Partition Your Data:** Before you begin, split your data into a **Training Set** (to build the model) and a **Test Set** (to independently validate it).
            
            2.  **Use Cross-Validation:** Within the training set, use cross-validation to choose the optimal number of Latent Variables. The goal is to find the number of LVs that maximizes the **predictive power (Q²)**, not the number that maximizes the R-squared.
            
            3.  **Final Verdict:** The ultimate test of the model is its performance on the held-out Test Set. This simulates how the model will perform in the future when it encounters new process data.
            
            A model that predicts well is useful. A model that is *proven* to predict well is valuable.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            PLS was developed in the 1960s and 70s by the brilliant Swedish statistician **Herman Wold**. He originally developed it for the complex, "data-rich but theory-poor" problems found in econometrics and social sciences.
            
            However, its true potential was unlocked by his son, **Svante Wold**, a chemist. In the late 1970s and 80s, Svante recognized that the problems his father was solving were mathematically identical to the challenges in **chemometrics**—the science of extracting information from chemical systems by data-driven means. Analytical instruments like spectrometers were producing huge, highly correlated datasets that traditional statistics couldn't handle.
            
            Svante Wold and his colleagues adapted and popularized PLS, turning it into the powerhouse of modern chemometrics. This father-son legacy created a tool that bridged disciplines and became the statistical engine for the PAT revolution in the pharmaceutical industry.
            
            #### How It Works: The Consensus Group Analogy
            How does PLS handle thousands of inputs? It doesn't use them directly.
            - **Standard Regression** is like trying to listen to 1000 people shouting at once. It's chaos.
            - **PLS is smarter.** It first tells the 1000 people (X variables) to form a few small "consensus groups" based on who is saying similar things. These groups are the **Latent Variables (LVs)**.
            - Then, PLS simply listens to the summary from these few group leaders to make its prediction about the Y variable. This process of creating a few informative LVs from thousands of inputs is called **dimensionality reduction**, and it's the core of how PLS works.
            """)

def render_clustering():
    """Renders the module for unsupervised clustering."""
    st.markdown("""
    #### Purpose & Application: The Data Archeologist
    **Purpose:** To act as a **data archeologist**, sifting through your process data to discover natural, hidden groupings or "regimes." Without any prior knowledge, it can uncover distinct "civilizations" within your data, answering the question: "Are all of my 'good' batches truly the same, or are there different ways to be good?"
    
    **Strategic Application:** This is a powerful exploratory tool for deep process understanding. It moves you from assumption to discovery.
    - **Process Regime Identification:** Can reveal that a process is secretly operating in two or three different states (e.g., due to different raw material suppliers, seasonal effects, or operator techniques), even when all batches are passing specification. This is often the first clue to a major process improvement.
    - **Root Cause Analysis:** If a failure occurs, clustering can help determine which "family" of normal operation the failed batch was most similar to, providing critical clues for the investigation.
    - **Customer Segmentation:** In a commercial context, it can be used to segment patients or customers into distinct groups based on their characteristics, enabling targeted strategies.
    """)
    
    st.info("""
    **Interactive Demo:** Use the sliders in the sidebar to change the underlying structure of the data.
    - **Increase `Cluster Separation`:** Watch the groups move farther apart. This makes the clustering task easier, and you will see the **Silhouette Score (cluster quality) increase** and a **sharper bend in the Elbow Plot**.
    - **Increase `Cluster Spread (Noise)`:** Watch the groups become wider and more diffuse. This makes the clustering task harder, and the **Silhouette Score will decrease** as the clusters begin to overlap.
    """)

    st.sidebar.subheader("Clustering Controls")
    separation_slider = st.sidebar.slider(
        "Cluster Separation",
        min_value=5, max_value=25, value=15, step=1,
        help="Controls how far apart the centers of the data clusters are."
    )
    spread_slider = st.sidebar.slider(
        "Cluster Spread (Noise)",
        min_value=1.0, max_value=10.0, value=2.5, step=0.5,
        help="Controls the standard deviation (spread) within each cluster. Higher spread means more overlap."
    )
    
    fig_scatter, fig_elbow, silhouette_val = plot_clustering(separation=separation_slider, spread=spread_slider)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    st.subheader("Analysis & Interpretation")
    tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History", "🧠 How It Works"])
    
    with tabs[0]:
        st.metric(label="🏺 Assumed 'Regimes' (k)", value="3", help="The number of clusters the K-Means algorithm was asked to find for the scatter plot.")
        st.metric(label="🗺️ Cluster Quality (Silhouette Score)", value=f"{silhouette_val:.3f}", help="A measure of how distinct the clusters are from each other. Higher is better (max 1.0).")
        st.metric(label="⛏️ Algorithm Used", value="K-Means", help="A classic and robust partitioning-based clustering algorithm.")
        
        st.markdown("""
        **The Dig Site Findings:**
        - **Left Plot:** The algorithm, without any help, has found three distinct groups in the data, color-coded for clarity. The black crosses mark the final calculated centers (centroids) of these groups.
        - **Right Plot:** The Elbow Method confirms that *k*=3 (the "elbow" of the curve) is the optimal number of clusters for this dataset.
        
        **The Core Strategic Insight:** The discovery of hidden clusters is one of the most valuable findings in data analysis. It proves that your single process is actually a collection of multiple sub-processes. Understanding the *causes* of this separation is the gateway to improved process control, robustness, and optimization.
        """)

    with tabs[1]:
        st.error("""
        🔴 **THE INCORRECT APPROACH: The "If It Ain't Broke..." Fallacy**
        This is the most common way to squander the value of a clustering analysis.
        
        - An analyst presents the discovery of three distinct clusters. A manager responds, *"Interesting, but all of those batches passed QC testing, so who cares? Let's move on."*
        - **The Flaw:** This treats a treasure map as a doodle. The fact that all batches passed is what makes the discovery so important! It means there are different—and potentially more or less robust—paths to success. One of those "regimes" might be living on the edge of a cliff (close to a specification limit), while another is safe in a valley.
        """)
        st.success("""
        🟢 **THE GOLDEN RULE: A Cluster is a Clue, Not a Conclusion**
        The discovery of clusters is the **start** of the investigation, not the end. The correct approach is a disciplined forensic analysis.
        
        1.  **Find the Clusters:** Use an algorithm like K-Means to partition the data.
        2.  **Validate the Clusters:** Use a metric like the Silhouette Score to ensure the clusters are meaningful.
        3.  **Profile the Clusters (This is the most important step!):** Treat each cluster as a separate group. Overlay other information. Ask:
            - Do batches in Cluster 1 use a different raw material lot than Cluster 2?
            - Were the batches in Cluster 3 all run by the night shift?
            - Is there a seasonal effect that separates the clusters?
        
        This profiling step is what turns a statistical finding into actionable process knowledge.
        """)

    with tabs[2]:
        st.markdown("""
        #### Historical Context & Origin
        The K-Means algorithm is a foundational pillar of machine learning, with a history that predates many modern techniques. While the core idea was explored by several researchers, it was first proposed by **Stuart Lloyd** at Bell Labs in 1957 as a technique for pulse-code modulation. The term "k-means" itself was first coined by **James MacQueen** in a 1967 paper.
        """)
        
        # --- NEW, EXTENDED CONTENT STARTS HERE ---
        st.markdown("""
        The algorithm's development was a direct consequence of the dawn of the digital computing age. For the first time, the iterative, computationally intensive process of repeatedly assigning points and updating centroids became feasible. Lloyd's original application was for optimizing the transmission of signals, a core problem for Bell Labs as they built out the world's telecommunications infrastructure. The goal was to find a small set of "codebook vectors" (cluster centroids) that could efficiently represent a much larger set of signals, thereby compressing the data.

        This concept—finding a small set of prototypes to represent a large, complex dataset—is the essence of what K-Means does. Its simplicity and intuitive geometric interpretation made it a go-to tool as computer science and data analysis grew. It became a canonical example of an **unsupervised learning** algorithm, a paradigm where the goal is not to predict a known label but to discover the inherent structure in the data itself. Along with Principal Component Analysis (PCA), K-Means helped lay the groundwork for the modern field of data mining and data science, representing a fundamental shift in data analysis—from testing pre-defined hypotheses to exploring data to generate *new* hypotheses.
        """)
        # --- NEW, EXTENDED CONTENT ENDS HERE ---

    with tabs[3]:
        st.markdown("""
        #### How do you choose the number of clusters (k)?
        
        This is the most common question in clustering. While in this demo we set *k*=3, in a real analysis, you wouldn't know the right number. The most common method to estimate *k* is the **Elbow Method**.
        
        1.  **Run K-Means multiple times:** You run the algorithm for a range of *k* values (e.g., from *k*=1 to *k*=10).
        2.  **Calculate the Inertia:** For each run, you calculate the **Within-Cluster Sum of Squares (WCSS)**, also called "inertia." This is a measure of how compact and tight the clusters are. A lower WCSS is better.
        3.  **Plot the results:** You plot WCSS (y-axis) vs. *k* (x-axis). The resulting curve typically looks like an arm. The point where the curve bends—the **"elbow"**—is considered the optimal number of clusters. It represents the point of diminishing returns, where adding another cluster doesn't significantly improve the compactness of the clusters.

        *Other advanced methods include the Silhouette Score analysis and the Gap Statistic, which provide more statistically rigorous ways to find the optimal k.*
        
        ---
        
        #### How are the borders between clusters established?

        The K-Means algorithm establishes the borders with a simple and ruthless rule: **every point in the space belongs to the closest cluster center (centroid).**

        This creates a geometric structure called a **Voronoi Tessellation**. Imagine planting a flag at each of the final cluster centroids (the black crosses in the plot). The border between any two clusters is the line that is exactly halfway between their two flags. Every location on one side of that line is closer to one flag, and every location on the other side is closer to the other. When you do this for all the cluster centers, you carve up the entire space into distinct territories, and these territories are the clusters. The borders are perfectly defined, straight lines.
        
        ---
        
        #### How is the "size" of a cluster determined?
        
        This is the most straightforward part. The size of a cluster is simply the **total number of data points** that have been assigned to it after the algorithm finishes. If you have 150 data points and the algorithm assigns 60 to Cluster 1, 50 to Cluster 2, and 40 to Cluster 3, then their sizes are 60, 50, and 40, respectively.
        """)

def render_classification_models():
    """Renders the module for Predictive QC (Classification)."""
    st.markdown("""
    #### Purpose & Application: The AI Gatekeeper
    **Purpose:** To build an **AI Gatekeeper** that can inspect in-process data and predict, with high accuracy, whether a batch will ultimately pass or fail its final QC specifications. This moves quality control from a reactive, end-of-line activity to a proactive, predictive science.
    
    **Strategic Application:** This is the foundation of real-time release and "lights-out" manufacturing. By predicting outcomes early, we can:
    - **Prevent Failures:** Intervene in a batch that is trending towards failure, saving it before it's too late.
    - **Optimize Resource Allocation:** Divert QC lab resources away from batches predicted to be good and focus on those with higher risk.
    - **Accelerate Release:** Provide the statistical evidence needed to release batches based on in-process data, rather than waiting for slow offline tests.
    """)
    
    # --- NEW: Added Interactive Demo explanation ---
    st.info("""
    **Interactive Demo:** Use the **Boundary Complexity** slider in the sidebar to change the true pass/fail relationship in the simulated data.
    - **High values (e.g., 20):** Creates a simple, almost linear boundary. Notice both models perform well.
    - **Low values (e.g., 8):** Creates a complex, non-linear "island" of failures. Watch the accuracy of the linear Logistic Regression model collapse, while the non-linear Random Forest continues to perform well.
    """)

    # --- NEW: Added slider gadget to the sidebar ---
    st.sidebar.subheader("Predictive QC Controls")
    complexity_slider = st.sidebar.slider(
        "Boundary Complexity",
        min_value=4, max_value=25, value=12, step=1,
        help="Controls how non-linear the true pass/fail boundary is. Lower values create a more complex 'island' that is harder for linear models to solve."
    )
    
    # --- MODIFIED: Call backend with slider value and unpack KPIs ---
    fig, lr_accuracy, rf_accuracy = plot_classification_models(boundary_radius=complexity_slider)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            # --- MODIFIED: KPIs are now dynamic ---
            st.metric(label="📈 Model 1: Logistic Regression Accuracy", value=f"{lr_accuracy:.2%}", help="Performance of the simpler, linear model.")
            st.metric(label="🚀 Model 2: Random Forest Accuracy", value=f"{rf_accuracy:.2%}", help="Performance of the more complex, non-linear model.")

            st.markdown("""
            **Reading the Decision Boundaries:**
            - The plots show how each model carves up the process space into "predicted pass" (blue regions) and "predicted fail" (red regions). The dots are the true outcomes.
            - **Logistic Regression (Left):** This classical statistical model can only draw a **straight line** to separate the groups. It struggles when the true boundary is curved.
            - **Random Forest (Right):** This powerful machine learning model can create a complex, **non-linear boundary**. It can learn the true "island" of failure, leading to much higher accuracy on complex problems.

            **The Core Strategic Insight:** For complex biological or chemical processes, the relationship between process parameters and final quality is rarely linear. Modern machine learning models like Random Forest or Gradient Boosting are often required to capture this complexity and build a truly effective AI Gatekeeper.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Garbage In, Garbage Out" Fallacy**
            An analyst takes all 500 available sensor tags, feeds them directly into a model, and trains it.
            
            - **The Flaw 1 (Curse of Dimensionality):** With more input variables than batches, the model is likely to find spurious correlations and will fail to generalize to new data.
            - **The Flaw 2 (Lack of Causality):** The model may learn that "Sensor A" is highly predictive, without understanding that Sensor A is only correlated with the true causal driver, "Raw Material B". If the correlation changes, the model breaks.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Feature Engineering is the Secret Ingredient**
            The success of a predictive model depends less on the algorithm and more on the quality of the inputs ("features").
            
            1.  **Collaborate with SMEs:** Work with scientists and engineers to identify which process parameters are *scientifically likely* to be causal drivers of quality.
            
            2.  **Engineer Smart Features:** Don't just use raw sensor values. Create more informative features. Examples:
                - The *slope* of the temperature profile during a key phase.
                - The *cumulative* feed volume.
                - The *time* spent above a certain pH.
            
            3.  **Validate on Unseen Data:** The model's true performance is only revealed when it is tested on a hold-out set of batches it has never seen before.
            
            A model built on a few, scientifically relevant, well-engineered features will always outperform a model built on hundreds of raw, noisy inputs.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            This module showcases the evolution from classical statistics to modern machine learning, representing what statistician Leo Breiman called **"The Two Cultures"** of statistical modeling.
            - **Logistic Regression (1958) - The Data Modeling Culture:** Developed by British statistician **David Cox**, it is a direct generalization of linear regression for binary (pass/fail) outcomes. It models the **log-odds** of the outcome as a linear combination of the input variables. It remains a powerful and highly interpretable baseline model. This approach comes from the classical statistics tradition, where the goal is to create an interpretable mathematical model that explains the relationship between the inputs and the output.

            - **Random Forest (2001) - The Algorithmic Modeling Culture:** Invented by **Leo Breiman and Adele Cutler**, this is a quintessential machine learning algorithm. It is an **ensemble method** that builds hundreds of individual decision trees on random subsets of the data and features, and then makes its final prediction based on a "majority vote" of all the trees. This "wisdom of the crowd" approach makes it highly accurate and robust to overfitting. This approach comes from the machine learning and computer science tradition, where the primary goal is predictive accuracy, even if the internal logic is complex.

            Tools like **Explainable AI (XAI)** are now bridging this gap, allowing us to use powerful algorithmic models like Random Forest while still gaining deep insights.
            
            #### How They Work
            - **Logistic Regression:** It fits a linear equation to the data and then passes the output through a **Sigmoid function**, which squashes the result into a probability between 0 and 1.
            - **Random Forest:** It creates a diverse "committee" of simple decision tree models. Each tree gets a vote, and the final prediction is the one that receives the most votes. This ensemble approach is why it can create complex, non-linear decision boundaries.
            """)
            
def render_anomaly_detection():
    """Renders the module for unsupervised anomaly detection."""
    st.markdown("""
    #### Purpose & Application: The AI Bouncer
    **Purpose:** To deploy an **AI Bouncer** for your data—a smart system that identifies rare, unexpected observations (anomalies) without any prior knowledge of what "bad" looks like. It doesn't need a list of troublemakers; it learns the "normal vibe" of the crowd and flags anything that stands out.
    
    **Strategic Application:** This is a game-changer for monitoring complex processes where simple rule-based alarms are blind to new problems.
    - **Novel Fault Detection:** The AI Bouncer's greatest strength. It can flag a completely new type of process failure the first time it occurs, because it looks for "weirdness," not pre-defined failures.
    - **Intelligent Data Cleaning:** Automatically identifies potential sensor glitches or data entry errors before they contaminate models or analyses.
    - **"Golden Batch" Investigation:** Can find which batches, even if they passed all specifications, were statistically unusual. These "weird-but-good" batches often hold the secrets to improving process robustness.
    """)

    st.info("""
    **Interactive Demo:** Use the **Expected Contamination** slider in the sidebar. This slider controls the model's sensitivity.
    - **Low values (e.g., 1%):** Makes the "AI Bouncer" very strict. It will only flag the most extreme and obvious outliers as anomalies.
    - **High values (e.g., 20%):** Makes the bouncer very lenient. It will start to flag points that are closer to the main "normal" crowd, increasing the number of detected anomalies.
    """)

    st.sidebar.subheader("Anomaly Detection Controls")
    contamination_slider = st.sidebar.slider(
        "Expected Contamination (%)",
        min_value=1, max_value=25, value=10, step=1,
        help="Your assumption about the percentage of anomalies in the data. This tunes the model's sensitivity."
    )

    # --- THIS LINE NOW CORRECTLY CALLS THE TOP-LEVEL HELPER FUNCTION ---
    fig, num_anomalies = plot_isolation_forest(contamination_rate=contamination_slider/100.0)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="Total Data Points Scanned", value="115")
            st.metric(label="Anomalies Flagged by Model", value=f"{num_anomalies}", help="The number of points classified as anomalies based on the selected contamination rate.")
            st.metric(label="Algorithm Used", value="Isolation Forest", help="An unsupervised machine learning method.")

            st.markdown("""
            **Reading the Chart:**
            - **The Blue Circles:** This is the "normal crowd" in your process data. They are dense and clustered together.
            - **The Red 'X's:** These are the anomalies. The algorithm has flagged them as "not belonging" to the main crowd.
            - **The Unsupervised Magic:** The key is that we never told the algorithm where the "normal" data was. It learned the data's structure on its own and identified the points that were easiest to isolate.

            **The Core Strategic Insight:** Anomaly detection is your early warning system for the **unknown unknowns**. While a control chart tells you if you've broken a known rule, an anomaly detector tells you that something you've never seen before just happened.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "The Glitch Hunter"**
            When an anomaly is detected, the immediate reaction is to dismiss it as a data error.
            
            - *"Oh, that's just a sensor glitch. Delete the point and move on."*
            - *"The model must be wrong. That batch passed all its QC tests, so it can't be an anomaly."*
            - *"Let's increase the contamination parameter until the alarms go away."*
            
            This approach treats valuable signals as noise. It's like the bouncer seeing a problem, shrugging, and looking the other way. You are deliberately blinding yourself to potentially critical process information.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: An Anomaly is a Question, Not an Answer**
            The goal is to treat every flagged anomaly as the start of a forensic investigation.
            
            - **The anomaly is the breadcrumb:** When the bouncer flags someone, you don't instantly throw them out. You ask questions. "What happened in the process at that exact time? Was it a specific operator? A new raw material lot? A strange environmental reading?"
            - **Investigate the weird-but-good:** If a batch that passed all specifications is flagged as an anomaly, it's a golden opportunity. What made it different? Did it run faster? With a higher yield? Understanding these "good" anomalies is a key to process optimization.
            
            The anomaly itself is not the conclusion; it is the starting pistol for discovery.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            For decades, "outlier detection" was a purely statistical affair, often done one variable at a time (e.g., using a boxplot). This falls apart in the world of modern, high-dimensional data where an event might be anomalous not because of one value, but because of a strange *combination* of many values.
            
            The **Isolation Forest** algorithm was a brilliant solution to this problem, introduced in a 2008 paper by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Their insight was elegantly counter-intuitive. Instead of trying to build a complex model of what "normal" data looks like, they decided to just try to **isolate** every data point.
            
            They reasoned that anomalous points are, by definition, "few and different." This makes them much easier to separate from the rest of the data points. Like finding a single red marble in a jar of blue ones, it's easy to "isolate" because it doesn't blend in. This approach turned out to be both highly effective and computationally fast, and it has become a go-to method for unsupervised anomaly detection.
            
            #### How it Works: The "20 Questions" Analogy
            Think of the algorithm playing a game of "20 Questions" to find a specific data point.
            1.  It builds a "forest" of many random decision trees.
            2.  Each "question" in a tree is a random split on a random feature (e.g., "Is temperature > 50?").
            3.  It counts the number of questions (the path length) it takes to uniquely identify each point.
            4.  **The Result:** Points in the heart of the normal cluster are hard to isolate and require many questions. Anomalous points are isolated very quickly with few questions. The algorithm calculates an "anomaly score" based on the average path length across all the trees in the forest.
            """)
            
def render_xai_shap():
    """Renders the module for Explainable AI (XAI) using SHAP."""
    st.markdown("""
    #### Purpose & Application: The AI Root Cause Investigator
    **Purpose:** To deploy an **AI Investigator** that forces a complex "black box" model to confess exactly *why* it predicted a specific assay run would fail. **Explainable AI (XAI)** cracks open the black box to reveal the model's reasoning.
    
    **Strategic Application:** This is a crucial tool for validating and deploying predictive models in a regulated GxP environment, especially for **tech transfer and continued process verification.** Instead of just getting a pass/fail prediction, you get a full root cause analysis for every run.
    - **🔬 Model Validation:** Confirm that the model is flagging runs for scientifically valid reasons (e.g., a low calibrator slope) and not due to spurious correlations (e.g., the day of the week).
    - **🎯 Proactive Troubleshooting:** If the model predicts a run has a high risk of failure, the SHAP plot immediately points to the most likely reasons, allowing technicians to intervene *before* the run is completed.
    - **⚖️ Tech Transfer Evidence:** Provides objective, data-driven evidence that a receiving lab's process is performing identically to the sending lab's, or pinpoints exactly which parameters are driving any observed differences.
    """)

    st.info("""
    **Interactive Demo:** Use the **"Select a Case to Investigate"** radio buttons in the sidebar to choose a specific, high-interest assay run.
    - The **Global Feature Importance** plot always shows the model's overall strategy.
    - The **Local Prediction Explanation** will update to show the specific root cause analysis for the case you've chosen to investigate. Compare the 'highest risk' and 'lowest risk' cases to see how the factors change.
    """)

    # --- Redesigned gadget using radio buttons for clear choices ---
    st.sidebar.subheader("XAI Investigation Controls")
    case_choice = st.sidebar.radio(
        "Select a Case to Investigate:",
        options=['highest_risk', 'lowest_risk', 'most_ambiguous'],
        format_func=lambda key: {
            'highest_risk': "Highest Predicted Failure Risk",
            'lowest_risk': "Lowest Predicted Failure Risk",
            'most_ambiguous': "Most Ambiguous Case (Prediction ≈ 50%)"
        }[key],
        help="Select a meaningful scenario to see its specific SHAP explanation."
    )
    
    # Call backend with the selected case
    summary_buf, force_html, selected_instance_df, actual_outcome, found_index = plot_xai_shap(case_to_explain=case_choice)
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.subheader("Global Feature Importance (The Model's General Strategy)")
        st.image(summary_buf, caption="This plot shows which factors have the biggest impact on run failure risk across all runs.")
        
        st.subheader(f"Local Prediction Explanation for Run #{found_index} ({case_choice.replace('_', ' ').title()})")
        st.dataframe(selected_instance_df)
        st.components.v1.html(force_html, height=150, scrolling=False)
        st.caption("This force plot explains why the model made its prediction for this specific run.")
        
    with col2:
        st.subheader("Analysis & Interpretation")
        # --- Added the new "SME Analysis" tab ---
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History", "🔬 SME Analysis"])

        with tabs[0]:
            st.metric("Actual Run Outcome", actual_outcome)
            st.markdown("""
            **Global Explanation (Top-Left Plot):**
            - **Feature Importance:** The model has correctly learned that `Operator Experience`, `Calibrator Slope`, and `Reagent Age` are the most important predictors of run failure.
            - **Impact Direction:** High values of `Reagent Age` (red dots) push the prediction towards failure (positive SHAP value), while high `Operator Experience` (red dots) pushes the prediction towards success (negative SHAP value). This confirms the model's logic is scientifically sound.
            
            **Local Explanation (Bottom-Left Plot):**
            - This is a root cause analysis for the selected run.
            - **Red Forces:** These are the factors pushing this specific run's risk **higher**.
            - **Blue Forces:** These are the factors pushing this run's risk **lower**.
            - **Final Prediction:** The model's final risk assessment. By switching between the cases in the sidebar, you can see a clear story: the 'highest risk' run is dominated by red factors, while the 'lowest risk' run is dominated by blue factors.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Accuracy is Everything" Fallacy**
            This is a dangerous mindset that leads to deploying untrustworthy models.
            
            - An analyst builds a model with 99% accuracy to predict run failures. They declare victory and push to put it into production without any further checks.
            - **The Flaw:** The model might be a "Clever Hans"—like the horse that could supposedly do math but was actually just reacting to its trainer's subtle cues. The model might have learned a nonsensical, spurious correlation in the training data (e.g., "runs performed on Mondays always fail"). The high accuracy is an illusion that will shatter when the model sees new data where that spurious correlation doesn't hold.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Explainability Builds Trust and Uncovers Flaws**
            The goal of XAI is not just to explain predictions, but to use those explanations to **validate the model's reasoning and build trust** in its decisions.
            
            1.  **Build the Model:** Train your powerful "black box" model (e.g., XGBoost, Random Forest) to achieve high predictive accuracy.
            2.  **Interrogate with SHAP:** Apply SHAP to the model's predictions on a validation set.
            3.  **Consult the Expert:** Show the SHAP plots to a Subject Matter Expert (SME) who knows the assay science. Ask them: *"Does this make sense? Is the model using the right features in the right way?"*
                - **If YES:** The model has likely learned real, scientifically valid relationships. You can now trust its predictions.
                - **If NO:** The model has learned a spurious correlation. XAI has just saved you from deploying a flawed model. Use the insight to improve your feature engineering and retrain.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: From Game Theory to AI
            The theoretical foundation of SHAP comes from a surprising place: **cooperative game theory**. In 1951, the brilliant mathematician and economist **Lloyd Shapley** developed a concept to solve the "fair payout" problem.
            
            Imagine a team of players collaborates to win a prize. How do you divide the winnings fairly based on each player's individual contribution? **Shapley values** provided a mathematically rigorous and unique solution.
            
            Fast forward to 2017. Scott Lundberg and Su-In Lee at the University of Washington had a genius insight. They realized that a machine learning model's prediction could be seen as a "game" and the model's features could be seen as the "players." They adapted Shapley's game theory concepts to create **SHAP (SHapley Additive exPlanations)**, a method to fairly distribute the "payout" (the prediction) among the features. This clever fusion of game theory and machine learning provided the first unified and theoretically sound framework for explaining the output of any machine learning model, a breakthrough that is driving the adoption of AI in high-stakes fields.
            """)

        with tabs[3]:
            st.markdown("""
            #### SME Analysis: From Raw Data to Actionable Intelligence

            As a Subject Matter Expert (SME) in process validation and tech transfer, this tool isn't just a data science curiosity; it's a powerful diagnostic and risk-management engine. Here’s how we would use this in a real-world GxP environment.

            ---

            ##### How is this data gathered and what are the parameters?

            The data used by this model is a simplified version of what we collect during **late-stage development, process characterization, and tech transfer validation runs**.

            -   **Data Gathering:** Every time an assay run is performed, we log key parameters in a Laboratory Information Management System (LIMS) or an Electronic Lab Notebook (ELN). This includes automated readings from the instrument and manual entries by the technician. The final "Pass/Fail" result of the run is the target we are trying to predict.

            -   **Parameters Considered:**
                *   **`Operator Experience`**: This is critical, especially during tech transfer. We track this from training records. A junior analyst at a receiving site might follow the SOP perfectly, but their inexperience can be a hidden source of variability.
                *   **`Reagent Age`**: We track this via lot numbers and expiration dates. Even within its expiry, a reagent that is 90 days old might perform differently than one that is 5 days old. This model helps quantify that risk.
                *   **`Calibrator Slope`**: This is a direct output from the instrument's software. It's a key health indicator for the assay. A decreasing slope over time often signals a systemic issue.
                *   **`QC Level 1 Value`**: This is the result for a known Quality Control sample. We monitor this using standard SPC charts (like Westgard Rules), but including it here allows the model to learn complex interactions, like how a slight drop in QC value is more dangerous when reagent age is also high.
                *   **`Instrument ID`**: In a lab with multiple instruments, we always log which machine was used. They are never perfectly identical, and this model can detect if one instrument is contributing disproportionately to run failures.

            ---

            ##### How do we interpret the plots and gain insights?

            The true power here is moving from "what happened" to "why it happened."

            -   **Global Plot (The Big Picture):** The summary plot is our first validation checkpoint for the model itself. As an SME, if I saw that `Instrument ID` was the most important factor and `Calibrator Slope` was irrelevant, I would immediately reject the model. It would mean the model learned a spurious correlation (e.g., our failing instrument is also where we train new operators) rather than the true science. The fact that `Operator Experience` and `Calibrator Slope` are top drivers gives me confidence that the AI's "thinking" aligns with scientific reality.

            -   **Local Plot (The Smoking Gun):** This is our **automated root cause investigation tool**.
                *   When I select the **"Highest Predicted Failure Risk"** case, the force plot instantly shows me the "root cause narrative." For the selected run, the story is clear: an inexperienced operator combined with a low calibrator slope created a high-risk situation. The fact that the reagent was fresh (a blue, risk-lowering factor) wasn't enough to save it.
                *   When I select the **"Lowest Predicted Failure Risk"** case, I see the "golden run" profile: an experienced operator, a perfect calibrator slope, and fresh reagents. This confirms what an ideal run looks like.

            ---

            ##### How would we implement this?

            Implementation is a phased process moving from monitoring to proactive control.

            1.  **Phase 1 (Silent Monitoring):** The model runs in the background. It predicts the failure risk for every run, and we use SHAP to analyze the reasons for high-risk predictions. This data is reviewed during weekly process monitoring meetings. It helps us spot trends—"Are we seeing more failures driven by `Reagent Age` lately?"—and guides our investigations.

            2.  **Phase 2 (Advisory Mode):** The system is integrated with the LIMS. When an operator starts a run, the model calculates a risk score based on the chosen reagents and their own logged experience. If the risk is high, it could generate an advisory: **"Warning: Reagent Lot XYZ is 85 days old. This significantly increases the risk of run failure. Consider using a newer lot."**

            3.  **Phase 3 (Proactive Control / Real-Time Release):** This is the ultimate goal of PAT. Once the model is fully validated and trusted, its predictions can become part of the official batch record. A run with a very low predicted risk and a favorable SHAP explanation could be eligible for **Real-Time Release Testing (RTRT)**, skipping certain redundant final QC tests. This dramatically accelerates production timelines and reduces costs, all while increasing quality assurance.
            """)
            
def render_advanced_ai_concepts():
    """Renders the module for advanced AI concepts with an improved, guided layout."""
    st.markdown("""
    #### Purpose & Application: A Glimpse into the AI Frontier
    **Purpose:** To provide a high-level, conceptual overview of cutting-edge AI architectures that represent the future of process analytics. While coding them is beyond the scope of this toolkit, understanding their capabilities is crucial for shaping future strategy and envisioning what's possible.
    """)

    st.markdown("---")
    st.subheader("Interactive Demo: Solving V&V Challenges with AI")
    st.info("""
    Follow the two steps below to explore how advanced AI can tackle some of the toughest problems in assay validation and tech transfer. The diagrams and explanations will update based on your selections.
    """)

    # --- Create a two-column layout for the controls and their description ---
    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        # --- Gadget 1: The Problem ---
        st.markdown("##### Step 1: Choose a Challenge")
        challenge_key = st.selectbox(
            "Select a V&V Tech Transfer Challenge:",
            options=[
                "Silent Process Drift",
                "System-Wide Failures",
                "Optimizing New Processes",
                "Lack of Failure Data"
            ],
            index=0,
            help="Choose a common, difficult problem faced during validation and tech transfer."
        )

        # --- Gadget 2: The Proposed Solution ---
        st.markdown("##### Step 2: Select an AI Solution")
        concept_key = st.radio(
            "Select an Advanced AI Concept:", 
            ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)", "Generative AI"],
            label_visibility="collapsed"
        )

    with col2:
        # Provide a dynamic description of the selected challenge
        st.markdown(f"**Challenge Description: `{challenge_key}`**")
        if challenge_key == "Silent Process Drift":
            st.write("A process slowly and subtly deviates from its target over a long period. Individual SPC charts fail to alarm because no single data point is extreme, but the cumulative effect leads to a batch failure.")
        elif challenge_key == "System-Wide Failures":
            st.write("A failure in one part of the process (e.g., a bad raw material lot) causes a cascade of issues in multiple, seemingly unrelated downstream batches, making root cause analysis extremely difficult.")
        elif challenge_key == "Optimizing New Processes":
            st.write("When developing a new assay or manufacturing process, the optimal operating conditions are unknown. Traditional one-factor-at-a-time experiments are slow and often miss the true optimum.")
        elif challenge_key == "Lack of Failure Data":
            st.write("Building a robust predictive model requires examples of both success and failure. For a high-quality process, failure events are rare, leading to a severe data imbalance that cripples standard machine learning models.")
            
    st.markdown("---")

    # --- Main Display Area ---
    # Generate the plot based on the selected AI concept
    fig = plot_advanced_ai_concepts(concept_key)
    
    st.subheader(f"AI Solution for: {challenge_key}")
    
    plot_col, analysis_col = st.columns([0.6, 0.4])

    with plot_col:
        st.plotly_chart(fig, use_container_width=True)

    with analysis_col:
        tabs = st.tabs(["💡 Application Insight", "✅ The Golden Rule", "📖 Origin Story"])
        
        # Dynamically populate tabs based on BOTH selections
        if concept_key == "Transformers":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Self-Attention")
                st.markdown("**The AI Historian for Your Batch Record**")
                if challenge_key == "Silent Process Drift":
                    st.success("**Application:** A Transformer can read the entire sequence of a batch's process parameters. It can learn long-range dependencies that a simple EWMA chart would miss, such as how a subtle change in cell growth during Phase 1 influences the final product purity 20 days later. It excels at finding these 'needle-in-a-haystack' temporal patterns.")
                elif challenge_key == "System-Wide Failures":
                    st.warning("**Application:** Less ideal for this challenge. Transformers focus on sequential relationships within a single entity (like one batch), not necessarily the connections between different entities across a system.")
                elif challenge_key == "Optimizing New Processes":
                    st.success("**Application:** By analyzing the full 'narrative' of dozens of successful development batches, a Transformer can identify the sequence of events and parameter profiles that are most predictive of a high-quality outcome, guiding the design of the optimal process.")
                elif challenge_key == "Lack of Failure Data":
                    st.success("**Application:** Transformers can be trained using self-supervised methods on vast amounts of *good* batch data to learn what 'normal' looks like. It can then flag a new batch as anomalous if its process narrative deviates significantly from the learned normal patterns.")
            with tabs[1]:
                st.success("🟢 **THE GOLDEN RULE:** Tokenize Your Process Narrative. Convert continuous data into a discrete sequence of meaningful events (e.g., `[Feed_Event, pH_Excursion, Operator_Shift]`).")
            with tabs[2]:
                st.markdown("**Origin:** Revolutionized AI with the 2017 Google Brain paper, **\"Attention Is All You Need,\"** forming the basis for models like GPT.")

        elif concept_key == "Graph Neural Networks (GNNs)":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Message Passing")
                st.markdown("**The System-Wide Process Cartographer**")
                if challenge_key == "Silent Process Drift":
                    st.warning("**Application:** Not the primary tool. GNNs are focused on network relationships, while time-series models like Transformers are better suited for detecting drift within a single process over time.")
                elif challenge_key == "System-Wide Failures":
                    st.success("**Application:** This is the GNN's superpower. It can model your entire facility as a graph (raw materials -> equipment -> batches). If a contaminant is found, the GNN can trace the most likely propagation path backward through the network to identify the root cause.")
                elif challenge_key == "Optimizing New Processes":
                    st.warning("**Application:** Less direct. GNNs are better for understanding existing, interconnected systems rather than designing a single new process from scratch.")
                elif challenge_key == "Lack of Failure Data":
                    st.success("**Application:** By modeling the connections between raw material lots and batches, a GNN can perform 'guilt-by-association.' If a few batches linked to a specific lot fail, the GNN can flag *all other batches* that used the same lot as high-risk, even before they show signs of failure.")
            with tabs[1]:
                st.success("🟢 **THE GOLDEN RULE:** Your Graph IS Your Model. The most important work is defining the nodes (e.g., equipment, lots) and edges (e.g., 'used-in' relationships).")
            with tabs[2]:
                st.markdown("**Origin:** Exploded in popularity around 2018, generalizing deep learning from grids (images) and sequences (text) to the more flexible structure of graphs.")

        elif concept_key == "Reinforcement Learning (RL)":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Reward Maximization")
                st.markdown("**The AI Process Optimization Pilot**")
                if challenge_key == "Silent Process Drift":
                    st.success("**Application:** An RL agent could be trained to actively *control* the process. If it detects the beginning of a drift, it can learn a policy (e.g., adjusting feed rates) to counteract the drift in real-time and keep the process centered on its target.")
                elif challenge_key == "System-Wide Failures":
                    st.warning("**Application:** Less ideal for root cause analysis of past failures. RL is forward-looking and focused on control, not backward-looking diagnosis.")
                elif challenge_key == "Optimizing New Processes":
                    st.success("**Application:** This is a prime use case. Given a digital twin of a new process, an RL agent can run millions of virtual experiments to discover a novel, non-obvious control strategy that maximizes yield or robustness.")
                elif challenge_key == "Lack of Failure Data":
                    st.warning("**Application:** RL requires a model of the world (a digital twin) to learn from. It doesn't directly solve the problem of having no historical failure data to build that model.")
            with tabs[1]:
                st.success("🟢 **THE GOLDEN RULE:** The Digital Twin is the Dojo. An RL agent must be trained in a high-fidelity simulation to learn optimal control strategies with zero real-world risk.")
            with tabs[2]:
                st.markdown("**Origin:** Deep roots in control theory, supercharged by DeepMind in the mid-2010s with AlphaGo.")
        
        elif concept_key == "Generative AI":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Distribution Learning")
                st.markdown("**The Synthetic Data Factory**")
                if challenge_key == "Silent Process Drift":
                    st.success("**Application:** We can generate thousands of synthetic batch records that simulate various types of process drift. This augmented dataset can then be used to train a more robust and sensitive drift detection model (like a Transformer).")
                elif challenge_key == "System-Wide Failures":
                    st.success("**Application:** We can generate synthetic data representing rare contamination events or equipment malfunctions. This data can then be used to train a GNN to become much better at recognizing the early warning signs of these system-wide failures.")
                elif challenge_key == "Optimizing New Processes":
                    st.warning("**Application:** Generative AI learns from existing data. For a truly novel process with no data, it has nothing to learn from.")
                elif challenge_key == "Lack of Failure Data":
                    st.success("**Application:** This is the quintessential use case. If you only have two real examples of a critical failure, you can train a Generative AI model on them. The model can then produce hundreds of new, statistically realistic synthetic failure examples, providing enough data to train a robust predictive QC model.")
            with tabs[1]:
                st.success("🟢 **THE GOLDEN RULE:** Validate the Forgeries. The generated data is only useful if it is proven to be statistically indistinguishable from real data.")
            with tabs[2]:
                st.markdown("**Origin:** Catalyzed by **Generative Adversarial Networks (GANs)** in 2014, with modern **Diffusion Models** (e.g., DALL-E 2) being state-of-the-art.")



# ==============================================================================
# MAIN APP LOGIC AND LAYOUT
# ==============================================================================

# --- Initialize Session State ---
# The default view will now be 'Introduction'.
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Introduction'

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    
    # FIX: A single, dedicated button for the introduction/framework page.
    if st.sidebar.button("🚀 Project Framework", use_container_width=True):
        st.session_state.current_view = 'Introduction'
        # We can add a rerun for immediate feedback if needed, but often not necessary.
        st.rerun()

    st.divider()

    # The dictionary now ONLY contains the tools, grouped by Act.
    all_tools = {
        "ACT I: FOUNDATION & CHARACTERIZATION": ["Confidence Interval Concept", "Core Validation Parameters", "Gage R&R / VCA", "LOD & LOQ", "Linearity & Range", "Non-Linear Regression (4PL/5PL)", "ROC Curve Analysis", "Equivalence Testing (TOST)", "Assay Robustness (DOE)", "Causal Inference"],
        "ACT II: TRANSFER & STABILITY": ["Process Stability (SPC)", "Process Capability (Cpk)", "Tolerance Intervals", "Method Comparison", "Pass/Fail Analysis", "Bayesian Inference"],
        "ACT III: LIFECYCLE & PREDICTIVE MGMT": ["Run Validation (Westgard)", "Multivariate SPC", "Small Shift Detection", "Time Series Analysis", "Stability Analysis (Shelf-Life)", "Reliability / Survival Analysis", "Multivariate Analysis (MVA)", "Clustering (Unsupervised)", "Predictive QC (Classification)", "Anomaly Detection", "Explainable AI (XAI)", "Advanced AI Concepts"]
    }

    # The loop for creating tool buttons remains the same.
    for act_title, act_tools in all_tools.items():
        st.subheader(act_title)
        for tool in act_tools:
            if st.button(tool, key=tool, use_container_width=True):
                st.session_state.current_view = tool
                st.rerun()

# --- Main Content Area Dispatcher ---
view = st.session_state.current_view

# FIX: The logic is now much simpler.
if view == 'Introduction':
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
        "Assay Robustness (DOE)": render_assay_robustness_doe,
        "Causal Inference": render_causal_inference,
        "Process Stability (SPC)": render_spc_charts,
        "Process Capability (Cpk)": render_capability,
        "Tolerance Intervals": render_tolerance_intervals,
        "Method Comparison": render_method_comparison,
        "Pass/Fail Analysis": render_pass_fail,
        "Bayesian Inference": render_bayesian,
        "Run Validation (Westgard)": render_multi_rule,
        "Multivariate SPC": render_multivariate_spc,
        "Small Shift Detection": render_ewma_cusum, # FIX: Added the new function here
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
        # Failsafe if state gets corrupted somehow
        st.error("Error: Could not find the selected tool to render.")
        st.session_state.current_view = 'Introduction'
        st.rerun()
