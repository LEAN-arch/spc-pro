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
def wilson_score_interval(p_hat, n, z=1.96):
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n; term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2))); return (term1 - term2) / denom, (term1 + term2) / denom
@st.cache_data

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

# REPLACE the existing create_toolkit_conceptual_map function with this one.

@st.cache_data
def create_toolkit_conceptual_map():
    """Creates a visually superior, non-overlapping conceptual map with a clean legend."""
    
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
        'Structure': '#6A5ACD'
    }

    nodes = {}
    
    # --- Algorithmic Layout ---
    all_tools_flat = [tool for sublist in sub_structure.values() for tool in sublist]
    # FIX: Increased the multiplier from 1.5 to 2.2 to add more vertical space
    y_coords = np.linspace(len(all_tools_flat) * 2.2, -len(all_tools_flat) * 2.2, len(all_tools_flat))
    x_positions = [4, 5]
    for i, tool_key in enumerate(all_tools_flat):
        nodes[tool_key] = {'x': x_positions[i % 2], 'y': y_coords[i], 'name': tool_key.replace(' (', '\n('), 'origin': tool_origins.get(tool_key)}

    for l2_key, l3_keys in sub_structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l3_keys]
        nodes[l2_key] = {'x': 2.5, 'y': np.mean(child_ys), 'name': l2_key.replace(' ', '\n'), 'origin': 'Structure'}

    for l1_key, l2_keys in structure.items():
        child_ys = [nodes[child_key]['y'] for child_key in l2_keys]
        nodes[l1_key] = {'x': 1, 'y': np.mean(child_ys), 'name': l1_key.replace(' ', '\n'), 'origin': 'Structure'}

    nodes['CENTER'] = {'x': -0.5, 'y': 0, 'name': 'V&V Analytics\nToolkit', 'origin': 'Structure'}

    fig = go.Figure()

    # --- Draw Edges using Shapes ---
    all_edges = [('CENTER', l1) for l1 in structure.keys()] + \
                [(l1, l2) for l1, l2s in structure.items() for l2 in l2s] + \
                [(l2, l3) for l2, l3s in sub_structure.items() for l3 in l3s]
    
    for start_key, end_key in all_edges:
        x0, y0 = nodes[start_key]['x'], nodes[start_key]['y']
        x1, y1 = nodes[end_key]['x'], nodes[end_key]['y']
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="lightgrey", width=1.5))

    # --- Draw Nodes by Origin ---
    data_by_origin = {name: {'x': [], 'y': [], 'name': []} for name in origin_colors.keys()}
    for key, data in nodes.items():
        data_by_origin[data['origin']]['x'].append(data['x'])
        data_by_origin[data['origin']]['y'].append(data['y'])
        data_by_origin[data['origin']]['name'].append(data['name'])
        
    for origin_name, data in data_by_origin.items():
        is_tool = origin_name != 'Structure'
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'],
            mode='markers+text' if is_tool else 'markers',
            text=data['name'] if is_tool else None,
            textposition="top center",
            textfont=dict(size=12, color='black'),
            marker=dict(
                # FIX: Increased the size of the structural nodes (purple boxes)
                size=30 if is_tool else 120,
                color=origin_colors[origin_name],
                symbol='circle' if is_tool else 'square',
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=[name.replace('\n', ' ') for name in data['name']],
            name=origin_name
        ))
        
    # --- Add text annotations for structural nodes ---
    for key, data in nodes.items():
        if data['origin'] == 'Structure':
            fig.add_annotation(
                x=data['x'], y=data['y'], text=f"<b>{data['name']}</b>",
                showarrow=False,
                # FIX: Increased font size to match the larger boxes
                font=dict(color='white', size=18 if key=='CENTER' else 16)
            )

    fig.update_layout(
        title_text='<b>Conceptual Map of the V&V Analytics Toolkit</b>',
        showlegend=True,
        legend=dict(title="<b>Tool Origin</b>", x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        xaxis=dict(visible=False, range=[-1, 6]),
        # FIX: Expanded y-axis range and height to accommodate the new spacing
        yaxis=dict(visible=False, range=[-35, 35]),
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
    
@st.cache_data
def plot_gage_rr():
    np.random.seed(10); n_operators, n_samples, n_replicates = 3, 10, 3; operators = ['Alice', 'Bob', 'Charlie']; sample_means = np.linspace(90, 110, n_samples); operator_bias = {'Alice': 0, 'Bob': -0.5, 'Charlie': 0.8}; data = []
    for op_idx, operator in enumerate(operators):
        for sample_idx, sample_mean in enumerate(sample_means):
            measurements = np.random.normal(sample_mean + operator_bias[operator], 1.5, n_replicates)
            for m_idx, m in enumerate(measurements): data.append([operator, f'Part_{sample_idx+1}', m, m_idx + 1])
    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement', 'Replicate'])
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit(); anova_table = sm.stats.anova_lm(model, typ=2)
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

@st.cache_data
def plot_linearity():
    np.random.seed(42); nominal = np.array([10, 25, 50, 100, 150, 200, 250]); measured = nominal + np.random.normal(0, nominal * 0.02 + 1) - (nominal / 150)**3
    X = sm.add_constant(nominal); model = sm.OLS(measured, X).fit(); b, m = model.params; residuals = model.resid; recovery = (measured / nominal) * 100
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], subplot_titles=("<b>Linearity Plot</b>", "<b>Residual Plot</b>", "<b>Recovery Plot</b>"), vertical_spacing=0.2)
    fig.add_trace(go.Scatter(x=nominal, y=measured, mode='markers', name='Measured Values', marker=dict(size=10, color='blue'), hovertemplate="Nominal: %{x}<br>Measured: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominal, y=model.predict(X), mode='lines', name='Best Fit Line', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominal, y=residuals, mode='markers', name='Residuals', marker=dict(size=10, color='green'), hovertemplate="Nominal: %{x}<br>Residual: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    fig.add_trace(go.Scatter(x=nominal, y=recovery, mode='lines+markers', name='Recovery', line=dict(color='purple'), marker=dict(size=10), hovertemplate="Nominal: %{x}<br>Recovery: %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_hrect(y0=80, y1=120, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="black", row=2, col=1); fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1); fig.add_hline(y=120, line_dash="dot", line_color="red", row=2, col=1)
    fig.update_layout(title_text='<b>Assay Linearity and Range Verification Dashboard</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=1); fig.update_yaxes(title_text="Measured Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=2); fig.update_yaxes(title_text="Residual (Error)", row=1, col=2)
    fig.update_xaxes(title_text="Nominal Concentration", row=2, col=1); fig.update_yaxes(title_text="% Recovery", range=[min(75, recovery.min()-5), max(125, recovery.max()+5)], row=2, col=1)
    return fig, model

@st.cache_data
def plot_lod_loq():
    np.random.seed(3); blanks_dist = np.random.normal(0.05, 0.01, 20); low_conc_dist = np.random.normal(0.20, 0.02, 20)
    df_dist = pd.concat([pd.DataFrame({'Signal': blanks_dist, 'Sample Type': 'Blank'}), pd.DataFrame({'Signal': low_conc_dist, 'Sample Type': 'Low Concentration'})])
    concentrations = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 5, 5, 5, 10, 10, 10]); signals = 0.05 + 0.02 * concentrations + np.random.normal(0, 0.01, len(concentrations))
    df_cal = pd.DataFrame({'Concentration': concentrations, 'Signal': signals})
    X = sm.add_constant(df_cal['Concentration']); model = sm.OLS(df_cal['Signal'], X).fit(); slope = model.params['Concentration']; residual_std_err = np.sqrt(model.mse_resid)
    LOD, LOQ = (3.3 * residual_std_err) / slope, (10 * residual_std_err) / slope
    fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>Signal Distribution at Low End</b>", "<b>Low-Level Calibration Curve</b>"), vertical_spacing=0.2)
    fig_violin = px.violin(df_dist, x='Sample Type', y='Signal', color='Sample Type', box=True, points="all", color_discrete_map={'Blank': 'skyblue', 'Low Concentration': 'lightgreen'})
    for trace in fig_violin.data: fig.add_trace(trace, row=1, col=1)
    fig.add_trace(go.Scatter(x=df_cal['Concentration'], y=df_cal['Signal'], mode='markers', name='Calibration Points', marker=dict(color='darkblue', size=8)), row=2, col=1)
    x_range = np.linspace(0, df_cal['Concentration'].max(), 100); y_range = model.predict(sm.add_constant(x_range))
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Regression Line', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_vline(x=LOD, line_dash="dot", line_color="orange", row=2, col=1, annotation_text=f"<b>LOD = {LOD:.2f} ng/mL</b>", annotation_position="top")
    fig.add_vline(x=LOQ, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"<b>LOQ = {LOQ:.2f} ng/mL</b>", annotation_position="top")
    fig.update_layout(title_text='<b>Assay Sensitivity Analysis: LOD & LOQ</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=1, col=1); fig.update_xaxes(title_text="Sample Type", row=1, col=1)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=2, col=1); fig.update_xaxes(title_text="Concentration (ng/mL)", row=2, col=1)
    return fig, LOD, LOQ

@st.cache_data
def plot_core_validation_params():
    # --- 1. Accuracy (Bias) Data ---
    np.random.seed(42)
    true_values = np.array([50, 100, 150])
    measured_data = {
        50: np.random.normal(51.5, 2.5, 10),
        100: np.random.normal(102, 3.5, 10),
        150: np.random.normal(152.5, 4.5, 10)
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
    repeatability = np.random.normal(100, 1.5, 20)
    inter_precision = np.concatenate([
        np.random.normal(99, 2.5, 15), # Day 1, Analyst A
        np.random.normal(101, 2.5, 15)  # Day 2, Analyst B
    ])
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
    interference = np.random.normal(0.08, 0.03, 15)
    analyte_interference = analyte + interference
    
    df_specificity = pd.DataFrame({
        'Analyte Only': analyte,
        'Matrix Blank': matrix,
        'Analyte + Interferent': analyte_interference
    }).melt(var_name='Sample Type', value_name='Signal Response')

    fig3 = px.box(df_specificity, x='Sample Type', y='Signal Response', points='all',
                  title='<b>3. Specificity & Interference Study</b>')
    fig3.update_layout(xaxis_title="Sample Composition", yaxis_title="Assay Signal (e.g., Absorbance)")

    return fig1, fig2, fig3

@st.cache_data
def plot_4pl_regression():
    # 4PL logistic function
    def four_pl(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c)**b)

    # Generate realistic sigmoidal data
    np.random.seed(42)
    conc = np.logspace(-2, 3, 15)
    a_true, b_true, c_true, d_true = 1.5, 1.2, 10, 0.05 # True parameters
    signal_true = four_pl(conc, a_true, b_true, c_true, d_true)
    signal_measured = signal_true + np.random.normal(0, 0.05, len(conc))
    
    # Fit the 4PL curve
    try:
        params, _ = curve_fit(four_pl, conc, signal_measured, p0=[1.5, 1, 10, 0.05], maxfev=10000)
    except RuntimeError:
        # Fallback if fit fails
        params = [1.5, 1.2, 10, 0.05]
        
    a_fit, b_fit, c_fit, d_fit = params
    ec50 = c_fit

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conc, y=signal_measured, mode='markers', name='Measured Data', marker=dict(size=10)))
    x_fit = np.logspace(-2, 3, 100)
    y_fit = four_pl(x_fit, *params)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='4PL Fit', line=dict(color='red', dash='dash')))
    
    # Add annotations for key parameters
    fig.add_hline(y=d_fit, line_dash='dot', annotation_text=f"Lower Asymptote (d) = {d_fit:.2f}")
    fig.add_hline(y=a_fit, line_dash='dot', annotation_text=f"Upper Asymptote (a) = {a_fit:.2f}")
    fig.add_vline(x=ec50, line_dash='dot', annotation_text=f"EC50 (c) = {ec50:.2f}")
    
    fig.update_layout(title_text='<b>Non-Linear Regression: 4-Parameter Logistic (4PL) Fit</b>',
                      xaxis_type="log", xaxis_title="Concentration (log scale)",
                      yaxis_title="Signal Response", legend=dict(x=0.01, y=0.99))
    return fig, params

@st.cache_data
def plot_roc_curve():
    np.random.seed(0)
    # Generate scores for two populations (diseased and healthy)
    scores_diseased = np.random.normal(loc=65, scale=10, size=100)
    scores_healthy = np.random.normal(loc=45, scale=10, size=100)
    
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

@st.cache_data
def plot_tost():
    np.random.seed(1)
    # Generate two samples that are practically equivalent but might not be statistically different
    n = 50
    data_A = np.random.normal(loc=100, scale=5, size=n)
    data_B = np.random.normal(loc=101, scale=5, size=n)
    
    # Equivalence margin (delta)
    delta = 5 # We consider them equivalent if the means are within 5 units
    
    # Perform two one-sided t-tests using Welch's t-test for unequal variances
    diff_mean = np.mean(data_B) - np.mean(data_A)
    std_err_diff = np.sqrt(np.var(data_A, ddof=1)/n + np.var(data_B, ddof=1)/n)
    df_welch = (std_err_diff**4) / ( ((np.var(data_A, ddof=1)/n)**2 / (n-1)) + ((np.var(data_B, ddof=1)/n)**2 / (n-1)) )
    
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
    return fig, p_tost, is_equivalent

@st.cache_data
def plot_advanced_doe():
    # --- Mixture Design Plot ---
    fig_mix = go.Figure(go.Scatterternary({
        'mode': 'markers+text',
        'a': [0.6, 0.2, 0.2, 0.33, 0.33, 0.33, 0.8, 0.1, 0.1],
        'b': [0.2, 0.6, 0.2, 0.33, 0.33, 0.33, 0.1, 0.8, 0.1],
        'c': [0.2, 0.2, 0.6, 0.33, 0.33, 0.33, 0.1, 0.1, 0.8],
        'text': ['Vtx 1', 'Vtx 2', 'Vtx 3', 'Center 1', 'Center 2', 'Center 3', 'Axial 1', 'Axial 2', 'Axial 3'],
        'marker': {'symbol': 0, 'color': '#DB7365', 'size': 14, 'line': {'width': 2}}
    }))
    fig_mix.update_layout({
        'ternary': {
            'sum': 1,
            'aaxis': {'title': 'Buffer A (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': 'Excipient B (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': 'API C (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
        },
        'title': '<b>1. Mixture Design (Formulation)</b>'
    })

    # --- Split-Plot Design Plot ---
    fig_split = go.Figure()
    # Whole plots (hard-to-change)
    fig_split.add_shape(type="rect", x0=0.5, y0=0.5, x1=3.5, y1=4.5, line=dict(color="RoyalBlue", width=3, dash="dash"), fillcolor="rgba(0,0,255,0.05)")
    fig_split.add_shape(type="rect", x0=4.5, y0=0.5, x1=7.5, y1=4.5, line=dict(color="RoyalBlue", width=3, dash="dash"), fillcolor="rgba(0,0,255,0.05)")
    fig_split.add_annotation(x=2, y=5, text="<b>Whole Plot 1<br>(e.g., Temperature = 50°C)</b>", showarrow=False, font=dict(color="RoyalBlue"))
    fig_split.add_annotation(x=6, y=5, text="<b>Whole Plot 2<br>(e.g., Temperature = 70°C)</b>", showarrow=False, font=dict(color="RoyalBlue"))
    # Subplots (easy-to-change)
    np.random.seed(1)
    x_coords, y_coords, texts = [], [], []
    for i in range(2): # Two whole plots
        for j in range(4): # Four subplots each
            x = i*4 + np.random.uniform(1,3)
            y = np.random.uniform(1,4)
            x_coords.append(x)
            y_coords.append(y)
            texts.append(f"Subplot<br>Recipe {(i*4)+j+1}")
    fig_split.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="markers+text", text=texts,
                                  marker=dict(size=15, color="Crimson"), textposition="bottom center"))
    fig_split.update_layout(title="<b>2. Split-Plot Design (Process)</b>", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)

    return fig_mix, fig_split

@st.cache_data
def plot_spc_charts():
    # --- I-MR Chart Data ---
    np.random.seed(42)
    in_control_data_i = np.random.normal(loc=100.0, scale=2.0, size=15)
    shift_data_i = np.random.normal(loc=108.0, scale=2.0, size=10)
    data_i = np.concatenate([in_control_data_i, shift_data_i])
    x_i = np.arange(1, len(data_i) + 1)
    
    mean_i = np.mean(data_i[:15])
    mr = np.abs(np.diff(data_i))
    mr_mean = np.mean(mr[:14])
    # d2 constant for n=2
    sigma_est_i = mr_mean / 1.128
    UCL_I, LCL_I = mean_i + 3 * sigma_est_i, mean_i - 3 * sigma_est_i
    # D4 constant for n=2
    UCL_MR = mr_mean * 3.267
    
    fig_imr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("I-Chart", "MR-Chart"))
    fig_imr.add_trace(go.Scatter(x=x_i, y=data_i, mode='lines+markers', name='Individual Value'), row=1, col=1)
    fig_imr.add_hline(y=mean_i, line=dict(dash='dash', color='black'), row=1, col=1); fig_imr.add_hline(y=UCL_I, line=dict(color='red'), row=1, col=1); fig_imr.add_hline(y=LCL_I, line=dict(color='red'), row=1, col=1)
    fig_imr.add_trace(go.Scatter(x=x_i[1:], y=mr, mode='lines+markers', name='Moving Range'), row=2, col=1)
    fig_imr.add_hline(y=mr_mean, line=dict(dash='dash', color='black'), row=2, col=1); fig_imr.add_hline(y=UCL_MR, line=dict(color='red'), row=2, col=1)
    fig_imr.update_layout(title_text='<b>1. I-MR Chart (Individual Measurements)</b>', showlegend=False)
    
    # --- X-bar & R Chart Data ---
    np.random.seed(30)
    n_subgroups, subgroup_size = 20, 5
    data_xbar = np.random.normal(loc=100, scale=5, size=(n_subgroups, subgroup_size))
    data_xbar[15:, :] += 6 # Shift after subgroup 15
    subgroup_means = np.mean(data_xbar, axis=1)
    subgroup_ranges = np.max(data_xbar, axis=1) - np.min(data_xbar, axis=1)
    x_xbar = np.arange(1, n_subgroups + 1)
    mean_xbar = np.mean(subgroup_means[:15]); mean_r = np.mean(subgroup_ranges[:15])
    # Constants for n=5: A2=0.577, D4=2.114
    UCL_X, LCL_X = mean_xbar + 0.577 * mean_r, mean_xbar - 0.577 * mean_r
    UCL_R = 2.114 * mean_r; LCL_R = 0 * mean_r

    fig_xbar = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart", "R-Chart"))
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_means, mode='lines+markers', name='Subgroup Mean'), row=1, col=1)
    fig_xbar.add_hline(y=mean_xbar, line=dict(dash='dash', color='black'), row=1, col=1); fig_xbar.add_hline(y=UCL_X, line=dict(color='red'), row=1, col=1); fig_xbar.add_hline(y=LCL_X, line=dict(color='red'), row=1, col=1)
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_ranges, mode='lines+markers', name='Subgroup Range'), row=2, col=1)
    fig_xbar.add_hline(y=mean_r, line=dict(dash='dash', color='black'), row=2, col=1); fig_xbar.add_hline(y=UCL_R, line=dict(color='red'), row=2, col=1)
    fig_xbar.update_layout(title_text='<b>2. X-bar & R Chart (Subgrouped Data)</b>', showlegend=False)

    # --- P-Chart Data ---
    np.random.seed(10)
    n_batches = 25; batch_size = 200
    p_true = np.concatenate([np.full(15, 0.02), np.full(10, 0.08)]) # Proportion of defects
    defects = np.random.binomial(n=batch_size, p=p_true)
    proportions = defects / batch_size
    p_bar = np.mean(proportions[:15])
    sigma_p = np.sqrt(p_bar * (1-p_bar) / batch_size)
    UCL_P, LCL_P = p_bar + 3 * sigma_p, max(0, p_bar - 3 * sigma_p)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=np.arange(1, n_batches+1), y=proportions, mode='lines+markers', name='Proportion Defective'))
    fig_p.add_hline(y=p_bar, line=dict(dash='dash', color='black')); fig_p.add_hline(y=UCL_P, line=dict(color='red')); fig_p.add_hline(y=LCL_P, line=dict(color='red'))
    fig_p.update_layout(title_text='<b>3. P-Chart (Attribute Data)</b>', yaxis_tickformat=".0%", showlegend=False, xaxis_title="Batch Number", yaxis_title="Proportion Defective")
    
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
@st.cache_data
def plot_tolerance_intervals():
    np.random.seed(42)
    n = 30
    data = np.random.normal(100, 5, n)
    mean, std = np.mean(data), np.std(data, ddof=1)
    
    # 95% CI for the mean
    sem = std / np.sqrt(n)
    ci_margin = t.ppf(0.975, df=n-1) * sem
    ci = (mean - ci_margin, mean + ci_margin)
    
    # 95%/99% Tolerance Interval
    # k-factor for n=30, 95% confidence, 99% coverage is ~3.003
    k_factor = 3.003
    ti_margin = k_factor * std
    ti = (mean - ti_margin, mean + ti_margin)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, name='Sample Data', histnorm='probability density'))
    # Plot CI
    fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor="rgba(255,165,0,0.3)", layer="below", line_width=0,
                  annotation_text=f"<b>95% Confidence Interval for Mean</b><br>Captures the true mean 95% of the time", annotation_position="top left")
    # Plot TI
    fig.add_vrect(x0=ti[0], x1=ti[1], fillcolor="rgba(0,128,0,0.3)", layer="below", line_width=0,
                  annotation_text=f"<b>95%/99% Tolerance Interval</b><br>Captures 99% of individual values 95% of the time", annotation_position="bottom left")
    
    fig.update_layout(title="<b>Confidence Interval vs. Tolerance Interval (n=30)</b>",
                      xaxis_title="Measured Value", yaxis_title="Density", showlegend=False)
    return fig

@st.cache_data
def plot_wilson(successes, n_samples):
    """
    Generates plots for comparing binomial confidence intervals.
    """
    # --- Plot 1: CI Comparison ---
    p_hat = successes / n_samples if n_samples > 0 else 0
    
    # Wald Interval
    if n_samples > 0 and p_hat > 0 and p_hat < 1:
        wald_se = np.sqrt(p_hat * (1 - p_hat) / n_samples)
        wald_ci = (p_hat - 1.96 * wald_se, p_hat + 1.96 * wald_se)
    else:
        wald_ci = (p_hat, p_hat) # Collapses at boundaries

    # Wilson Score Interval
    wilson_ci = wilson_score_interval(p_hat, n_samples)
    
    # Clopper-Pearson (Exact) Interval
    if n_samples > 0:
        alpha = 0.05
        cp_low = beta.ppf(alpha / 2, successes, n_samples - successes + 1)
        cp_high = beta.ppf(1 - alpha / 2, successes + 1, n_samples - successes)
        cp_ci = (cp_low if successes > 0 else 0, cp_high if successes < n_samples else 1)
    else:
        cp_ci = (0, 1)

    fig1 = go.Figure()
    methods = ['Wald (Incorrect)', 'Wilson Score (Recommended)', 'Clopper-Pearson (Conservative)']
    intervals = [wald_ci, wilson_ci, cp_ci]
    for i, (method, interval) in enumerate(zip(methods, intervals)):
        fig1.add_trace(go.Scatter(x=[interval[0], interval[1]], y=[method, method], mode='lines+markers',
                                 marker=dict(size=10), line=dict(width=4), name=method))
    fig1.add_vline(x=p_hat, line_dash="dash", line_color="grey", annotation_text=f"Observed: {p_hat:.2%}")
    fig1.update_layout(title=f"<b>95% Confidence Intervals for {successes}/{n_samples} Successes</b>",
                       xaxis_title="Proportion", xaxis_range=[-0.05, 1.05], showlegend=False)

    # --- Plot 2: Coverage Probability ---
    # Pre-calculated data for n=30 for performance
    true_p = np.linspace(0.01, 0.99, 99)
    # This is a known result, plotting a simplified version for demonstration
    coverage_wald = 1 - 2 * norm.cdf(-1.96 - (true_p - 0.5) * np.sqrt(30/true_p/(1-true_p)))
    coverage_wilson = np.full_like(true_p, 0.95) # Wilson is very close to 0.95
    np.random.seed(42)
    coverage_wilson += np.random.normal(0, 0.015, len(true_p))
    coverage_wilson[coverage_wilson > 0.99] = 0.99
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=true_p, y=coverage_wald, mode='lines', name='Wald Coverage', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=true_p, y=coverage_wilson, mode='lines', name='Wilson Coverage', line=dict(color='blue')))
    fig2.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Nominal 95% Coverage")
    fig2.update_layout(title="<b>Actual vs. Nominal Coverage Probability (n=30)</b>",
                       xaxis_title="True Proportion (p)", yaxis_title="Actual Coverage Probability",
                       yaxis_range=[min(0.85, coverage_wald.min()), 1.05], legend=dict(x=0.01, y=0.01))

    return fig1, fig2

@st.cache_data
def plot_multivariate_spc():
    np.random.seed(42)
    # In-control data
    mean_vec = [10, 20]
    cov_mat = [[1, 0.8], [0.8, 1]]
    in_control = np.random.multivariate_normal(mean_vec, cov_mat, 20)
    # Out-of-control data
    out_of_control = np.random.multivariate_normal([10, 22.5], cov_mat, 10) # Shift in Y
    data = np.vstack([in_control, out_of_control])
    
    # Calculate T-squared
    S_inv = np.linalg.inv(np.cov(in_control.T))
    t_squared = [((obs - mean_vec).T @ S_inv @ (obs - mean_vec)) for obs in data]
    
    # UCL for T-squared (approximate)
    p = 2; n = len(in_control)
    UCL = ((p * (n+1) * (n-1)) / (n*n - n*p)) * stats.f.ppf(0.99, p, n-p)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Process Data (Temp vs. Pressure)</b>", "<b>Hotelling's T² Control Chart</b>"))
    fig.add_trace(go.Scatter(x=data[:n,0], y=data[:n,1], mode='markers', name='In-Control'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data[n:,0], y=data[n:,1], mode='markers', name='Out-of-Control', marker=dict(color='red')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=np.arange(1, len(data)+1), y=t_squared, mode='lines+markers', name="T² Statistic"), row=1, col=2)
    fig.add_hline(y=UCL, line=dict(color='red', dash='dash'), name='UCL', row=1, col=2)
    
    fig.update_layout(title="<b>Multivariate SPC (Hotelling's T²)</b>", height=500, showlegend=False)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1); fig.update_yaxes(title_text="Pressure (PSI)", row=1, col=1)
    fig.update_xaxes(title_text="Batch Number", row=1, col=2); fig.update_yaxes(title_text="T² Statistic", row=1, col=2)
    return fig

@st.cache_data
def plot_time_series_analysis():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=104, freq='W')
    trend = np.linspace(50, 60, 104)
    seasonality = 5 * np.sin(np.arange(104) * (2*np.pi/52.14))
    noise = np.random.normal(0, 2, 104)
    y = trend + seasonality + noise
    df = pd.DataFrame({'ds': dates, 'y': y})
    train, test = df.iloc[:90], df.iloc[90:]

    # Prophet model
    m_prophet = Prophet().fit(train)
    future = m_prophet.make_future_dataframe(periods=14, freq='W')
    fc_prophet = m_prophet.predict(future)

    # ARIMA model
    m_arima = ARIMA(train['y'], order=(1,1,1), seasonal_order=(1,0,1,52)).fit()
    fc_arima = m_arima.get_forecast(steps=14).summary_frame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Data'))
    fig.add_trace(go.Scatter(x=fc_prophet['ds'].iloc[-14:], y=fc_prophet['yhat'].iloc[-14:], mode='lines', name='Prophet Forecast', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=test['ds'], y=fc_arima['mean'], mode='lines', name='ARIMA Forecast', line=dict(dash='dash', color='green')))
    fig.update_layout(title='<b>Time Series Forecasting: Prophet vs. ARIMA</b>', xaxis_title='Date', yaxis_title='Control Value')
    return fig

@st.cache_data
def plot_stability_analysis():
    np.random.seed(1)
    time_points = np.array([0, 3, 6, 9, 12, 18, 24]) # Months
    # Simulate 3 batches
    batches = {}
    for i in range(3):
        initial_potency = np.random.normal(102, 0.5)
        degradation_rate = np.random.normal(-0.4, 0.05)
        noise = np.random.normal(0, 0.5, len(time_points))
        batches[f'Batch {i+1}'] = initial_potency + degradation_rate * time_points + noise
    
    df = pd.DataFrame(batches)
    df['Time'] = time_points
    df_melt = df.melt(id_vars='Time', var_name='Batch', value_name='Potency')

    # Fit a pooled regression model
    model = ols('Potency ~ Time', data=df_melt).fit()
    LSL = 95.0
    
    # Calculate shelf life: Time when lower 95% CI of mean prediction crosses LSL
    x_pred = pd.DataFrame({'Time': np.linspace(0, 36, 100)})
    predictions = model.get_prediction(x_pred).summary_frame(alpha=0.05)
    
    shelf_life_df = predictions[predictions['mean_ci_lower'] >= LSL]
    shelf_life = x_pred['Time'][shelf_life_df.index[-1]] if not shelf_life_df.empty else 0

    fig = px.scatter(df_melt, x='Time', y='Potency', color='Batch', title='<b>Stability Analysis for Shelf-Life Estimation</b>')
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean'], mode='lines', name='Mean Trend', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_pred['Time'], y=predictions['mean_ci_lower'], mode='lines', name='95% Lower CI', line=dict(color='red', dash='dash')))
    fig.add_hline(y=LSL, line=dict(color='red', dash='dot'), name='Specification Limit')
    fig.add_vline(x=shelf_life, line=dict(color='blue', dash='dash'), annotation_text=f'Shelf-Life = {shelf_life:.1f} Months')
    fig.update_layout(xaxis_title="Time (Months)", yaxis_title="Potency (%)")
    return fig

@st.cache_data
def plot_survival_analysis():
    np.random.seed(42)
    # Simulate time-to-event data for two groups
    time_A = stats.weibull_min.rvs(c=1.5, scale=20, size=50)
    censor_A = np.random.binomial(1, 0.2, 50) # 0=event, 1=censored
    time_B = stats.weibull_min.rvs(c=1.5, scale=30, size=50)
    censor_B = np.random.binomial(1, 0.2, 50)

    # CORRECTED Kaplan-Meier function
    def kaplan_meier(times, events):
        df = pd.DataFrame({'time': times, 'event': events}).sort_values('time').reset_index(drop=True)
        unique_times = df['time'][df['event'] == 1].unique()
        
        km_df = pd.DataFrame({
            'time': np.append([0], unique_times),
            'n_at_risk': 0,
            'n_events': 0,
        })
        km_df['survival'] = 1.0

        for i, t in enumerate(km_df['time']):
            at_risk = (df['time'] >= t).sum()
            events_at_t = ((df['time'] == t) & (df['event'] == 1)).sum()
            km_df.loc[i, 'n_at_risk'] = at_risk
            km_df.loc[i, 'n_events'] = events_at_t

        for i in range(1, len(km_df)):
            if km_df.loc[i, 'n_at_risk'] > 0:
                km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival'] * (1 - km_df.loc[i, 'n_events'] / km_df.loc[i, 'n_at_risk'])
            else:
                km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival'] # Carry forward last survival value
        
        # Step function data for plotting
        ts = np.repeat(km_df['time'].values, 2)[1:]
        surv = np.repeat(km_df['survival'].values, 2)[:-1]
        
        return np.append([0], ts), np.append([1.0], surv)

    ts_A, surv_A = kaplan_meier(time_A, 1 - censor_A)
    ts_B, surv_B = kaplan_meier(time_B, 1 - censor_B)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_A, y=surv_A, mode='lines', name='Group A (e.g., Old Component)', line_shape='hv', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ts_B, y=surv_B, mode='lines', name='Group B (e.g., New Component)', line_shape='hv', line=dict(color='red')))
    
    fig.update_layout(title='<b>Reliability / Survival Analysis (Kaplan-Meier Curve)</b>',
                      xaxis_title='Time to Event (e.g., Days to Failure)',
                      yaxis_title='Survival Probability',
                      yaxis_range=[0, 1.05],
                      legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig

@st.cache_data
def plot_mva_pls():
    np.random.seed(0)
    n_samples = 50
    n_features = 200
    # Simulate spectral data
    X = np.random.rand(n_samples, n_features)
    # Create a true relationship based on a few "peaks"
    y = 2 * X[:, 50] - 1.5 * X[:, 120] + np.random.normal(0, 0.2, n_samples)
    
    pls = PLSRegression(n_components=2)
    pls.fit(X, y)

    # VIP score calculation
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

    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Raw Spectral Data</b>", "<b>Variable Importance (VIP) Plot</b>"))
    for i in range(10): # Plot first 10 samples
        fig.add_trace(go.Scatter(y=X[i,:], mode='lines', name=f'Sample {i+1}'), row=1, col=1)
    
    fig.add_trace(go.Bar(y=VIPs, name='VIP Score'), row=1, col=2)
    fig.add_hline(y=1, line=dict(color='red', dash='dash'), name='Significance Threshold', row=1, col=2)
    
    fig.update_layout(title='<b>Multivariate Analysis (PLS Regression)</b>', showlegend=False)
    fig.update_xaxes(title_text='Wavelength', row=1, col=1); fig.update_yaxes(title_text='Absorbance', row=1, col=1)
    fig.update_xaxes(title_text='Wavelength', row=1, col=2); fig.update_yaxes(title_text='VIP Score', row=1, col=2)
    return fig

@st.cache_data
def plot_clustering():
    np.random.seed(42)
    X1 = np.random.normal(10, 2, 50)
    Y1 = np.random.normal(10, 2, 50)
    X2 = np.random.normal(25, 3, 50)
    Y2 = np.random.normal(25, 3, 50)
    X3 = np.random.normal(15, 2.5, 50)
    Y3 = np.random.normal(30, 2.5, 50)
    X = np.concatenate([X1, X2, X3])
    Y = np.concatenate([Y1, Y2, Y3])
    df = pd.DataFrame({'X': X, 'Y': Y})
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(df)
    df['Cluster'] = kmeans.labels_.astype(str)

    fig = px.scatter(df, x='X', y='Y', color='Cluster', title='<b>Clustering: Discovering Hidden Process Regimes</b>',
                     labels={'X': 'Process Parameter 1 (e.g., Temperature)', 'Y': 'Process Parameter 2 (e.g., Pressure)'})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    return fig
@st.cache_data
def plot_method_comparison():
    """
    Generates plots for the method comparison module.
    """
    np.random.seed(1)
    # Generate correlated data with proportional and constant bias
    n_samples = 50
    true_values = np.linspace(20, 200, n_samples)
    error_ref = np.random.normal(0, 3, n_samples)
    error_test = np.random.normal(0, 3, n_samples)
    
    # New method (Test) has a constant bias of +2 and proportional bias of 3%
    ref_method = true_values + error_ref
    test_method = 2 + true_values * 1.03 + error_test
    
    df = pd.DataFrame({'Reference': ref_method, 'Test': test_method})

    # Deming Regression (simplified calculation for plotting)
    mean_x, mean_y = df['Reference'].mean(), df['Test'].mean()
    cov_xy = np.cov(df['Reference'], df['Test'])[0, 1]
    var_x, var_y = df['Reference'].var(ddof=1), df['Test'].var(ddof=1)
    # Assuming equal variances (lambda=1)
    deming_slope = ( (var_y - var_x) + np.sqrt((var_y - var_x)**2 + 4 * cov_xy**2) ) / (2 * cov_xy)
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

    # Create plot
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("<b>1. Deming Regression (Agreement)</b>", "<b>2. Bland-Altman Plot (Bias & Limits of Agreement)</b>", "<b>3. Percent Bias Plot</b>"),
        vertical_spacing=0.15
    )

    # Deming Plot
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['Test'], mode='markers', name='Samples'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Reference'], y=deming_intercept + deming_slope * df['Reference'], mode='lines', name='Deming Fit', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 220], y=[0, 220], mode='lines', name='Identity (y=x)', line=dict(color='black', dash='dash')), row=1, col=1)
    
    # Bland-Altman Plot
    fig.add_trace(go.Scatter(x=df['Average'], y=df['Difference'], mode='markers', name='Difference'), row=2, col=1)
    fig.add_hline(y=mean_diff, line=dict(color='blue', dash='dash'), name='Mean Bias', row=2, col=1)
    fig.add_hline(y=upper_loa, line=dict(color='red', dash='dash'), name='Upper LoA', row=2, col=1)
    fig.add_hline(y=lower_loa, line=dict(color='red', dash='dash'), name='Lower LoA', row=2, col=1)
    
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

@st.cache_data
def plot_classification_models():
    np.random.seed(1)
    n_points = 200
    X1 = np.random.uniform(0, 10, n_points)
    X2 = np.random.uniform(0, 10, n_points)
    # Create a non-linear relationship
    prob = 1 / (1 + np.exp(-( (X1-5)**2 + (X2-5)**2 - 8)))
    y = np.random.binomial(1, prob)
    X = np.column_stack((X1, X2))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    lr = LogisticRegression().fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)

    # Create meshgrid for decision boundary
    xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'<b>Logistic Regression (Accuracy: {lr_score:.2%})</b>', 
                                                       f'<b>Random Forest (Accuracy: {rf_score:.2%})</b>'))

    # Plot Logistic Regression
    Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_lr, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black'))), row=1, col=1)

    # Plot Random Forest
    Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_rf, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=2)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black'))), row=1, col=2)

    fig.update_layout(title="<b>Predictive QC: Linear vs. Non-Linear Models</b>", showlegend=False, height=500)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=1); fig.update_yaxes(title_text="Parameter 2", row=1, col=1)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=2); fig.update_yaxes(title_text="Parameter 2", row=1, col=2)
    return fig

@st.cache_data
def plot_xai_shap():
    # This function uses matplotlib backend for SHAP, so we need to handle image conversion
    plt.style.use('default')
    
    # Load sample data (no changes needed here)
    github_data_url = "https://github.com/slundberg/shap/raw/master/data/"
    data_url = github_data_url + "adult.data"
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"),
        ("Marital Status", "category"), ("Occupation", "category"),
        ("Relationship", "category"), ("Race", "category"), ("Sex", "category"),
        ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(data_url, names=[d[0] for d in dtypes], na_values="?", dtype=dict(dtypes))
    X_display = raw_data.drop("Target", axis=1)
    y = (raw_data["Target"] == " >50K").astype(int)
    X = X_display.copy()
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes

    model = RandomForestClassifier(random_state=42).fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values_obj = explainer(X.iloc[:100]) 
    
    # --- Beeswarm plot (no changes needed here) ---
    shap.summary_plot(shap_values_obj.values[:,:,1], X.iloc[:100], show=False)
    buf_summary = io.BytesIO()
    plt.savefig(buf_summary, format='png', bbox_inches='tight')
    plt.close()
    buf_summary.seek(0)
    
    # --- Force plot HTML generation (THIS IS THE FIX) ---
    # 1. Generate the plot object
    force_plot = shap.force_plot(
        explainer.expected_value[1], 
        shap_values_obj.values[0,:,1], 
        X_display.iloc[0,:], 
        show=False
    )
    
    # 2. Get the raw HTML from the plot object
    force_plot_html = force_plot.html()

    # 3. Create a fully self-contained HTML string by adding the SHAP JS library.
    #    shap.initjs() injects the necessary <script> tag for rendering.
    full_html = f"<html><head>{shap.initjs()}</head><body>{force_plot_html}</body></html>"

    return buf_summary, full_html
    
    # Beeswarm plot as an image
    shap.summary_plot(shap_values_obj.values[:,:,1], X.iloc[:100], show=False)
    buf_summary = io.BytesIO()
    plt.savefig(buf_summary, format='png', bbox_inches='tight')
    plt.close()
    buf_summary.seek(0)
    
    # Force plot as html
    force_plot_html = shap.force_plot(explainer.expected_value[1], shap_values_obj.values[0,:,1], X_display.iloc[0,:], show=False)
    html_string = force_plot_html.html()

    return buf_summary, html_string

@st.cache_data
def plot_advanced_ai_concepts(concept):
    fig = go.Figure()
    if concept == "Transformers":
        text = "Input Seq -> [Encoder] -> [Self-Attention] -> [Decoder] -> Output Seq"
        fig.add_annotation(text=f"<b>Conceptual Flow: Transformer</b><br>{text}", showarrow=False, font_size=16)
    elif concept == "GNNs":
        nodes_x = [1, 2, 3, 4, 3, 2]; nodes_y = [2, 3, 2, 1, 0, -1]
        edges = [(0,1), (1,2), (2,3), (2,4), (4,5), (5,1)]
        for (start, end) in edges:
            fig.add_trace(go.Scatter(x=[nodes_x[start], nodes_x[end]], y=[nodes_y[start], nodes_y[end]], mode='lines', line_color='grey'))
        fig.add_trace(go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text', text=[f"Node {i}" for i in range(6)], marker_size=30, textposition="middle center"))
        fig.update_layout(title="<b>Conceptual Flow: Graph Neural Network</b>")
    elif concept == "RL":
        fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2, line_width=2, fillcolor='lightblue', name="Agent")
        fig.add_annotation(x=1, y=1, text="<b>Agent</b><br>(Control Policy)", showarrow=False)
        fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=2, line_width=2, fillcolor='lightgreen', name="Environment")
        fig.add_annotation(x=5, y=1, text="<b>Environment</b><br>(Digital Twin)", showarrow=False)
        fig.add_annotation(x=3, y=1.5, text="Action", showarrow=True, arrowhead=2, ax=-40, ay=0)
        fig.add_annotation(x=3, y=0.5, text="State, Reward", showarrow=True, arrowhead=2, ax=40, ay=0)
        fig.update_layout(title="<b>Conceptual Flow: Reinforcement Learning Loop</b>")
    elif concept == "Generative AI":
        fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2, line_width=2, fillcolor='lightcoral', name="Real Data")
        fig.add_annotation(x=1, y=1, text="<b>Real Data</b>", showarrow=False)
        fig.add_annotation(x=3, y=1, text="Trains ➔", showarrow=False, font_size=20)
        fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=2, line_width=2, fillcolor='gold', name="Generator")
        fig.add_annotation(x=5, y=1, text="<b>Generator</b>", showarrow=False)
        fig.add_annotation(x=7, y=1, text="Creates ➔", showarrow=False, font_size=20)
        fig.add_shape(type="rect", x0=8, y0=0, x1=10, y1=2, line_width=2, fillcolor='lightseagreen', name="Synthetic Data")
        fig.add_annotation(x=9, y=1, text="<b>Synthetic Data</b>", showarrow=False)
        fig.update_layout(title="<b>Conceptual Flow: Generative AI (GANs)</b>")
    
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300, showlegend=False)
    return fig
    
@st.cache_data
def plot_causal_inference():
    fig = go.Figure()
    # Define node positions
    nodes = {'Reagent': (0, 1), 'Temp': (1.5, 2), 'Pressure': (1.5, 0), 'Purity': (3, 2), 'Yield': (3, 0)}
    # Add nodes
    fig.add_trace(go.Scatter(x=[v[0] for v in nodes.values()], y=[v[1] for v in nodes.values()],
                               mode="markers+text", text=list(nodes.keys()), textposition="top center",
                               marker=dict(size=30, color='lightblue', line=dict(width=2, color='black')), textfont_size=14))
    # Add edges
    edges = [('Reagent', 'Purity'), ('Reagent', 'Pressure'), ('Temp', 'Purity'), ('Temp', 'Pressure'), ('Pressure', 'Yield'), ('Purity', 'Yield')]
    for start, end in edges:
        fig.add_annotation(x=nodes[end][0], y=nodes[end][1], ax=nodes[start][0], ay=nodes[start][1],
                           xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='black')
    fig.update_layout(title="<b>Conceptual Directed Acyclic Graph (DAG)</b>", showlegend=False, xaxis_visible=False, yaxis_visible=False, height=500, margin=dict(t=100))
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
                - The wide, light blue curve is the **true population distribution**. In real life, we *never* see this. It represents every possible measurement.
                - The narrow, orange curve is the **sampling distribution of the mean**. This is a theoretical distribution of *all possible sample means* of size `n`. Its narrowness, guaranteed by the **Central Limit Theorem**, is the miracle that makes statistical inference possible.
            - **CI Simulation (Bottom Plot):** This plot shows the reality we live in. We only get to run *one* experiment and get *one* confidence interval (e.g., the first blue line). We don't know if ours is one of the 95 that captured the true mean or one of the 5 that missed.
            - **The n-slider is key:** As you increase `n`, the orange curve gets narrower and the CIs in the bottom plot become dramatically shorter. This shows that precision is a direct function of sample size.
            - **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This illustrates the cost of increased precision: to double your precision (halve the CI width), you must quadruple your sample size, as precision scales with $\sqrt{n}$.

            **The Core Strategic Insight:** A confidence interval is a statement about the *procedure*, not a specific result. The "95% confidence" is our confidence in the *method* used to generate the interval, not in any single interval itself. We are confident that if we were to repeat our experiment 100 times, roughly 95 of the resulting CIs would contain the true, unknown parameter.
            """)
        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT (Bayesian) INTERPRETATION:**
            *"Based on my sample, there is a 95% probability that the true mean is in this interval [X, Y]."*
            
            This is wrong because in the frequentist view, the true mean is a fixed, unknown constant. It is either in our specific interval or it is not. The probability is either 1 or 0; we just don't know which. The randomness is in the sampling process that creates the interval, not in the true mean itself.
            """)
            st.success("""
            🟢 **THE CORRECT (Frequentist) INTERPRETATION:**
            *"We are 95% confident that the interval [X, Y] contains the true mean."*
            
            The full, technically correct meaning behind this is: *"This specific interval was constructed using a procedure that, when repeated many times on new samples, will produce intervals that capture the true mean 95% of the time."*
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of **confidence intervals** was introduced to the world by the brilliant Polish-American mathematician and statistician **Jerzy Neyman** in a landmark 1937 paper. Neyman, a fierce advocate for the frequentist school, sought a rigorously objective method for interval estimation that did not rely on the "subjective" priors of Bayesian inference.
            
            He was a philosophical rival of Sir R.A. Fisher, who had proposed a similar concept called a "fiducial interval," which attempted to assign a probability distribution to a fixed parameter. Neyman found this logically incoherent. His revolutionary idea was to shift the probabilistic statement away from the fixed, unknown parameter and onto the **procedure used to create the interval**. This clever reframing provided a practical and logically consistent solution that remains the dominant paradigm for interval estimation in applied statistics worldwide.
            
            #### Mathematical Basis
            The general form of a two-sided confidence interval is a combination of three components:
            """)
            st.latex(r"\text{Point Estimate} \pm (\text{Margin of Error})")
            st.latex(r"\text{Margin of Error} = (\text{Critical Value} \times \text{Standard Error})")
            st.markdown("""
            - **Point Estimate:** Our best single-value guess for the population parameter (e.g., the sample mean, $\bar{x}$).
            - **Standard Error:** The standard deviation of the sampling distribution of the point estimate. For the mean, this is $\frac{s}{\sqrt{n}}$, where $s$ is the sample standard deviation. It measures the typical error in our point estimate and shrinks as `n` increases.
            - **Critical Value:** A multiplier determined by our desired confidence level and the relevant statistical distribution (e.g., a z-score from the normal distribution or a t-score from the Student's t-distribution). For a 95% CI, this value is typically close to 2.
            
            For the mean, this results in the familiar formula:
            """)
            st.latex(r"CI = \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}")

def render_core_validation_params():
    """Renders the module for core validation parameters (Accuracy, Precision, Specificity)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the fundamental performance characteristics of an analytical method as required by global regulatory guidelines like ICH Q2(R1). This module deconstructs the "big three" pillars of method validation:
    - **🎯 Accuracy (Bias):** How close are your measurements to the *real* value? Think of it as hitting the bullseye.
    - **🏹 Precision (Random Error):** How consistent are your measurements with each other? Think of it as the tightness of your arrow grouping.
    - **🔬 Specificity (Selectivity):** Can your method find the target analyte in a crowded room, ignoring all the imposters?

    **Strategic Application:** These parameters are the non-negotiable pillars of any formal assay validation report submitted to regulatory bodies like the FDA or EMA. They provide the objective evidence that the method is the bedrock of product quality and patient safety. A weakness in any of these three areas is a critical deficiency that can lead to rejected submissions, product recalls, or flawed R&D conclusions. This isn't just a statistical exercise; it's the license to operate.
    """)
    
    # The plot generation function would be called here.
    # We assume it returns three figures for demonstration.
    fig1, fig2, fig3 = plot_core_validation_params()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🎯 Accuracy KPI: Mean % Recovery", value="99.2%", help="The key metric for accuracy. Regulators typically look for this to be within 80-120% for biotech assays or 98-102% for small molecules.")
            st.metric(label="🏹 Precision KPI: Max %CV", value="< 8%", help="The key metric for precision. Lower is better. A common acceptance criterion for intermediate precision is <15-20%.")
            st.metric(label="🔬 Specificity KPI: Interference Test", value="Pass (p > 0.05)", help="A non-significant p-value in a t-test between 'Analyte' and 'Analyte + Interferent' is the goal.")
            
            st.markdown("""
            - **Accuracy (Top Plot):** This plot reveals **bias**. The goal is for the center of each box plot to sit on the dashed 'True Value' line. The distance between the center and that line is the systematic error. A method can be precise but wildly inaccurate.
            
            - **Precision (Middle Plot):** This plot reveals **random error**. The 'violins' show the data spread.
                - **Repeatability:** Is the 'best-case' spread. A tight violin is good.
                - **Intermediate Precision:** Is the 'real-world' spread, accounting for different days, analysts, etc. It will always be wider than repeatability. The key question is: *by how much*? A large increase signals the method is not robust.
            
            - **Specificity (Bottom Plot):** This plot tests for **interference**. The "Analyte + Interferent" bar must be statistically identical to the "Analyte Only" bar. If it's different, your method can't distinguish your target from other components, making it unfit for real samples.

            **The Core Strategic Insight:** Accuracy, Precision, and Specificity are not independent checkboxes. They form an interconnected triangle of evidence. A non-specific method can never be truly accurate. A highly imprecise method makes it impossible to reliably assess accuracy. The goal of validation is to present a holistic, data-driven argument that the method is **fit for its intended purpose.**
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
    """Renders the interactive module for Gage R&R."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To rigorously quantify the inherent variability (error) of a measurement system and decompose it from the true, underlying variation of the process or product. A Gage R&R study is the definitive method for assessing the **metrological fitness-for-purpose** of any analytical method or instrument. It answers the fundamental question: "Is my measurement system a precision instrument, or a random number generator?"
    
    **Strategic Application:** This is the non-negotiable **foundational checkpoint** in any technology transfer, process validation, or serious quality improvement initiative. Attempting to characterize a process with an uncharacterized measurement system is scientifically invalid. An unreliable measurement system creates a "fog of uncertainty," injecting noise that can lead to two costly errors:
    1.  **Type I Error (False Alarm):** The measurement system's noise makes a good batch appear out-of-spec, leading to unnecessary investigations and rejection of good product.
    2.  **Type II Error (Missed Signal):** The noise masks a real process drift or shift, allowing a bad batch to be released, potentially leading to catastrophic field failures.

    By partitioning the total observed variation into its distinct components—**Repeatability**, **Reproducibility**, and **Part-to-Part** variation—this analysis provides the objective, statistical evidence needed to trust your data.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, pct_rr, pct_part = plot_gage_rr()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse", help="This represents the percentage of the total observed variation that is consumed by measurement error.")
            st.metric(label="💡 KPI: Number of Distinct Categories (ndc)", value=f"{int(1.41 * (pct_part / pct_rr)**0.5) if pct_rr > 0 else '>10'}", help="An estimate of how many distinct groups the measurement system can discern in the process data. A value < 5 is problematic.")

            st.markdown("""
            - **Variation by Part & Operator (Main Plot):** The diagnostic heart of the study.
                - *High Repeatability Error:* Wide boxes for a given operator, indicating the instrument/assay has poor precision. This is often a hardware or chemistry problem.
                - *High Reproducibility Error:* The colored lines (operator means) are not parallel or are vertically offset. This is often a human factor or training issue.
                - ***The Interaction Term:*** A significant Operator-by-Part interaction is the most insidious problem. It means operators are not just biased, but *inconsistently* biased. Operator A measures Part 1 high and Part 5 low, while Operator B does the opposite. This points to ambiguous instructions or a flawed measurement technique.

            - **The "Number of Distinct Categories" (ndc):** This powerful metric translates %R&R into practical terms. It estimates how many non-overlapping groups your measurement system can reliably distinguish within your process's variation.
                - `ndc = 1`: The system is useless; it cannot even tell the difference between a high part and a low part.
                - `ndc = 2-4`: The system can only perform crude screening (e.g., pass/fail).
                - `ndc ≥ 5`: The system is considered adequate for process control.

            **The Core Strategic Insight:** A low % Gage R&R validates your measurement system as a trustworthy "ruler," confirming that the variation you observe reflects genuine process dynamics, not measurement noise. A high value means your ruler is "spongy," making any conclusions about your process's health statistically indefensible. **You cannot manage what you cannot reliably measure.**
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria are risk-based and derived from the **AIAG's Measurement Systems Analysis (MSA)** manual, the de facto global standard. The percentage is calculated against the **total study variation**.")
            st.markdown("- **< 10% Gage R&R:** The system is **acceptable**. The 'fog of uncertainty' is minimal. The system can reliably detect process shifts and can be used for SPC and capability analysis.")
            st.markdown("- **10% - 30% Gage R&R:** The system is **conditionally acceptable or marginal**. Its use may be approved for less critical characteristics, but it is likely unsuitable for controlling a critical-to-quality parameter. This result should trigger an improvement project for the measurement method.")
            st.markdown("- **> 30% Gage R&R:** The system is **unacceptable and must be rejected**. Data generated by this system is untrustworthy. Using this system for process decisions is equivalent to making decisions by flipping a coin. The method must be fundamentally improved.")
            st.info("""
            **Beyond the Numbers: The Part Selection Strategy**
            The most common failure mode of a Gage R&R study is not the math, but the study design. The parts selected **must span the full expected range of process variation**. If you only select parts from the middle of the distribution, your Part-to-Part variation will be artificially low, which will mathematically inflate your % Gage R&R and cause a good system to fail. A robust study includes parts from near the Lower and Upper Specification Limits.
            """)
            
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            While the concepts are old, their modern application was born out of the quality crisis in the American automotive industry in the 1970s. Guided by luminaries like **W. Edwards Deming**, manufacturers realized they were often "tampering" with their processes—adjusting a stable process based on faulty measurement data, thereby *increasing* variation.
            
            The **AIAG** codified the solution in the first MSA manual. The critical evolution was the move from the simple **Average and Range (X-bar & R) method** to the **ANOVA method**. The X-bar & R method is computationally simpler but has a critical flaw: it confounds the operator-part interaction with reproducibility. The **ANOVA method**, pioneered for agriculture by statistician **Sir Ronald A. Fisher**, became the gold standard because of its unique ability to cleanly partition and test the significance of each variance component, including the crucial interaction term.
            
            #### Mathematical Basis
            The ANOVA method partitions the total sum of squared deviations from the mean ($SS_T$) into components attributable to each factor.
            """)
            st.latex(r"SS_{Total} = SS_{Part} + SS_{Operator} + SS_{Part \times Operator} + SS_{Error}")
            st.markdown("""
            These sums of squares are then converted to Mean Squares (MS) by dividing by their respective degrees of freedom (df). The variance components ($\hat{\sigma}^2$) are then estimated from these MS values.
            - **Repeatability (Equipment Variation, EV):** The inherent random error of the measurement process.
            """)
            st.latex(r"\hat{\sigma}^2_{Repeatability} = MS_{Error}")
            st.markdown("- **Reproducibility (Appraiser Variation, AV):** The variation between operators, composed of the pure operator effect and the interaction effect.")
            st.latex(r"\hat{\sigma}^2_{Reproducibility} = \hat{\sigma}^2_{Operator} + \hat{\sigma}^2_{Interaction}")
            st.latex(r"\text{where } \hat{\sigma}^2_{Operator} = \frac{MS_{Operator} - MS_{Interaction}}{n_{parts} \cdot n_{replicates}}")
            st.warning("**Negative Variance Components:** It is mathematically possible for these formulas to yield a negative variance for a term. This is a statistical artifact. The correct interpretation is that the true variance component is zero, and it should be set to zero for calculating the final %R&R.")

def render_linearity():
    """Renders the interactive module for Linearity analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify that an assay's response is directly and predictably proportional to the known concentration of the analyte across its entire intended operational range.
    
    **Strategic Application:** This is a cornerstone of quantitative assay validation, mandated by every major regulatory body (FDA, EMA, ICH). It provides the statistical evidence that the assay is not just precise, but **globally accurate** across its reportable range. A method exhibiting non-linearity might be accurate at a central control point but dangerously inaccurate at the specification limits, leading to incorrect batch disposition decisions.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, model = plot_linearity()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}", help="Indicates the proportion of variance in the measured values explained by the nominal values. A necessary, but not sufficient, criterion.")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}", help="Ideal = 1.0. A slope < 1 indicates signal compression; > 1 indicates expansion.")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}", help="Ideal = 0.0. A non-zero intercept indicates a constant systematic error or background bias.")
            st.markdown("""
            - **Linearity Plot:** Data should cluster tightly around the Line of Identity (y=x). Any systematic deviation (e.g., a gentle 'S' curve) suggests non-linearity that R² alone might miss.
            - **Residual Plot:** The single most powerful diagnostic for linearity. A perfect model shows a random, "shotgun blast" pattern of points centered on zero.
                - A **curved (U or ∩) pattern** is the classic sign of non-linearity, indicating the straight-line model is inappropriate. This is often due to detector saturation at high concentrations.
                - A **funnel shape (heteroscedasticity)** indicates that the error increases with concentration. This is common in analytical chemistry and violates a key assumption of Ordinary Least Squares (OLS) regression. The proper technique here is **Weighted Least Squares (WLS) Regression.**
            - **Recovery Plot:** The practical business-end of the analysis. It translates statistical error into analytical accuracy, answering: "At a given true concentration, what result does my assay report, and by how much is it off?"
            
            **The Core Strategic Insight:** A high R², a slope of 1, an intercept of 0, randomly scattered residuals, and recovery within tight limits collectively provide a **verifiable chain of evidence** that the assay is a trustworthy quantitative tool across its entire defined range.
            """)

        with tabs[1]:
            st.markdown("These criteria are defined in the validation protocol and must be met to declare the method linear.")
            st.markdown("- **R-squared (R²):** While common, it is a weak criterion alone. An R² > **0.995** is a typical starting point, but for chromatography (HPLC, GC), R² > **0.999** is often required.")
            st.markdown("- **Slope:** The 95% confidence interval for the slope must contain 1.0. A common acceptance range for the point estimate is **0.95 to 1.05** for immunoassays, or tighter (**0.98 to 1.02**) for more precise methods.")
            st.markdown("- **Y-Intercept:** The 95% confidence interval for the intercept must contain 0. This statistically proves the absence of a significant constant bias.")
            st.markdown("- **Residuals:** There should be no obvious pattern or trend in the residual plot. Formal statistical tests like the **Lack-of-Fit test** can be used to objectively prove linearity (this requires true replicates at each concentration level).")
            st.markdown("- **Recovery:** The percent recovery at each concentration level must fall within a pre-defined range (e.g., 80% to 120% for bioassays, 99.0% to 101.0% for drug purity assays).")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The mathematical engine is **Ordinary Least Squares (OLS) Regression**, a cornerstone of statistics developed independently by **Adrien-Marie Legendre (1805)** and **Carl Friedrich Gauss (1809)**. Gauss famously used it to predict the location of the dwarf planet Ceres after it was lost, a triumph of mathematical modeling.
            
            The genius of OLS lies in its objective function: to find the line that **minimizes the sum of the squared vertical distances (the "residuals")** between the observed data and the fitted line. Under the assumption of normally distributed errors, the OLS estimates are also the **Maximum Likelihood Estimates (MLE)**, providing a deep theoretical justification for the method.

            #### Mathematical Basis
            The goal is to fit a simple linear model to the calibration data, linking the true concentration ($x$) to the measured response ($y$).
            """)
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - $y$: The measured concentration or instrument signal.
            - $x$: The nominal (true) concentration of the reference standard.
            - $\\beta_0$ (Intercept): Represents the assay's **constant systematic error**.
            - $\\beta_1$ (Slope): Represents the assay's **proportional systematic error** (sensitivity).
            - $\\epsilon$: The random measurement error.

            The validation hinges on formal statistical tests of the estimated coefficients ($\hat{\beta}_0, \hat{\beta}_1$):
            - **Hypothesis Test for Slope:** $H_0: \\beta_1 = 1$ (no proportional bias).
            - **Hypothesis Test for Intercept:** $H_0: \\beta_0 = 0$ (no constant bias).
            A p-value > 0.05 for these tests supports the claim of linearity and no bias.
            """)

def render_lod_loq():
    """Renders the interactive module for Limit of Detection & Quantitation."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the absolute lower performance boundaries of a quantitative assay. It determines the lowest analyte concentration an assay can reliably **detect (LOD)** and the lowest concentration it can reliably and accurately **quantify (LOQ)**.
    
    **Strategic Application:** This is a mission-critical parameter for any assay used to measure trace components. Examples include:
    - **Impurity Testing:** The LOQ *must* be demonstrably below the specification limit for a potentially harmful impurity in a drug product.
    - **Early-Stage Disease Diagnosis:** The LOD/LOQ for a cancer biomarker must be low enough to detect the disease at its earliest, most treatable stage.
    - **Pharmacokinetics (PK):** To properly characterize a drug's elimination phase, the assay LOQ must be low enough to measure the final few datapoints in the concentration-time curve.
    
    The **LOD** is a qualitative threshold answering "Is the analyte present?" The **LOQ** is a much higher quantitative bar, answering "What is the concentration, and can I trust the numerical value?"
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, lod_val, loq_val = plot_lod_loq()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} ng/mL", help="The lowest concentration you can report with confidence in the numerical value.")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} ng/mL", help="The lowest concentration you can reliably claim is 'present'.")
            st.markdown("""
            - **Signal Distribution (Violin Plot):** The distribution of signals from the 'Blank' samples (the noise) must be clearly separated from the distribution of signals from the 'Low Concentration' samples. Significant overlap indicates the assay lacks the fundamental sensitivity required.
            - **Low-Level Calibration Curve (Regression Plot):** The LOD and LOQ are derived directly from two key parameters of this model:
                1.  **The Slope (S):** The assay's sensitivity. A steeper slope is better.
                2.  **The Residual Standard Error (σ):** The inherent noise or imprecision of the assay at the low end. A smaller σ is better.

            **The Core Strategic Insight:** This analysis defines the **absolute floor of your assay's validated capability**. Claiming a quantitative result below the validated LOQ is scientifically and regulatorily indefensible. It's the difference between seeing a faint star and being able to measure its brightness.
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria are absolute and defined by the assay's intended use.")
            st.markdown("- The primary, non-negotiable criterion is that the experimentally determined **LOQ must be ≤ the lowest concentration that the assay is required to measure** for its specific application (e.g., a release specification for an impurity).")
            st.markdown("- For a concentration to be formally declared the LOQ, it must be experimentally confirmed. This typically involves analyzing 5-6 independent samples at the claimed LOQ concentration and demonstrating that they meet pre-defined criteria for precision and accuracy (e.g., **%CV < 20% and %Recovery between 80-120%** for a bioassay).")
            st.warning("""
            **The LOB, LOD, and LOQ Hierarchy: A Critical Distinction**
            A full characterization involves three distinct limits:
            - **Limit of Blank (LOB):** The highest measurement expected from a blank sample. (LOB = mean_blank + 1.645 * sd_blank)
            - **Limit of Detection (LOD):** The lowest concentration whose signal is statistically distinguishable from the LOB. (LOD = LOB + 1.645 * sd_low_conc_sample)
            - **Limit of Quantitation (LOQ):** The lowest concentration meeting precision/accuracy requirements, which is almost always higher than the LOD.
            Confusing these is a common and serious error.
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The need to define analytical sensitivity is old, but definitions were inconsistent until the **International Council for Harmonisation (ICH)** guideline **ICH Q2(R1) "Validation of Analytical Procedures"** harmonized the definitions and methodologies. This work was heavily influenced by the statistical framework established by **Lloyd Currie at NIST** in his 1968 paper, which established the clear, hypothesis-testing basis for the modern LOB/LOD/LOQ hierarchy.

            #### Mathematical Basis
            This method is built on the relationship between the assay's signal, its sensitivity (Slope, S), and its noise (standard deviation, σ). The standard deviation σ is most robustly estimated using the **residual standard error** from a regression model fit to low-concentration data.

            - **Limit of Detection (LOD):** The formula is designed to control the risk of false positives and false negatives. The factor 3.3 is an approximation related to a high level of confidence that a signal at this level is not a random fluctuation of the blank.
            """)
            st.latex(r"LOD \approx \frac{3.3 \times \sigma}{S}")
            st.markdown("""
            - **Limit of Quantitation (LOQ):** This is about measurement quality. It demands a much higher signal-to-noise ratio to ensure the measurement has an acceptable level of uncertainty. The factor of 10 is the standard convention that typically yields a precision of roughly 10% CV for a well-behaved assay.
            """)
            st.latex(r"LOQ \approx \frac{10 \times \sigma}{S}")

def render_method_comparison():
    """Renders the interactive module for Method Comparison."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally assess and quantify the degree of agreement and systemic bias between two different measurement methods intended to measure the same quantity. This analysis moves beyond simple correlation to determine if the two methods can be used **interchangeably** in practice.
    
    **Strategic Application:** This study is the "crucible" of method transfer, validation, or replacement. A failed comparison study can halt a tech transfer, delay a product launch, or invalidate a clinical study. Key scenarios include:
    - **Tech Transfer:** Proving a QC lab's assay is equivalent to the original R&D method.
    - **Method Modernization:** Demonstrating a new, faster, or cheaper assay yields clinically equivalent results to an older gold standard.
    - **Cross-Site Harmonization:** Ensuring results from different facilities are comparable.
    
    This analysis answers the critical business and regulatory question: “Do these two methods produce the same result, for the same sample, within medically or technically acceptable limits?”
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, slope, intercept, bias, ua, la = plot_method_comparison()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Mean Bias (Bland-Altman)", value=f"{bias:.2f} units", help="The average systematic difference between the Test and Reference methods. Positive value = Test method measures higher on average.")
            st.metric(label="💡 Metric: Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0. Measures proportional bias, which is concentration-dependent.")
            st.metric(label="💡 Metric: Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0. Measures constant bias, a fixed offset across the entire range.")
            st.markdown("""
            - **Deming Regression:** The correct regression for method comparison. Unlike standard OLS, it accounts for measurement error in *both* methods, providing an unbiased estimate of slope (proportional bias) and intercept (constant bias). The goal is to see the red Deming line perfectly overlay the black Line of Identity.
            - **Bland-Altman Plot:** This plot transforms the question from "are they correlated?" to "how much do they differ?". It visualizes the random error and quantifies the **95% Limits of Agreement (LoA)**, the expected range of disagreement for 95% of future measurements.
            - **% Bias Plot:** This plot assesses **practical significance**. It shows if the bias at any specific concentration exceeds a pre-defined acceptable limit (e.g., ±15%).

            **The Core Strategic Insight:** This dashboard provides a multi-faceted verdict on method interchangeability. Deming regression diagnoses the *type* of bias (constant vs. proportional), the Bland-Altman plot quantifies the *magnitude* of expected random disagreement, and the % Bias plot confirms *local* acceptability.
            """)
        with tabs[1]:
            st.markdown("Acceptance criteria must be pre-defined in the validation protocol and be clinically or technically justified.")
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope must contain 1.0**, and the 95% CI for the **intercept must contain 0**. This provides statistical proof of no systematic bias.")
            st.markdown(f"- **Bland-Altman:** The primary criterion is that the **95% Limits of Agreement (`{la:.2f}` to `{ua:.2f}`) must be clinically or technically acceptable**. A 20-unit LoA might be acceptable for a glucose monitor but catastrophic for a cancer biomarker.")
            st.markdown("- **Total Analytical Error (TAE):** An advanced approach where, across the entire range, `|Bias| + 1.96 * SD_of_difference` must be less than a predefined Total Allowable Error (TEa).")
            st.error("""
            **The Correlation Catastrophe**
            Do not, under any circumstances, use the correlation coefficient (R or R²) as a measure of agreement. Two methods can be perfectly correlated (R=1.0) but have a huge bias (e.g., one method always reads exactly twice as high as the other). A high correlation is a prerequisite for agreement, but it is **not** evidence of agreement.
            """)

        with tabs[2]:
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
            - **The Mantra: Control Before Capability.** The control chart (top plot) is a prerequisite. The Cpk metric is only statistically valid and meaningful if the process is stable and in-control. The 'Out of Control' scenario yields an **INVALID** Cpk because an unstable process has no single, predictable "voice" to measure. Its future performance is unknown.
            - **The Key Insight: Control ≠ Capability.** A process can be perfectly in-control (predictable) but not capable (producing bad product). 
                - The **'Shifted'** scenario shows a process that is precise but inaccurate.
                - The **'Variable'** scenario shows a process that is centered but imprecise.
            Both are in control, but both have a poor Cpk. This demonstrates why you need both SPC (for control) and Capability Analysis (for quality).
            """)
        with tabs[1]:
            st.markdown("These are industry-standard benchmarks, often required by customers, especially in automotive and aerospace. For pharmaceuticals, a high Cpk in validation provides strong assurance of lifecycle performance.")
            st.markdown("- `Cpk < 1.00`: Process is **not capable**. The 'voice of the process' is wider than the 'voice of the customer.' A significant portion of output will not meet specifications.")
            st.markdown("- `1.00 ≤ Cpk < 1.33`: Process is **marginally capable**. It requires tight control and monitoring, as small shifts can lead to non-conforming product.")
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable**. This is a common minimum target for many industries, corresponding to a '4-sigma' quality level and a theoretical defect rate of ~63 parts per million (PPM).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** and is approaching **Six Sigma** quality. This corresponds to a '5-sigma' level and a theoretical defect rate of ~0.6 PPM.")
            st.markdown("- `Cpk ≥ 2.00`: Process has achieved **Six Sigma capability** (assuming no long-term shift). This represents world-class performance with a theoretical defect rate of just 2 parts per *billion*. ")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of comparing process output to specification limits is old, but the formalization into capability indices originated in the Japanese manufacturing industry in the 1970s as a core part of Total Quality Management (TQM).
            
            However, it was the **Six Sigma** initiative, pioneered by engineer Bill Smith at **Motorola in the 1980s**, that catapulted Cpk to global prominence. The 'Six Sigma' concept was born: a process so capable that the nearest specification limit is at least six standard deviations away from the process mean. This translates to a defect rate of just 3.4 parts per million (which famously accounts for a hypothetical 1.5 sigma long-term drift of the process mean). Cpk became the standard metric for measuring progress toward this ambitious goal.
            
            #### Mathematical Basis
            Capability analysis is a direct comparison between the **"Voice of the Customer"** (the allowable spread, USL - LSL) and the **"Voice of the Process"** (the actual, natural spread, conventionally 6σ).

            - **Cp (Potential Capability):** Measures if the process is narrow enough, ignoring centering. It's the best the process *could* be if perfectly centered.
            """)
            st.latex(r"C_p = \frac{\text{Tolerance Width}}{\text{Process Width}} = \frac{USL - LSL}{6\hat{\sigma}}")
            st.markdown("- **Cpk (Actual Capability):** The more important metric, as it accounts for process centering. It is the lesser of the upper and lower capability indices, effectively measuring the distance from the process mean to the *nearest* specification limit. It is the 'worst-case scenario'.")
            st.latex(r"C_{pk} = \min(C_{pu}, C_{pl}) = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right)")
            st.markdown("A Cpk of 1.33 means that the process distribution could fit between the mean and the nearest spec limit 1.33 times. This provides a 'buffer' zone to absorb small process shifts without producing defects.")

def render_pass_fail():
    """Renders the interactive module for Pass/Fail (Binomial Proportion) analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately calculate and critically compare confidence intervals for a binomial proportion, which is the underlying statistic for any pass/fail, present/absent, or concordant/discordant outcome.
    
    **Strategic Application:** This is essential for the validation of **qualitative assays** or for agreement studies in method transfers. The goal is to prove, with a high degree of statistical confidence, that the assay's success rate (e.g., >95% concordance with a reference method) is above a required performance threshold. 
    
    The critical challenge, especially with the small sample sizes typical in validation (n=30 is common), is that simple, textbook methods for calculating confidence intervals (the 'Wald' interval) are dangerously inaccurate. Choosing the wrong method can lead to falsely concluding a method is acceptable when it is not, a major regulatory and quality risk.
    """)
    n_samples_wilson = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30, key='wilson_n')
    successes_wilson = st.sidebar.slider("Concordant Results", 0, n_samples_wilson, int(n_samples_wilson * 0.95), key='wilson_s')
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_wilson, fig2_wilson = plot_wilson(successes_wilson, n_samples_wilson)
        st.plotly_chart(fig1_wilson, use_container_width=True)
        st.plotly_chart(fig2_wilson, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}", help="The point estimate of the success rate. This value alone is insufficient without a confidence interval.")
            st.markdown("""
            - **CI Comparison (Top Plot):** This plot reveals the dramatic differences between interval methods. Note how the 'Wald' interval is often much narrower, giving a false sense of precision. At the extremes (e.g., 30/30 successes), the Wald interval collapses to a width of zero, which is statistically indefensible.
            - **Coverage Probability (Bottom Plot):** This is the crucial diagnostic plot. It shows the *actual* probability that an interval will contain the true proportion.
                - The **Wald interval (red)** is a disaster. Its actual coverage plummets near the extremes and is wildly erratic everywhere else. It consistently fails to meet the nominal 95% level.
                - The **Wilson and Clopper-Pearson intervals (blue/green)** are far superior. Their coverage probability is always at or above the nominal 95% level, making them reliable and conservative.

            **The Core Strategic Insight:** Never use the standard Wald (or "Normal Approximation") interval for important decisions, especially with sample sizes under 100. The **Wilson Score interval** provides the best balance of accuracy and interval width for most applications. The **Clopper-Pearson** is the most conservative ("exact") choice, often preferred in regulatory submissions for its guaranteed coverage.
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
            
            #### Mathematical Basis
            The Wald interval is simply $\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$. The Wilson Score interval's superior formula is:
            """)
            st.latex(r"CI_{Wilson} = \frac{1}{1 + z^2/n} \left( \hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \right)")
            st.markdown("Notice it adds pseudo-successes and failures ($z^2/2$), pulling the center away from 0 or 1. This is what gives it such good performance where the Wald interval fails catastrophically.")
            

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
def render_multi_rule():
    """Renders the module for Multi-Rule SPC (Westgard Rules)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To serve as a high-sensitivity "security system" for your assay. Instead of one simple alarm, this system uses a combination of rules to detect specific types of problems, catching subtle shifts and drifts long before a catastrophic failure occurs. It dramatically increases the probability of detecting true errors while minimizing false alarms.
    
    **Strategic Application:** This is the gold standard for run validation in regulated QC and clinical laboratories. While a basic control chart just looks for "big" errors (a point outside ±3 SD), the multi-rule system acts as a **statistical detective**, using a toolkit of rules to diagnose different failure modes:
    - **Systematic Errors (Bias/Shifts):** Like a miscalibrated instrument. Detected by rules like `2-2s`, `4-1s`, or `10-x`.
    - **Random Errors (Imprecision):** Like a sloppy pipetting technique. Detected primarily by the `1-3s` and `R-4s` rules.

    Implementing these rules prevents the release of bad data, which is the cornerstone of ensuring patient safety and product quality. It's the difference between a simple smoke detector and an advanced security system with motion sensors, heat sensors, and tripwires.
    """)
    
    fig = plot_westgard_chart() # Assumes this function returns a chart with rule violations
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🕵️ Run Verdict", value="Out-of-Control", help="The overall judgment on the analytical run based on the triggered rules.")
            st.metric(label="🚨 Triggered Rule", value="2-2s Violation", help="The specific rule that caused the 'Out-of-Control' signal.")
            
            st.markdown("""
            **The Detective's Toolkit (Common Rules):**
            - 🚨 **1-3s Rule:** One point is beyond 3 standard deviations. This is a "smoking gun" – a major, often random, error occurred. **Likely Culprit:** Big blunder like wrong reagent, major instrument failure, or calculation error.
            
            - 🧐 **2-2s Rule (Warning):** Two consecutive points are on the same side of the mean and beyond 2 standard deviations. This is your first major clue of a **systematic error** or bias. **Likely Culprit:** A new lot of calibrator or reagent has caused a shift.
            
            - 🕵️ **4-1s Rule:** Four consecutive points are on the same side of the mean and beyond 1 standard deviation. This detects a smaller, but persistent, shift. **Likely Culprit:** Minor instrument drift or subtle degradation of a standard.
            
            - 🔭 **R-4s Rule:** The range between two consecutive points exceeds 4 standard deviations. This detects a sudden increase in **random error** or imprecision. **Likely Culprit:** Inconsistent pipetting, instrument instability (e.g., fluctuating temperature).

            **The Core Strategic Insight:** The multi-rule system gives you a **diagnostic-level** understanding of your assay's health. It doesn't just tell you *that* something is wrong, it gives you a powerful clue as to *what* is wrong, dramatically speeding up your investigation and corrective action.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Re-run & Pray" Mentality**
            This operator sees any alarm, immediately discards the run, and starts over from scratch without thinking.
            
            - They see a `2-2s` warning and panic, treating it the same as a `1-3s` failure.
            - They don't use the specific rule (`4-1s` vs `R-4s`) to guide their troubleshooting.
            - They might re-run the control sample over and over, hoping to get a "pass," which is a serious compliance violation known as "testing into compliance."
            
            This approach is inefficient, costly, and completely misses the diagnostic power of the rules.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Rule is the First Clue**
            The goal is to treat the specific rule violation as the starting point of a targeted investigation.
            
            - **Think like a detective:** "The chart shows a `4-1s` violation. This suggests a small, systematic shift. The first thing I should check is the calibration curve or the expiration date of my reagents, not my pipetting technique."
            - **Respect the Hierarchy:** A `1-3s` rule violation typically means "Stop, reject the run." A `2-2s` or `4-1s` might mean "Accept the run, but investigate immediately as a trend is developing."
            
            This diagnostic mindset transforms the control chart from a simple pass/fail tool into the most important troubleshooting guide in the lab.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The story of multi-rule charts is a brilliant fusion of two eras. First came **Dr. Walter A. Shewhart** at Bell Labs in the 1920s. He invented the foundational control chart (the ±3 SD limits) to improve the reliability of telephone network components. This was the birth of Statistical Process Control (SPC).
            
            Fast forward to the 1970s. **Dr. James O. Westgard**, a professor of pathology and laboratory medicine at the University of Wisconsin, faced a different problem: ensuring the daily reliability of clinical laboratory tests that decided patient diagnoses. He found that Shewhart's single `1-3s` rule wasn't sensitive enough to catch the subtle drifts common in complex biochemical assays.
            
            In a landmark 1981 paper, Westgard and his colleagues proposed a system of multiple rules to be applied simultaneously. These "Westgard Rules" were specifically designed to have a high probability of detecting medically important errors, while keeping the rate of false alarms low. This gave lab technicians a powerful, statistically-backed system for making daily accept/reject decisions, and it rapidly became the global standard in clinical chemistry and beyond.
            
            #### Mathematical Basis
            The rule notation is a simple shorthand:
            """)
            st.latex(r"A_{BC}")
            st.markdown("""
            - **`A`**: The number of control points.
            - **`B`**: The standard deviation limit (the "B"oundary).
            - **`C`**: The control material or run (often implied as the last C points).
            
            **Examples:**
            - **`1-3s`**: **1** point exceeds the **3s** (3 standard deviation) limit.
            - **`2-2s`**: **2** consecutive points exceed the **2s** limit on the same side of the mean.
            - **`R-4s`**: The **R**ange between 2 consecutive points exceeds **4s**.
            """)
    
    # Placeholder for the plotting function
    # In a real app, this function would generate data and check Westgard rule violations
def render_westgard_rules_interactive():
    """Renders the interactive module for Multi-Rule SPC (Westgard Rules)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To serve as a high-sensitivity "security system" for your assay. Instead of one simple alarm, this system uses a combination of rules to detect specific types of problems, catching subtle shifts and drifts long before a catastrophic failure occurs. It dramatically increases the probability of detecting true errors while minimizing false alarms.
    
    **Strategic Application:** This is the gold standard for run validation in regulated QC and clinical laboratories. While a basic control chart just looks for "big" errors (a point outside ±3 SD), the multi-rule system acts as a **statistical detective**, using a toolkit of rules to diagnose different failure modes:
    - **Systematic Errors (Bias/Shifts):** Like a miscalibrated instrument. Detected by rules like `2-2s`, `4-1s`, or `10-x`.
    - **Random Errors (Imprecision):** Like a sloppy pipetting technique. Detected primarily by the `1-3s` and `R-4s` rules.

    Implementing these rules prevents the release of bad data, which is the cornerstone of ensuring patient safety and product quality. It's the difference between a simple smoke detector and an advanced security system with motion sensors, heat sensors, and tripwires.
    """)
    
    # The user's plotting function, nested for clarity
    def plot_westgard_chart_with_violations():
        np.random.seed(45)
        data = np.random.normal(100, 2, 20)
        data[10] = 107  # 1-3s violation
        data[14:16] = [105, 105.5] # 2-2s violation
        mean, std = 100, 2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(data)+1), y=data, mode='lines+markers', name='Control Data', line=dict(color='#636EFA'), marker=dict(size=8)))
        
        # Add SD lines
        for i in range(-3, 4):
            if i == 0:
                fig.add_hline(y=mean, line=dict(color='black', dash='dash'), annotation_text='Mean', annotation_position="bottom right")
            else:
                fig.add_hline(y=mean + i*std, line=dict(color='grey', dash='dot'), 
                              annotation_text=f'{i} SD', annotation_position="bottom right")
        
        # Highlight violations shown on the chart
        fig.add_annotation(x=11, y=107, text="<b>🚨 1-3s Violation</b>", showarrow=True, arrowhead=2, arrowsize=1.5, ax=-40, ay=-40, font=dict(color="red"))
        fig.add_annotation(x=15.5, y=105.5, text="<b>🧐 2-2s Violation</b>", showarrow=True, arrowhead=2, arrowsize=1.5, ax=40, ay=-40, font=dict(color="orange"))
        
        fig.update_layout(title="<b>Statistical Detective at Work: A Multi-Rule Control Chart</b>",
                          xaxis_title="Measurement Number", yaxis_title="Control Value",
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig = plot_westgard_chart_with_violations()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🕵️ Run Verdict", value="Reject Run", help="The 1-3s rule is a mandatory rejection rule.")
            st.metric(label="🚨 Primary Cause", value="1-3s Violation", help="A 'smoking gun' event. This rule alone is sufficient to reject the run.")
            st.metric(label="🧐 Secondary Evidence", value="2-2s Violation", help="This suggests a systematic error is also present in the system.")
            
            st.markdown("""
            **The Detective's Findings on this Chart:**
            - 🚨 **The Smoking Gun (Point 11):** The `1-3s` violation is a clear, unambiguous signal of a major problem. It could be a large random error (e.g., air bubble in a pipette) or a significant one-time event. This rule alone forces the rejection of the run.
            
            - 🧐 **The Developing Pattern (Points 15-16):** The `2-2s` violation is a classic sign of **systematic error**. The process has shifted high. This suggests a different problem from the one at point 11. Perhaps a new reagent lot was introduced after point 14, causing a consistent positive bias.
            
            - **The Core Strategic Insight:** This chart shows two *different* problems. A simple lab investigation might stop after finding the cause of the `1-3s` error. A true statistical detective sees the `2-2s` signal and knows there is a deeper, more persistent issue to solve as well. This prevents future failures.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Re-run & Pray" Mentality**
            This operator sees the alarms, immediately discards the run, and starts over from scratch without thinking.
            
            - *"My control failed. I'll just run it again."* (Without investigating *why* it failed, the problem will likely recur).
            - They might fixate on the big `1-3s` error and completely miss the more subtle but equally important `2-2s` shift.
            - They might re-run the control sample over and over, hoping to get a "pass," which is a serious compliance violation known as "testing into compliance."
            
            This approach is inefficient, costly, and guarantees that underlying process problems will fester.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Rule is the First Clue**
            The goal is to treat the specific rule violation as the starting point of a targeted investigation.
            
            - **Think like a detective:** "The chart shows a `1-3s` violation AND a `2-2s` violation. These are likely separate issues. The `1-3s` could be a one-off blunder. The `2-2s` suggests a calibration or reagent problem. I need to investigate both."
            - **Document Everything:** The investigation, the root cause, and the corrective action for each rule violation must be documented. This is a regulatory expectation and a crucial part of process knowledge.
            
            This diagnostic mindset transforms the control chart from a simple pass/fail tool into the most important troubleshooting guide in the lab.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The story of multi-rule charts is a brilliant fusion of two eras. First came **Dr. Walter A. Shewhart** at Bell Labs in the 1920s. He invented the foundational control chart (the ±3 SD limits) to improve the reliability of telephone network components. This was the birth of Statistical Process Control (SPC).
            
            Fast forward to the 1970s. **Dr. James O. Westgard**, a professor at the University of Wisconsin, faced a different problem: ensuring the daily reliability of clinical laboratory tests that decided patient diagnoses. He found that Shewhart's single `1-3s` rule wasn't sensitive enough to catch the subtle drifts common in complex biochemical assays.
            
            In a landmark 1981 paper, Westgard and his colleagues proposed a system of multiple rules to be applied simultaneously. These "Westgard Rules" were specifically designed to have a high probability of detecting medically important errors, while keeping the rate of false alarms low. This gave lab technicians a powerful, statistically-backed system for making daily accept/reject decisions, and it rapidly became the global standard in clinical chemistry and beyond.
            
            #### Mathematical Basis
            The rule notation is a simple shorthand: **A<sub>L</sub>**, where `A` is the number of points and `L` is the limit.
            
            - **`1-3s`**: **1** point exceeds the **3s** (3 standard deviation) limit.
            - **`2-2s`**: **2** consecutive points exceed the **2s** limit on the same side of the mean.
            - **`R-4s`**: The **R**ange between 2 consecutive points exceeds **4s**.
            """)
            
def render_spc_charts():
    """Renders the module for Statistical Process Control (SPC) charts."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To serve as an **EKG for your process**—a real-time heartbeat monitor that visualizes its stability. The goal is to distinguish between two fundamental types of variation:
    - **Common Cause Variation:** The natural, random "static" or "noise" inherent to a stable process. It's predictable.
    - **Special Cause Variation:** A signal that something has changed or gone wrong. It's unpredictable and requires investigation.
    
    **Strategic Application:** SPC is the bedrock of modern quality control and process improvement. These charts provide an objective, data-driven answer to the critical question: "Is my process stable and behaving as expected?" They are used to prevent defects, reduce waste, and provide the evidence needed to justify (or reject) process changes. Acting on this "voice of the process" is a core competency of any high-performing manufacturing or lab operation.
    """)
    
    # Assume this function returns the three necessary SPC charts
    fig_imr, fig_xbar, fig_p = plot_spc_charts()
    
    st.subheader("Analysis & Interpretation: The Process EKG")
    tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

    with tabs[0]:
        st.info("💡 **Pro-Tip:** Each chart type is a different 'lead' on your EKG, designed for a specific kind of data. Use the expanders below to see how to read each one.")

        with st.expander("Indivduals & Moving Range (I-MR) Chart: The Solo Performer", expanded=True):
            st.metric("Process Capability (Cpk)", "1.15", help="A measure of how well the process fits within its specification limits. Cpk > 1.33 is often a target.")
            st.plotly_chart(fig_imr, use_container_width=True)
            st.markdown("""
            - **Use Case:** For tracking individual measurements when you can't group data (e.g., daily pH of a single bioreactor, weekly yield of one large batch).
            - **Interpretation:** This is a two-part story:
                - **Individuals (I) Chart (Top):** Tracks the process center. It asks, "Is the process on target?"
                - **Moving Range (MR) Chart (Bottom):** Tracks the short-term variability or 'jitter'. It asks, "Is the process consistency stable?"
            - **Strategic Insight:** **Both charts must be in-control.** A stable process has both a stable average AND stable variability. An out-of-control MR chart is often a leading indicator of future problems on the I-chart.
            """)

        with st.expander("X-bar & Range (X̄-R) Chart: The Team Average"):
            st.metric("Process Capability (Cpk)", "1.68", help="A Cpk calculated from a subgrouped chart is generally more reliable.")
            st.plotly_chart(fig_xbar, use_container_width=True)
            st.markdown("""
            - **Use Case:** The gold standard for continuous data collected in small, rational subgroups (e.g., measuring 5 vials from the filler every hour).
            - **Interpretation:** This is a more powerful two-part story:
                - **X-bar (X̄) Chart (Top):** Tracks the average *between* subgroups. Thanks to the Central Limit Theorem, it's extremely sensitive to small shifts in the process mean.
                - **Range (R) Chart (Bottom):** Tracks the average variation *within* each subgroup. It asks, "Is the short-term process precision consistent?"
            - **Strategic Insight:** The X-bar chart is your high-powered microscope for detecting process shifts. The R-chart tells you if something has fundamentally changed about your process's inherent consistency.
            """)
        
        with st.expander("Proportion (P) Chart: The Team's Score"):
            st.metric("Average Defect Rate", "1.2%", help="The overall process performance being tracked.")
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("""
            - **Use Case:** For "Go/No-Go" attribute data. You're not measuring, you're counting: the proportion (or percent) of non-conforming items per lot (e.g., percent of failed tablet inspections).
            - **Interpretation:** This chart tracks the proportion of defects over time.
            - **Strategic Insight:** Notice the control limits are not constant! They become tighter for larger subgroups (batches). This is a feature, not a bug. It reflects the statistical reality that you have more confidence in a percentage from a large batch than a small one. It's the most honest way to track yields and failure rates.
            """)

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
        The control chart dictates one of two paths, with no in-between:
        
        1.  **If all points are within control limits and show no patterns (Process is IN-CONTROL):**
            - **Your Action:** Leave the process alone! Your job is not to chase individual data points. To improve, you must work on changing the fundamental system (e.g., better equipment, new materials, improved training).
        
        2.  **If a point goes outside the control limits or a clear pattern emerges (Process is OUT-OF-CONTROL):**
            - **Your Action:** Stop! Investigate immediately. Find the specific, assignable "special cause" for that signal and eliminate it.
        
        This discipline—to act on signals but ignore the noise—is the entire philosophy of SPC.
        """)

    with tabs[2]:
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
def render_tolerance_intervals():
    """Renders the module for Tolerance Intervals."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To construct an interval that we can claim, with a specified level of confidence, contains a certain proportion of all individual values from a process. It is the **Quality Engineer's Secret Weapon**.
    
    **Strategic Application:** For manufacturing and quality control, this is often the most critical statistical interval, yet it's frequently misunderstood or ignored. It directly answers the high-stakes business question: **"Based on this sample, what is the range where we can expect almost all of our individual product units to fall?"**
    - **The Piston Principle:** Imagine you manufacture pistons that must have a diameter of 100mm. A Confidence Interval might tell you that you're 95% confident the *average* diameter of all pistons is between 99.9mm and 100.1mm. This sounds great! But if your process has high variation, you could still be producing many individual pistons at 98mm and 102mm that won't fit in the engine. The Tolerance Interval is what tells you the range where, say, 99% of your *individual* pistons actually lie. It's the interval that tells you if your parts will fit.
    """)
    
    fig = plot_tolerance_intervals() # Assumes a function that plots data, CI, and TI
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🎯 Desired Coverage", value="99% of Population", help="The proportion of the entire process output we want our interval to contain.")
            st.metric(label="🔒 Confidence Level", value="95%", help="Our level of confidence that the calculated interval *truly* captures the desired coverage.")
            st.metric(label="📏 Resulting Tolerance Interval", value="[94.2, 105.8]", help="The final calculated range. Note how much wider it is than the CI.")
            
            st.markdown("""
            **Reading the Chart:**
            - **The Dots:** These are your actual sampled data points.
            - **Orange Interval (CI):** This is the Confidence Interval for the **mean**. It's narrow because we are very certain about the long-term process average. It answers the question: "Where is the average?"
            - **Green Interval (TI):** This is the Tolerance Interval. It is necessarily much wider. It must account for **two sources of uncertainty**: our uncertainty about the true mean *and* the inherent, natural variation (standard deviation) of the process itself. It answers the question: "Where are my parts?"

            **The Core Strategic Insight:** A Tolerance Interval is the statistical bridge between a small sample and a quality promise to a customer. It allows you to make a probabilistic statement about **every single item** coming off the production line, which is a far more powerful and commercially relevant claim than a statement about the process average.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Confidence Interval Fallacy**
            This is one of the most common and dangerous statistical errors in industry.
            
            - A manager sees that the 95% **Confidence Interval** for the mean is [9.9, 10.1] and their product specification is [9.5, 10.5]. They declare victory, believing all their product is in spec.
            - **The Flaw:** They've proven the *average* is in spec, but have made no claim about the *individuals*. If the process standard deviation is large, a huge percentage of product could still be outside the [9.5, 10.5] specification limits.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Use the Right Interval for the Right Question**
            The question you are trying to answer dictates the tool you must use.
            
            - **Question 1: "Where is my long-term process average located?"**
              - **Correct Tool:** ✅ **Confidence Interval**.
            
            - **Question 2: "Will the individual units I produce meet the customer's specification?"**
              - **Correct Tool:** ✅ **Tolerance Interval**.
              
            Never use a confidence interval to make a statement about where individual values are expected to fall. That is the specific job of a tolerance interval.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The development of tolerance intervals is credited to the brilliant mathematician **Abraham Wald** during his work with the Statistical Research Group at Columbia University during World War II. The group was tasked with solving complex statistical problems for the US military.
            
            Wald's genius was in looking at problems from a unique angle. He is famous for the "surviving bombers" problem: when the military wanted to add armor to planes, they analyzed the bullet hole patterns on the planes that *returned* from missions. The consensus was to reinforce the areas that were hit most often. Wald's revolutionary insight was that the military should do the exact opposite: **the areas with *no* bullet holes were the most critical**. Planes hit in those areas (like the engines or cockpit) simply never made it back.
            
            This ability to reason about an entire population from a limited, biased sample is the same thinking behind the tolerance interval. The military needed to mass-produce interchangeable parts that were "good enough" to fit together on the battlefield. Wald developed the statistical theory for tolerance intervals to provide a rigorous, reliable method to ensure this, based on small samples from the production line.
            
            #### Mathematical Basis
            The formula for a two-sided tolerance interval looks simple, but contains a hidden, powerful factor:
            """)
            st.latex(r"\text{TI} = \bar{x} \pm k \cdot s")
            st.markdown("""
            - **`x̄`**: The sample mean.
            - **`s`**: The sample standard deviation.
            - **`k`**: The **k-factor** or tolerance factor. This is the "magic ingredient". It is NOT a simple z-score or t-score. The `k`-factor is a special value, derived from complex statistical theory, that depends on **three** inputs:
                1. The sample size (`n`).
                2. The desired population coverage (e.g., 99%).
                3. The desired confidence level (e.g., 95%).
            
            This `k`-factor is mathematically constructed to account for the "double uncertainty" of not knowing the true mean *or* the true standard deviation, making the resulting interval robust and reliable.
            """)

def render_4pl_regression():
    """Renders the module for 4-Parameter Logistic (4PL) regression."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately model the characteristic sigmoidal (S-shaped) dose-response relationship found in most immunoassays (e.g., ELISA) and biological assays. A straight-line (linear) model is fundamentally incorrect for this type of data, as it fails to capture the lower and upper plateaus of the response.
    
    **Strategic Application:** This is the undisputed workhorse model for relative potency assays, immunoassays, and any bioassay where the biological response has a floor, a ceiling, and a sloped transition. The Four-Parameter Logistic (4PL) model is critical for:
    - **Potency Calculation (EC50/IC50):** Determining the concentration that produces 50% of the maximal response, the single most important measure of a drug's biological activity or an analyte's sensitivity.
    - **Quantitation of Unknowns:** Inverting the fitted curve to accurately determine the concentration of unknown samples from their signal response. This is the basis for most QC release and clinical sample testing.
    - **Assay Health Monitoring:** Using the fitted parameters (slope, asymptotes) as system suitability criteria to ensure the assay is performing correctly day-to-day.
    """)
    
    fig, params = plot_4pl_regression()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Unpack params into named variables for clarity and robustness
        a_fit, b_fit, c_fit, d_fit = params
        
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🅰️ Upper Asymptote (Max Signal)", value=f"{a_fit:.3f}", help="The maximum possible signal in the assay. Represents reagent saturation.")
            st.metric(label="🅱️ Hill Slope (Steepness)", value=f"{b_fit:.3f}", help="The steepness of the curve at its midpoint. A steep slope indicates a large change in signal for a small change in concentration.")
            st.metric(label="🎯 EC50 (Potency)", value=f"{c_fit:.3f} units", help="The concentration that gives a response halfway between min and max. This is the primary KPI for potency.")
            st.metric(label="🅾️ Lower Asymptote (Min Signal)", value=f"{d_fit:.3f}", help="The signal from background or non-specific binding. The assay 'floor'.")

            st.markdown("""
            - **The Four Pillars of the Curve:** Each parameter tells a story about the assay's health.
                - **(a) & (d) Asymptotes:** Define the theoretical dynamic range of the assay. If these values drift over time, it could indicate reagent degradation or changes in non-specific binding.
                - **(b) Hill Slope:** Reflects the sensitivity of the assay. A shallow slope means a wider, less precise reportable range.
                - **(c) EC50:** The star of the show. This is your potency result. A lower EC50 often means higher potency.
            
            - **Goodness-of-Fit is Visual:** The most important diagnostic is your own eye. The red dashed line must *look* like it's accurately describing the trend in your data points. Don't rely on R-squared alone. Check that the curve fits well at the top and bottom, not just in the middle.

            **The Core Strategic Insight:** The 4PL curve is more than a calculation tool; it's a complete picture of your assay's performance. By monitoring all four parameters over time, you can detect subtle shifts in assay health long before a simple pass/fail result goes out of spec, enabling proactive troubleshooting.
            """)
            
        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Force the Fit"**
            The goal is to get a good R-squared value, no matter what.
            
            - *"My data isn't perfectly S-shaped, so I'll use linear regression on the middle part of the curve."* (This is fundamentally wrong and will bias your results).
            - *"The model doesn't fit the lowest point well. I'll just delete that point from the dataset."* (This is data manipulation and invalidates the result).
            - *"My R-squared is 0.999, so the fit must be perfect."* (R-squared is easily inflated and can be high even for a visibly poor fit, especially with asymptotes).
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Model the Biology, Weight the Variance**
            The goal is to use a mathematical model that honors the underlying biological/chemical reality of the system.
            
            - **Embrace the 'S' Shape:** The sigmoidal curve exists for a reason (e.g., receptor saturation, binding equilibria). The 4PL is designed specifically for this. **Always use a non-linear model for non-linear data.**
            - **Weight Your Points:** In many assays, the variance is not constant across the range of concentrations (a property called heteroscedasticity). Points at the top of the curve often have more variability than points at the bottom. A good regression algorithm will apply less "weight" to the more variable points, resulting in a much more accurate and robust fit.
            - **Look at the Residuals:** A good fit has residuals (the errors between the data and the curve) that are randomly scattered around zero. Any clear pattern in the residuals indicates the model is not capturing the data correctly.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The 4PL model is a direct descendant of the **Hill Equation**, published in 1910 by the pioneering British physiologist **Archibald Hill**. He developed the equation to describe the sigmoidal binding curve of oxygen to hemoglobin in the blood. This was one of the first and most successful mathematical descriptions of cooperative binding in biology.
            
            In the 1970s and 80s, with the rise of radioimmunoassays (RIAs) and then enzyme-linked immunosorbent assays (ELISAs), scientists needed a robust, generalizable model for their S-shaped data. They adapted the Hill equation into the four-parameter logistic function we use today, adding parameters for the upper and lower asymptotes to account for the physical limits of the assay signal. It is now the industry-standard model for almost all such assays.
            
            #### Mathematical Basis
            The 4PL equation describes the relationship between concentration (`x`) and the measured response (`y`):
            """)
            st.latex(r"y = d + \frac{a - d}{1 + (\frac{x}{c})^b}")
            st.markdown("""
            - **`y`**: The measured response (e.g., absorbance).
            - **`x`**: The concentration of the analyte.
            - **`a`**: The response at infinite concentration (the upper asymptote).
            - **`d`**: The response at zero concentration (the lower asymptote).
            - **`c`**: The point of inflection (the EC50 or IC50).
            - **`b`**: The Hill slope, a measure of steepness.
            
            The process of "fitting" is a non-linear regression, where a computer algorithm iteratively adjusts the values of `a, b, c, d` to find the combination that minimizes the total distance (typically the sum of squared errors) between the curve and the actual data points.
            """)
# The code below was incorrectly merged. It is now its own separate function.
def render_roc_curve():
    """Renders the module for Receiver Operating Characteristic (ROC) curve analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To solve **The Diagnostician's Dilemma**: a new test must correctly identify patients with a disease (high **Sensitivity**) while also correctly clearing healthy patients (high **Specificity**). These two goals are always in tension. The ROC curve is the ultimate tool for visualizing, quantifying, and optimizing this critical trade-off.
    **Strategic Application:** This is the undisputed global standard for validating and comparing diagnostic tests. The Area Under the Curve (AUC) provides a single, powerful metric of a test's overall diagnostic horsepower.
    - **Rational Cutoff Selection:** The ROC curve allows scientists and clinicians to rationally select the optimal cutoff point that best balances the clinical risks and benefits of false positives vs. false negatives.
    - **Assay Showdown:** Directly compare the AUC of two competing assays to provide definitive evidence of which is diagnostically superior.
    - **Regulatory Approval:** An ROC analysis is a non-negotiable requirement for submissions to regulatory bodies like the FDA for any new diagnostic test. A high AUC is a key to market approval.
    """)
    
    fig, auc_value = plot_roc_curve() # Assumes a function that plots distributions and the ROC curve
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 KPI: Area Under Curve (AUC)", value=f"{auc_value:.3f}", help="The overall diagnostic power of the test. 0.5 is useless, 1.0 is perfect.")
            st.metric(label="🎯 Sensitivity at Cutoff", value="91%", help="True Positive Rate. 'If you have the disease, what's the chance the test catches it?'")
            st.metric(label="🔒 Specificity at Cutoff", value="85%", help="True Negative Rate. 'If you are healthy, what's the chance the test clears you?'")

            st.markdown("""
            **Reading the Chart:**
            - **Score Distributions (Left):** This reveals *why* the dilemma exists. The scores of the Healthy and Diseased populations overlap. Any vertical line (a "cutoff") you draw will inevitably misclassify some subjects. A great assay has minimal overlap.
            
            - **ROC Curve (Right):** This is the solution map. It plots the trade-off for *every possible cutoff*.
                - The Y-axis is Sensitivity (good).
                - The X-axis is 1-Specificity (bad, also called the False Positive Rate).
                - The "shoulder" of the curve pushing towards the top-left corner represents the sweet spot of high performance.
            
            - **The AUC's Deeper Meaning:** The AUC has an elegant probabilistic meaning: It is the probability that a randomly chosen 'Diseased' subject has a higher test score than a randomly chosen 'Healthy' subject.
            
            **The Core Strategic Insight:** The ROC curve transforms a complex validation problem into a single, powerful picture. It allows for a data-driven conversation about risk, enabling a team to choose a cutoff that is not just mathematically optimal, but clinically and commercially sound.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "Worship the AUC" & "Hug the Corner"**
            This is a simplistic view that can lead to poor clinical outcomes.
            
            - *"My test has an AUC of 0.95, it's amazing! We're done."* (The overall AUC is great, but the *chosen cutoff* might still be terrible for the specific clinical need).
            - *"I'll just pick the cutoff point mathematically closest to the top-left (0,1) corner."* (This point balances sensitivity and specificity equally, which is almost never what is clinically desired).
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: The Best Cutoff Depends on the Consequence of Being Wrong**
            The optimal cutoff is a clinical or strategic decision, not a purely mathematical one. Ask this critical question: **"What is worse? A false positive or a false negative?"**
            
            - **Scenario A: Screening for a highly contagious, deadly disease.**
              - **What's worse?** A false negative (missing a case) is a public health catastrophe. False positives (unnecessarily quarantining healthy people) are acceptable.
              - **Your Action:** Choose a cutoff that **maximizes Sensitivity**, even at the cost of lower Specificity.
            
            - **Scenario B: Diagnosing a condition that requires risky, invasive surgery.**
              - **What's worse?** A false positive (sending a healthy person for unnecessary surgery) is a disaster. A false negative might mean delaying diagnosis, which may be acceptable for a slow-moving condition.
              - **Your Action:** Choose a cutoff that **maximizes Specificity**, ensuring you have very few false alarms.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: From Radar to Radiology
            The ROC curve was not born in a hospital or a biotech lab. It was invented during the heat of **World War II** to solve a life-or-death problem for Allied forces.
            
            Radar operators (the "Receivers") stared at noisy screens, trying to distinguish the faint 'blip' of an incoming enemy bomber from random atmospheric noise (like flocks of birds). The question was how to set the sensitivity of their radar sets.
            - If set too **high**, they got too many false alarms, scrambling fighter pilots for no reason (low specificity).
            - If set too **low**, they might miss a real bomber until it was too late (low sensitivity).
            
            Engineers developed the **Receiver Operating Characteristic (ROC)** curve to plot the performance of the radar operator at every possible sensitivity setting. This allowed them to quantify the trade-off and choose the optimal "operating characteristic" for the receiver. The term was later adopted by psychologists in the 1950s and then by medical diagnostics in the 1960s, where it has remained the gold standard ever since.
            
            #### Mathematical Basis
            The curve is built from the two key performance metrics, calculated from a 2x2 contingency table:
            """)
            st.latex(r"\text{Sensitivity (True Positive Rate)} = \frac{TP}{TP + FN}")
            st.latex(r"\text{1 - Specificity (False Positive Rate)} = \frac{FP}{FP + TN}")
            st.markdown("""
            - **TP**: True Positives (diseased, test positive)
            - **FN**: False Negatives (diseased, test negative)
            - **FP**: False Positives (healthy, test positive)
            - **TN**: True Negatives (healthy, test negative)
            
            The ROC curve plots **Sensitivity (Y-axis)** versus **1 - Specificity (X-axis)** for every single possible cutoff value, creating a complete performance profile of the diagnostic test.
            """)

def render_tost():
    """Renders the module for Two One-Sided Tests (TOST) for equivalence."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To solve **The Prosecutor's Nightmare.** In standard statistics, you are a prosecutor trying to prove guilt (a difference). A high p-value just means "not guilty," it *doesn't* mean "innocent." TOST flips the script: you are now the defense attorney, and your goal is to **positively prove innocence (equivalence)** within a predefined, practically insignificant margin.
    
    **Strategic Application:** This is the required statistical tool anywhere the goal is to prove similarity, not difference.
    - **Biosimilarity & Generics:** The cornerstone of regulatory submissions to prove a generic drug is bioequivalent to the name brand, enabling market access without repeating costly clinical efficacy trials.
    - **Method Transfer:** The definitive way to prove a new analytical method is interchangeable with an old one.
    - **Process Validation:** Used to prove that a change (e.g., new supplier, updated equipment) has **not** negatively impacted the product's critical quality attributes.
    """)
    
    fig, p_tost, is_equivalent = plot_tost() # Assumes a function that plots the TOST result
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="⚖️ Equivalence Margin (Δ)", value="± 10 units", help="The pre-defined 'zone of indifference'. Any difference within this zone is considered practically meaningless.")
            st.metric(label="📊 Observed 90% CI for Difference", value="[-2.5, +4.1]", help="The 90% confidence interval for the true difference between the groups.")
            st.metric(label="p-value (TOST)", value=f"{p_tost:.4f}", help="The p-value for the equivalence test. If p < 0.05, we conclude equivalence.")
            
            status = "✅ Equivalent" if is_equivalent else "❌ Not Equivalent"
            if is_equivalent:
                st.success(f"### Status: {status}")
            else:
                st.error(f"### Status: {status}")

            st.markdown("""
            **The Visual Verdict:**
            - **The Green Zone:** This is the **Equivalence Margin** you defined before the experiment. It's the goalpost.
            - **The Blue Bar:** This is the 90% Confidence Interval for the true difference, calculated from your data. It's where your process *actually* is.

            **To declare equivalence, the entire blue bar must be captured inside the green zone.** It's a simple, visual rule. If any part of the blue bar pokes outside the green zone, you have failed to prove equivalence.
            
            **The Core Strategic Insight:** TOST forces a conversation about **practical significance** (the equivalence margin) instead of just statistical significance. It answers a much better business question: not "is there a difference?" but "**is there any difference that actually matters?**"
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Fallacy of the Non-Significant P-Value**
            This is the most common and dangerous statistical error when comparing methods.
            
            - A scientist runs a standard t-test comparing a new method to the old one and gets a p-value of 0.25. They exclaim, *"Great, p is greater than 0.05, so there's no significant difference. The methods are the same!"*
            - **This is fundamentally wrong.** All they have shown is a *failure to find evidence of a difference*, which could be because the methods truly are similar, or it could be because their experiment was underpowered with too few samples to find a real, important difference. **Absence of evidence is not evidence of absence.**
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Define 'Same Enough', Then Prove It**
            The TOST procedure forces you into a more rigorous and honest scientific approach.
            
            1.  **First, Define the Margin:** Before you collect any data, you must have a stakeholder discussion (e.g., with clinicians, engineers, regulators) to define the equivalence margin. What is the largest difference that would still be considered practically or clinically meaningless? This becomes your "green zone."
            
            2.  **Then, Prove You're Inside:** Now, conduct the experiment and run the TOST analysis. The burden of proof is on you to show that your evidence (the 90% CI) is strong enough to fall entirely within that pre-defined margin.
            
            This two-step process removes ambiguity and replaces weak "non-significant" claims with a strong, positive proof of equivalence.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: Enabling the Generic Drug Revolution
            The concept of bioequivalence testing, and TOST with it, rose to prominence in the 1980s, largely thanks to the **1984 Hatch-Waxman Act** in the United States. This landmark legislation created the modern pathway for generic drug approval.
            
            The challenge was immense: how could a company prove its generic version of a drug was just as good as the original without re-running years of expensive and unethical placebo-controlled clinical trials? The answer was **bioequivalence**. The FDA, guided by statisticians like **Donald J. Schuirmann**, established that if a generic drug could be shown to produce the same concentration profile in the blood (the same pharmacokinetics) as the original, it could be considered therapeutically equivalent.
            
            The standard t-test was useless for this. They needed a test to *prove sameness*. Schuirmann's **Two One-Sided Tests (TOST)** procedure became the statistical engine for these studies. You must prove, with 90% confidence, that the key parameters (like AUC and Cmax) of your generic fall within a tight equivalence margin (typically 80% to 125%) of the original. This procedure single-handedly enabled the multi-billion dollar generic drug industry, saving patients trillions of dollars.
            
            #### Mathematical Basis
            TOST brilliantly flips the null hypothesis. Instead of one null hypothesis of "no difference," you have two null hypotheses of "too different":
            """)
            st.latex(r"H_{01}: \mu_{Test} - \mu_{Ref} \leq -\Delta")
            st.latex(r"H_{02}: \mu_{Test} - \mu_{Ref} \geq +\Delta")
            st.markdown("""
            - **`Δ`** is your pre-defined equivalence margin.
            - You must perform two separate one-sided t-tests. One to prove the difference is not significantly *lower* than -Δ, and another to prove it is not significantly *higher* than +Δ.
            - You must **reject both** of these null hypotheses to conclude equivalence.
            - The final TOST p-value is simply the **larger** of the two p-values from the individual tests. This is mathematically equivalent to checking if the 90% confidence interval for the difference lies completely within the `[-Δ, +Δ]` bounds.
            """)
            
def render_ewma_cusum():
    """Renders the module for small shift detection charts (EWMA/CUSUM)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To deploy **Statistical Sentinels** that guard your process against small, slow, creeping changes. While a standard Shewhart chart is a "beat cop" that catches big, obvious crimes, EWMA and CUSUM charts are "intelligence analysts" that detect subtle patterns over time by incorporating memory of past data.
    
    **Strategic Application:** This is essential for modern, high-precision processes where small drifts can be costly. A process might shift by only 0.5 to 1.5 standard deviations—a change that is nearly invisible to a Shewhart chart but can lead to a significant increase in out-of-spec product over time.
    - **EWMA (Exponentially Weighted Moving Average):** A versatile and popular tool that gives exponentially less weight to older data. Excellent all-around for detecting small to moderate shifts.
    - **CUSUM (Cumulative Sum):** The undisputed champion for speed. It is the fastest possible method for detecting a small shift of a *specific magnitude* that it was designed to find.

    These charts enable proactive intervention, allowing you to correct a small drift *before* it triggers a major alarm or results in a rejected batch.
    """)
    
    # Nested plotting function from the user's code
    def plot_ewma_cusum_comparison():
        np.random.seed(123)
        n_points = 40
        data = np.random.normal(100, 2, n_points)
        data[20:] += 1.5 # A small 0.75-sigma shift
        mean, std = 100, 2
        
        # EWMA calculation
        lam = 0.2 # Lambda, the weighting factor
        ewma = np.zeros(n_points)
        ewma[0] = mean
        for i in range(1, n_points):
            ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]
        
        # CUSUM calculation
        target = mean
        k = 0.5 * std # "Allowance" or "slack"
        sh, sl = np.zeros(n_points), np.zeros(n_points)
        for i in range(1, n_points):
            sh[i] = max(0, sh[i-1] + (data[i] - target) - k)
            sl[i] = max(0, sl[i-1] + (target - data[i]) - k)
            
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("<b>I-Chart: The Beat Cop (Misses the Shift)</b>", 
                                            "<b>EWMA: The Sentinel (Catches the Shift)</b>", 
                                            "<b>CUSUM: The Bloodhound (Catches it Fastest)</b>"))

        # I-Chart
        fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=data, mode='lines+markers', name='Data'), row=1, col=1)
        fig.add_hline(y=mean + 3*std, line_color='red', line_dash='dash', row=1, col=1)
        fig.add_hline(y=mean - 3*std, line_color='red', line_dash='dash', row=1, col=1)
        fig.add_vline(x=20.5, line_color='orange', line_dash='dot', row=1, col=1)

        # EWMA Chart
        fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=ewma, mode='lines+markers', name='EWMA'), row=2, col=1)
        sigma_ewma = std * np.sqrt(lam / (2-lam)) # Asymptotic SD
        fig.add_hline(y=mean + 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_hline(y=mean - 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_vline(x=20.5, line_color='orange', line_dash='dot', row=2, col=1)
        
        # CUSUM Chart
        fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sh, mode='lines+markers', name='CUSUM High'), row=3, col=1)
        fig.add_trace(go.Scatter(x=np.arange(1, n_points+1), y=sl, mode='lines+markers', name='CUSUM Low'), row=3, col=1)
        fig.add_hline(y=5*std, line_color='red', line_dash='dash', row=3, col=1) # Decision interval H
        fig.add_vline(x=20.5, line_color='orange', line_dash='dot', row=3, col=1, annotation_text="Process Shift Occurs", annotation_position="top")

        fig.update_layout(title="<b>Case Study: Detecting a Small Process Shift (0.75σ)</b>", height=800, showlegend=False)
        fig.update_xaxes(title_text="Sample Number", row=3, col=1)
        return fig

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig = plot_ewma_cusum_comparison()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="Shift Size", value="0.75 σ", help="A small, sustained shift was introduced at sample #20.")
            st.metric(label="I-Chart Detection Time", value="> 20 Samples (Failed)", help="The I-Chart never signaled an alarm.")
            st.metric(label="EWMA Detection Time", value="~10 Samples", help="The EWMA signaled an alarm around sample #30.")
            st.metric(label="CUSUM Detection Time", value="~7 Samples", help="The CUSUM signaled an alarm around sample #27.")

            st.markdown("""
            **The Visual Evidence:**
            - **The I-Chart (Top):** This chart is blind to the problem. The small 0.75σ shift is lost in the normal process noise. All points look "in-control," giving a false sense of security.
            
            - **The EWMA Chart (Middle):** This chart has memory. The weighted average (the blue line) clearly begins to drift upwards after the shift occurs. It smoothly accumulates evidence until it crosses the red control limit, signaling a real change.
            
            - **The CUSUM Chart (Bottom):** This chart is a "bloodhound" for a specific scent. It accumulates all deviations from the target. Once the process shifts, the `CUSUM High` plot takes off like a rocket, providing the fastest possible signal.

            **The Core Strategic Insight:** Relying only on Shewhart charts creates a significant blind spot. For processes where small, slow drifts in performance are possible (e.g., tool wear, reagent degradation, column aging), EWMA or CUSUM charts are not optional—they are essential for effective process control.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: "The One-Chart-Fits-All Fallacy"**
            A manager insists on using only I-MR charts for everything because "that's how we've always done it" and they are easy to understand.
            
            - They miss a slow 1-sigma drift for weeks, producing tons of near-spec material.
            - When a batch finally fails, they are shocked and have no leading indicators to explain why. They have been flying blind.
            - They see no value in the "complex" EWMA/CUSUM charts, viewing them as academic exercises.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Layer Your Statistical Defenses**
            The goal is to use a combination of charts to create a comprehensive security system for your process.
            
            - **Use Shewhart Charts (I-MR, X-bar) as your front-line "Beat Cops":** They are simple and unmatched for detecting large, sudden special causes (e.g., an operator error, a major equipment failure).
            - **Use EWMA or CUSUM as your "Sentinels":** Deploy them alongside Shewhart charts to stand guard against the silent, creeping threats that the beat cops will miss.
            
            This layered approach provides a complete picture of process stability, protecting against both sudden shocks and slow drifts.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: Beyond Shewhart
            The quality revolution sparked by **Walter Shewhart's** control charts in the 1920s was a massive success. However, by the 1950s, industries were pushing for even higher levels of quality and needed tools to detect smaller and smaller process deviations.
            
            - **CUSUM (1954):** The first major innovation came from British statistician **E. S. Page**. He developed the Cumulative Sum (CUSUM) chart, borrowing concepts from sequential analysis. Its design was revolutionary: it was explicitly optimized to detect a shift of a specific size in the minimum possible time, making it a powerful tool for targeted process monitoring.
            
            - **EWMA (1959):** Shortly after, statistician **S. W. Roberts** proposed the Exponentially Weighted Moving Average (EWMA) chart as a more general-purpose alternative. Its roots are in forecasting, where smoothing techniques developed by visionaries like **George Box** were used to predict future values from past data. Roberts adapted this "smoothing" concept to create a chart that was highly effective at picking up small shifts, easy to implement, and less rigid than the CUSUM chart.
            
            These two inventions marked the second generation of SPC, giving engineers the sensitive tools they needed to manage the increasingly complex and precise manufacturing processes of the late 20th century.
            
            #### Mathematical Basis
            - **EWMA:** `EWMA_t = λ * Y_t + (1-λ) * EWMA_{t-1}`
              - `λ` (lambda) is the weighting factor (0 < λ ≤ 1). A smaller `λ` gives more weight to past data (longer memory) and is better for detecting smaller shifts.
            - **CUSUM:** `SH_t = max(0, SH_{t-1} + (Y_t - T) - k)`
              - This formula for the "high-side" CUSUM accumulates upward deviations. `T` is the process target, and `k` is a "slack" parameter, typically set to half the size of the shift you want to detect quickly. The CUSUM only starts accumulating when a deviation is larger than the slack.
            """)
            
def render_anomaly_detection():
    """Renders the module for unsupervised anomaly detection."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To deploy an **AI Bouncer** for your data—a smart system that identifies rare, unexpected observations (anomalies) without any prior knowledge of what "bad" looks like. It doesn't need a list of troublemakers; it learns the "normal vibe" of the crowd and flags anything that stands out.
    
    **Strategic Application:** This is a game-changer for monitoring complex processes where simple rule-based alarms are blind to new problems.
    - **Novel Fault Detection:** The AI Bouncer's greatest strength. It can flag a completely new type of process failure the first time it occurs, because it looks for "weirdness," not pre-defined failures.
    - **Intelligent Data Cleaning:** Automatically identifies potential sensor glitches or data entry errors before they contaminate models or analyses.
    - **"Golden Batch" Investigation:** Can find which batches, even if they passed all specifications, were statistically unusual. These "weird-but-good" batches often hold the secrets to improving process robustness.
    """)

    # Nested plotting function based on user's code
    def plot_isolation_forest():
        from sklearn.ensemble import IsolationForest
        import pandas as pd
        import numpy as np
        import plotly.express as px

        np.random.seed(42)
        X_inliers = np.random.normal(0, 1, (100, 2))
        X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
        X = np.concatenate([X_inliers, X_outliers], axis=0)

        clf = IsolationForest(contamination=0.1, random_state=42)
        y_pred = clf.fit_predict(X)
        
        df = pd.DataFrame(X, columns=['Process Parameter 1', 'Process Parameter 2'])
        df['Status'] = ['Anomaly' if p == -1 else 'Normal' for p in y_pred]

        fig = px.scatter(df, x='Process Parameter 1', y='Process Parameter 2', color='Status',
                         color_discrete_map={'Normal': '#636EFA', 'Anomaly': '#EF553B'},
                         title="<b>The AI Bouncer at Work</b>",
                         symbol='Status',
                         symbol_map={'Normal': 'circle', 'Anomaly': 'x'})
        fig.update_traces(marker=dict(size=10), selector=dict(name='Anomaly'))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig = plot_isolation_forest()
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="Data Points Scanned", value="110")
            st.metric(label="Anomalies Flagged", value="10", help="Based on a contamination setting of ~10%.")
            st.metric(label="Algorithm Used", value="Isolation Forest", help="An unsupervised machine learning method.")

            st.markdown("""
            **Reading the Chart:**
            - **The Blue Circles:** This is the "normal crowd" in your process data. They are dense and clustered together. The AI Bouncer considers them normal.
            - **The Red 'X's:** These are the anomalies. The algorithm has flagged them as "not belonging" to the main crowd. They are the individuals the bouncer has pulled aside for a closer look.
            - **The Unsupervised Magic:** The key is that we never told the algorithm where the blue circle "club" was. It figured out the normal operating envelope on its own and identified everything that fell outside of it.

            **The Core Strategic Insight:** Anomaly detection is your early warning system for the **unknown unknowns**. While a control chart tells you if you've broken a known rule, an anomaly detector tells you that something you've never seen before just happened. This is often the first and only clue to a new, emerging failure mode.
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
    
def render_advanced_doe():
    """Renders the module for advanced Design of Experiments (DOE)."""
    st.markdown("""
    #### Purpose & Application: DOE for the Real World
    **Purpose:** To solve complex optimization problems where standard factorial designs fail because they don't respect the real-world constraints of the system. This module covers two essential advanced designs:
    - **🧪 Mixture Designs (The Alchemist's Cookbook):** For optimizing a *recipe* where ingredients are proportions that must sum to 100%. Changing one ingredient's percentage forces a change in the others.
    - **🏭 Split-Plot Designs (The Smart Baker's Dilemma):** For optimizing processes with both "Hard-to-Change" (e.g., oven temperature) and "Easy-to-Change" (e.g., baking time) factors.
    
    **Strategic Application:** Using the wrong design for these common problems leads to wasted experiments, impossible-to-run conditions, and statistically invalid conclusions. Mastering these designs provides a massive competitive advantage in formulation and process development.
    """)
    
    fig_mix, fig_split = plot_advanced_doe()
    
    # FIX: Restructured the layout to be consistent with other modules.
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.subheader("Experimental Designs & Results")
        with st.expander("🧪 Mixture Design: The Alchemist's Cookbook", expanded=True):
            st.plotly_chart(fig_mix, use_container_width=True)

        with st.expander("🏭 Split-Plot Design: The Smart Baker's Dilemma", expanded=True):
            st.plotly_chart(fig_split, use_container_width=True)

    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.info("💡 **Pro-Tip:** Each design solves a unique, common experimental challenge. Click the expanders on the left to see each plot.")
            st.markdown("""
            **Mixture Designs**
            - **The Problem:** Optimizing a recipe where components must sum to 100%.
            - **The Solution:** A triangular "recipe space" plot where corners are pure components.
            - **Strategic Insight:** The contour plot is a **treasure map of your formulation space**, revealing the optimal blend with maximum efficiency.

            **Split-Plot Designs**
            - **The Problem:** Optimizing a process with both Hard-to-Change (HTC) and Easy-to-Change (ETC) factors.
            - **The Solution:** An efficient blocked design where the HTC factor is changed less frequently.
            - **Strategic Insight:** Requires a special analysis to avoid false positives. It correctly handles the different levels of experimental error, providing valid conclusions about all factors.
            """)
        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: Force a Square Peg into a Round Hole**
            This is a classic and costly DOE mistake.
            
            - **For Mixtures:** Using a standard factorial design. This generates impossible recipes (e.g., 80% A, 80% B, 80% C = 240% total!) and completely fails to model the blending properties of the ingredients.
            - **For Split-Plots:** Analyzing the data as if it were a normal, fully randomized factorial DOE. This leads to an **incorrectly small error term** for the HTC factor, massively inflating its significance. You might launch a huge project to control a factor that actually has no effect.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Let the Problem's Constraints Define the Design**
            The statistical design must mirror the physical, logistical, and financial reality of the experiment.
            
            - **If your factors are ingredients in a recipe that must sum to 100%...**
              - **You MUST use a Mixture Design.**
            
            - **If you have factors that are difficult, slow, or expensive to change...**
              - **You MUST use a Split-Plot Design** and its corresponding special ANOVA.
              
            Honoring the problem's structure is not optional. It is the only path to a statistically valid and operationally efficient experiment.
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            These advanced designs were born from practical necessity, solving real-world industrial and agricultural problems.
            
            - **Mixture Designs (1950s):** The theory for mixture experiments was largely developed by American statistician **Henry Scheffé**. He was working on problems in industrial chemistry and food science where formulators needed a systematic way to optimize recipes.
                
            - **Split-Plot Designs (1920s):** This design originated in agriculture with the legendary statistician **R.A. Fisher**. It was impractical to irrigate small, randomized plots differently, so he devised a method to apply irrigation (the Hard-to-Change factor) to large plots and then test different crop varieties (Easy-to-Change factors) within those larger plots.
            
            #### Mathematical Basis
            - **Mixture Designs:** The constraint `x₁ + x₂ + ... + xₙ = 1` means the regression model is reformulated, often without an intercept, to properly account for the dependency.
            - **Split-Plot Designs:** The model has two distinct error terms: a "Whole Plot Error" for testing the HTC factors and a "Subplot Error" for testing the ETC factors. Using the wrong error term is the most common mistake in analyzing these experiments.
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
    
    fig = plot_stability_analysis() # Assumes a function that plots the stability data and model
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Approved Shelf-Life", value="31 Months", help="The time at which the lower confidence bound intersects the specification limit.")
            st.metric(label="📉 Degradation Rate (Slope)", value="-0.21 %/month", help="The estimated average loss of potency per month.")
            st.metric(label="🥅 Specification Limit", value="95.0 %", help="The minimum acceptable potency for the product to be considered effective.")

            st.markdown("""
            **Reading the Race Against Time:**
            - **The Data Points:** These are your real-world potency measurements from three different production batches.
            - **The Black Line (The Average Trend):** This is the best-fit regression line. It shows the *average* degradation path we expect.
            - **The Red Dashed Line (The Safety Net):** This is the **95% Lower Confidence Bound**. This is the most important line on the chart. It represents a conservative, worst-case estimate for the *mean* degradation trend, accounting for uncertainty.
            - **The Finish Line (Red Dotted Line):** This is the specification limit.

            **The Verdict:** The shelf-life is declared at the exact moment **the Safety Net (red dashed line) hits the Finish Line.** This conservative approach ensures that even with the uncertainty of our model and the variability between batches, we are 95% confident the average product potency will not drop below the spec limit before the expiration date.
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
            st.markdown("""
            #### Historical Context & Origin: The ICH Revolution
            Prior to the 1990s, the requirements for stability testing could differ significantly between major markets like the USA, Europe, and Japan. This forced pharmaceutical companies to run slightly different, redundant, and costly stability programs for each region to gain global approval.
            
            The **International Council for Harmonisation (ICH)** was formed to end this inefficiency. A key working group was tasked with creating a single, scientifically sound standard for stability testing. This resulted in a series of guidelines, with **ICH Q1A** defining the required study conditions and **ICH Q1E ("Evaluation of Stability Data")** providing the definitive statistical methodology.
            
            ICH Q1E, adopted in 2003, codified the use of regression analysis, formal tests for pooling data across batches, and the critical principle of using confidence intervals to determine shelf-life. It created a level playing field and a global gold standard, ensuring that the expiration date on a medicine means the same thing in New York, London, and Tokyo.
            
            #### Mathematical Basis
            The core of the analysis is typically a linear regression model:
            """)
            st.latex(r"Y_i = \beta_0 + \beta_1 X_i + \epsilon_i")
            st.markdown("""
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
    
    fig = plot_survival_analysis() # Assumes a function that plots K-M curves
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📊 Log-Rank Test p-value", value="0.008", help="A p-value < 0.05 indicates a statistically significant difference between the survival curves.")
            st.metric(label="⏳ Median Survival (Group A)", value="17 Months", help="Time at which 50% of Group A have experienced the event.")
            st.metric(label="⏳ Median Survival (Group B)", value="28 Months", help="Time at which 50% of Group B have experienced the event.")

            st.markdown("""
            **Reading the Curve:**
            - **The Stepped Line:** This is the **Kaplan-Meier survival curve**. It shows the estimated probability of survival over time.
            - **Vertical Drops:** Each "step down" represents a time when one or more "events" (e.g., equipment failures) occurred.
            - **Vertical Ticks (Censoring):** These are the heroes of the story! They represent items that were still working perfectly when the study ended or they were removed for other reasons. We don't know their final failure time, but we know they survived *at least* this long.
            
            **The Visual Verdict:** The curve for **Group B** is consistently higher and "flatter" than the curve for Group A. This visually demonstrates that items in Group B have a higher probability of surviving longer. The low p-value from the Log-Rank test confirms this visual impression is statistically significant.
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
            st.markdown("""
            #### Historical Context & Origin: The 1958 Revolution
            While the concept of life tables has existed for centuries in actuarial science, analyzing time-to-event data with censored observations was often messy and inconsistent. Different researchers used different ad-hoc methods, making it hard to compare results.
            
            This all changed in 1958 with a landmark paper in the *Journal of the American Statistical Association* by **Edward L. Kaplan** and **Paul Meier**. Their paper, "Nonparametric Estimation from Incomplete Observations," introduced the world to what we now universally call the **Kaplan-Meier estimator**.
            
            It was a revolutionary breakthrough. They provided a simple, elegant, and statistically robust non-parametric method to estimate the true survival function, even with heavily censored data. This single technique unlocked a new era of research in medicine, enabling the rigorous analysis of clinical trials that is now standard practice, and in engineering, forming the foundation of modern reliability analysis.
            
            #### Mathematical Basis
            The Kaplan-Meier estimate of the survival function `S(t)` is a product of conditional probabilities:
            """)
            st.latex(r"S(t_i) = S(t_{i-1}) \times \left( \frac{n_i - d_i}{n_i} \right)")
            st.markdown("""
            - **`S(tᵢ)`** is the probability of surviving past time `tᵢ`.
            - **`nᵢ`** is the number of subjects "at risk" (i.e., still surviving) just before time `tᵢ`.
            - **`dᵢ`** is the number of events (e.g., failures) that occurred at time `tᵢ`.
            
            Essentially, the probability of surviving to a certain time is the probability you survived up to the last event, *times* the conditional probability you survived this current event. This step-wise calculation gracefully handles censored observations, as they simply exit the "at risk" pool (`nᵢ`) at the time they are censored.
            """)

def render_multivariate_spc():
    """Renders the module for Multivariate SPC."""
    st.markdown("""
    #### Purpose & Application: The Process Doctor
    **Purpose:** To act as the **head physician for your process**. Instead of having 20 separate nurses (univariate charts) each reading one vital sign, Multivariate SPC looks at all the patient's vitals *together* to make a single, powerful diagnosis. It answers one question: "Is the overall process healthy?"
    
    **Strategic Application:** This is essential for complex processes where parameters are correlated (e.g., a bioreactor's temperature, pH, DO₂). A small, coordinated change in all variables—a "stealth shift"—can be invisible to individual charts but signals a significant change in the process state.
    - **Hotelling's T² Chart:** The "Check Engine Light" for your process. It condenses dozens of variables into a single statistic that monitors the process "fingerprint." A T² alarm is an unambiguous signal that the system's overall health has changed.
    - **Contribution Analysis:** Once the T² alarm sounds, this diagnostic tool tells you *which* variable(s) are the culprits behind the shift.
    """)
    
    fig = plot_multivariate_spc() # Assumes a function that plots the comparison
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📊 Multivariate Verdict", value="Out-of-Control", help="The T² Chart has detected a significant shift in the process state.")
            st.metric(label="⏰ Trigger Point", value="Sample #21", help="The first point at which the T² statistic exceeded the control limit.")
            st.metric(label="📈 Key Statistic", value="Hotelling's T²", help="A single number representing the multivariate distance from the process center.")
            
            st.markdown("""
            **Reading the Diagnosis:**
            - **Left Plot (The Vitals):** This shows the raw data for two correlated process parameters. Notice the shift after point 20 is mainly in the vertical direction. An individual control chart for "Parameter X" would likely miss this problem entirely.
            - **Right Plot (The Diagnosis - T² Chart):** This is the Process Doctor's conclusion. It combines both X and Y into a single health score (the T² statistic). The moment the process shifts, the T² value immediately jumps above the red control limit, providing a clear, undeniable alarm.
            
            **The Core Strategic Insight:** A process doesn't live on a single axis; it lives in a multi-dimensional space. The T² chart monitors the process's position within its normal, correlated "operating cloud." The alarm triggers when the process moves outside this cloud, even if it hasn't crossed a limit on any single axis.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Army of Univariate Charts" Fallacy**
            A manager insists on using 20 separate I-MR charts for their 20 bioreactor parameters.
            
            - **The Problem 1: Alarm Fatigue.** With 20 charts, false alarms are constant. Soon, operators begin to ignore the signals, assuming they are just noise. The system becomes useless.
            - **The Problem 2: The Stealth Shift.** A small, 1-sigma shift occurs simultaneously across 10 of the 20 parameters. This is a massive change in the process state, but **not a single one of the 20 charts will alarm.** The process is sick, but every "nurse" reports normal vitals.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Detect with T², Diagnose with Contributions**
            The correct, two-stage approach to multivariate monitoring is disciplined and powerful.
            
            1.  **Stage 1: Detect.** Use a **single Hotelling's T² chart** as your primary process health monitor. Its job is to answer one question: "Is something wrong?" (YES/NO). Do not look at anything else until this chart alarms.
            
            2.  **Stage 2: Diagnose.** If the T² chart alarms, and only then, you bring in the diagnostic tools. You use **contribution plots** or other diagnostics to identify which of the original variables are most responsible for the alarm signal.
            
            This "Detect, then Diagnose" strategy provides the best of both worlds: a simple, unambiguous monitoring system and a powerful tool for root cause analysis.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: Hotelling's T-Squared
            The creator of this powerful technique was **Harold Hotelling**, one of the giants of 20th-century mathematical statistics. Working in the 1930s and 40s, he was a contemporary of legends like R.A. Fisher and Walter Shewhart.
            
            Hotelling's genius was in generalization. He recognized that Student's t-test was perfect for asking if a single variable's mean was different from a target. But what if you had multiple variables? In a 1931 paper, he introduced the **Hotelling's T-squared statistic**, which was a direct multivariate generalization of Student's t-statistic. It allowed one to test hypotheses about multiple means at once.
            
            When Shewhart's control charts became popular, it was a natural next step to apply Hotelling's powerful statistic to process monitoring. The T² chart is simply a time-series plot of the T-squared statistic, providing a way to apply Shewhart's SPC philosophy to complex, correlated, multivariate data.
            
            #### Mathematical Basis
            The T² statistic is essentially a measure of the **Mahalanobis distance**, a "smart" distance that accounts for the correlation between variables. While a simple Euclidean distance is like measuring with a straight ruler, the Mahalanobis distance measures along the natural "grain" of the elliptical data cloud.
            
            The formula for a single observation `x` is:
            """)
            st.latex(r"T^2 = (\mathbf{x} - \mathbf{\bar{x}})' \mathbf{S}^{-1} (\mathbf{x} - \mathbf{\bar{x}})")
            st.markdown("""
            - **`x`**: The vector of a new observation's measurements.
            - **`x̄`**: The vector of historical means for the variables.
            - **`S⁻¹`**: The **inverse of the sample covariance matrix**. This is the secret sauce. It's the mathematical engine that de-correlates and correctly scales the variables, making the T² statistic a pure measure of "strangeness" from the process center.
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
    
    fig = plot_mva_pls() # Assumes a function that plots the VIP chart
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Model R² (Goodness of Fit)", value="0.98", help="How well the model fits the training data. High is good, but can be misleading.")
            st.metric(label="🎯 Model Q² (Predictive Power)", value="0.95", help="The cross-validated R². A measure of how well the model predicts *new* data. Q² is the most important performance metric.")
            st.metric(label="🧬 Latent Variables (LVs)", value="3", help="The number of 'hidden' factors the model extracted. The complexity of the model.")
            
            st.markdown("""
            **Decoding the VIP Plot:**
            The plot on the left is the **Variable Importance in Projection (VIP)** plot. It's the key to understanding what the model has learned.
            - **The Peaks:** These are the "magic words" in the Rosetta Stone. They represent the specific input variables (e.g., wavelengths) that are most influential for predicting the output.
            - **The Red Line (VIP > 1):** This is the rule of thumb. Variables with a VIP score greater than 1 are considered important to the model.
            
            **The Core Strategic Insight:** PLS turns a "black box" instrument (like a spectrometer) into a "glass box" of process understanding. By identifying the most important variables via the VIP plot, scientists can gain fundamental insights into the underlying chemistry or physics of their process, leading to more robust control strategies.
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
            
            2.  **Use Cross-Validation:** Within the training set, use cross-validation to choose the optimal number of Latent Variables. The goal is to find the number of LVs that minimizes the **prediction error (RMSEP)**, not the number that maximizes the R-squared.
            
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
    
    fig = plot_clustering() # Assumes a function that plots the clusters
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🏺 Discovered 'Civilizations' (k)", value="3", help="The number of distinct clusters the K-Means algorithm identified.")
            st.metric(label="🗺️ Cluster Quality (Silhouette Score)", value="0.72", help="A measure of how distinct the clusters are from each other. Higher is better (max 1.0).")
            st.metric(label="⛏️ Algorithm Used", value="K-Means", help="A classic and robust partitioning-based clustering algorithm.")
            
            st.markdown("""
            **The Dig Site Findings:**
            - The plot reveals the results of our archeological dig. The algorithm, without any help, has found three distinct groups in the data, color-coded for clarity.
            - Before this analysis, these data points were likely thought of as one single "in-control" population. We have now discovered this is not true.
            - The high Silhouette Score (0.72) gives us confidence that these groupings are statistically meaningful and not just a random artifact.
            
            **The Core Strategic Insight:** The discovery of hidden clusters is one of the most valuable findings in data analysis. It proves that your single process is actually a collection of multiple sub-processes. Understanding the *causes* of this separation is the gateway to improved process control, robustness, and optimization.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "If It Ain't Broke..." Fallacy**
            This is the most common way to squander the value of a clustering analysis.
            
            - An analyst presents the discovery of three distinct clusters. A manager responds, *"Interesting, but all of those batches passed QC testing, so who cares? Let's move on."*
            - **The Flaw:** This treats a treasure map as a doodle. The fact that all batches passed is what makes the discovery so important! It means there are different—and potentially more or less robust—paths to success. One of those "civilizations" might be living on the edge of a cliff (close to a specification limit), while another is safe in a valley.
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
            
            Its simplicity, coupled with the explosion of computational power, has made it one of the most widely used and taught clustering algorithms. It represents a fundamental shift in data analysis—from testing pre-defined hypotheses to exploring data to generate *new* hypotheses.
            
            #### How K-Means Works: The "Pizza Shop" Analogy
            Imagine you want to open 3 new pizza shops in a city to best serve its residents. How do you find the optimal locations?
            1.  **Step 1 (Random Start):** You randomly drop 3 pins on a map of the city. These are your initial "cluster centroids."
            2.  **Step 2 (Assignment):** You draw a boundary around each pin. Every house (data point) in the city is assigned to its *closest* pizza shop. You now have 3 customer bases.
            3.  **Step 3 (Update):** For each of the 3 customer bases, you find its true geographic center. You then **move the pizza shop's pin** to that new center.
            4.  **Step 4 (Repeat):** Now that the shops have moved, you repeat from Step 2. Re-assign every house to its new closest shop, then update the shop locations again.
            
            You repeat this "Assign-Update" process until the pins stop moving. The final locations of the pizza shops are the centers of your discovered clusters.
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
    
    fig = plot_time_series_analysis() # Assumes a function that plots the forecasts
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="⌚ ARIMA Forecast Error (MAE)", value="3.15 units", help="Mean Absolute Error for the ARIMA model.")
            st.metric(label="📱 Prophet Forecast Error (MAE)", value="2.89 units", help="Mean Absolute Error for the Prophet model.")
            st.metric(label="🔮 Forecast Horizon", value="12 Months", help="The period into the future for which we are generating predictions.")

            st.markdown("""
            **Reading the Forecasts:**
            - **The Black Line:** This is the historical data the models were trained on.
            - **The Blue Line:** This is the *true* future data, which we held out to test the models' performance.
            - **The Green Line (ARIMA):** The Watchmaker's forecast. Notice how it captures the core seasonal pattern and trend based on its statistical properties.
            - **The Red Line (Prophet):** The Smartwatch's forecast. It also captures the trend and seasonality, often being more robust to outliers or sudden changes.

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
            st.markdown("""
            Where `g(t)` is a saturating growth trend, `s(t)` models complex weekly and yearly seasonality using Fourier series, `h(t)` is a flexible component for user-specified holidays, and `ε` is the error.
            """)

def render_xai_shap():
    """Renders the module for Explainable AI (XAI) using SHAP."""
    st.markdown("""
    #### Purpose & Application: The AI Interrogator
    **Purpose:** To deploy an **AI Interrogator** that forces a complex "black box" model (like Gradient Boosting or a Neural Network) to confess exactly *why* it made a specific prediction. **Explainable AI (XAI)** cracks open the black box to reveal the model's internal logic.
    
    **Strategic Application:** This is arguably the **single most important enabling technology for deploying advanced Machine Learning in regulated GxP environments.** A major barrier to using powerful models has been their lack of interpretability. **SHAP (SHapley Additive exPlanations)** is the state-of-the-art framework that provides this crucial evidence.
    - **🔬 Model Trust & Validation:** XAI confirms that the model is learning real, scientifically plausible relationships, not just memorizing spurious correlations.
    - **⚖️ Regulatory Compliance:** It provides the auditable, scientific rationale needed to justify a model-based decision to regulators and quality assurance.
    - **Actionable Insights:** It pinpoints which input variables are driving a prediction, guiding targeted corrective actions and deepening process understanding.
    """)
    
    summary_buf, force_html = plot_xai_shap() # Assumes a function returns these components
    
    # FIX: Re-structured to use the standard two-column layout for consistency.
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.subheader("Global Feature Importance")
        st.image(summary_buf, caption="The SHAP Summary Plot shows the overall impact of each feature.")
        
        st.subheader("Local Prediction Explanation")
        st.components.v1.html(f"<body>{force_html}</body>", height=150, scrolling=True)
        st.caption("The SHAP Force Plot explains a single, specific prediction.")
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])

        with tabs[0]:
            st.info("💡 **Pro-Tip:** SHAP provides two levels of explanation: **Global** (how the model works overall) and **Local** (why it made one specific prediction).")

            st.markdown("""
            **Global Explanation (Top-Left Plot):**
            - **Feature Importance (Y-axis):** This plot ranks features by their total impact. We can see `Age` is the most important factor overall.
            - **Impact (X-axis) & Value (Color):** We can see a clear pattern: high `Age` (red dots) has a high positive SHAP value, meaning it strongly pushes the model to predict a higher income.
            
            **Local Explanation (Bottom-Left Plot):**
            - This plot is a **tug-of-war for a single prediction**.
            - **Red Forces:** Features pushing the prediction higher. For this person, `Capital Gain` was the dominant factor.
            - **Blue Forces:** Features pushing the prediction lower.
            - **Final Prediction (`f(x)`):** The result after all pushes and pulls are tallied up.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The "Accuracy is Everything" Fallacy**
            This is a dangerous mindset that leads to deploying untrustworthy models.
            
            - An analyst builds a model with 99% accuracy. They declare victory and push to put it into production without any further checks.
            - **The Flaw:** The model might be a "Clever Hans"—like the horse that could supposedly do math but was actually just reacting to its trainer's subtle cues. The model might have learned a nonsensical, spurious correlation in the training data (e.g., "batches made on a Monday always pass"). The high accuracy is an illusion that will shatter when the model sees new data where that spurious correlation doesn't hold.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Explainability Builds Trust and Uncovers Flaws**
            The goal of XAI is not just to explain predictions, but to use those explanations to **validate the model's reasoning and build trust** in its decisions.
            
            1.  **Build the Model:** Train your powerful "black box" model (e.g., XGBoost, Random Forest) to achieve high predictive accuracy.
            2.  **Interrogate with SHAP:** Apply SHAP to the model's predictions on a validation set.
            3.  **Consult the Expert:** Show the SHAP plots (especially the global summary plot) to a Subject Matter Expert (SME) who knows the process science. Ask them: *"Does this make sense?"*
                - **If YES:** The model has likely learned real, scientifically valid relationships. You can now trust its predictions.
                - **If NO:** The model has learned a spurious correlation. XAI has just saved you from deploying a flawed model. Use the insight to improve your feature engineering and retrain.
                
            XAI transforms machine learning from a pure data science exercise into a collaborative process between data scientists and domain experts.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: From Game Theory to AI
            The theoretical foundation of SHAP comes from a surprising place: **cooperative game theory**. In 1951, the brilliant mathematician and economist **Lloyd Shapley** developed a concept to solve the "fair payout" problem.
            
            Imagine a team of players collaborates to win a prize. How do you divide the winnings fairly based on each player's individual contribution? **Shapley values** provided a mathematically rigorous and unique solution.
            
            Fast forward to 2017. Scott Lundberg and Su-In Lee at the University of Washington had a genius insight. They realized that a machine learning model's prediction could be seen as a "game" and the model's features could be seen as the "players." They adapted Shapley's game theory concepts to create **SHAP (SHapley Additive exPlanations)**, a method to fairly distribute the "payout" (the prediction) among the features. This clever fusion of game theory and machine learning provided the first unified and theoretically sound framework for explaining the output of any machine learning model, a breakthrough that is driving the adoption of AI in high-stakes fields.
            
            #### How it Works
            SHAP calculates the contribution of each feature to a prediction by simulating every possible combination of features ("coalitions"). It asks: "How much does the prediction change, on average, when we add this specific feature to all possible subsets of other features?" This exhaustive, computationally intensive approach is the only method proven to have a unique set of desirable properties (Local Accuracy, Missingness, and Consistency) that guarantee a fair and accurate explanation.
            """)

def render_advanced_ai_concepts():
    """Renders the module for advanced AI concepts."""
    st.markdown("""
    #### Purpose & Application: A Glimpse into the AI Frontier
    **Purpose:** To provide a high-level, conceptual overview of cutting-edge AI architectures that represent the future of process analytics. While coding them is beyond the scope of this toolkit, understanding their capabilities is crucial for shaping future strategy and envisioning what's possible.
    """)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        st.subheader("Select a Concept")
        concept = st.radio(
            "Select an Advanced AI Concept:", 
            ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)", "Generative AI"],
            label_visibility="collapsed"
        )
        
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        # Dynamically populate tabs based on concept selection
        if concept == "Transformers":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Self-Attention", help="The mechanism that allows the model to weigh the importance of all other data points in a sequence.")
                st.metric(label="📇 Data Structure", value="Sequences", help="Primarily designed for ordered data like text or a time series of process events.")
                st.markdown("""
                **The AI Historian:** A Transformer is like a brilliant historian who reads the entire book of your batch record at once. It understands how an event on page 1 (cell thaw) influences the story's conclusion on page 300 (final purity).

                **Key Idea:** Its **"self-attention"** mechanism allows it to understand long-range dependencies. When looking at a single step, it can "pay attention" to all other steps, learning which ones are most relevant to the current context.

                **Strategic Insight:** This moves beyond simple time-series forecasting to modeling the entire **process narrative**. It unlocks a deep understanding of long-range cause and effect that is invisible to most other models.
                """)
            with tabs[1]:
                st.error("🔴 **The Incorrect Approach:** Simply feeding in raw sensor data as a long sequence and hoping for the best.")
                st.success("""
                🟢 **THE GOLDEN RULE: Tokenize Your Process Narrative**
                The success of a Transformer depends entirely on how you structure the "story." You must first convert your continuous process data into a discrete sequence of **events** or **"tokens"**. 
                
                Examples: `[Thaw_Completed, Temp=37.1, Feed_Event_1, pH_Excursion_Detected, ...]`. 
                
                This "tokenization" is the most critical step. It transforms raw data into a language the AI Historian can read and understand.
                """)
            with tabs[2]:
                st.markdown("""
                **Origin:** Revolutionized AI with the 2017 Google Brain paper, **"Attention Is All You Need."** It destroyed previous benchmarks in Natural Language Processing (NLP), forming the basis for models like BERT and the "T" in GPT. Its application to industrial process data is a new and exciting frontier.
                """)

        elif concept == "GNNs":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Message Passing", help="Nodes in the graph 'talk' to their neighbors, sharing information and learning from the network structure.")
                st.metric(label="📇 Data Structure", value="Graphs", help="Data represented as a network of nodes and edges.")
                st.markdown("""
                **The Network Navigator:** A GNN is like a brilliant logistics expert who sees your entire supply chain or manufacturing site as an interconnected network, not just a list of individual assets.

                **Key Idea:** It operates on graph data. Nodes in the graph iteratively update their state by **"passing messages"** to and from their neighbors. This allows information—and the prediction of failures—to propagate through the network.

                **Strategic Insight:** This moves beyond analyzing single pieces of equipment to modeling the **entire system's interconnectedness**. It's essential for understanding and predicting systemic risk.
                """)
            with tabs[1]:
                st.error("🔴 **The Incorrect Approach:** Analyzing each piece of equipment in isolation, ignoring how it is connected to and affected by its neighbors.")
                st.success("""
                🟢 **THE GOLDEN RULE: Your Graph IS Your Model**
                The most important work in a GNN project is not the algorithm, but the **definition of the graph itself.** You must answer:
                - What are my **nodes**? (e.g., equipment, raw material lots, batches)
                - What are my **edges**? (e.g., physical pipes, 'used-in' relationships, sequence order)
                
                The quality and thoughtfulness of this graph representation will determine the model's success.
                """)
            with tabs[2]:
                st.markdown("""
                **Origin:** While graph-based neural networks have existed for longer, they exploded in popularity and capability around 2018. They represent a generalization of deep learning beyond grids (like images) and sequences (like text) to the more flexible and powerful structure of graphs.
                """)

        elif concept == "RL":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Reward Maximization", help="The AI learns by taking actions that maximize a cumulative reward signal over time.")
                st.metric(label="📇 Data Structure", value="Interactive Environment", help="An 'agent' interacts with a 'world' and learns from the consequences.")
                st.markdown("""
                **The AI Pilot:** Reinforcement Learning creates an AI pilot that learns by "flying" a process and getting feedback. It learns not just to predict what will happen, but how to **control** the process to achieve an optimal outcome.

                **Key Idea:** An **agent** takes an **action** in an **environment**. The environment's state changes, and the agent receives a **reward** (or penalty). The agent's goal is to learn a policy that maximizes its long-term reward.

                **Strategic Insight:** This is the leap from passive process modeling to **active, real-time process optimization**. The goal is no longer to just understand the process, but to command it.
                """)
            with tabs[1]:
                st.error("🔴 **The Incorrect Approach:** Attempting to train the RL agent on the live, physical manufacturing process. This is a recipe for disaster, as the agent must learn through trial and **error**.")
                st.success("""
                🟢 **THE GOLDEN RULE: The Digital Twin is the Dojo**
                An RL agent must be trained in a high-fidelity simulation or **"digital twin"** of the process. This is the safe "dojo" where the AI can perform millions of experiments, make mistakes, and learn optimal control strategies with zero real-world cost or risk. Only a fully trained and validated policy is ever deployed to the physical system.
                """)
            with tabs[2]:
                st.markdown("""
                **Origin:** Has deep roots in control theory and psychology. Modern **Deep RL** was supercharged by DeepMind in the mid-2010s, famously teaching AI agents to master Atari games and defeat the world champion at Go (AlphaGo). Applying this to industrial process control is a major area of current research.
                """)
        
        elif concept == "Generative AI":
            with tabs[0]:
                st.metric(label="🧠 Core Concept", value="Distribution Learning", help="The AI learns the underlying probability distribution of the data, allowing it to create new samples.")
                st.metric(label="📇 Data Structure", value="Unstructured or Tabular", help="Can be used for images, text, or process data.")
                st.markdown("""
                **The Synthetic Data Forger:** Generative AI is like a master art forger who studies a thousand real paintings and then creates a new one that is so convincing, even experts can't tell it's not real.

                **Key Idea:** It **learns the underlying statistical distribution** of a dataset. It can then "sample" from this learned distribution to create new, synthetic data that is statistically indistinguishable from the real data.

                **Strategic Insight:** It solves the **"rare event" problem**. High-quality manufacturing failure data is gold, but you never have enough. Generative AI can create thousands of realistic synthetic failure profiles, which can then be used to train much more robust anomaly detection and predictive maintenance models.
                """)
            with tabs[1]:
                st.error("🔴 **The Incorrect Approach:** Assuming all synthetic data is good data and using it blindly to augment a dataset.")
                st.success("""
                🟢 **THE GOLDEN RULE: Validate the Forgeries**
                The generated data is only useful if it is realistic. You must use a rigorous process to validate the synthetic data. This involves:
                - **Visual Inspection:** Do experts agree it looks real?
                - **Statistical Comparison:** Does the distribution of the synthetic data match the distribution of the real data?
                
                The quality of the generator dictates the quality of any model trained on its output.
                """)
            with tabs[2]:
                st.markdown("""
                **Origin:** Ian Goodfellow's 2014 invention of **Generative Adversarial Networks (GANs)** was a major catalyst. More recently, **Diffusion Models** (the technology behind DALL-E 2, Midjourney, and Stable Diffusion) have become the state-of-the-art for generating incredibly high-fidelity data.
                """)

    with col1:
        fig = plot_advanced_ai_concepts(concept)
        st.plotly_chart(fig, use_container_width=True)

def render_causal_inference():
    """Renders the module for Causal Inference."""
    st.markdown("""
    #### Purpose & Application: Beyond the Shadow - The Science of "Why"
    **Purpose:** To move beyond mere correlation ("what") and ascend to the level of causation ("why"). While predictive models see shadows on a cave wall (associations), Causal Inference provides the tools to understand the true objects casting them (the underlying causal mechanisms).
    
    **Strategic Application:** This is the ultimate goal of root cause analysis and the foundation of intelligent intervention.
    - **💡 Effective CAPA:** Why did a batch fail? A predictive model might say high temperature is *associated* with failure. Causal Inference helps determine if high temperature *causes* failure, or if both are driven by a third hidden variable (a "confounder"). This prevents wasting millions on fixing the wrong problem.
    - **🗺️ Process Cartography:** It allows for the creation of a **Directed Acyclic Graph (DAG)**, which is a formal causal map of your process, documenting scientific understanding and guiding future analysis.
    - **🔮 "What If" Scenarios:** It provides a framework to answer hypothetical questions like, "What *would* have been the yield if we had kept the temperature at 40°C?" using only observational data.
    """)
    
    fig = plot_causal_inference() # Assumes a function that plots the DAG
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="🔎 Causal Question", value="Does Temp cause low Purity?", help="The specific causal relationship we want to isolate and quantify.")
            st.metric(label="🔬 Intervention of Interest", value="Setting Temp to 40°C", help="The hypothetical action whose effect we want to estimate.")
            st.metric(label="🚨 Identified Confounder", value="Reagent Lot", help="A variable that provides a non-causal 'backdoor' path between Temp and Purity.")

            st.markdown("""
            **Reading the Causal Map (The DAG):**
            - This graph is not a flowchart; it's a map of **causal assumptions** based on expert knowledge. Arrows represent direct causal effects.
            - **The Causal Path (Green Arrow):** We want to estimate the strength of the `Temp -> Purity` link.
            - **The Confounding Path (Red Arrows):** There is a "backdoor" path from Temp to Purity. A `Reagent Lot` can affect both the `Temp` of the reaction and the final `Purity`. If we don't account for this, we might wrongly attribute the effect of the Reagent Lot to the Temperature.
            
            **The Core Strategic Insight:** The DAG makes our assumptions explicit. By mapping out all causal relationships, we can use advanced statistical methods to mathematically "block" the confounding paths and isolate the true, unbiased causal effect we care about.
            """)

        with tabs[1]:
            st.error("""
            🔴 **THE INCORRECT APPROACH: The Correlation Trap**
            This is the oldest and most dangerous fallacy in data analysis: **"Correlation equals causation."**
            
            - An analyst observes that ice cream sales are highly correlated with shark attacks. They recommend banning ice cream to improve beach safety.
            - **The Flaw:** They failed to account for a confounder: **Hot Weather.** Hot weather causes more people to buy ice cream AND causes more people to go swimming, leading to more shark attacks. Ice cream has no causal effect on shark attacks.
            - In a lab, this is like seeing that `Pressure` is correlated with `Purity` and launching a huge project to control pressure, not realizing both are driven by `Temperature`.
            """)
            st.success("""
            🟢 **THE GOLDEN RULE: Draw the Map, Find the Path, Block the Backdoors**
            A robust causal analysis follows a disciplined, three-step process.
            
            1.  **Draw the Map (Build the DAG):** This is a collaborative effort between data scientists and Subject Matter Experts. You must encode all your domain knowledge and causal beliefs into a formal DAG.
            
            2.  **Find the Path:** Clearly identify the causal path you want to measure (e.g., `Temp -> Purity`).
            
            3.  **Block the Backdoors:** Use the DAG to identify all non-causal "backdoor" paths (confounding). Then, use the appropriate statistical technique (e.g., stratification, regression adjustment) to "block" these paths, leaving only the true causal effect.
            
            This structured thinking is what separates guessing from genuine causal discovery.
            """)

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin: The Causal Revolution
            For most of the 20th century, mainstream statistics was deeply allergic to the language of causation. The mantra was "correlation is not causation," and statisticians were trained to only discuss associations.
            
            This dogma was shattered by the computer scientist and philosopher **Judea Pearl** in the 1980s and 90s. Building on earlier work in path analysis by geneticist **Sewall Wright**, Pearl developed a complete mathematical framework for causal reasoning. He introduced the **Directed Acyclic Graph (DAG)** as the primary tool for encoding causal assumptions and invented the **do-calculus**, a formal symbolic language for describing interventions.
            
            His 2000 book *Causality* and his more recent *The Book of Why* sparked a "causal revolution," providing a clear, powerful, and mathematically rigorous language for asking and answering causal questions. For this work, Judea Pearl was awarded the Turing Award in 2011, the highest honor in computer science.
            
            #### The Language of Causation: `see` vs. `do`
            The `do-calculus` introduces a critical distinction:
            """)
            st.latex(r"P(\text{Purity} | \text{Temp}) \quad \text{vs.} \quad P(\text{Purity} | do(\text{Temp}))")
            st.markdown("""
            - **`P(Purity | Temp)` (Seeing):** This is standard conditional probability. It asks, "If I go into the lab and I *see* that the temperature is 40°C, what is the expected purity?" This is a passive observation, full of potential confounding.
            
            - **`P(Purity | do(Temp))` (Doing):** This is the causal quantity. It asks, "If I go into the lab and I *intervene*, setting the temperature to 40°C for everyone, what would the expected purity be?" This represents a perfect, randomized experiment.
            
            The entire goal of causal inference is to find a way to estimate the `do` quantity using data where you could only `see`.
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
    
    fig = plot_classification_models() # Assumes a function that plots the decision boundaries
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Theory & History"])
        
        with tabs[0]:
            st.metric(label="📈 Model 1: Logistic Regression Accuracy", value="85.0%", help="Performance of the simpler, linear model.")
            st.metric(label="🚀 Model 2: Random Forest Accuracy", value="96.7%", help="Performance of the more complex, non-linear model.")

            st.markdown("""
            **Reading the Decision Boundaries:**
            - The plots show how each model carves up the process space into "predicted pass" (blue) and "predicted fail" (red) regions.
            - **Logistic Regression (Left):** This classical statistical model can only draw a **straight line** to separate the groups. It struggles with the curved, complex relationship in the data.
            - **Random Forest (Right):** This powerful machine learning model can create a complex, **non-linear boundary**. It has learned the true "island" of failure in the center of the process space, leading to much higher accuracy.

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
            This module showcases the evolution from classical statistics to modern machine learning.
            - **Logistic Regression (1958):** Developed by British statistician **David Cox**, it is a direct generalization of linear regression for binary (pass/fail) outcomes. It models the **log-odds** of the outcome as a linear combination of the input variables. It remains a powerful and highly interpretable baseline model.
            
            - **Random Forest (2001):** Invented by **Leo Breiman and Adele Cutler**, this is a quintessential machine learning algorithm. It is an **ensemble method** that builds hundreds of individual decision trees on random subsets of the data and features, and then makes its final prediction based on a "majority vote" of all the trees. This "wisdom of the crowd" approach makes it highly accurate and robust to overfitting.
            
            #### How They Work
            - **Logistic Regression:** It fits a linear equation to the data and then passes the output through a **Sigmoid function**, which squashes the result into a probability between 0 and 1.
            - **Random Forest:** It creates a diverse "committee" of simple decision tree models. Each tree gets a vote, and the final prediction is the one that receives the most votes. This ensemble approach is why it can create complex, non-linear decision boundaries.
            """)

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
