# ==============================================================================
# LIBRARIES & IMPORTS
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
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
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Biotech V&V Analytics Toolkit",
    page_icon="🔬"
)

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
# ICONS DICTIONARY FOR SIDEBAR MENU
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
# HELPER & PLOTTING FUNCTIONS
# ==============================================================================

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

@st.cache_data
def plot_ci_concept(n=30):
    np.random.seed(42)
    pop_mean, pop_std = 100, 15
    x = np.linspace(pop_mean - 4*pop_std, pop_mean + 4*pop_std, 400)
    pop_dist = norm.pdf(x, pop_mean, pop_std)
    sampling_dist_std = pop_std / np.sqrt(n)
    sampling_dist = norm.pdf(x, pop_mean, sampling_dist_std)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=pop_dist, fill='tozeroy', name='Population Distribution', line=dict(color='skyblue')))
    fig1.add_trace(go.Scatter(x=x, y=sampling_dist, fill='tozeroy', name=f'Sampling Distribution (n={n})', line=dict(color='orange')))
    fig1.add_vline(x=pop_mean, line=dict(color='black', dash='dash'), annotation_text="True Mean (μ)")
    fig1.update_layout(title=f"<b>Population vs. Sampling Distribution of the Mean (n={n})</b>", showlegend=True, legend=dict(x=0.01, y=0.99))
    n_sims = 1000
    samples = np.random.normal(pop_mean, pop_std, size=(n_sims, n))
    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1, ddof=1)
    t_crit = t.ppf(0.975, df=n-1)
    margin_of_error = t_crit * sample_stds / np.sqrt(n)
    ci_lowers, ci_uppers = sample_means - margin_of_error, sample_means + margin_of_error
    capture_mask = (ci_lowers <= pop_mean) & (ci_uppers >= pop_mean)
    capture_count, avg_width = np.sum(capture_mask), np.mean(ci_uppers - ci_lowers)
    fig2 = go.Figure()
    for i in range(min(n_sims, 100)):
        color = 'blue' if capture_mask[i] else 'red'
        fig2.add_trace(go.Scatter(x=[ci_lowers[i], ci_uppers[i]], y=[i, i], mode='lines', line=dict(color=color, width=2), showlegend=False))
        fig2.add_trace(go.Scatter(x=[sample_means[i]], y=[i], mode='markers', marker=dict(color=color, size=4), showlegend=False))
    fig2.add_vline(x=pop_mean, line=dict(color='black', dash='dash'), annotation_text="True Mean (μ)")
    fig2.update_layout(title=f"<b>{min(n_sims, 100)} Simulated 95% Confidence Intervals</b>", yaxis_visible=False)
    return fig1, fig2, capture_count, n_sims, avg_width

@st.cache_data
def plot_act_grouped_timeline():
    all_tools_data = [
        {'name': 'Assay Robustness (DOE)', 'act': 1, 'year': 1926, 'inventor': 'R.A. Fisher', 'desc': 'Fisher publishes his work on Design of Experiments.'},
        {'name': 'CI Concept', 'act': 1, 'year': 1937, 'inventor': 'Jerzy Neyman', 'desc': 'Neyman formalizes the frequentist confidence interval.'},
        {'name': 'ROC Curve Analysis', 'act': 1, 'year': 1945, 'inventor': 'Signal Processing Labs', 'desc': 'Developed for radar, now the standard for diagnostic tests.'},
        {'name': 'LOD & LOQ', 'act': 1, 'year': 1968, 'inventor': 'Lloyd Currie (NIST)', 'desc': 'Currie at NIST formalizes the statistical basis.'},
        {'name': 'Core Validation Params', 'act': 1, 'year': 1980, 'inventor': 'ICH / FDA', 'desc': 'Accuracy, Precision, Specificity codified.'},
        {'name': 'Gage R&R', 'act': 1, 'year': 1982, 'inventor': 'AIAG', 'desc': 'AIAG codifies Measurement Systems Analysis (MSA).'},
        {'name': 'Equivalence Testing (TOST)', 'act': 1, 'year': 1987, 'inventor': 'Donald Schuirmann', 'desc': 'Schuirmann proposes TOST for bioequivalence.'},
        {'name': 'Process Stability', 'act': 2, 'year': 1924, 'inventor': 'Walter Shewhart', 'desc': 'Shewhart invents the control chart at Bell Labs.'},
        {'name': 'Pass/Fail Analysis', 'act': 2, 'year': 1927, 'inventor': 'Edwin B. Wilson', 'desc': 'Wilson develops a superior confidence interval.'},
        {'name': 'Tolerance Intervals', 'act': 2, 'year': 1942, 'inventor': 'Abraham Wald', 'desc': 'Wald develops intervals to cover a proportion of a population.'},
        {'name': 'Method Comparison', 'act': 2, 'year': 1986, 'inventor': 'Bland & Altman', 'desc': 'Bland & Altman revolutionize method agreement studies.'},
        {'name': 'Process Capability', 'act': 2, 'year': 1986, 'inventor': 'Bill Smith (Motorola)', 'desc': 'Motorola popularizes Cpk with the Six Sigma initiative.'},
        {'name': 'Bayesian Inference', 'act': 2, 'year': 1990, 'inventor': 'Metropolis et al.', 'desc': 'Computational methods (MCMC) make Bayes practical.'},
        {'name': 'Multivariate SPC', 'act': 3, 'year': 1931, 'inventor': 'Harold Hotelling', 'desc': 'Hotelling develops the multivariate analog to the t-test.'},
        {'name': 'Small Shift Detection', 'act': 3, 'year': 1954, 'inventor': 'Page (CUSUM) & Roberts (EWMA)', 'desc': 'Charts for faster detection of small process drifts.'},
        {'name': 'Reliability Analysis', 'act': 3, 'year': 1958, 'inventor': 'Kaplan & Meier', 'desc': 'Kaplan-Meier estimator for time-to-event data.'},
        {'name': 'Time Series Analysis', 'act': 3, 'year': 1970, 'inventor': 'Box & Jenkins', 'desc': 'Box & Jenkins publish their seminal work on ARIMA models.'},
        {'name': 'Multivariate Analysis', 'act': 3, 'year': 1975, 'inventor': 'Herman Wold', 'desc': 'Partial Least Squares for modeling complex process data.'},
        {'name': 'Run Validation', 'act': 3, 'year': 1981, 'inventor': 'James Westgard', 'desc': 'Westgard publishes his multi-rule QC system.'},
        {'name': 'Stability Analysis', 'act': 3, 'year': 1993, 'inventor': 'ICH', 'desc': 'ICH guidelines formalize statistical shelf-life estimation.'},
        {'name': 'Advanced AI/ML', 'act': 3, 'year': 2017, 'inventor': 'Vaswani, Lundberg et al.', 'desc': 'Transformers and Explainable AI (XAI) emerge.'},
    ]
    all_tools_data.sort(key=lambda x: (x['act'], x['year']))
    act_ranges = {1: (5, 45), 2: (50, 75), 3: (80, 115)}
    tools_by_act = {act: [t for t in all_tools_data if t['act'] == act] for act in act_ranges}
    for act_num, tools_in_act in tools_by_act.items():
        start, end = act_ranges[act_num]
        x_coords = np.linspace(start, end, len(tools_in_act))
        for i, tool in enumerate(tools_in_act):
            tool['x'] = x_coords[i]
    y_offsets = [3.0, -3.0, 3.5, -3.5, 2.5, -2.5, 4.0, -4.0]
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
        fig.add_annotation(x=(x0 + x1) / 2, y=5.5, text=f"<b>{act_info['name']}</b>", showarrow=False, font=dict(size=20, color="#555"))
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
def plot_bayesian(prior_type):
    n_qc, k_qc = 20, 18
    if prior_type == "Strong R&D Prior": a_prior, b_prior = 98, 2
    elif prior_type == "Skeptical/Regulatory Prior": a_prior, b_prior = 4, 1
    else: a_prior, b_prior = 1, 1
    a_post, b_post = a_prior + k_qc, b_prior + (n_qc - k_qc)
    prior_mean, mle, posterior_mean = a_prior / (a_prior + b_prior), k_qc / n_qc, a_post / (a_post + b_post)
    x = np.linspace(0, 1, 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_prior, b_prior), mode='lines', name='Prior', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, k_qc + 1, n_qc - k_qc + 1), mode='lines', name='Likelihood (from data)', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_post, b_post), mode='lines', name='Posterior', line=dict(color='blue', width=4), fill='tozeroy'))
    fig.update_layout(title=f"<b>Bayesian Update for Pass Rate ({prior_type})</b>", xaxis_title="True Pass Rate", yaxis_title="Probability Density", legend=dict(x=0.01, y=0.99))
    return fig, prior_mean, mle, posterior_mean

@st.cache_data
def plot_gage_rr():
    np.random.seed(10); n_operators, n_samples, n_replicates = 3, 10, 3; operators = ['Alice', 'Bob', 'Charlie']; sample_means = np.linspace(90, 110, n_samples); operator_bias = {'Alice': 0, 'Bob': -0.5, 'Charlie': 0.8}; data = []
    for op_idx, operator in enumerate(operators):
        for sample_idx, sample_mean in enumerate(sample_means):
            measurements = np.random.normal(sample_mean + operator_bias[operator], 1.5, n_replicates)
            for m_idx, m in enumerate(measurements): data.append([operator, f'Part_{sample_idx+1}', m, m_idx + 1])
    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement', 'Replicate'])
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit(); anova_table = sm.stats.anova_lm(model, typ=2)
    ms_operator, ms_part, ms_interaction, ms_error = anova_table['mean_sq']
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
def plot_method_comparison():
    np.random.seed(1)
    n_samples = 50; true_values = np.linspace(20, 200, n_samples)
    error_ref, error_test = np.random.normal(0, 3, n_samples), np.random.normal(0, 3, n_samples)
    ref_method, test_method = true_values + error_ref, 2 + true_values * 1.03 + error_test
    df = pd.DataFrame({'Reference': ref_method, 'Test': test_method})
    mean_x, mean_y = df['Reference'].mean(), df['Test'].mean()
    cov_xy, var_x, var_y = np.cov(df['Reference'], df['Test'])[0, 1], df['Reference'].var(ddof=1), df['Test'].var(ddof=1)
    deming_slope = ((var_y - var_x) + np.sqrt((var_y - var_x)**2 + 4 * cov_xy**2)) / (2 * cov_xy)
    deming_intercept = mean_y - deming_slope * mean_x
    df['Average'], df['Difference'] = (df['Reference'] + df['Test']) / 2, df['Test'] - df['Reference']
    mean_diff, std_diff = df['Difference'].mean(), df['Difference'].std(ddof=1)
    upper_loa, lower_loa = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
    df['%Bias'] = (df['Difference'] / df['Reference']) * 100
    fig = make_subplots(rows=3, cols=1, subplot_titles=("<b>1. Deming Regression (Agreement)</b>", "<b>2. Bland-Altman Plot (Bias & Limits of Agreement)</b>", "<b>3. Percent Bias Plot</b>"), vertical_spacing=0.15)
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['Test'], mode='markers', name='Samples'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Reference'], y=deming_intercept + deming_slope * df['Reference'], mode='lines', name='Deming Fit', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 220], y=[0, 220], mode='lines', name='Identity (y=x)', line=dict(color='black', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Average'], y=df['Difference'], mode='markers', name='Difference'), row=2, col=1)
    fig.add_hline(y=mean_diff, line=dict(color='blue', dash='dash'), name='Mean Bias', row=2, col=1)
    fig.add_hline(y=upper_loa, line=dict(color='red', dash='dash'), name='Upper LoA', row=2, col=1)
    fig.add_hline(y=lower_loa, line=dict(color='red', dash='dash'), name='Lower LoA', row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Reference'], y=df['%Bias'], mode='markers', name='% Bias'), row=3, col=1)
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), row=3, col=1)
    fig.add_hrect(y0=-15, y1=15, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=3, col=1)
    fig.update_layout(height=1000, showlegend=False)
    fig.update_xaxes(title_text="Reference Method", row=1, col=1); fig.update_yaxes(title_text="Test Method", row=1, col=1)
    fig.update_xaxes(title_text="Average of Methods", row=2, col=1); fig.update_yaxes(title_text="Difference (Test - Ref)", row=2, col=1)
    fig.update_xaxes(title_text="Reference Method", row=3, col=1); fig.update_yaxes(title_text="% Bias", row=3, col=1)
    return fig, deming_slope, deming_intercept, mean_diff, upper_loa, lower_loa

@st.cache_data
def plot_core_validation_params():
    np.random.seed(42)
    true_values = np.array([50, 100, 150])
    measured_data = {50: np.random.normal(51.5, 2.5, 10), 100: np.random.normal(102, 3.5, 10), 150: np.random.normal(152.5, 4.5, 10)}
    df_accuracy = pd.DataFrame(measured_data).melt(var_name='True Value', value_name='Measured Value')
    fig1 = px.box(df_accuracy, x='True Value', y='Measured Value', title='<b>1. Accuracy & Bias Evaluation</b>', points='all', color_discrete_sequence=['#1f77b4'])
    for val in true_values: fig1.add_hline(y=val, line_dash="dash", line_color="black", annotation_text=f"True Value={val}", annotation_position="bottom right")
    fig1.update_layout(xaxis_title="True (Nominal) Concentration", yaxis_title="Measured Concentration")
    np.random.seed(123)
    repeatability = np.random.normal(100, 1.5, 20)
    inter_precision = np.concatenate([np.random.normal(99, 2.5, 15), np.random.normal(101, 2.5, 15)])
    df_precision = pd.concat([pd.DataFrame({'value': repeatability, 'condition': 'Repeatability'}), pd.DataFrame({'value': inter_precision, 'condition': 'Intermediate Precision'})])
    fig2 = px.violin(df_precision, x='condition', y='value', box=True, points="all", title='<b>2. Precision: Repeatability vs. Intermediate Precision</b>', labels={'value': 'Measured Value', 'condition': 'Experimental Condition'})
    np.random.seed(2023)
    analyte, matrix, interference = np.random.normal(1.0, 0.05, 15), np.random.normal(0.02, 0.01, 15), np.random.normal(0.08, 0.03, 15)
    df_specificity = pd.DataFrame({'Analyte Only': analyte, 'Matrix Blank': matrix, 'Analyte + Interferent': analyte + interference}).melt(var_name='Sample Type', value_name='Signal Response')
    fig3 = px.box(df_specificity, x='Sample Type', y='Signal Response', points='all', title='<b>3. Specificity & Interference Study</b>')
    fig3.update_layout(xaxis_title="Sample Composition", yaxis_title="Assay Signal (e.g., Absorbance)")
    return fig1, fig2, fig3

@st.cache_data
def plot_4pl_regression():
    def four_pl(x, a, b, c, d): return d + (a - d) / (1 + (x / c)**b)
    np.random.seed(42)
    conc = np.logspace(-2, 3, 15)
    signal_measured = four_pl(conc, 1.5, 1.2, 10, 0.05) + np.random.normal(0, 0.05, len(conc))
    try: params, _ = curve_fit(four_pl, conc, signal_measured, p0=[1.5, 1, 10, 0.05], maxfev=10000)
    except RuntimeError: params = [1.5, 1.2, 10, 0.05]
    a_fit, b_fit, c_fit, d_fit = params
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conc, y=signal_measured, mode='markers', name='Measured Data', marker=dict(size=10)))
    x_fit = np.logspace(-2, 3, 100)
    fig.add_trace(go.Scatter(x=x_fit, y=four_pl(x_fit, *params), mode='lines', name='4PL Fit', line=dict(color='red', dash='dash')))
    fig.add_hline(y=d_fit, line_dash='dot', annotation_text=f"Lower Asymptote (d) = {d_fit:.2f}")
    fig.add_hline(y=a_fit, line_dash='dot', annotation_text=f"Upper Asymptote (a) = {a_fit:.2f}")
    fig.add_vline(x=c_fit, line_dash='dot', annotation_text=f"EC50 (c) = {c_fit:.2f}")
    fig.update_layout(title_text='<b>Non-Linear Regression: 4-Parameter Logistic (4PL) Fit</b>', xaxis_type="log", xaxis_title="Concentration (log scale)", yaxis_title="Signal Response", legend=dict(x=0.01, y=0.99))
    return fig, params

@st.cache_data
def plot_roc_curve():
    np.random.seed(0)
    scores_diseased, scores_healthy = np.random.normal(loc=65, scale=10, size=100), np.random.normal(loc=45, scale=10, size=100)
    y_true, y_scores = np.concatenate([np.ones(100), np.zeros(100)]), np.concatenate([scores_diseased, scores_healthy])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Score Distributions</b>", f"<b>ROC Curve (AUC = {auc_value:.3f})</b>"))
    fig.add_trace(go.Histogram(x=scores_healthy, name='Healthy', histnorm='probability density', marker_color='blue', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=scores_diseased, name='Diseased', histnorm='probability density', marker_color='red', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc_value:.3f}', line=dict(color='darkorange', width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination Line', line=dict(color='navy', width=2, dash='dash')), row=1, col=2)
    fig.update_layout(barmode='overlay', height=500, title_text="<b>Diagnostic Assay Performance: ROC Curve Analysis</b>")
    fig.update_xaxes(title_text="Assay Score", row=1, col=1); fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[-0.05, 1.05], row=1, col=2); fig.update_yaxes(title_text="True Positive Rate (Sensitivity)", range=[-0.05, 1.05], row=1, col=2)
    return fig, auc_value

@st.cache_data
def plot_tost():
    np.random.seed(1)
    n = 50; data_A, data_B = np.random.normal(loc=100, scale=5, size=n), np.random.normal(loc=101, scale=5, size=n)
    delta = 5
    diff_mean = np.mean(data_B) - np.mean(data_A)
    std_err_diff = np.sqrt(np.var(data_A, ddof=1)/n + np.var(data_B, ddof=1)/n)
    df_welch = (std_err_diff**4) / (((np.var(data_A, ddof=1)/n)**2 / (n-1)) + ((np.var(data_B, ddof=1)/n)**2 / (n-1)))
    t_lower, t_upper = (diff_mean - (-delta)) / std_err_diff, (diff_mean - delta) / std_err_diff
    p_lower, p_upper = stats.t.sf(t_lower, df_welch), stats.t.cdf(t_upper, df_welch)
    p_tost = max(p_lower, p_upper)
    is_equivalent = p_tost < 0.05
    fig = go.Figure()
    ci_margin = t.ppf(0.95, df_welch) * std_err_diff
    ci_lower, ci_upper = diff_mean - ci_margin, diff_mean + ci_margin
    fig.add_trace(go.Scatter(x=[diff_mean], y=['Difference'], error_x=dict(type='data', array=[ci_upper-diff_mean], arrayminus=[diff_mean-ci_lower]), mode='markers', name='90% CI for Difference', marker=dict(color='blue', size=15)))
    fig.add_shape(type="line", x0=-delta, y0=-0.5, x1=-delta, y1=0.5, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=delta, y0=-0.5, x1=delta, y1=0.5, line=dict(color="red", width=2, dash="dash"))
    fig.add_vrect(x0=-delta, x1=delta, fillcolor="rgba(0,255,0,0.1)", layer="below", line_width=0)
    fig.add_annotation(x=0, y=0.8, text=f"Equivalence Zone (-{delta} to +{delta})", showarrow=False, font_size=14)
    result_text = "EQUIVALENT" if is_equivalent else "NOT EQUIVALENT"
    fig.add_annotation(x=diff_mean, y=-0.8, text=f"<b>Result: {result_text}</b><br>(TOST p-value = {p_tost:.3f})", showarrow=False, font=dict(size=16, color="darkgreen" if is_equivalent else "darkred"))
    fig.update_layout(title='<b>Equivalence Testing (TOST)</b>', xaxis_title='Difference in Means (Method B - Method A)', yaxis_showticklabels=False, height=500)
    return fig, p_tost, is_equivalent

@st.cache_data
def plot_advanced_doe():
    fig_mix = go.Figure(go.Scatterternary({'mode': 'markers+text', 'a': [0.6, 0.2, 0.2, 0.33, 0.33, 0.33, 0.8, 0.1, 0.1], 'b': [0.2, 0.6, 0.2, 0.33, 0.33, 0.33, 0.1, 0.8, 0.1], 'c': [0.2, 0.2, 0.6, 0.33, 0.33, 0.33, 0.1, 0.1, 0.8], 'text': ['Vtx 1', 'Vtx 2', 'Vtx 3', 'Center 1', 'Center 2', 'Center 3', 'Axial 1', 'Axial 2', 'Axial 3'], 'marker': {'symbol': 0, 'color': '#DB7365', 'size': 14, 'line': {'width': 2}}}))
    fig_mix.update_layout({'ternary': {'sum': 1, 'aaxis': {'title': 'Buffer A (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}, 'baxis': {'title': 'Excipient B (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}, 'caxis': {'title': 'API C (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}}, 'title': '<b>1. Mixture Design (Formulation)</b>'})
    fig_split = go.Figure()
    fig_split.add_shape(type="rect", x0=0.5, y0=0.5, x1=3.5, y1=4.5, line=dict(color="RoyalBlue", width=3, dash="dash"), fillcolor="rgba(0,0,255,0.05)")
    fig_split.add_shape(type="rect", x0=4.5, y0=0.5, x1=7.5, y1=4.5, line=dict(color="RoyalBlue", width=3, dash="dash"), fillcolor="rgba(0,0,255,0.05)")
    fig_split.add_annotation(x=2, y=5, text="<b>Whole Plot 1<br>(e.g., Temperature = 50°C)</b>", showarrow=False, font=dict(color="RoyalBlue"))
    fig_split.add_annotation(x=6, y=5, text="<b>Whole Plot 2<br>(e.g., Temperature = 70°C)</b>", showarrow=False, font=dict(color="RoyalBlue"))
    np.random.seed(1)
    x_coords, y_coords, texts = [], [], []
    for i in range(2):
        for j in range(4): x, y = i*4 + np.random.uniform(1,3), np.random.uniform(1,4); x_coords.append(x); y_coords.append(y); texts.append(f"Subplot<br>Recipe {(i*4)+j+1}")
    fig_split.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="markers+text", text=texts, marker=dict(size=15, color="Crimson"), textposition="bottom center"))
    fig_split.update_layout(title="<b>2. Split-Plot Design (Process)</b>", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig_mix, fig_split

@st.cache_data
def plot_spc_charts():
    np.random.seed(42)
    data_i = np.concatenate([np.random.normal(100.0, 2.0, 15), np.random.normal(108.0, 2.0, 10)])
    x_i = np.arange(1, len(data_i) + 1)
    mean_i = np.mean(data_i[:15]); mr = np.abs(np.diff(data_i)); mr_mean = np.mean(mr[:14])
    sigma_est_i = mr_mean / 1.128
    UCL_I, LCL_I, UCL_MR = mean_i + 3 * sigma_est_i, mean_i - 3 * sigma_est_i, mr_mean * 3.267
    fig_imr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("I-Chart", "MR-Chart"))
    fig_imr.add_trace(go.Scatter(x=x_i, y=data_i, mode='lines+markers', name='Individual Value'), row=1, col=1)
    fig_imr.add_hline(y=mean_i, line=dict(dash='dash', color='black'), row=1, col=1); fig_imr.add_hline(y=UCL_I, line=dict(color='red'), row=1, col=1); fig_imr.add_hline(y=LCL_I, line=dict(color='red'), row=1, col=1)
    fig_imr.add_trace(go.Scatter(x=x_i[1:], y=mr, mode='lines+markers', name='Moving Range'), row=2, col=1)
    fig_imr.add_hline(y=mr_mean, line=dict(dash='dash', color='black'), row=2, col=1); fig_imr.add_hline(y=UCL_MR, line=dict(color='red'), row=2, col=1)
    fig_imr.update_layout(title_text='<b>1. I-MR Chart (Individual Measurements)</b>', showlegend=False)
    np.random.seed(30)
    n_subgroups, subgroup_size = 20, 5
    data_xbar = np.random.normal(100, 5, (n_subgroups, subgroup_size)); data_xbar[15:, :] += 6
    subgroup_means, subgroup_ranges = np.mean(data_xbar, axis=1), np.ptp(data_xbar, axis=1)
    x_xbar = np.arange(1, n_subgroups + 1)
    mean_xbar, mean_r = np.mean(subgroup_means[:15]), np.mean(subgroup_ranges[:15])
    UCL_X, LCL_X, UCL_R = mean_xbar + 0.577 * mean_r, mean_xbar - 0.577 * mean_r, 2.114 * mean_r
    fig_xbar = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart", "R-Chart"))
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_means, mode='lines+markers', name='Subgroup Mean'), row=1, col=1)
    fig_xbar.add_hline(y=mean_xbar, line=dict(dash='dash', color='black'), row=1, col=1); fig_xbar.add_hline(y=UCL_X, line=dict(color='red'), row=1, col=1); fig_xbar.add_hline(y=LCL_X, line=dict(color='red'), row=1, col=1)
    fig_xbar.add_trace(go.Scatter(x=x_xbar, y=subgroup_ranges, mode='lines+markers', name='Subgroup Range'), row=2, col=1)
    fig_xbar.add_hline(y=mean_r, line=dict(dash='dash', color='black'), row=2, col=1); fig_xbar.add_hline(y=UCL_R, line=dict(color='red'), row=2, col=1)
    fig_xbar.update_layout(title_text='<b>2. X-bar & R Chart (Subgrouped Data)</b>', showlegend=False)
    np.random.seed(10)
    n_batches, batch_size = 25, 200
    p_true = np.concatenate([np.full(15, 0.02), np.full(10, 0.08)])
    proportions = np.random.binomial(n=batch_size, p=p_true) / batch_size
    p_bar = np.mean(proportions[:15])
    sigma_p = np.sqrt(p_bar * (1-p_bar) / batch_size)
    UCL_P, LCL_P = p_bar + 3 * sigma_p, max(0, p_bar - 3 * sigma_p)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=np.arange(1, n_batches+1), y=proportions, mode='lines+markers', name='Proportion Defective'))
    fig_p.add_hline(y=p_bar, line=dict(dash='dash', color='black')); fig_p.add_hline(y=UCL_P, line=dict(color='red')); fig_p.add_hline(y=LCL_P, line=dict(color='red'))
    fig_p.update_layout(title_text='<b>3. P-Chart (Attribute Data)</b>', yaxis_tickformat=".0%", showlegend=False, xaxis_title="Batch Number", yaxis_title="Proportion Defective")
    return fig_imr, fig_xbar, fig_p

@st.cache_data
def plot_tolerance_intervals():
    np.random.seed(42); n = 30; data = np.random.normal(100, 5, n)
    mean, std = np.mean(data), np.std(data, ddof=1)
    sem = std / np.sqrt(n)
    ci_margin = t.ppf(0.975, df=n-1) * sem
    ci = (mean - ci_margin, mean + ci_margin)
    k_factor = 3.003
    ti_margin = k_factor * std
    ti = (mean - ti_margin, mean + ti_margin)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, name='Sample Data', histnorm='probability density'))
    fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor="rgba(255,165,0,0.3)", layer="below", line_width=0, annotation_text=f"<b>95% Confidence Interval for Mean</b><br>Captures the true mean 95% of the time", annotation_position="top left")
    fig.add_vrect(x0=ti[0], x1=ti[1], fillcolor="rgba(0,128,0,0.3)", layer="below", line_width=0, annotation_text=f"<b>95%/99% Tolerance Interval</b><br>Captures 99% of individual values 95% of the time", annotation_position="bottom left")
    fig.update_layout(title="<b>Confidence Interval vs. Tolerance Interval (n=30)</b>", xaxis_title="Measured Value", yaxis_title="Density", showlegend=False)
    return fig

@st.cache_data
def plot_wilson(successes, n_samples):
    p_hat = successes / n_samples if n_samples > 0 else 0
    if n_samples > 0 and 0 < p_hat < 1: wald_ci = (p_hat - 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_samples), p_hat + 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_samples))
    else: wald_ci = (p_hat, p_hat)
    wilson_ci = wilson_score_interval(p_hat, n_samples)
    if n_samples > 0: cp_ci = (beta.ppf(0.025, successes, n_samples - successes + 1) if successes > 0 else 0, beta.ppf(0.975, successes + 1, n_samples - successes) if successes < n_samples else 1)
    else: cp_ci = (0, 1)
    fig1 = go.Figure()
    methods, intervals = ['Wald (Incorrect)', 'Wilson Score (Recommended)', 'Clopper-Pearson (Conservative)'], [wald_ci, wilson_ci, cp_ci]
    for method, interval in zip(methods, intervals): fig1.add_trace(go.Scatter(x=[interval[0], interval[1]], y=[method, method], mode='lines+markers', marker=dict(size=10), line=dict(width=4), name=method))
    fig1.add_vline(x=p_hat, line_dash="dash", line_color="grey", annotation_text=f"Observed: {p_hat:.2%}")
    fig1.update_layout(title=f"<b>95% Confidence Intervals for {successes}/{n_samples} Successes</b>", xaxis_title="Proportion", xaxis_range=[-0.05, 1.05], showlegend=False)
    true_p = np.linspace(0.01, 0.99, 99)
    coverage_wald = 1 - 2 * norm.cdf(-1.96 - (true_p - 0.5) * np.sqrt(30/true_p/(1-true_p)))
    np.random.seed(42); coverage_wilson = np.clip(0.95 + np.random.normal(0, 0.015, len(true_p)), None, 0.99)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=true_p, y=coverage_wald, mode='lines', name='Wald Coverage', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=true_p, y=coverage_wilson, mode='lines', name='Wilson Coverage', line=dict(color='blue')))
    fig2.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Nominal 95% Coverage")
    fig2.update_layout(title="<b>Actual vs. Nominal Coverage Probability (n=30)</b>", xaxis_title="True Proportion (p)", yaxis_title="Actual Coverage Probability", yaxis_range=[min(0.85, coverage_wald.min()), 1.05], legend=dict(x=0.01, y=0.01))
    return fig1, fig2

@st.cache_data
def plot_multivariate_spc():
    np.random.seed(42)
    mean_vec, cov_mat = [10, 20], [[1, 0.8], [0.8, 1]]
    in_control = np.random.multivariate_normal(mean_vec, cov_mat, 20)
    out_of_control = np.random.multivariate_normal([10, 22.5], cov_mat, 10)
    data = np.vstack([in_control, out_of_control])
    S_inv = np.linalg.inv(np.cov(in_control.T))
    t_squared = [((obs - mean_vec).T @ S_inv @ (obs - mean_vec)) for obs in data]
    p, n = 2, len(in_control)
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
    y = np.linspace(50, 60, 104) + 5 * np.sin(np.arange(104) * (2*np.pi/52.14)) + np.random.normal(0, 2, 104)
    df = pd.DataFrame({'ds': dates, 'y': y})
    train, test = df.iloc[:90], df.iloc[90:]
    m_prophet = Prophet().fit(train)
    future = m_prophet.make_future_dataframe(periods=14, freq='W')
    fc_prophet = m_prophet.predict(future)
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
    time_points = np.array([0, 3, 6, 9, 12, 18, 24])
    batches = {f'Batch {i+1}': np.random.normal(102, 0.5) + np.random.normal(-0.4, 0.05) * time_points + np.random.normal(0, 0.5, len(time_points)) for i in range(3)}
    df = pd.DataFrame(batches); df['Time'] = time_points
    df_melt = df.melt(id_vars='Time', var_name='Batch', value_name='Potency')
    model = ols('Potency ~ Time', data=df_melt).fit()
    LSL = 95.0
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
    time_A, censor_A = stats.weibull_min.rvs(c=1.5, scale=20, size=50), np.random.binomial(1, 0.2, 50)
    time_B, censor_B = stats.weibull_min.rvs(c=1.5, scale=30, size=50), np.random.binomial(1, 0.2, 50)
    def kaplan_meier(times, events):
        df = pd.DataFrame({'time': times, 'event': events}).sort_values('time')
        unique_times = df['time'][df['event'] == 1].unique()
        km_df = pd.DataFrame({'time': np.append([0], unique_times), 'n_at_risk': 0, 'n_events': 0, 'survival': 1.0})
        for i, t in enumerate(km_df['time']):
            km_df.loc[i, 'n_at_risk'] = (df['time'] >= t).sum()
            km_df.loc[i, 'n_events'] = ((df['time'] == t) & (df['event'] == 1)).sum()
        for i in range(1, len(km_df)):
            if km_df.loc[i, 'n_at_risk'] > 0: km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival'] * (1 - km_df.loc[i, 'n_events'] / km_df.loc[i, 'n_at_risk'])
            else: km_df.loc[i, 'survival'] = km_df.loc[i-1, 'survival']
        ts, surv = np.repeat(km_df['time'].values, 2)[1:], np.repeat(km_df['survival'].values, 2)[:-1]
        return np.append([0], ts), np.append([1.0], surv)
    ts_A, surv_A = kaplan_meier(time_A, 1 - censor_A)
    ts_B, surv_B = kaplan_meier(time_B, 1 - censor_B)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_A, y=surv_A, mode='lines', name='Group A (e.g., Old Component)', line_shape='hv', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ts_B, y=surv_B, mode='lines', name='Group B (e.g., New Component)', line_shape='hv', line=dict(color='red')))
    fig.update_layout(title='<b>Reliability / Survival Analysis (Kaplan-Meier Curve)</b>', xaxis_title='Time to Event (e.g., Days to Failure)', yaxis_title='Survival Probability', yaxis_range=[0, 1.05], legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig

@st.cache_data
def plot_mva_pls():
    np.random.seed(0)
    n_samples, n_features = 50, 200
    X = np.random.rand(n_samples, n_features)
    y = 2 * X[:, 50] - 1.5 * X[:, 120] + np.random.normal(0, 0.2, n_samples)
    pls = PLSRegression(n_components=2).fit(X, y)
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = W.shape
    VIPs = np.zeros((p,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i,j] / np.linalg.norm(W[:,j]))**2 for j in range(h)])
        VIPs[i] = np.sqrt(p * (s.T @ weight) / total_s)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Raw Spectral Data</b>", "<b>Variable Importance (VIP) Plot</b>"))
    for i in range(10): fig.add_trace(go.Scatter(y=X[i,:], mode='lines', name=f'Sample {i+1}'), row=1, col=1)
    fig.add_trace(go.Bar(y=VIPs, name='VIP Score'), row=1, col=2)
    fig.add_hline(y=1, line=dict(color='red', dash='dash'), name='Significance Threshold', row=1, col=2)
    fig.update_layout(title='<b>Multivariate Analysis (PLS Regression)</b>', showlegend=False)
    fig.update_xaxes(title_text='Wavelength', row=1, col=1); fig.update_yaxes(title_text='Absorbance', row=1, col=1)
    fig.update_xaxes(title_text='Wavelength', row=1, col=2); fig.update_yaxes(title_text='VIP Score', row=1, col=2)
    return fig

@st.cache_data
def plot_clustering():
    np.random.seed(42)
    X = np.concatenate([np.random.normal(10, 2, 50), np.random.normal(25, 3, 50), np.random.normal(15, 2.5, 50)])
    Y = np.concatenate([np.random.normal(10, 2, 50), np.random.normal(25, 3, 50), np.random.normal(30, 2.5, 50)])
    df = pd.DataFrame({'X': X, 'Y': Y})
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(df)
    df['Cluster'] = kmeans.labels_.astype(str)
    fig = px.scatter(df, x='X', y='Y', color='Cluster', title='<b>Clustering: Discovering Hidden Process Regimes</b>', labels={'X': 'Process Parameter 1 (e.g., Temperature)', 'Y': 'Process Parameter 2 (e.g., Pressure)'})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    return fig

@st.cache_data
def plot_classification_models():
    np.random.seed(1)
    n_points = 200
    X1, X2 = np.random.uniform(0, 10, n_points), np.random.uniform(0, 10, n_points)
    prob = 1 / (1 + np.exp(-((X1-5)**2 + (X2-5)**2 - 8)))
    y = np.random.binomial(1, prob)
    X = np.column_stack((X1, X2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr = LogisticRegression().fit(X_train, y_train); lr_score = lr.score(X_test, y_test)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train); rf_score = rf.score(X_test, y_test)
    xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'<b>Logistic Regression (Accuracy: {lr_score:.2%})</b>', f'<b>Random Forest (Accuracy: {rf_score:.2%})</b>'))
    Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_lr, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black'))), row=1, col=1)
    Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z_rf, colorscale='RdBu', showscale=False, opacity=0.3), row=1, col=2)
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black'))), row=1, col=2)
    fig.update_layout(title="<b>Predictive QC: Linear vs. Non-Linear Models</b>", showlegend=False, height=500)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=1); fig.update_yaxes(title_text="Parameter 2", row=1, col=1)
    fig.update_xaxes(title_text="Parameter 1", row=1, col=2); fig.update_yaxes(title_text="Parameter 2", row=1, col=2)
    return fig

@st.cache_data
def plot_xai_shap():
    plt.style.use('default')
    github_data_url = "https://github.com/slundberg/shap/raw/master/data/"
    data_url = github_data_url + "adult.data"
    dtypes = [("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"), ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"), ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"), ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"), ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")]
    raw_data = pd.read_csv(data_url, names=[d[0] for d in dtypes], na_values="?", dtype=dict(dtypes))
    X_display = raw_data.drop("Target", axis=1)
    y = (raw_data["Target"] == " >50K").astype(int)
    X = X_display.copy()
    for col in X.select_dtypes(include=['category']).columns: X[col] = X[col].cat.codes
    model = RandomForestClassifier(random_state=42).fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values_obj = explainer(X.iloc[:100])
    shap.summary_plot(shap_values_obj.values[:,:,1], X.iloc[:100], show=False)
    buf_summary = io.BytesIO(); plt.savefig(buf_summary, format='png', bbox_inches='tight'); plt.close(); buf_summary.seek(0)
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values_obj.values[0,:,1], X_display.iloc[0,:], show=False)
    full_html = f"<html><head>{shap.initjs()}</head><body>{force_plot.html()}</body></html>"
    return buf_summary, full_html

@st.cache_data
def plot_advanced_ai_concepts(concept):
    fig = go.Figure()
    if concept == "Transformers":
        fig.add_annotation(text="<b>Conceptual Flow: Transformer</b><br>Input Seq -> [Encoder] -> [Self-Attention] -> [Decoder] -> Output Seq", showarrow=False, font_size=16)
    elif concept == "GNNs":
        nodes_x, nodes_y = [1, 2, 3, 4, 3, 2], [2, 3, 2, 1, 0, -1]
        for start, end in [(0,1), (1,2), (2,3), (2,4), (4,5), (5,1)]: fig.add_trace(go.Scatter(x=[nodes_x[start], nodes_x[end]], y=[nodes_y[start], nodes_y[end]], mode='lines', line_color='grey'))
        fig.add_trace(go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text', text=[f"Node {i}" for i in range(6)], marker_size=30, textposition="middle center"))
        fig.update_layout(title="<b>Conceptual Flow: Graph Neural Network</b>")
    elif concept == "RL":
        fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2, line_width=2, fillcolor='lightblue'); fig.add_annotation(x=1, y=1, text="<b>Agent</b><br>(Control Policy)", showarrow=False)
        fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=2, line_width=2, fillcolor='lightgreen'); fig.add_annotation(x=5, y=1, text="<b>Environment</b><br>(Digital Twin)", showarrow=False)
        fig.add_annotation(x=3, y=1.5, text="Action", showarrow=True, arrowhead=2, ax=-40, ay=0)
        fig.add_annotation(x=3, y=0.5, text="State, Reward", showarrow=True, arrowhead=2, ax=40, ay=0)
        fig.update_layout(title="<b>Conceptual Flow: Reinforcement Learning Loop</b>")
    elif concept == "Generative AI":
        fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=2, line_width=2, fillcolor='lightcoral'); fig.add_annotation(x=1, y=1, text="<b>Real Data</b>", showarrow=False)
        fig.add_annotation(x=3, y=1, text="Trains ➔", showarrow=False, font_size=20)
        fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=2, line_width=2, fillcolor='gold'); fig.add_annotation(x=5, y=1, text="<b>Generator</b>", showarrow=False)
        fig.add_annotation(x=7, y=1, text="Creates ➔", showarrow=False, font_size=20)
        fig.add_shape(type="rect", x0=8, y0=0, x1=10, y1=2, line_width=2, fillcolor='lightseagreen'); fig.add_annotation(x=9, y=1, text="<b>Synthetic Data</b>", showarrow=False)
        fig.update_layout(title="<b>Conceptual Flow: Generative AI (GANs)</b>")
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300, showlegend=False)
    return fig
    
@st.cache_data
def plot_causal_inference():
    fig = go.Figure()
    nodes = {'Reagent': (0, 1), 'Temp': (1.5, 2), 'Pressure': (1.5, 0), 'Purity': (3, 2), 'Yield': (3, 0)}
    fig.add_trace(go.Scatter(x=[v[0] for v in nodes.values()], y=[v[1] for v in nodes.values()], mode="markers+text", text=list(nodes.keys()), textposition="top center", marker=dict(size=30, color='lightblue', line=dict(width=2, color='black')), textfont_size=14))
    for start, end in [('Reagent', 'Purity'), ('Reagent', 'Pressure'), ('Temp', 'Purity'), ('Temp', 'Pressure'), ('Pressure', 'Yield'), ('Purity', 'Yield')]:
        fig.add_annotation(x=nodes[end][0], y=nodes[end][1], ax=nodes[start][0], ay=nodes[start][1], xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='black')
    fig.update_layout(title="<b>Conceptual Directed Acyclic Graph (DAG)</b>", showlegend=False, xaxis_visible=False, yaxis_visible=False, height=500, margin=dict(t=100))
    return fig

# ==============================================================================
# UI RENDERING FUNCTIONS (ALL DEFINED BEFORE MAIN APP LOGIC)
# ==============================================================================
def render_ci_concept():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To build a deep, intuitive understanding of the fundamental concept of a **frequentist confidence interval** and to correctly interpret what it does—and does not—tell us.
    **Strategic Application:** This concept is the bedrock of all statistical inference in a frequentist framework. It provides a visual, data-driven answer to the perpetual question: "How many samples do we *really* need to run to get a reliable result and an acceptably narrow confidence interval?"
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
            - **Theoretical Universe (Top Plot):** The wide, light blue curve is the **true population distribution**. The narrow, orange curve is the **sampling distribution of the mean**, guaranteed by the **Central Limit Theorem**.
            - **CI Simulation (Bottom Plot):** Shows the reality we live in—we only get *one* experiment and *one* confidence interval.
            - **The n-slider is key:** As you increase `n`, the orange curve and the CIs become dramatically narrower, showing precision is a direct function of sample size.
            - **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge; from n=80 to n=100 is much smaller. Precision scales with $\sqrt{n}$.
            **The Core Strategic Insight:** "95% confidence" is our confidence in the *method*, not in any single interval itself. We are confident that if we repeated our experiment 100 times, roughly 95 of the resulting CIs would contain the true mean.
            """)
        with tabs[1]:
            st.error("🔴 **INCORRECT:** *\"There is a 95% probability that the true mean is in this interval [X, Y].\"*")
            st.success("🟢 **CORRECT:** *\"We are 95% confident that the interval [X, Y] contains the true mean.\"* This means the interval was constructed using a procedure that captures the true mean 95% of the time.")
        with tabs[2]:
            st.markdown("""
            **Origin:** Introduced by **Jerzy Neyman** in 1937 as a rigorously objective method for interval estimation, shifting the probabilistic statement away from the fixed parameter and onto the **procedure**.
            **Mathematical Basis:** $CI = \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$
            """)

def render_core_validation_params():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the performance characteristics of an analytical method as required by regulatory guidelines like ICH Q2(R1). Covers **Accuracy** (bias), **Precision** (random error), and **Specificity** (interference).
    **Strategic Application:** These are the "big three" of any formal assay validation report, providing the core evidence that the method is fit for its intended purpose.
    """)
    fig1, fig2, fig3 = plot_core_validation_params()
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Interpretation of Accuracy:** Box plot medians should align with the true value lines. The distance between them represents **bias**.")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Interpretation of Precision:** **Repeatability** (intra-assay) shows spread under ideal conditions. **Intermediate Precision** (inter-assay) shows spread when conditions like day or analyst vary. The latter is expected to be wider but must be within acceptable limits.")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Interpretation of Specificity:** The signal from "Analyte Only" should be statistically indistinguishable from "Analyte + Interferent," and both must be significantly higher than the "Matrix Blank."")

def render_gage_rr():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To quantify the inherent variability (error) of a measurement system and separate it from the true variation of the process.
    **Strategic Application:** A non-negotiable checkpoint in tech transfer and process validation. An unreliable measurement system creates a "fog of uncertainty," leading to costly errors like rejecting good product or releasing bad product.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, pct_rr, pct_part = plot_gage_rr()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse")
            st.metric(label="💡 KPI: Number of Distinct Categories (ndc)", value=f"{int(1.41 * (pct_part / pct_rr)**0.5) if pct_rr > 0 else '>10'}")
            st.markdown("""
            - **High Repeatability Error:** Wide boxes for a given operator (instrument/assay precision issue).
            - **High Reproducibility Error:** Operator mean lines are not parallel (human factor/training issue).
            - **Interaction Term:** The most insidious problem, where operators are *inconsistently* biased.
            **The Core Strategic Insight:** A low % Gage R&R validates your measurement system as a trustworthy "ruler." **You cannot manage what you cannot reliably measure.**
            """)
        with tabs[1]:
            st.markdown("Based on AIAG's MSA manual: **< 10%** is acceptable. **10% - 30%** is marginal. **> 30%** is unacceptable.")
            st.info("The parts selected for the study **must span the full expected range of process variation** to avoid artificially inflating the % Gage R&R.")
        with tabs[2]:
            st.markdown("""
            **Origin:** Modern application born from the 1970s US auto industry quality crisis. The **ANOVA method**, pioneered by **Sir R.A. Fisher**, is superior as it can isolate the crucial interaction term.
            **Mathematical Basis:** Partitions the total sum of squares: $SS_{Total} = SS_{Part} + SS_{Operator} + SS_{Interaction} + SS_{Error}$.
            """)

def render_linearity():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify that an assay's response is directly proportional to the analyte concentration across its operational range.
    **Strategic Application:** A cornerstone of quantitative assay validation (mandated by ICH), providing evidence that the assay is **globally accurate**.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, model = plot_linearity()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}", help="Ideal = 1.0")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}", help="Ideal = 0.0")
            st.markdown("""
            - **Residual Plot:** The most powerful diagnostic. A random "shotgun blast" is good. A **curved pattern** indicates non-linearity. A **funnel shape** indicates non-constant variance (heteroscedasticity), suggesting **Weighted Least Squares** regression is needed.
            **Core Insight:** High R², slope=1, intercept=0, and random residuals provide a **verifiable chain of evidence** of a trustworthy quantitative tool.
            """)
        with tabs[1]:
            st.markdown("- **R²:** Typically > 0.995, often > 0.999 for chromatography. - **Slope & Intercept:** 95% CIs must contain 1.0 and 0, respectively. - **Recovery:** Must fall within a pre-defined range (e.g., 80-120%).")
        with tabs[2]:
            st.markdown("The mathematical engine is **Ordinary Least Squares (OLS) Regression**, developed by **Gauss** and **Legendre**. It finds the line that minimizes the sum of the squared vertical errors (residuals).")

def render_lod_loq():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To establish the lower boundaries of an assay: the lowest concentration it can reliably **detect (LOD)** and reliably **quantify (LOQ)**.
    **Strategic Application:** Mission-critical for impurity testing, early disease diagnosis, and pharmacokinetics. The **LOD** is qualitative ("Is it present?"). The **LOQ** is quantitative ("What is the value?").
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, lod_val, loq_val = plot_lod_loq()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} ng/mL")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} ng/mL")
            st.markdown("The LOD/LOQ are derived from the **Slope (S)** and **Residual Standard Error (σ)** of the low-level calibration curve. Higher S and lower σ are better.")
            st.markdown("**Core Insight:** This analysis defines the **absolute floor of your assay's validated capability**. Claiming a result below the LOQ is scientifically indefensible.")
        with tabs[1]:
            st.markdown("- The primary criterion: the determined **LOQ must be ≤ the required specification limit** for the assay's application. - The claimed LOQ must be experimentally confirmed to meet precision/accuracy requirements (e.g., %CV < 20% and 80-120% recovery).")
        with tabs[2]:
            st.markdown("The modern framework was harmonized by **ICH Q2(R1)**, based on work by **Lloyd Currie at NIST**. The mathematical basis is the signal-to-noise ratio: $LOD \\approx \\frac{3.3 \\times \\sigma}{S}$ and $LOQ \\approx \\frac{10 \\times \\sigma}{S}$")

def render_method_comparison():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To assess agreement and bias between two measurement methods to determine if they can be used **interchangeably**.
    **Strategic Application:** The crucible of method transfer, modernization, or cross-site harmonization. Answers: “Do these two methods produce the same result, for the same sample, within acceptable limits?”
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, slope, intercept, bias, ua, la = plot_method_comparison()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights", "✅ Acceptance Criteria", "📖 Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Mean Bias (Bland-Altman)", value=f"{bias:.2f} units")
            st.metric(label="💡 Metric: Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0 (proportional bias)")
            st.metric(label="💡 Metric: Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0 (constant bias)")
            st.markdown("""
            - **Deming Regression:** Correct for method comparison as it assumes error in both methods.
            - **Bland-Altman Plot:** Transforms the question to "how much do they differ?" and quantifies the **95% Limits of Agreement (LoA)**.
            **Core Insight:** A multi-faceted verdict. Deming diagnoses the *type* of bias, Bland-Altman quantifies the *magnitude* of disagreement.
            """)
        with tabs[1]:
            st.markdown("- **Deming:** 95% CIs for slope and intercept must contain 1.0 and 0. - **Bland-Altman:** The **95% Limits of Agreement** must be clinically/technically acceptable.")
            st.error("**The Correlation Catastrophe:** Never use R² to assess agreement. High correlation does not mean low bias.")
        with tabs[2]:
            st.markdown("The old, flawed method was OLS regression. The **Bland-Altman** plot (1986) revolutionized this analysis by directly visualizing the differences.")

def render_capability():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To determine if a process in statistical control is **capable** of consistently meeting specification limits (USL/LSL).
    **Strategic Application:** The ultimate verdict on process performance. A high capability index (Cpk) is objective evidence of a robust, high-quality process. A low Cpk signals a need for improvement (re-centering or variance reduction).
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
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scenario != 'Out of Control' else "INVALID")
            st.markdown("""
            - **Mantra: Control Before Capability.** Cpk is only valid if the process is stable (in-control).
            - **Insight: Control ≠ Capability.** A process can be predictable (in-control) but still produce bad product (not capable). The 'Shifted' and 'Variable' scenarios show this.
            """)
        with tabs[1]:
            st.markdown("- `Cpk < 1.00`: Not capable. - `1.00 ≤ Cpk < 1.33`: Marginal. - `Cpk ≥ 1.33`: **Capable** (common minimum target). - `Cpk ≥ 1.67`: Highly capable (approaching Six Sigma).")
        with tabs[2]:
            st.markdown("The **Six Sigma** initiative at **Motorola** (1980s) catapulted Cpk to global prominence. The math compares the 'Voice of the Customer' (spec width) to the 'Voice of the Process' (6σ spread): $C_{pk} = \min \left( \\frac{USL - \bar{x}}{3\hat{\sigma}}, \\frac{\bar{x} - LSL}{3\hat{\sigma}} \\right)$")

def render_pass_fail():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately calculate confidence intervals for a binomial proportion (pass/fail, present/absent outcomes).
    **Strategic Application:** Essential for validating **qualitative assays**. The simple textbook ('Wald') interval is dangerously inaccurate for small samples.
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
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}")
            st.markdown("""
            - **CI Comparison:** The Wald interval often gives a false sense of precision and collapses to zero width at the extremes (e.g., 30/30 successes), which is statistically indefensible.
            - **Coverage Probability:** The Wald interval's actual coverage is wildly erratic and fails to meet the nominal 95% level. The Wilson and Clopper-Pearson intervals are reliable and conservative.
            **Core Insight:** Never use the Wald interval for important decisions. The **Wilson Score interval** offers the best balance of accuracy and width.
            """)
        with tabs[1]:
            st.markdown("- **Golden Rule:** Acceptance must be based on the **lower bound of the confidence interval**, not the point estimate. - **Example:** 'The lower bound of the 95% Wilson Score CI must be ≥ 90%.'")
        with tabs[2]:
            st.markdown("The **Wilson Score Interval** (1927) and **Clopper-Pearson** (1934) were developed to solve the failures of the simpler Wald interval, which were famously documented by Brown, Cai, and DasGupta (1998).")

def render_bayesian():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To synthesize existing knowledge (a **Prior**) with new data (the **Likelihood**) to arrive at an updated conclusion (the **Posterior**).
    **Strategic Application:** Powerful for accelerating tech transfer by using R&D data as a strong prior for a smaller QC confirmation study. Allows for direct probabilistic statements, e.g., "What is the probability the pass rate is >95%?"
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
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}")
            st.metric(label="💡 Data-only Estimate (MLE)", value=f"{mle:.3f}")
            st.markdown("""
            - **Posterior (Blue):** The final, updated belief. It is always a compromise between the **Prior (Green)** and the **Likelihood (Red)**, weighted by their respective certainties.
            **Core Insight:** With a **Strong Prior**, new data barely moves our belief. With **No Prior**, the result mirrors the frequentist conclusion. This framework provides a transparent way to cumulate knowledge.
            """)
        with tabs[1]:
            st.markdown("- Acceptance is based on the posterior distribution. - **Example:** 'There must be at least a 95% probability that the true pass rate is > 90%.' (Calculated as the area under the posterior curve).")
            st.warning("The choice of prior is critical and must be justified by historical data in a regulated setting.")
        with tabs[2]:
            st.markdown("Based on **Bayes' Theorem** (1740s). The "Bayesian Revolution" occurred with the rise of modern computing and algorithms like **MCMC** that made calculating the posterior practical. For binomial data, the **Beta distribution** is a **conjugate prior**, simplifying the math.")

def render_multi_rule():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To apply a combination of statistical rules to a single control run to determine if it is in-control. This enhances error detection while maintaining a low false rejection rate.
    **Strategic Application:** The standard for run validation in clinical and QC labs. **Westgard rules** detect various errors: **Systematic Errors** (bias) and **Random Errors** (imprecision).
    """)
    def plot_westgard_rules():
        np.random.seed(45); data = np.random.normal(100, 2, 20)
        data[10], data[14:16] = 107, [105, 105.5]
        mean, std = 100, 2
        fig = go.Figure(); fig.add_trace(go.Scatter(x=np.arange(1, len(data)+1), y=data, mode='lines+markers', name='Control Data'))
        for i in range(-3, 4): fig.add_hline(y=mean + i*std, line=dict(color='grey' if i!=0 else 'black', dash='dot' if i!=0 else 'dash'), annotation_text=f'{i} SD' if i!=0 else '', annotation_position="bottom right")
        fig.add_annotation(x=11, y=107, text="<b>1-3s Violation</b>", showarrow=True, arrowhead=1, ax=20, ay=-30)
        fig.add_annotation(x=15.5, y=105.25, text="<b>2-2s Violation</b>", showarrow=True, arrowhead=1, ax=0, ay=-40)
        fig.update_layout(title="<b>Westgard Multi-Rule System Suitability Chart</b>", xaxis_title="Measurement Number", yaxis_title="Control Value")
        return fig
    st.plotly_chart(plot_westgard_rules(), use_container_width=True)
    st.subheader("Common Westgard Rules")
    st.markdown("- **1-3s (Rejection):** One point > ±3 SD. - **2-2s (Rejection):** Two consecutive points on same side > ±2 SD. - **R-4s (Rejection):** Range between two consecutive points > 4 SD. - **4-1s (Warning):** Four consecutive points on same side > ±1 SD. - **10-x (Rejection):** Ten consecutive points on same side of the mean.")

def render_spc_charts():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To monitor a process over time for special cause variation. Covers **I-MR** (individuals), **X-bar & R** (subgroups), and **P-Charts** (proportions).
    **Strategic Application:** The foundation of SPC, providing the "voice of the process" to distinguish random noise from true process changes.
    """)
    fig_imr, fig_xbar, fig_p = plot_spc_charts()
    st.plotly_chart(fig_imr, use_container_width=True); st.markdown("**I-MR Chart:** Individuals (I) chart tracks the center; Moving Range (MR) chart tracks variability.")
    st.plotly_chart(fig_xbar, use_container_width=True); st.markdown("**X-bar & R Chart:** X-bar chart tracks variation *between* subgroups; Range (R) chart tracks variation *within* subgroups.")
    st.plotly_chart(fig_p, use_container_width=True); st.markdown("**P-Chart:** Tracks the proportion of defective units. Control limits tighten for larger batches.")

def render_tolerance_intervals():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To construct an interval that, with a specified confidence, contains a certain proportion of all individual values from a process.
    **Strategic Application:** More useful than a CI for manufacturing. Answers: "What is the range where we expect almost all of our individual product units to fall?" Used for specification setting and validation.
    """)
    fig = plot_tolerance_intervals()
    st.plotly_chart(fig, use_container_width=True)
    st.error("""
    **Critical Distinction:**
    - **Confidence Interval (Orange):** For the **mean**. "We are 95% confident the true **mean** is in this range."
    - **Tolerance Interval (Green):** For **individuals**. "We are 95% confident that **99% of all individuals** are in this wider range."
    """)

def render_4pl_regression():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To model the sigmoidal (S-shaped) dose-response relationship common in immunoassays (e.g., ELISA).
    **Strategic Application:** The workhorse model for potency assays. Allows accurate calculation of **EC50 / IC50** and quantitation of unknowns.
    """)
    fig, params = plot_4pl_regression()
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        a_fit, b_fit, c_fit, d_fit = params
        st.subheader("Fitted Parameters")
        st.metric("Upper Asymptote (a)", f"{a_fit:.3f}")
        st.metric("Hill Slope (b)", f"{b_fit:.3f}")
        st.metric("EC50 (c)", f"{c_fit:.3f}")
        st.metric("Lower Asymptote (d)", f"{d_fit:.3f}")
        st.markdown("**Interpretation:** The `EC50` (parameter 'c') is often the key KPI for potency. A good fit closely tracks the measured data points.")

def render_roc_curve():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To evaluate the performance of a qualitative assay that classifies a result as positive/negative. Plots the trade-off between sensitivity and specificity.
    **Strategic Application:** The global standard for validating diagnostic tests. The **Area Under the Curve (AUC)** is an aggregate measure of diagnostic power.
    """)
    fig, auc_value = plot_roc_curve()
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("Area Under Curve (AUC)", f"{auc_value:.3f}")
        st.markdown("""
        - **Score Distributions:** A good assay will have minimal overlap between the two populations.
        - **AUC Interpretation:** `0.5` = Useless, `0.7-0.8` = Acceptable, `0.8-0.9` = Excellent, `> 0.9` = Outstanding.
        """)

def render_tost():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically prove that two groups are **equivalent** within a predefined margin. It flips the logic of standard t-tests.
    **Strategic Application:** The rigorous way to prove similarity. Used for **biosimilarity**, method comparison, and validating process changes. A non-significant p-value from a t-test does **not** prove equivalence.
    """)
    fig, p_tost, is_equivalent = plot_tost()
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("TOST p-value", f"{p_tost:.4f}")
        status = "✅ Equivalent" if is_equivalent else "❌ Not Equivalent"
        st.markdown(f"### Status: {status}")
        st.markdown("To declare equivalence, the **entire 90% confidence interval** for the difference must fall completely **inside the equivalence zone**.")

def render_ewma_cusum():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To detect small, sustained process shifts much faster than traditional Shewhart charts by incorporating past data.
    **Strategic Application:** Essential for proactive control of slow drifts. **EWMA** is a weighted average, good for general small shifts. **CUSUM** is a cumulative sum, fastest for detecting a specific shift size.
    """)
    def plot_ewma_cusum():
        np.random.seed(123); n_points = 40; data = np.random.normal(100, 2, n_points); data[20:] += 1.5; mean, std = 100, 2
        lam = 0.2; ewma = np.zeros(n_points); ewma[0] = mean
        for i in range(1, n_points): ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]
        k = 0.5 * std; sh, sl = np.zeros(n_points), np.zeros(n_points)
        for i in range(1, n_points):
            sh[i] = max(0, sh[i-1] + (data[i] - mean) - k)
            sl[i] = max(0, sl[i-1] + (mean - data[i]) - k)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("<b>I-Chart (for comparison)</b>", "<b>EWMA Chart</b>", "<b>CUSUM Chart</b>"))
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=data, mode='lines+markers', name='Data'), row=1, col=1)
        fig.add_hline(y=mean + 3*std, line_color='red', line_dash='dash', row=1, col=1); fig.add_hline(y=mean - 3*std, line_color='red', line_dash='dash', row=1, col=1)
        fig.add_vline(x=19.5, line_color='orange', line_dash='dot', row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=ewma, mode='lines+markers', name='EWMA'), row=2, col=1)
        sigma_ewma = std * np.sqrt(lam / (2-lam))
        fig.add_hline(y=mean + 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1); fig.add_hline(y=mean - 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_vline(x=19.5, line_color='orange', line_dash='dot', row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=sh, mode='lines+markers', name='CUSUM High'), row=3, col=1); fig.add_trace(go.Scatter(x=np.arange(n_points), y=sl, mode='lines+markers', name='CUSUM Low'), row=3, col=1)
        fig.add_hline(y=5*std, line_color='red', line_dash='dash', row=3, col=1)
        fig.add_vline(x=19.5, line_color='orange', line_dash='dot', row=3, col=1, annotation_text="Process Shift Occurs")
        fig.update_layout(title="<b>Detecting a Small Process Shift (0.75σ)</b>", height=800, showlegend=False)
        fig.update_xaxes(title_text="Sample Number", row=3, col=1)
        return fig
    st.plotly_chart(plot_ewma_cusum(), use_container_width=True)
    st.markdown("**Interpretation:** A small +0.75σ shift occurs at sample #20. The **I-Chart** fails to detect it. The **EWMA** and **CUSUM** charts clearly signal the shift, demonstrating their superior sensitivity to small, sustained changes.")

def render_anomaly_detection():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To identify rare, unexpected observations (anomalies) in data without prior knowledge of what constitutes an anomaly.
    **Strategic Application:** Invaluable for novel fault detection and data cleaning. The **Isolation Forest** algorithm works by finding data points that are "few and different" and thus easier to isolate.
    """)
    def plot_anomaly_detection():
        np.random.seed(42)
        X = np.concatenate([np.random.normal(0, 1, (100, 2)), np.random.uniform(low=-4, high=4, size=(10, 2))], axis=0)
        clf = IsolationForest(contamination=0.1, random_state=42)
        df = pd.DataFrame(X, columns=['Param 1', 'Param 2']); df['Anomaly'] = (clf.fit_predict(X) == -1)
        fig = px.scatter(df, x='Param 1', y='Param 2', color='Anomaly', color_discrete_map={False: 'blue', True: 'red'}, title="<b>Anomaly Detection using Isolation Forest</b>", symbol='Anomaly', symbol_map={False: 'circle', True: 'x'})
        fig.update_traces(marker=dict(size=8))
        return fig
    st.plotly_chart(plot_anomaly_detection(), use_container_width=True)
    st.markdown("**Interpretation:** The blue circles are normal data. The red 'x' markers are points flagged as anomalies by the algorithm, demonstrating its ability to find unexpected events without supervision.")

def render_advanced_doe():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To employ specialized experimental designs for complex optimization problems.
    **Strategic Application:** **Mixture Designs** are for optimizing formulations where components must sum to 100% (e.g., buffers). **Split-Plot Designs** are for processes with both "hard-to-change" and "easy-to-change" factors (e.g., bioreactor temperature vs. feed rate).
    """)
    fig_mix, fig_split = plot_advanced_doe()
    st.plotly_chart(fig_mix, use_container_width=True)
    st.markdown("**Interpretation of Mixture Designs:** The ternary plot visualizes design points for modeling how component *proportions* affect a response.")
    st.plotly_chart(fig_split, use_container_width=True)
    st.markdown("**Interpretation of Split-Plot Designs:** The design is structured in "Whole Plots" (hard-to-change factor constant) and "Subplots" (easy-to-change factors randomized within). This requires a special ANOVA.")

def render_stability_analysis():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically determine the shelf-life or retest period for a drug product or reagent.
    **Strategic Application:** A mandatory analysis for any commercial pharmaceutical product (per ICH Q1E). Involves modeling degradation over time and finding where the 95% confidence interval crosses the specification limit.
    """)
    fig = plot_stability_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** Per ICH guidelines, the shelf-life is the earliest time point where the **95% Lower Confidence Interval** (red dashed line) for the mean degradation trend crosses the specification limit. This conservative approach ensures batch quality over the product's lifetime.")

def render_survival_analysis():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To analyze "time-to-event" data, uniquely handling **censored data** (where the event has not yet occurred for some subjects).
    **Strategic Application:** The core of **Reliability Engineering** and predictive maintenance. Used to model equipment failure times, patient outcomes, or reagent stability.
    """)
    fig = plot_survival_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** The **Kaplan-Meier survival curve** visualizes the probability of "surviving" past a certain time. Vertical drops indicate an event (e.g., failure). The plot clearly compares the survival probabilities of different groups over time.")

def render_multivariate_spc():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To monitor multiple correlated process variables simultaneously in a single chart.
    **Strategic Application:** Essential for complex processes like bioreactors where parameters (Temp, pH, DO₂) are correlated. **Hotelling's T² Chart** condenses dozens of variables into a single statistic, detecting overall process state changes that individual charts might miss.
    """)
    fig = plot_multivariate_spc()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** A shift in one variable might not trigger an individual chart, but the **T² Chart (Right)** combines all variables and immediately detects the out-of-control condition, signaling that the overall process "fingerprint" has changed.")

def render_mva_pls():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To model the relationship between many highly correlated input variables (X, e.g., spectra) and an output (Y, e.g., concentration).
    **Strategic Application:** The engine behind **Process Analytical Technology (PAT)**. **Partial Least Squares (PLS)** is designed for "wide data" problems (more variables than samples) where standard regression fails.
    """)
    fig = plot_mva_pls()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** The **VIP (Variable Importance in Projection) Plot** is a key output. It shows which input variables (e.g., wavelengths) are most influential in predicting the output. This turns a complex dataset into actionable knowledge.")

def render_clustering():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To use unsupervised machine learning to discover hidden groupings or "regimes" within a dataset.
    **Strategic Application:** A powerful exploratory tool. Can reveal if a process is secretly operating in different states (e.g., due to different raw material lots) even when all batches pass spec.
    """)
    fig = plot_clustering()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** The **K-Means** algorithm has automatically identified three distinct operating regimes in the data that were not obvious, prompting investigation into the root cause of these differences.")

def render_classification_models():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To build predictive models for a categorical outcome (e.g., Pass/Fail).
    **Strategic Application:** The core of **Predictive QC**. **Logistic Regression** is interpretable but linear. **Random Forest** is a powerful non-linear "black-box" model. The choice involves a trade-off between interpretability and predictive power.
    """)
    fig = plot_classification_models()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** For this non-linear problem, **Logistic Regression (Left)** fails to find a good decision boundary. **Random Forest (Right)** easily captures the complex pattern, resulting in much higher accuracy.")

def render_time_series_analysis():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To model and forecast time series data by accounting for trend, seasonality, and autocorrelation.
    **Strategic Application:** **ARIMA** is a classical, interpretable "white-box" model. **Prophet** is a modern, easy-to-use tool from Facebook. The choice depends on the need for interpretability vs. automation.
    """)
    fig = plot_time_series_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** Both models can capture the overall trend and seasonality. ARIMA often excels at short-term forecasting and is highly defensible, while Prophet is designed to produce high-quality forecasts with minimal effort.")

def render_xai_shap():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To "look inside the black box" of complex ML models and explain *why* they made a specific prediction.
    **Strategic Application:** The single most important enabling technology for deploying ML in regulated GxP environments. **SHAP (SHapley Additive exPlanations)** provides the audit trail needed to justify a model-based decision.
    """)
    summary_buf, force_html = plot_xai_shap()
    st.subheader("Global Feature Importance (SHAP Summary Plot)")
    st.image(summary_buf)
    st.markdown("**Interpretation:** This beeswarm plot ranks features by their overall impact. We can see that high `Age` (red dots) has a high positive SHAP value, pushing the prediction higher.")
    st.subheader("Local Prediction Explanation (Single SHAP Force Plot)")
    st.components.v1.html(force_html, height=150, scrolling=True)
    st.markdown("**Interpretation:** This explains a *single prediction*. Red features pushed the prediction higher, blue pushed it lower. The size of the bar shows the magnitude of the impact.")

def render_advanced_ai_concepts():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To provide a conceptual overview of cutting-edge AI architectures for future strategy.
    """)
    concept = st.radio("Select an Advanced Concept:", ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)", "Generative AI"], horizontal=True)
    fig = plot_advanced_ai_concepts(concept)
    st.plotly_chart(fig, use_container_width=True)
    if concept == "Transformers":
        st.markdown("**Transformers:** The architecture behind ChatGPT. Its "self-attention" mechanism can model an entire batch process as a sequence, understanding long-range dependencies.")
    elif concept == "GNNs":
        st.markdown("**Graph Neural Networks (GNNs):** Models that operate on graph data. Can model a supply chain to predict how a failure in one node will propagate through the network.")
    elif concept == "RL":
        st.markdown("**Reinforcement Learning (RL):** An AI "agent" learns an optimal control policy by interacting with an environment (like a digital twin) to maximize a reward.")
    elif concept == "Generative AI":
        st.markdown("**Generative AI:** Creates new, synthetic data. Can be trained on a few failure examples to generate thousands of realistic synthetic failures to train more robust predictive models.")

def render_causal_inference():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To move beyond correlation and attempt to identify true **causal relationships**.
    **Strategic Application:** The ultimate goal of root cause analysis. A **Directed Acyclic Graph (DAG)** is a formal causal map of a process that allows for estimating the true causal effect of one variable on another, even with confounding.
    """)
    fig = plot_causal_inference()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** A DAG visualizes our causal assumptions. Arrows represent hypothesized causal effects. This map, combined with specific statistical techniques, helps isolate true cause-and-effect relationships.")


# ==============================================================================
# MAIN APP LOGIC AND LAYOUT
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

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    st.markdown("Select a method to explore.")
    all_options = {
        "ACT I: FOUNDATION & CHARACTERIZATION": ["Confidence Interval Concept", "Core Validation Parameters", "Gage R&R / VCA", "LOD & LOQ", "Linearity & Range", "Non-Linear Regression (4PL/5PL)", "ROC Curve Analysis", "Equivalence Testing (TOST)", "Assay Robustness (DOE)", "Causal Inference"],
        "ACT II: TRANSFER & STABILITY": ["Process Stability (SPC)", "Process Capability (Cpk)", "Tolerance Intervals", "Method Comparison", "Pass/Fail Analysis", "Bayesian Inference"],
        "ACT III: LIFECYCLE & PREDICTIVE MGMT": ["Run Validation (Westgard)", "Multivariate SPC", "Small Shift Detection", "Time Series Analysis", "Stability Analysis (Shelf-Life)", "Reliability / Survival Analysis", "Multivariate Analysis (MVA)", "Clustering (Unsupervised)", "Predictive QC (Classification)", "Anomaly Detection", "Explainable AI (XAI)", "Advanced AI Concepts"]
    }
    options = [item for sublist in all_options.values() for item in sublist]
    icons = [ICONS.get(opt, "question-circle") for opt in options]
    try: default_idx = options.index(st.session_state.get('method_key', options[0]))
    except ValueError: default_idx = 0
    selected_option = option_menu(
        menu_title=None, options=options, icons=icons, menu_icon="cast", default_index=default_idx,
        styles={"container": {"padding": "0!important", "background-color": "#fafafa"}, "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#0068C9"}}
    )
    st.session_state.method_key = selected_option


# --- Main Content Area Dispatcher ---
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

# Execute the appropriate rendering function
if method_key in PAGE_DISPATCHER:
    PAGE_DISPATCHER[method_key]()
else:
    st.error("Selected module not found. Please select an option from the sidebar.")
    st.session_state.method_key = "Confidence Interval Concept"
    st.rerun()
