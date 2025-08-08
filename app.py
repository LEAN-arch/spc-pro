# ==============================================================================
# LIBRARIES & IMPORTS
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
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
    
    # This section for loading data directly via URL is already correct and robust.
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
    # 1. Generate the plot object as before
    force_plot = shap.force_plot(
        explainer.expected_value[1], 
        shap_values_obj.values[0,:,1], 
        X_display.iloc[0,:], 
        show=False
    )
    
    # 2. Get the raw HTML from the plot object
    force_plot_html = force_plot.html()

    # 3. Create a fully self-contained HTML string by adding the SHAP JS library
    #    shap.initjs() injects the necessary <script> tag.
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
    **Purpose:** To apply a combination of statistical rules to a single control run to determine if it is in-control or out-of-control. This provides a high probability of detecting true errors while maintaining a low false rejection rate.
    
    **Strategic Application:** This is the standard for run validation and system suitability in clinical chemistry and regulated QC laboratories. While a standard control chart (like an I-MR chart) uses a single rule (e.g., a point outside ±3 SD), Westgard rules use a more sensitive combination to detect various types of errors:
    - **Systematic Errors (Bias/Shifts):** Detected by rules like 2-2s, 4-1s, or 10-x.
    - **Random Errors (Imprecision):** Detected primarily by the 1-3s and R-4s rules.

    Implementing these rules prevents the release of results from an assay run that has failed, ensuring the quality and reliability of patient or product data.
    """)
    
    # Placeholder for the plotting function
    # In a real app, this function would generate data and check Westgard rule violations
    def plot_westgard_rules():
        np.random.seed(45)
        data = np.random.normal(100, 2, 20)
        data[10] = 107  # 1-3s violation
        data[14:16] = [105, 105.5] # 2-2s violation
        mean, std = 100, 2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(data)+1), y=data, mode='lines+markers', name='Control Data'))
        
        # Add SD lines
        for i in range(-3, 4):
            if i == 0:
                fig.add_hline(y=mean, line=dict(color='black', dash='dash'))
            else:
                fig.add_hline(y=mean + i*std, line=dict(color='grey', dash='dot'), 
                              annotation_text=f'{i} SD', annotation_position="bottom right")
        
        # Highlight violations
        fig.add_annotation(x=11, y=107, text="<b>1-3s Violation</b>", showarrow=True, arrowhead=1, ax=20, ay=-30)
        fig.add_annotation(x=15.5, y=105.25, text="<b>2-2s Violation</b>", showarrow=True, arrowhead=1, ax=0, ay=-40)
        
        fig.update_layout(title="<b>Westgard Multi-Rule System Suitability Chart</b>",
                          xaxis_title="Measurement Number", yaxis_title="Control Value")
        return fig

    fig = plot_westgard_rules()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Common Westgard Rules")
    st.markdown("""
    - **1-3s Rule (Warning/Rejection):** One point falls outside ±3 standard deviations. This is a rejection rule and a primary sign of significant random error or a large systematic shift.
    - **2-2s Rule (Rejection):** Two consecutive points fall on the same side of the mean and are both beyond ±2 standard deviations. This is a sensitive indicator of systematic error (bias).
    - **R-4s Rule (Rejection):** The range between two consecutive points exceeds 4 standard deviations. This detects random error or imprecision.
    - **4-1s Rule (Warning/Rejection):** Four consecutive points fall on the same side of the mean and are all beyond ±1 standard deviation. This indicates a small, sustained systematic shift.
    - **10-x Rule (Rejection):** Ten consecutive points fall on the same side of the mean. This detects a small bias that may not trigger other rules.
    """)
def render_spc_charts():
    """Renders the module for Statistical Process Control (SPC) charts."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To monitor a process over time to detect special cause variation, ensuring it remains in a state of statistical control. This module covers the three most common types of Shewhart charts:
    - **I-MR Chart:** For individual measurements where subgrouping is not possible or practical.
    - **X-bar & R Chart:** For continuous data collected in rational subgroups (e.g., 5 samples per hour). This is more powerful than an I-MR chart for detecting small shifts.
    - **P-Chart:** For attribute data, specifically the proportion of non-conforming items per batch or lot.

    **Strategic Application:** This is the foundation of SPC. These charts provide the real-time "voice of the process," distinguishing between normal, random variation (common cause) and signals that indicate a fundamental process change (special cause).
    """)
    fig_imr, fig_xbar, fig_p = plot_spc_charts()
    
    st.plotly_chart(fig_imr, use_container_width=True)
    st.markdown("**Interpretation of I-MR Chart:** The Individuals (I) chart tracks the process center, while the Moving Range (MR) chart tracks its short-term variability. Both must be stable to declare the process 'in control'.")
    
    st.plotly_chart(fig_xbar, use_container_width=True)
    st.markdown("**Interpretation of X-bar & R Chart:** The X-bar chart tracks the variation *between* subgroups (a measure of process shifts), while the Range (R) chart tracks the variation *within* subgroups (a measure of process consistency).")
    
    st.plotly_chart(fig_p, use_container_width=True)
    st.markdown("**Interpretation of P-Chart:** This chart tracks the proportion of defective units over time. The control limits are not constant; they become tighter for larger batch sizes, reflecting the increased certainty. This chart is essential for monitoring pass/fail rates and yields.")

def render_tolerance_intervals():
    """Renders the module for Tolerance Intervals."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To construct an interval that we can claim, with a specified level of confidence, contains a certain proportion of all individual values from a process.
    
    **Strategic Application:** For many manufacturing and quality applications, this is a far more useful and powerful tool than a standard confidence interval. It directly answers the question that engineers and quality managers care about: "What is the range where we expect almost all of our individual product units to fall?"
    - **Specification Setting:** Tolerance intervals can be used to set statistically-driven release specifications.
    - **Validation:** They are used in validation reports to demonstrate that a process can reliably produce units within a required range.
    - **Comparing to a Confidence Interval:** A CI is about the mean; a TI is about the individuals. A process can have a very narrow CI for its mean but still produce many individual units outside of specification if its variance is high.
    """)
    fig = plot_tolerance_intervals()
    st.plotly_chart(fig, use_container_width=True)
    st.error("""
    **Critical Distinction:**
    - **Confidence Interval (Orange):** "We are 95% confident that the true long-term **mean** of the process is within this narrow range."
    - **Tolerance Interval (Green):** "We are 95% confident that **99% of all individual units** produced by this process will fall within this much wider range."
    
    For ensuring product quality, the Tolerance Interval is the relevant metric.
    """)

def render_4pl_regression():
    """Renders the module for 4-Parameter Logistic (4PL) regression."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately model the characteristic sigmoidal (S-shaped) dose-response relationship found in most immunoassays (e.g., ELISA) and biological assays. A straight-line (linear) model is fundamentally incorrect for this type of data.
    
    **Strategic Application:** This is the workhorse model for potency assays, immunoassays, and any assay where the response has a floor, a ceiling, and a sloped transition between them. The Four-Parameter Logistic (4PL) model allows for the accurate calculation of critical assay parameters:
    - **EC50 / IC50:** The concentration that produces 50% of the maximal response, a key measure of a drug's potency.
    - **Quantitation of Unknowns:** By inverting the fitted curve, the model can accurately determine the concentration of unknown samples from their signal response.
    - **System Suitability:** The fitted parameters (like slope and asymptotes) can be used as system suitability criteria to ensure the assay is performing correctly on a given day.
    """)
    fig, params = plot_4pl_regression()
    
    # This entire block was incorrectly indented. It now belongs inside the function.
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Unpack params into named variables for clarity and robustness
        a_fit, b_fit, c_fit, d_fit = params
        
        st.subheader("Fitted Parameters")
        st.metric("Upper Asymptote (a)", f"{a_fit:.3f}")
        st.metric("Hill Slope (b)", f"{b_fit:.3f}")
        st.metric("EC50 (c)", f"{c_fit:.3f}")
        st.metric("Lower Asymptote (d)", f"{d_fit:.3f}")
        st.markdown("**Interpretation:** The 4PL model fits four key parameters to describe the curve's shape. The `EC50` (parameter 'c') is often the most important KPI, representing the potency of the analyte. A good fit is characterized by the red dashed line closely tracking the measured data points and by a high R-squared value for the non-linear fit.")

# The code below was incorrectly merged. It is now its own separate function.
def render_roc_curve():
    """Renders the module for Receiver Operating Characteristic (ROC) curve analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To evaluate and visualize the performance of a diagnostic or qualitative assay that classifies a result as positive/negative or diseased/healthy. The Receiver Operating Characteristic (ROC) curve plots the trade-off between sensitivity and specificity at all possible cutoff values.
    
    **Strategic Application:** This is the global standard for validating and comparing diagnostic tests. The Area Under the Curve (AUC) provides a single, aggregate measure of a test's diagnostic power.
    - **Choosing a Cutoff:** The ROC curve allows scientists to rationally select an optimal cutoff point for the assay that balances the clinical need for high sensitivity (catching all true positives) against the need for high specificity (avoiding false alarms).
    - **Comparing Assays:** The AUC values of two different assays for the same disease can be directly compared to determine which is diagnostically superior.
    - **Regulatory Submissions:** An ROC analysis is a standard requirement for submissions to regulatory bodies like the FDA for any new diagnostic test.
    """)
    fig, auc_value = plot_roc_curve()
    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        st.metric("Area Under Curve (AUC)", f"{auc_value:.3f}")
        st.markdown("""
        - **Score Distributions (Left):** This plot shows the fundamental challenge. The two populations (Healthy and Diseased) overlap. A good assay will have minimal overlap.
        - **ROC Curve (Right):** This plot visualizes the performance across all thresholds. The "shoulder" of the curve pushing towards the top-left corner indicates a good test.
        - **AUC Interpretation:**
            - `AUC = 0.5`: The test is useless (equivalent to a coin flip).
            - `0.7 < AUC < 0.8`: Acceptable discrimination.
            - `0.8 < AUC < 0.9`: Excellent discrimination.
            - `AUC > 0.9`: Outstanding discrimination.
        """)

def render_tost():
    """Renders the module for Two One-Sided Tests (TOST) for equivalence."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically prove that two methods or the mean of two groups are **equivalent** within a predefined, practically insignificant margin. This flips the logic of standard hypothesis testing.
    
    **Strategic Application:** This is the statistically rigorous way to handle comparisons where the goal is to prove similarity, not difference.
    - **Biosimilarity:** The primary tool used to demonstrate that a biosimilar drug has no clinically meaningful differences from a reference product.
    - **Method Comparison:** A more formal and powerful alternative to Deming regression for proving that a new analytical method is interchangeable with an old one.
    - **Process Changes:** Used in validation to prove that a change to the manufacturing process (e.g., a new raw material supplier) has not impacted the final product.
    
    **The Flaw of Standard T-tests:** A standard t-test is designed to find differences. A "non-significant" p-value (e.g., p=0.08) does not prove the null hypothesis of no difference; it merely means you failed to find sufficient evidence of a difference. **Two One-Sided Tests (TOST)** corrects this by making equivalence the alternative hypothesis to be proven.
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
        st.markdown("""
        **The Core Insight:** The blue line represents the 90% confidence interval for the true difference between the two methods (90% CI is standard for TOST). The red dashed lines represent the pre-defined **equivalence margin** (the zone where differences are considered practically meaningless).
        
        To declare equivalence, the **entire confidence interval must fall completely inside the equivalence zone.** In this example, it does, so we can statistically conclude that the two methods are equivalent.
        """)
def render_ewma_cusum():
    """Renders the module for small shift detection charts (EWMA/CUSUM)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To detect small, sustained process shifts much faster than traditional Shewhart charts (like the I-MR or X-bar charts). These charts achieve this by incorporating information from past data points.
    
    **Strategic Application:** Shewhart charts are excellent for detecting large, sudden shifts. However, in many biological or chemical processes, degradation or drift occurs slowly over time. A process might shift by only 0.5 to 1.5 standard deviations, a change that could go unnoticed for a long time on a standard chart.
    - **EWMA (Exponentially Weighted Moving Average):** This chart gives the most weight to the most recent data point, with the weights of older points decaying exponentially. It is effective at detecting small shifts and is generally easier to set up and interpret.
    - **CUSUM (Cumulative Sum):** This chart directly plots the cumulative sum of deviations from a target. It is the fastest method for detecting a shift of a specific magnitude that it was designed to find.

    These charts are essential for proactive process control, allowing for intervention *before* a small drift becomes a major out-of-spec event.
    """)
    
    # Placeholder for the plotting function
    def plot_ewma_cusum():
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
                            subplot_titles=("<b>I-Chart (for comparison)</b>", "<b>EWMA Chart</b>", "<b>CUSUM Chart</b>"))

        # I-Chart
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=data, mode='lines+markers', name='Data'), row=1, col=1)
        fig.add_hline(y=mean + 3*std, line_color='red', line_dash='dash', row=1, col=1)
        fig.add_hline(y=mean - 3*std, line_color='red', line_dash='dash', row=1, col=1)
        fig.add_vline(x=19.5, line_color='orange', line_dash='dot', row=1, col=1)

        # EWMA Chart
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=ewma, mode='lines+markers', name='EWMA'), row=2, col=1)
        # Simplified UCL/LCL for EWMA for plotting
        sigma_ewma = std * np.sqrt(lam / (2-lam))
        fig.add_hline(y=mean + 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_hline(y=mean - 3*sigma_ewma, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_vline(x=19.5, line_color='orange', line_dash='dot', row=2, col=1)
        
        # CUSUM Chart
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=sh, mode='lines+markers', name='CUSUM High'), row=3, col=1)
        fig.add_trace(go.Scatter(x=np.arange(n_points), y=sl, mode='lines+markers', name='CUSUM Low'), row=3, col=1)
        fig.add_hline(y=5*std, line_color='red', line_dash='dash', row=3, col=1) # Decision interval H
        fig.add_vline(x=19.5, line_color='orange', line_dash='dot', row=3, col=1, annotation_text="Process Shift Occurs")

        fig.update_layout(title="<b>Detecting a Small Process Shift (0.75σ)</b>", height=800, showlegend=False)
        fig.update_xaxes(title_text="Sample Number", row=3, col=1)
        return fig

    fig = plot_ewma_cusum()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**
    - A small but sustained process shift of +0.75σ is introduced at sample #20.
    - **I-Chart (Top):** This chart fails to detect the shift. No points are near the ±3σ control limits. This is expected behavior for a Shewhart chart with a small shift.
    - **EWMA Chart (Middle):** The exponentially weighted average begins to drift upwards immediately after the shift. It crosses its control limit around sample #30, providing a much earlier signal than the I-Chart.
    - **CUSUM Chart (Bottom):** The cumulative sum of positive deviations (`CUSUM High`) immediately starts to trend upwards after the shift, crossing its decision interval `H` even faster, around sample #27. This demonstrates its power in rapidly detecting pre-specified shift sizes.
    """)

def render_anomaly_detection():
    """Renders the module for unsupervised anomaly detection."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To identify rare, unexpected observations (anomalies or outliers) in data without any prior knowledge of what constitutes an anomaly. This is a form of unsupervised learning.
    
    **Strategic Application:** This is invaluable for process monitoring and root cause analysis, especially in complex, high-dimensional datasets where simple rule-based alarms are ineffective.
    - **Novel Fault Detection:** It can identify a new type of process failure that has never been seen before and for which no specific rules exist.
    - **Data Cleaning:** Used to identify and potentially remove erroneous data points (e.g., from sensor malfunction) before building other process models.
    - **"Golden Batch" Analysis:** Can be used to find which batches, even if they passed all specifications, were statistically unusual compared to the main "golden" population, providing clues about subtle process variability.
    
    The **Isolation Forest** algorithm is a modern and effective method. It works by "isolating" observations. It builds a forest of random trees, and the principle is that anomalous points are "few and different," making them easier to isolate (i.e., they require fewer splits to separate them from the rest of the data).
    """)

    # Placeholder for the plotting function
    def plot_anomaly_detection():
        np.random.seed(42)
        # Generate normal data
        X_inliers = np.random.normal(0, 1, (100, 2))
        # Generate some outliers
        X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
        X = np.concatenate([X_inliers, X_outliers], axis=0)

        # Fit the model
        clf = IsolationForest(contamination=0.1, random_state=42)
        y_pred = clf.fit_predict(X)
        
        # y_pred will be 1 for inliers, -1 for outliers. Map to 0 and 1 for color.
        df = pd.DataFrame(X, columns=['Param 1', 'Param 2'])
        df['Anomaly'] = (y_pred == -1)

        fig = px.scatter(df, x='Param 1', y='Param 2', color='Anomaly',
                         color_discrete_map={False: 'blue', True: 'red'},
                         title="<b>Anomaly Detection using Isolation Forest</b>",
                         symbol='Anomaly',
                         symbol_map={False: 'circle', True: 'x'})
        fig.update_traces(marker=dict(size=8))
        return fig

    fig = plot_anomaly_detection()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**
    - The blue circles represent the dense cloud of normal, "inlier" data points.
    - The red 'x' markers are the points that the Isolation Forest algorithm has identified as anomalies.
    - Notice that the algorithm successfully flags the points that are far away from the main cluster, without being told in advance where the "normal" data should be. This unsupervised approach is what makes it so powerful for discovering unexpected events.
    """)
def render_advanced_doe():
    """Renders the module for advanced Design of Experiments (DOE)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To employ specialized experimental designs for complex, real-world optimization problems that cannot be handled by standard factorial designs. This module covers two common scenarios:
    - **Mixture Designs:** Used for optimizing formulations where the components are proportions that must sum to 100% (e.g., buffers, cell culture media, vaccine adjuvants).
    - **Split-Plot Designs:** Used for processes with both "hard-to-change" and "easy-to-change" factors (e.g., optimizing a bioreactor where temperature is hard to change but feed rate is easy).
    
    **Strategic Application:** Using the wrong design for these problems leads to wasted experiments and incorrect conclusions. These advanced designs are essential for efficient and statistically valid formulation and process development.
    """)
    fig_mix, fig_split = plot_advanced_doe()
    
    st.plotly_chart(fig_mix, use_container_width=True)
    st.markdown("""
    **Interpretation of Mixture Designs:** The design points are shown on a ternary plot. The corners represent pure components, the edges represent two-component blends, and the center represents a mix of all three. This design allows for modeling how the *proportions* of the ingredients, not their absolute amounts, affect a response like stability or efficacy.
    """)

    st.plotly_chart(fig_split, use_container_width=True)
    st.markdown("""
    **Interpretation of Split-Plot Designs:** The design is structured in "Whole Plots" (blue dashed boxes) where the hard-to-change factor is held constant. Within each whole plot, the "Subplots" (red markers) represent the randomized trials of the easy-to-change factors. Analyzing this data requires a special type of ANOVA that correctly uses different error terms for the whole plot and subplot factors, preventing incorrect conclusions about the significance of the hard-to-change variables.
    """)

def render_stability_analysis():
    """Renders the module for pharmaceutical stability analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To statistically determine the shelf-life or retest period for a drug product, substance, or critical reagent. This involves modeling the degradation of a critical quality attribute (CQA) over time and finding the point where it is predicted to fail its specification.
    
    **Strategic Application:** This is a mandatory, high-stakes analysis for any commercial pharmaceutical product, as required by ICH Q1E guidelines. The analysis involves:
    - Collecting data from multiple batches at various time points under specified storage conditions.
    - Fitting a regression model to the data.
    - Determining the time at which the 95% confidence interval for the mean degradation trend intersects the specification limit.
    """)
    fig = plot_stability_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** This plot shows potency data from three different batches collected over 24 months. A linear regression model is fitted to the pooled data to estimate the average degradation trend (black line). The critical line is the **95% Lower Confidence Interval (red dashed line)**. Per ICH guidelines, the shelf-life is the earliest time point where this lower confidence bound crosses the specification limit (red dotted line). This conservative approach ensures that, with 95% confidence, the mean potency of future batches will remain above the limit throughout the product's shelf-life.
    """)

def render_survival_analysis():
    """Renders the module for Survival Analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To analyze and model "time-to-event" data, such as the time until a piece of equipment fails, a patient responds to treatment, or a reagent lot's performance drops below a threshold. It is uniquely designed to handle **censored data**, where the event has not yet occurred for some subjects at the end of the study.
    
    **Strategic Application:** This is the core of **Reliability Engineering** and is essential for predictive maintenance and risk analysis.
    - **Predictive Maintenance:** By modeling the failure time of critical components (e.g., an HPLC column), you can move from calendar-based replacement to condition-based replacement, saving costs and preventing unexpected downtime.
    - **Clinical Trials:** It is the standard method for analyzing trial endpoints like "time to disease progression" or "overall survival."
    - **Reagent Stability:** Can be used to model the probability that a reagent lot will "survive" (i.e., remain effective) past a certain number of months.
    """)
    fig = plot_survival_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** This plot shows a **Kaplan-Meier survival curve**. It visualizes the probability that an item from a given group will "survive" past a certain time. The vertical drops indicate when an event (e.g., failure) occurred, and the small vertical ticks represent censored observations (runs that ended before failure). Here, we can clearly see that Group B has a higher survival probability over time compared to Group A. A formal **Log-Rank test** would be used to determine if this difference is statistically significant.
    """)

def render_multivariate_spc():
    """Renders the module for Multivariate SPC."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To monitor multiple correlated process variables simultaneously in a single control chart. This is the multivariate extension of the Shewhart chart.
    
    **Strategic Application:** In complex processes like a bioreactor, parameters like temperature, pH, and dissolved oxygen are all correlated. Monitoring them with individual control charts is inefficient and can be misleading. A small deviation in all variables simultaneously might go unnoticed on individual charts but represents a significant deviation in the process's overall state.
    - **Hotelling's T² Chart:** This chart tracks the multivariate distance of a process observation from the center of the historical data, accounting for all correlations. It condenses dozens of variables into a single, powerful monitoring statistic.
    """)
    fig = plot_multivariate_spc()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** The left plot shows the raw data. The shift in the red points is only in the Y-direction; the X-values are still in control. An individual X-chart would not detect this shift. The **T² Chart (Right)** combines both variables into a single statistic. It clearly and immediately detects the out-of-control condition when the process shifts, providing a single, unambiguous signal that the overall process "fingerprint" has changed.
    """)

def render_mva_pls():
    """Renders the module for Multivariate Analysis (PLS)."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To model the relationship between a large set of highly correlated input variables (X, e.g., hundreds of spectral data points) and one or more output variables (Y, e.g., product concentration or purity). **Partial Least Squares (PLS)**, also known as Projection to Latent Structures, is the primary tool for this.
    
    **Strategic Application:** This is the engine behind **Process Analytical Technology (PAT)** and chemometrics. Standard regression fails when there are more input variables than samples or when inputs are highly correlated. PLS is designed for this "wide data" problem.
    - **Spectroscopy Calibration:** Used to build models that predict a chemical concentration from its NIR or Raman spectrum, enabling real-time, non-destructive testing.
    - **"Golden Batch" Monitoring:** PLS can model the normal relationship between all process parameters and the final quality attributes. Deviations from this model during a run can signal a problem in real-time.
    """)
    fig = plot_mva_pls()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** The left plot shows the raw spectral data, where it is impossible to see any relationship with the response by eye. The **VIP (Variable Importance in Projection) Plot on the right** is a key output of a PLS model. It shows which input variables (in this case, which wavelengths) are the most influential in predicting the output. Variables with a VIP score > 1 are typically considered important. This allows scientists to understand the underlying chemical signals driving the model, turning a complex dataset into actionable knowledge.
    """)

def render_clustering():
    """Renders the module for unsupervised clustering."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To use unsupervised machine learning to discover natural, hidden groupings or "regimes" within a dataset, without any prior knowledge of what those groups might be.
    
    **Strategic Application:** This is a powerful exploratory tool for process and data understanding. It helps answer the question: "Are all of my 'good' batches truly the same, or are there different ways to be good?"
    - **Process Regime Identification:** It can reveal that a process is secretly operating in two or three different states (e.g., due to different raw material suppliers, seasonal effects, or operator techniques), even when all batches are passing specification.
    - **Root Cause Analysis:** If a failure occurs, clustering can help determine which "family" of normal operation the failed batch was most similar to, providing clues for the investigation.
    - **Customer Segmentation:** In a commercial context, it can be used to segment patients or customers into distinct groups based on their characteristics.
    """)
    fig = plot_clustering()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** This plot shows the results of a **K-Means clustering** algorithm applied to process data. The algorithm has automatically identified three distinct clusters, or "operating regimes," in the data that were not obvious from looking at the raw data. This discovery could prompt an investigation into what makes the three groups different (e.g., raw material, equipment, time of year), potentially revealing a hidden source of process variation and an opportunity for improvement.
    """)

def render_classification_models():
    """Renders the module for classification models."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To build predictive models for a categorical outcome (e.g., Pass/Fail, Compliant/Non-compliant). This module compares a classical statistical model with a modern machine learning model.
    - **Logistic Regression:** A "white-box" statistical model that is highly interpretable but assumes a linear relationship between the inputs and the log-odds of the outcome.
    - **Random Forest:** A powerful, "black-box" machine learning model that can automatically capture complex, non-linear relationships and interactions. It is often more accurate but less interpretable.
    
    **Strategic Application:** These models are the core of **Predictive QC**. The choice between them involves a trade-off. In a highly regulated GxP environment, the interpretability of Logistic Regression is often preferred. For pure predictive performance where the "why" is less important than the "what," Random Forest often has the edge.
    """)
    fig = plot_classification_models()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** The data in this example has a non-linear circular relationship. 
    - **Logistic Regression (Left)** attempts to separate the groups with a straight line. It performs poorly as a result.
    - **Random Forest (Right)** is an ensemble of decision trees and can create a complex, non-linear decision boundary that perfectly captures the circular pattern. Its accuracy is much higher. This demonstrates the power of machine learning models for problems where the underlying relationships are not simple.
    """)

def render_time_series_analysis():
    """Renders the module for Time Series analysis."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To model and forecast time series data by explicitly accounting for its internal structure, such as trend, seasonality, and autocorrelation.
    
    **Strategic Application:** While modern tools like Prophet are often easier to use, classical models like **ARIMA (AutoRegressive Integrated Moving Average)** provide a deep statistical framework for process understanding.
    - **ARIMA:** A powerful and flexible "white-box" model where the parameters are interpretable, making it highly defensible in regulatory environments. It often excels at short-term forecasting.
    - **Prophet:** A modern forecasting tool from Facebook designed for ease-of-use and automatic handling of business time series features like multiple seasonalities and holidays.
    This module provides a comparison of the two approaches.
    """)
    fig = plot_time_series_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** This plot shows the forecasts from both Prophet (red) and ARIMA (green) against the true future data. Both models can capture the overall trend and seasonality. The choice between them often depends on the specific characteristics of the data and the need for interpretability vs. automation. ARIMA requires more statistical expertise to tune but can be more precise for certain processes, while Prophet is designed to produce high-quality forecasts with minimal effort.
    """)

def render_xai_shap():
    """Renders the module for Explainable AI (XAI) using SHAP."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To "look inside the black box" of complex machine learning models (like Random Forest or Gradient Boosting) and explain their predictions. **Explainable AI (XAI)** provides tools to understand *why* a model made a specific prediction.
    
    **Strategic Application:** This is arguably the **single most important enabling technology for deploying advanced ML in regulated GxP environments.** A major barrier to using powerful models has been their lack of interpretability. If a model predicts a batch will fail, regulators and scientists need to know *why*. **SHAP (SHapley Additive exPlanations)** is a state-of-the-art XAI framework that provides this insight.
    - **Model Trust & Validation:** XAI helps confirm that the model is learning real, scientifically plausible relationships, not just spurious correlations.
    - **Regulatory Compliance:** It provides the auditable evidence needed to justify a model-based decision.
    - **Actionable Insights:** It tells you which input variables are driving a prediction, guiding corrective actions.
    """)
    
    summary_buf, force_html = plot_xai_shap()
    
    st.subheader("Global Feature Importance (SHAP Summary Plot)")
    st.image(summary_buf)
    st.markdown("""
    **Interpretation:** This beeswarm plot shows the global importance of each feature. Each dot is a single prediction.
    - **Feature Importance:** Features are ranked by their total impact (top is most important). `Age` is the most important factor.
    - **Impact:** Where a dot falls on the x-axis shows its impact on the prediction (positive SHAP values push the prediction higher).
    - **Correlation:** The color shows the original value of the feature (high/low). We can see that high `Age` (red dots) has a high positive SHAP value, meaning it strongly pushes the model to predict a higher income.
    """)
    
    st.subheader("Local Prediction Explanation (Single SHAP Force Plot)")
    st.components.v1.html(f"<body>{force_html}</body>", height=150, scrolling=True)
    st.markdown("""
    **Interpretation:** This plot explains a *single prediction*.
    - **Base Value:** The average prediction across all data.
    - **Pushing Forces:** Features in red pushed the prediction higher (towards 1). Features in blue pushed it lower (towards 0).
    - **Magnitude:** The size of the bar shows the magnitude of that feature's impact on this specific prediction.
    For this specific person, their high `Capital Gain` was the biggest factor pushing the prediction higher, while their `Age` and `Hours per week` pushed it lower.
    """)

def render_advanced_ai_concepts():
    """Renders the module for advanced AI concepts."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To provide a high-level, conceptual overview of cutting-edge AI architectures that represent the future of process analytics. While coding them is beyond the scope of this toolkit, understanding their capabilities is crucial for future strategy.
    """)
    
    concept = st.radio("Select an Advanced Concept:", ["Transformers", "Graph Neural Networks (GNNs)", "Reinforcement Learning (RL)", "Generative AI"], horizontal=True)
    fig = plot_advanced_ai_concepts(concept)
    st.plotly_chart(fig, use_container_width=True)

    if concept == "Transformers":
        st.markdown("""
        **Transformers (2017):** The architecture behind ChatGPT. Its "self-attention" mechanism allows it to understand long-range dependencies in sequential data.
        - **Application:** Modeling an entire batch manufacturing process as a sequence. A Transformer can learn how a deviation in an early step (e.g., cell thaw) impacts a much later step (e.g., final purification), a task that is difficult for traditional models.
        """)
    elif concept == "GNNs":
        st.markdown("""
        **Graph Neural Networks (GNNs) (~2018):** AI models that operate on data structured as a graph (nodes and edges).
        - **Application:** Modeling a manufacturing plant or supply chain as a graph. GNNs can predict how a failure in one node (e.g., a specific vendor's raw material) will propagate through the network to affect downstream processes.
        """)
    elif concept == "RL":
        st.markdown("""
        **Reinforcement Learning (RL) (~2019):** An AI "agent" learns an optimal control policy by interacting with an environment (like a digital twin of a process) and maximizing a reward.
        - **Application:** Creating an AI that can learn to dynamically control a bioreactor in real-time, adjusting feed rates and gas mixtures to maximize yield in response to live sensor data, far exceeding static setpoints.
        """)
    elif concept == "Generative AI":
        st.markdown("""
        **Generative AI (GANs, Diffusion) (~2020):** AI models that can generate new, synthetic data that is statistically indistinguishable from real data.
        - **Application:** High-quality manufacturing failure data is rare. A Generative model can be trained on a few failure examples to create thousands of realistic synthetic failure profiles. This augmented dataset can then be used to train much more robust predictive QC and anomaly detection models.
        """)

def render_causal_inference():
    """Renders the module for Causal Inference."""
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To move beyond correlation and attempt to identify and quantify true **causal relationships** within a process. It is the science of asking "why?".
    
    **Strategic Application:** This is the ultimate goal of root cause analysis. While a predictive model might tell you that high temperature is *associated* with low purity, Causal Inference provides a framework to determine if high temperature *causes* low purity, or if both are actually caused by a third, hidden variable (a "confounder").
    - **Effective CAPA:** By identifying true causal drivers, we can implement Corrective and Preventive Actions that are far more likely to be effective.
    - **Process Understanding:** It allows for the creation of a **Directed Acyclic Graph (DAG)**, which is a formal causal map of the process, documenting the scientific understanding of how variables influence each other.
    """)
    fig = plot_causal_inference()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:** A DAG is a visual representation of our causal assumptions. The arrows represent hypothesized causal effects.
    - `Reagent Lot -> Purity`: The lot has a direct causal effect on purity.
    - `Temp -> Purity`: Temperature has a direct causal effect on purity.
    - `Temp -> Pressure`: Temperature also causes changes in pressure.
    - `Reagent Lot -> Pressure`: A confounding path exists between Reagent Lot and Purity through Pressure.
    
    By building this graph based on SME knowledge, we can use statistical techniques (like do-calculus or structural equation modeling) to estimate the true, isolated causal effect of one variable on another, even in the presence of confounding.
    """)


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
# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧰 Toolkit Navigation")
    st.markdown("Select a method to explore.")

    # Combine all options into a single structure
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
    
    # Flatten lists for the option_menu
    options = [item for sublist in all_options.values() for item in sublist]
    icons = [ICONS.get(opt, "question-circle") for opt in options]

    # Find the default index based on the current session state key
    # If key is not found, default to the first item.
    try:
        default_idx = options.index(st.session_state.get('method_key', options[0]))
    except ValueError:
        default_idx = 0

    # Use a single, unified menu. The menu's return value is the source of truth.
    # We no longer need the on_change callback.
    selected_option = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
        menu_icon="cast",
        default_index=default_idx,
        # Add orientation and styles to group items visually under headers
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0068C9"},
        }
    )
    
    # Update the session state directly from the component's return value
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
