# ==============================================================================
# LIBRARIES & IMPORTS
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import io
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from scipy import stats
from scipy.stats import beta
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

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

    /* Main container styling for better spacing */
    .main .block-container {
        padding: 2rem 5rem;
        max-width: 1600px;
    }
    
    /* Tab styling for a more professional look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        padding: 0px 24px;
        border-bottom: 2px solid transparent;
        transition: background-color 0.3s, border-bottom 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        font-weight: 600;
        border-bottom: 2px solid #0068C9; /* Professional blue */
    }

    /* Metric styling for a clean, card-like appearance */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        padding: 15px 20px;
        border-radius: 8px;
    }

    /* Sidebar styling - NOTE: This class name is subject to change in future Streamlit versions */
    .st-emotion-cache-16txtl3 { padding: 2rem 1.5rem; }
    
    /* Custom section header class for content panes */
    .section-header {
        font-weight: 600;
        color: #0068C9;
        padding-bottom: 4px;
        border-bottom: 1px solid #E0E0E0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ICONS DICTIONARY FOR SIDEBAR MENU
# ==============================================================================
ICONS = {
    "Gage R&R": "rulers", "Linearity and Range": "graph-up", "LOD & LOQ": "search",
    "Method Comparison": "people-fill", "Assay Robustness (DOE/RSM)": "shield-check",
    "Process Stability (Shewhart)": "activity", "Small Shift Detection": "graph-up-arrow",
    "Run Validation": "check2-circle", "Process Capability (Cpk)": "gem",
    "Anomaly Detection (ML)": "eye-fill", "Predictive QC (ML)": "cpu-fill",
    "Control Forecasting (AI)": "robot", "Pass/Fail Analysis": "toggles",
    "Bayesian Inference": "bullseye", "Confidence Interval Concept": "arrows-angle-expand"
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
def create_conceptual_map_plotly():
    nodes = { 'DS': ('Data Science', 0, 3.5), 'BS': ('Biostatistics', 0, 2.5), 'ST': ('Statistics', 0, 1.5), 'IE': ('Industrial Engineering', 0, 0.5), 'SI': ('Statistical Inference', 1, 2.5), 'SPC': ('SPC', 1, 0.5), 'CC': ('Control Charts', 2, 0), 'PC': ('Process Capability', 2, 1), 'WR': ('Westgard Rules', 2, 2), 'NR': ('Nelson Rules', 2, 3), 'HT': ('Hypothesis Testing', 2, 4), 'CI': ('Confidence Intervals', 2, 5), 'BAY': ('Bayesian Statistics', 2, 6), 'SWH': ('Shewhart Charts', 3, -0.5), 'EWM': ('EWMA', 3, 0), 'CSM': ('CUSUM', 3, 0.5), 'MQA': ('Manufacturing QA', 3, 1.5), 'CL': ('Clinical Labs', 3, 2.5), 'TAV': ('T-tests / ANOVA', 3, 3.5), 'ZME': ('Z-score / Margin of Error', 3, 4.5), 'WS': ('Wilson Score', 3, 5.5), 'PP': ('Posterior Probabilities', 3, 6.5), 'PE': ('Proportion Estimates', 4, 6.0) }
    edges = [('IE', 'SPC'), ('ST', 'SPC'), ('ST', 'SI'), ('BS', 'SI'), ('DS', 'SI'), ('SPC', 'CC'), ('SPC', 'PC'), ('SI', 'HT'), ('SI', 'CI'), ('SI', 'BAY'), ('SI', 'WR'), ('SI', 'NR'), ('CC', 'SWH'), ('CC', 'EWM'), ('CC', 'CSM'), ('PC', 'MQA'), ('WR', 'CL'), ('NR', 'MQA'), ('HT', 'TAV'), ('CI', 'ZME'), ('CI', 'WS'), ('BAY', 'PP'), ('WS', 'PE')]
    fig = go.Figure()
    for start, end in edges: fig.add_trace(go.Scatter(x=[nodes[start][1], nodes[end][1]], y=[nodes[start][2], nodes[end][2]], mode='lines', line=dict(color='lightgrey', width=1.5), hoverinfo='none'))
    node_x, node_y, node_text = [v[1] for v in nodes.values()], [v[2] for v in nodes.values()], [v[0] for v in nodes.values()]
    colors = ["#e0f2f1"]*4 + ["#b2dfdb"]*2 + ["#80cbc4"]*8 + ["#4db6ac"]*10
    fig.add_trace(go.Scatter(x=node_x, y=node_y, text=node_text, mode='markers+text', textposition="middle center", marker=dict(size=45, color=colors, line=dict(width=2, color='black')), textfont=dict(size=10, color='black', family="Arial", weight="bold"), hoverinfo='text'))
    fig.update_layout(title_text='<b>Hierarchical Map of Statistical Concepts</b>', title_x=0.5, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 7.5]), height=700, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
    return fig

def wilson_score_interval(p_hat, n, z=1.96):
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n; term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2))); return (term1 - term2) / denom, (term1 + term2) / denom

# ==============================================================================
# INDIVIDUAL PLOTTING FUNCTIONS (COLLAPSED FOR BREVITY IN PROMPT)
# ==============================================================================
@st.cache_data
def plot_gage_rr():
    np.random.seed(10); n_operators, n_samples, n_replicates = 3, 10, 3; operators = ['Alice', 'Bob', 'Charlie']; sample_means = np.linspace(90, 110, n_samples); operator_bias = {'Alice': 0, 'Bob': -0.5, 'Charlie': 0.8}; data = []
    for op_idx, operator in enumerate(operators):
        for sample_idx, sample_mean in enumerate(sample_means):
            measurements = np.random.normal(sample_mean + operator_bias[operator], 1.5, n_replicates)
            for m_idx, m in enumerate(measurements): data.append([operator, f'Part_{sample_idx+1}', m, m_idx + 1])
    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement', 'Replicate'])
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit(); anova_table = sm.stats.anova_lm(model, typ=2)
    ms_operator, ms_part, ms_interaction, ms_error = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df'], anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df'], anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df'], anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    var_repeatability, var_reproducibility, var_part = ms_error, ((ms_operator - ms_interaction) / (n_samples * n_replicates)) + ((ms_interaction - ms_error) / n_replicates), (ms_part - ms_interaction) / (n_operators * n_replicates)
    variances = {k: max(0, v) for k, v in locals().items() if 'var_' in k}; var_rr = variances['var_repeatability'] + variances['var_reproducibility']; var_total = var_rr + variances['var_part']
    pct_rr, pct_part = (var_rr / var_total) * 100 if var_total > 0 else 0, (variances['var_part'] / var_total) * 100 if var_total > 0 else 0
    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], row_heights=[0.5, 0.5], specs=[[{"rowspan": 2}, {}], [None, {}]], subplot_titles=("<b>Variation by Part & Operator</b>", "<b>Overall Variation by Operator</b>", "<b>Variance Contribution</b>"))
    fig_box = px.box(df, x='Part', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_box.data: trace.update(hoverinfo='none', hovertemplate=None); fig.add_trace(trace, row=1, col=1)
    for operator in operators:
        operator_df = df[df['Operator'] == operator]; part_means = operator_df.groupby('Part')['Measurement'].mean()
        fig.add_trace(go.Scatter(x=part_means.index, y=part_means.values, mode='lines', line=dict(width=2), name=f'{operator} Mean', hoverinfo='none', hovertemplate=None, marker_color=fig_box.data[operators.index(operator)].marker.color), row=1, col=1)
    fig_op_box = px.box(df, x='Operator', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_op_box.data: fig.add_trace(trace, row=1, col=2)
    fig.add_trace(go.Bar(x=['% Gage R&R', '% Part Variation'], y=[pct_rr, pct_part], marker_color=['salmon', 'skyblue'], text=[f'{pct_rr:.1f}%', f'{pct_part:.1f}%'], textposition='auto'), row=2, col=2)
    fig.add_hline(y=10, line_dash="dash", line_color="darkgreen", annotation_text="Acceptable < 10%", annotation_position="bottom right", row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="darkorange", annotation_text="Unacceptable > 30%", annotation_position="top right", row=2, col=2)
    fig.update_layout(title_text='<b>Gage R&R Study: A Multi-View Dashboard</b>', title_x=0.5, height=800, showlegend=False, bargap=0.1, boxmode='group'); fig.update_xaxes(tickangle=45, row=1, col=1)
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
    np.random.seed(42); x = np.linspace(20, 150, 50); y = 0.98 * x + 1.5 + np.random.normal(0, 2.5, 50)
    delta = np.var(y, ddof=1) / np.var(x, ddof=1); x_mean, y_mean = np.mean(x), np.mean(y); Sxx = np.sum((x - x_mean)**2); Sxy = np.sum((x - x_mean)*(y - y_mean))
    beta1_deming = (np.sum((y-y_mean)**2) - delta*Sxx + np.sqrt((np.sum((y-y_mean)**2) - delta*Sxx)**2 + 4*delta*Sxy**2)) / (2*Sxy); beta0_deming = y_mean - beta1_deming*x_mean
    avg, diff = (x + y) / 2, y - x; mean_diff = np.mean(diff); std_diff = np.std(diff, ddof=1); upper_loa, lower_loa = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff; percent_bias = (diff / x) * 100
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], subplot_titles=("<b>Deming Regression</b>", "<b>Bland-Altman Agreement Plot</b>", "<b>Percent Bias vs. Concentration</b>"), vertical_spacing=0.2)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Sample Results', marker=dict(color='blue')), row=1, col=1); fig.add_trace(go.Scatter(x=x, y=beta0_deming + beta1_deming*x, mode='lines', name='Deming Fit', line=dict(color='red')), row=1, col=1); fig.add_trace(go.Scatter(x=[0, 160], y=[0, 160], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=avg, y=diff, mode='markers', name='Difference', marker=dict(color='purple')), row=1, col=2); fig.add_hline(y=mean_diff, line_color="red", annotation_text=f"Mean Bias={mean_diff:.2f}", row=1, col=2); fig.add_hline(y=upper_loa, line_dash="dash", line_color="blue", annotation_text=f"Upper LoA={upper_loa:.2f}", row=1, col=2); fig.add_hline(y=lower_loa, line_dash="dash", line_color="blue", annotation_text=f"Lower LoA={lower_loa:.2f}", row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=percent_bias, mode='markers', name='% Bias', marker=dict(color='orange')), row=2, col=1); fig.add_hrect(y0=-15, y1=15, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1); fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1); fig.add_hline(y=15, line_dash="dot", line_color="red", row=2, col=1); fig.add_hline(y=-15, line_dash="dot", line_color="red", row=2, col=1)
    fig.update_layout(title_text='<b>Method Comparison Dashboard: R&D Lab vs QC Lab</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_xaxes(title_text="R&D Lab (Reference)", row=1, col=1); fig.update_yaxes(title_text="QC Lab (Test)", row=1, col=1)
    fig.update_xaxes(title_text="Average of Methods", row=1, col=2); fig.update_yaxes(title_text="Difference (QC - R&D)", row=1, col=2)
    fig.update_xaxes(title_text="R&D Lab (Reference Concentration)", row=2, col=1); fig.update_yaxes(title_text="% Bias", range=[-25, 25], row=2, col=1)
    return fig, beta1_deming, beta0_deming, mean_diff, upper_loa, lower_loa

@st.cache_data
def plot_robustness_rsm():
    np.random.seed(42); factors = {'Temp': [-1, 1, -1, 1, -1.414, 1.414, 0, 0, 0, 0, 0, 0, 0], 'pH': [-1, -1, 1, 1, 0, 0, -1.414, 1.414, 0, 0, 0, 0, 0]}; df = pd.DataFrame(factors)
    df['Response'] = 95 - 5*df['Temp'] + 2*df['pH'] - 4*(df['Temp']**2) - 2*(df['pH']**2) + 3*df['Temp']*df['pH'] + np.random.normal(0, 1.5, len(df))
    rsm_model = ols('Response ~ Temp + pH + I(Temp**2) + I(pH**2) + Temp:pH', data=df).fit()
    screening_model = ols('Response ~ Temp * pH', data=df).fit(); effects = screening_model.params.iloc[1:].sort_values(key=abs, ascending=False); p_values = screening_model.pvalues.iloc[1:][effects.index]
    effect_data = pd.DataFrame({'Effect': effects.index, 'Value': effects.values, 'p_value': p_values}); effect_data['color'] = np.where(effect_data['p_value'] < 0.05, 'salmon', 'skyblue'); significance_threshold = np.abs(effects.values).mean() * 1.5
    fig_pareto = px.bar(effect_data, x='Value', y='Effect', orientation='h', title="<b>Pareto Plot of Factor Effects</b>", text=np.round(effect_data['Value'], 2), labels={'Value': 'Standardized Effect Magnitude', 'Effect': 'Factor or Interaction'}, custom_data=['p_value'])
    fig_pareto.update_traces(marker_color=effect_data['color'], hovertemplate="<b>%{y}</b><br>Effect Value: %{x:.3f}<br>P-value: %{customdata[0]:.3f}<extra></extra>")
    fig_pareto.add_vline(x=significance_threshold, line_dash="dash", line_color="red", annotation_text="Significance Threshold"); fig_pareto.add_vline(x=-significance_threshold, line_dash="dash", line_color="red"); fig_pareto.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
    temp_range, ph_range = np.linspace(-2, 2, 50), np.linspace(-2, 2, 50); grid_temp, grid_ph = np.meshgrid(temp_range, ph_range); grid_df = pd.DataFrame({'Temp': grid_temp.ravel(), 'pH': grid_ph.ravel()}); grid_df['Predicted_Response'] = rsm_model.predict(grid_df); predicted_response_grid = grid_df['Predicted_Response'].values.reshape(50, 50)
    opt_idx = grid_df['Predicted_Response'].idxmax(); opt_temp, opt_ph, opt_response = grid_df.loc[opt_idx]
    fig_contour = go.Figure(data=go.Contour(z=predicted_response_grid, x=temp_range, y=ph_range, colorscale='Viridis', contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=12, color='white')), line=dict(width=2), hoverinfo='x+y+z'))
    fig_contour.add_trace(go.Scatter(x=df['Temp'], y=df['pH'], mode='markers', marker=dict(color='white', size=10, symbol='x', line=dict(color='black', width=2)), name='Design Points', hovertemplate="Temp: %{x:.2f}<br>pH: %{y:.2f}<extra></extra>"))
    fig_contour.add_trace(go.Scatter(x=[opt_temp], y=[opt_ph], mode='markers', marker=dict(color='red', size=16, symbol='star', line=dict(color='white', width=2)), name='Optimal Point', hovertemplate=f"<b>Optimal Point</b><br>Temp: {opt_temp:.2f}<br>pH: {opt_ph:.2f}<br>Response: {opt_response:.2f}<extra></extra>"))
    fig_contour.update_layout(title="<b>2D Contour Plot of Response Surface</b>", title_x=0.5, xaxis_title="Temperature (coded units)", yaxis_title="pH (coded units)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig_surface = go.Figure(data=[go.Surface(z=predicted_response_grid, x=temp_range, y=ph_range, colorscale='Viridis', contours = {"x": {"show": True, "start": -2, "end": 2, "size": 0.5, "color":"white"}, "y": {"show": True, "start": -2, "end": 2, "size": 0.5, "color":"white"}, "z": {"show": True, "start": predicted_response_grid.min(), "end": predicted_response_grid.max(), "size": 5}}, hoverinfo='x+y+z')])
    fig_surface.add_trace(go.Scatter3d(x=df['Temp'], y=df['pH'], z=df['Response'], mode='markers', marker=dict(color='red', size=5, symbol='diamond'), name='Design Points'))
    fig_surface.add_trace(go.Scatter3d(x=[opt_temp], y=[opt_ph], z=[opt_response], mode='markers', marker=dict(color='yellow', size=10, symbol='diamond'), name='Optimal Point'))
    fig_surface.update_layout(title='<b>3D Response Surface Plot</b>', title_x=0.5, scene=dict(xaxis_title="Temperature", yaxis_title="pH", zaxis_title="Assay Response", camera=dict(eye=dict(x=1.8, y=-1.8, z=1.5))), margin=dict(l=0, r=0, b=0, t=40))
    return fig_pareto, fig_contour, fig_surface, effects

@st.cache_data
def plot_shewhart():
    np.random.seed(42); in_control_data = np.random.normal(loc=100.0, scale=2.0, size=15); reagent_shift_data = np.random.normal(loc=108.0, scale=2.0, size=10); data = np.concatenate([in_control_data, reagent_shift_data]); x = np.arange(1, len(data) + 1)
    mean = np.mean(data[:15]); mr = np.abs(np.diff(data)); mr_mean = np.mean(mr[:14]); sigma_est = mr_mean / 1.128; UCL_I, LCL_I = mean + 3 * sigma_est, mean - 3 * sigma_est; UCL_MR = mr_mean * 3.267
    out_of_control_I_idx = np.where((data > UCL_I) | (data < LCL_I))[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>I-Chart: Monitors Accuracy (Bias)</b>", "<b>MR-Chart: Monitors Precision (Variability)</b>"), vertical_spacing=0.1, row_heights=[0.7, 0.3])
    for i, color in zip([1, 2, 3], ['#a5d6a7', '#fff59d', '#ef9a9a']): fig.add_hrect(y0=mean - i*sigma_est, y1=mean + i*sigma_est, fillcolor=color, opacity=0.3, layer="below", line_width=0, row=1, col=1)
    fig.add_hline(y=mean, line=dict(dash='dash', color='black'), annotation_text=f"Mean={mean:.1f}", row=1, col=1); fig.add_hline(y=UCL_I, line=dict(color='red'), annotation_text=f"UCL={UCL_I:.1f}", row=1, col=1); fig.add_hline(y=LCL_I, line=dict(color='red'), annotation_text=f"LCL={LCL_I:.1f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data, mode='lines+markers', name='Control Value', line=dict(color='royalblue'), marker=dict(color='royalblue', size=6), hovertemplate="Run %{x}<br>Value: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[out_of_control_I_idx], y=data[out_of_control_I_idx], mode='markers', name='Out of Control Signal', marker=dict(color='red', size=12, symbol='x-thin', line=dict(width=3)), hovertemplate="<b>VIOLATION</b><br>Run %{x}<br>Value: %{y:.2f}<extra></extra>"), row=1, col=1)
    for idx in out_of_control_I_idx: fig.add_annotation(x=x[idx], y=data[idx], text="Rule 1 Violation", showarrow=True, arrowhead=2, ax=20, ay=-40, row=1, col=1, font=dict(color="red"))
    fig.add_vrect(x0=15.5, x1=25.5, fillcolor="rgba(255,165,0,0.2)", layer="below", line_width=0, annotation_text="New Reagent Lot", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Scatter(x=x[1:], y=mr, mode='lines+markers', name='Moving Range', line=dict(color='teal'), hovertemplate="Range (Run %{x}-%{x_prev})<br>Value: %{y:.2f}<extra></extra>".replace('%{x_prev}', str(list(x[:-1])))), row=2, col=1)
    fig.add_hline(y=mr_mean, line=dict(dash='dash', color='black'), annotation_text=f"Mean={mr_mean:.1f}", row=2, col=1); fig.add_hline(y=UCL_MR, line=dict(color='red'), annotation_text=f"UCL={UCL_MR:.1f}", row=2, col=1)
    fig.update_layout(title_text='<b>Process Stability Monitoring: Shewhart I-MR Chart</b>', title_x=0.5, height=800, showlegend=False, margin=dict(t=100))
    fig.update_yaxes(title_text="Concentration (ng/mL)", row=1, col=1); fig.update_yaxes(title_text="Range (ng/mL)", row=2, col=1); fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1)
    return fig

@st.cache_data
def plot_ewma_cusum(chart_type, lmbda, k_sigma, H_sigma):
    np.random.seed(101); in_control_data = np.random.normal(50, 2, 25); shift_data = np.random.normal(52.5, 2, 15); data = np.concatenate([in_control_data, shift_data]); target = np.mean(in_control_data); sigma = np.std(in_control_data, ddof=1); x_axis = np.arange(1, len(data) + 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Raw Process Data</b>", f"<b>{chart_type} Chart</b>"), vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', name='Daily Control', marker=dict(color='grey'), line=dict(color='lightgrey'), hovertemplate="Run %{x}<br>Value: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=target, line_dash="dash", line_color="black", annotation_text=f"Target Mean={target:.1f}", row=1, col=1); fig.add_vrect(x0=25.5, x1=40.5, fillcolor="orange", opacity=0.2, layer="below", line_width=0, annotation_text="1.25σ Shift Introduced", annotation_position="top left", row=1, col=1)
    if chart_type == 'EWMA':
        ewma_vals = np.zeros_like(data); ewma_vals[0] = target;
        for i in range(1, len(data)): ewma_vals[i] = lmbda * data[i] + (1 - lmbda) * ewma_vals[i-1]
        L = 3; UCL = [target + L * sigma * np.sqrt((lmbda / (2 - lmbda)) * (1 - (1 - lmbda)**(2 * i))) for i in range(1, len(data) + 1)]; out_idx = np.where(ewma_vals > UCL)[0]
        fig.add_trace(go.Scatter(x=x_axis, y=ewma_vals, mode='lines+markers', name=f'EWMA (λ={lmbda})', line=dict(color='purple'), hovertemplate="Run %{x}<br>EWMA: %{y:.2f}<extra></extra>"), row=2, col=1); fig.add_trace(go.Scatter(x=x_axis, y=UCL, mode='lines', name='EWMA UCL', line=dict(color='red', dash='dash')), row=2, col=1)
        if len(out_idx) > 0: signal_idx = out_idx[0]; fig.add_trace(go.Scatter(x=[x_axis[signal_idx]], y=[ewma_vals[signal_idx]], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Signal'), row=2, col=1); fig.add_annotation(x=x_axis[signal_idx], y=ewma_vals[signal_idx], text="<b>Signal!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, row=2, col=1)
        fig.update_yaxes(title_text="EWMA Value", row=2, col=1)
    else: # CUSUM
        k = k_sigma * sigma; H = H_sigma * sigma; SH, SL = np.zeros_like(data), np.zeros_like(data)
        for i in range(1, len(data)): SH[i], SL[i] = max(0, SH[i-1] + (data[i] - target) - k), max(0, SL[i-1] + (target - data[i]) - k)
        out_idx = np.where((SH > H) | (SL > H))[0]
        fig.add_trace(go.Scatter(x=x_axis, y=SH, mode='lines+markers', name='High-Side CUSUM (SH)', line=dict(color='darkcyan')), row=2, col=1); fig.add_trace(go.Scatter(x=x_axis, y=SL, mode='lines+markers', name='Low-Side CUSUM (SL)', line=dict(color='darkorange')), row=2, col=1); fig.add_hline(y=H, line_dash="dash", line_color="red", annotation_text=f"Limit H={H:.1f}", row=2, col=1)
        if len(out_idx) > 0: signal_idx = out_idx[0]; fig.add_trace(go.Scatter(x=[x_axis[signal_idx]], y=[SH[signal_idx]], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Signal'), row=2, col=1); fig.add_annotation(x=x_axis[signal_idx], y=SH[signal_idx], text="<b>Signal!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Sum", row=2, col=1)
    fig.update_layout(title_text=f'<b>Small Shift Detection Dashboard ({chart_type})</b>', title_x=0.5, height=800, showlegend=False); fig.update_yaxes(title_text="Assay Response", row=1, col=1); fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1)
    return fig

@st.cache_data
def plot_multi_rule():
    np.random.seed(3); mean, std = 100, 2; data = np.array([100.5, 99.8, 101.2, 98.9, 100.2, 104.5, 105.1, 100.1, 99.5, 102.3, 102.8, 103.1, 102.5, 99.9, 106.5, 100.8, 98.5, 104.2, 95.5, 100.0]); x = np.arange(1, len(data) + 1); z_scores = (data - mean) / std
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Levey-Jennings Chart with Westgard Violations</b>", "<b>Distribution of QC Data</b>"), vertical_spacing=0.15, row_heights=[0.7, 0.3])
    for i, color in zip([3, 2, 1], ['#ef9a9a', '#fff59d', '#a5d6a7']): fig.add_hrect(y0=mean - i*std, y1=mean + i*std, fillcolor=color, opacity=0.3, layer="below", line_width=0, row=1, col=1)
    for i in [-3, -2, -1, 1, 2, 3]: fig.add_hline(y=mean + i*std, line_dash="dot", line_color="gray", annotation_text=f"{i}s", row=1, col=1)
    fig.add_hline(y=mean, line_dash="dash", line_color="black", annotation_text="Mean", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data, mode='lines+markers', name='QC Sample', line=dict(color='darkblue'), hovertemplate="Run: %{x}<br>Value: %{y:.2f}<br>Z-Score: %{customdata:.2f}s<extra></extra>", customdata=z_scores), row=1, col=1)
    violations = [];
    if np.any(np.abs(z_scores) > 3): idx = np.where(np.abs(z_scores) > 3)[0][0]; violations.append({'x': x[idx], 'y': data[idx], 'rule': '1_3s Violation'})
    for i in range(1, len(z_scores)):
        if (z_scores[i] > 2 and z_scores[i-1] > 2) or (z_scores[i] < -2 and z_scores[i-1] < -2): violations.append({'x': x[i], 'y': data[i], 'rule': '2_2s Violation'})
    for i in range(3, len(z_scores)):
        if np.all(z_scores[i-3:i+1] > 1) or np.all(z_scores[i-3:i+1] < -1): violations.append({'x': x[i], 'y': data[i], 'rule': '4_1s Violation'})
    for i in range(1, len(z_scores)):
        if (z_scores[i] > 2 and z_scores[i-1] < -2) or (z_scores[i] < -2 and z_scores[i-1] > 2): violations.append({'x': x[i], 'y': data[i], 'rule': 'R_4s Violation'})
    violation_points = pd.DataFrame(violations)
    if not violation_points.empty:
        fig.add_trace(go.Scatter(x=violation_points['x'], y=violation_points['y'], mode='markers', name='Violation', marker=dict(color='red', size=15, symbol='x-thin', line=dict(width=3))), row=1, col=1)
        for _, row in violation_points.iterrows(): fig.add_annotation(x=row['x'], y=row['y'], text=f"<b>{row['rule']}</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="red"), row=1, col=1)
    fig.add_trace(go.Histogram(x=data, name='Distribution', histnorm='probability density', marker_color='darkblue'), row=2, col=1); x_norm = np.linspace(mean - 4*std, mean + 4*std, 100); y_norm = stats.norm.pdf(x_norm, mean, std); fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Curve', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.update_layout(title_text='<b>QC Run Validation Dashboard</b>', title_x=0.5, height=800, showlegend=False)
    fig.update_yaxes(title_text="Measured Value", row=1, col=1); fig.update_yaxes(title_text="Density", row=2, col=1); fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1)
    return fig

@st.cache_data
def plot_capability(scenario):
    np.random.seed(42); LSL, USL = 90, 110
    if scenario == 'Ideal': data = np.random.normal(100, (USL-LSL)/(6*1.67), 200)
    elif scenario == 'Shifted': data = np.random.normal(105, (USL-LSL)/(6*1.67), 200)
    elif scenario == 'Variable': data = np.random.normal(100, (USL-LSL)/(6*0.9), 200)
    else: data = np.concatenate([np.random.normal(97, 2, 100), np.random.normal(103, 2, 100)])
    sigma_hat = np.std(data, ddof=1); Cpu = (USL - data.mean()) / (3 * sigma_hat); Cpl = (data.mean() - LSL) / (3 * sigma_hat); Cpk = np.min([Cpu, Cpl])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Process Control (I-Chart)</b>", "<b>Process Capability (Histogram)</b>"), vertical_spacing=0.1, row_heights=[0.4, 0.6])
    x_axis = np.arange(1, len(data) + 1); mean_i = data.mean(); mr = np.abs(np.diff(data)); mr_mean = np.mean(mr); UCL_I, LCL_I = mean_i + 3*(mr_mean/1.128), mean_i - 3*(mr_mean/1.128)
    fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines', line=dict(color='lightgrey'), name='Control Value'), row=1, col=1)
    out_of_control_idx = np.where((data > UCL_I) | (data < LCL_I))[0]; fig.add_trace(go.Scatter(x=x_axis[out_of_control_idx], y=data[out_of_control_idx], mode='markers', marker=dict(color='red', size=8), name='Signal'), row=1, col=1)
    fig.add_hline(y=mean_i, line_dash="dash", line_color="black", row=1, col=1); fig.add_hline(y=UCL_I, line_color="red", row=1, col=1); fig.add_hline(y=LCL_I, line_color="red", row=1, col=1)
    fig_hist = px.histogram(data, nbins=30, histnorm='probability density'); fig.add_trace(fig_hist.data[0], row=2, col=1)
    fig.add_vline(x=LSL, line_dash="dash", line_color="red", annotation_text="LSL", row=2, col=1); fig.add_vline(x=USL, line_dash="dash", line_color="red", annotation_text="USL", row=2, col=1); fig.add_vline(x=data.mean(), line_dash="dot", line_color="black", annotation_text="Mean", row=2, col=1)
    color = "darkgreen" if Cpk >= 1.33 and scenario != 'Out of Control' else "darkred"; text = f"Cpk = {Cpk:.2f}" if scenario != 'Out of Control' else "Cpk: INVALID"
    fig.add_annotation(text=text, align='left', showarrow=False, xref='paper', yref='paper', x=0.05, y=0.45, bordercolor="black", borderwidth=1, bgcolor=color, font=dict(color="white"))
    fig.update_layout(title_text=f'<b>Process Capability Analysis - Scenario: {scenario}</b>', title_x=0.5, height=800, showlegend=False);
    return fig, Cpk, scenario

@st.cache_data
def plot_anomaly_detection():
    np.random.seed(42); X_normal = np.random.multivariate_normal([100, 20], [[5, 2],[2, 1]], 200); X_anomalies = np.array([[95, 25], [110, 18], [115, 28]]); X = np.vstack([X_normal, X_anomalies])
    model = IsolationForest(n_estimators=100, contamination=0.015, random_state=42); model.fit(X); y_pred = model.predict(X)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-5, X[:, 0].max()+5, 100), np.linspace(X[:, 1].min()-5, X[:, 1].max()+5, 100)); Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure(); fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z, colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(0, 0, 255, 0.2)']], showscale=False, hoverinfo='none'))
    df_plot = pd.DataFrame(X, columns=['x', 'y']); df_plot['status'] = ['Anomaly' if p == -1 else 'Normal' for p in y_pred]
    normal_df, anomaly_df = df_plot[df_plot['status'] == 'Normal'], df_plot[df_plot['status'] == 'Anomaly']
    fig.add_trace(go.Scatter(x=normal_df['x'], y=normal_df['y'], mode='markers', marker=dict(color='royalblue', size=8, line=dict(width=1, color='black')), name='Normal Run', hovertemplate="<b>Status: Normal</b><br>Response: %{x:.2f}<br>Time: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=anomaly_df['x'], y=anomaly_df['y'], mode='markers', marker=dict(color='red', size=12, symbol='x-thin', line=dict(width=3)), name='Anomaly', hovertemplate="<b>Status: Anomaly</b><br>Response: %{x:.2f}<br>Time: %{y:.2f}<extra></extra>"))
    fig.update_layout(title_text='<b>Multivariate Anomaly Detection (Isolation Forest)</b>', title_x=0.5, xaxis_title='Assay Response (Fluorescence Units)', yaxis_title='Incubation Time (min)', height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

@st.cache_data
def plot_predictive_qc():
    np.random.seed(1); n_points = 150; X1 = np.random.normal(5, 2, n_points); X2 = np.random.normal(25, 3, n_points)
    logit_p = -15 + 1.5 * X1 + 0.5 * X2 + np.random.normal(0, 2, n_points); p = 1 / (1 + np.exp(-logit_p)); y = np.random.binomial(1, p); X = np.vstack([X1, X2]).T
    model = LogisticRegression().fit(X, y); probabilities = model.predict_proba(X)[:, 1]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Decision Boundary Risk Map</b>", "<b>Model Performance: Probability Distributions</b>"), column_widths=[0.6, 0.4])
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)); Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z, colorscale='RdYlGn_r', colorbar=dict(title="P(Fail)"), showscale=True, hoverinfo='none'), row=1, col=1)
    df_plot = pd.DataFrame(X, columns=['Reagent Age', 'Incubation Temp']); df_plot['Outcome'] = ['Pass' if i == 0 else 'Fail' for i in y]; df_plot['P(Fail)'] = probabilities
    fig_scatter = px.scatter(df_plot, x='Reagent Age', y='Incubation Temp', color='Outcome', color_discrete_map={'Pass':'green', 'Fail':'red'}, symbol='Outcome', symbol_map={'Pass': 'circle', 'Fail': 'x'}, custom_data=['P(Fail)'])
    for trace in fig_scatter.data: trace.update(hovertemplate="<b>%{customdata[0]:.1%} P(Fail)</b><br>Age: %{x:.1f} days<br>Temp: %{y:.1f}°C<extra></extra>"); fig.add_trace(trace, row=1, col=1)
    fig.add_trace(go.Histogram(x=df_plot[df_plot['Outcome'] == 'Pass']['P(Fail)'], name='Actual Pass', histnorm='probability density', marker_color='green', opacity=0.7), row=1, col=2); fig.add_trace(go.Histogram(x=df_plot[df_plot['Outcome'] == 'Fail']['P(Fail)'], name='Actual Fail', histnorm='probability density', marker_color='red', opacity=0.7), row=1, col=2)
    fig.update_layout(title_text='<b>Predictive QC Dashboard: Identifying At-Risk Runs</b>', title_x=0.5, height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), barmode='overlay')
    fig.update_xaxes(title_text="Reagent Age (days)", row=1, col=1); fig.update_yaxes(title_text="Incubation Temp (°C)", row=1, col=1); fig.update_xaxes(title_text="Predicted Probability of Failure", row=1, col=2); fig.update_yaxes(title_text="Density", row=1, col=2)
    return fig

@st.cache_data
def plot_forecasting():
    np.random.seed(42); dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W'))
    trend1, trend2 = np.linspace(0, 2, 52), np.linspace(2.1, 8, 52); trend = np.concatenate([trend1, trend2]); seasonality = 1.5 * np.sin(np.arange(104) * (2 * np.pi / 52.14)); noise = np.random.normal(0, 0.5, 104); y = 50 + trend + seasonality + noise; df = pd.DataFrame({'ds': dates, 'y': y})
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.5); model.fit(df); future = model.make_future_dataframe(periods=26, freq='W'); forecast = model.predict(future)
    fig1 = plot_plotly(model, forecast); fig1.update_layout(title_text='<b>Control Performance Forecast vs. Specification Limit</b>', title_x=0.5, xaxis_title='Date', yaxis_title='Control Value (U/mL)', showlegend=True)
    spec_limit = 58; fig1.add_hline(y=spec_limit, line_dash="dash", line_color="red", annotation_text="Upper Spec Limit"); breaches = forecast[forecast['yhat_upper'] > spec_limit]
    if not breaches.empty: fig1.add_trace(go.Scatter(x=breaches['ds'], y=breaches['yhat'], mode='markers', name='Predicted Breach', marker=dict(color='red', size=10, symbol='diamond')))
    fig1.add_vrect(x0=forecast['ds'].iloc[-26], x1=forecast['ds'].iloc[-1], fillcolor="rgba(0,100,80,0.1)", layer="below", line_width=0, annotation_text="Forecast Horizon", annotation_position="top left")
    fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='navy')))
    if len(model.changepoints) > 0:
        signif_changepoints = model.changepoints[np.abs(np.nanmean(model.params['delta'], axis=0)) >= 0.01]
        if len(signif_changepoints) > 0:
            for cp in signif_changepoints: fig2.add_vline(x=cp, line_width=1, line_dash="dash", line_color="red")
    fig2.update_layout(title_text='<b>Decomposed Trend with Detected Changepoints</b>', title_x=0.5, xaxis_title='Date', yaxis_title='Trend Value')
    fig3_full = plot_components_plotly(model, forecast, figsize=(900, 200)); fig3 = go.Figure()
    for trace in fig3_full.select_traces(selector=dict(xaxis='x2')): fig3.add_trace(trace)
    fig3.update_layout(title_text='<b>Decomposed Yearly Seasonal Effect</b>', title_x=0.5, xaxis_title='Day of Year', yaxis_title='Seasonal Component', showlegend=False)
    return fig1, fig2, fig3

@st.cache_data
def plot_wilson(successes, n_samples):
    p_hat = successes / n_samples if n_samples > 0 else 0; wald_lower, wald_upper = stats.norm.interval(0.95, loc=p_hat, scale=np.sqrt(p_hat*(1-p_hat)/n_samples)) if n_samples > 0 else (0,0); wilson_lower, wilson_upper = wilson_score_interval(p_hat, n_samples); cp_lower, cp_upper = stats.beta.interval(0.95, successes + 1e-9, n_samples - successes + 1) if n_samples > 0 else (0,1)
    intervals = {"Wald (Approximate)": (wald_lower, wald_upper, 'red'), "Wilson Score": (wilson_lower, wilson_upper, 'blue'), "Clopper-Pearson (Exact)": (cp_lower, cp_upper, 'green')}
    fig1 = go.Figure()
    for name, (lower, upper, color) in intervals.items():
        fig1.add_trace(go.Scatter(x=[p_hat], y=[name], error_x=dict(type='data', array=[upper-p_hat], arrayminus=[p_hat-lower]), mode='markers', marker=dict(color=color, size=12), name=name, hovertemplate=f"<b>{name}</b><br>Observed: {p_hat:.2%}<br>Lower: {lower:.3f}<br>Upper: {upper:.3f}<extra></extra>"))
    fig1.add_vrect(x0=0.9, x1=1.0, fillcolor="rgba(0,255,0,0.1)", layer="below", line_width=0, annotation_text="Target Zone > 90%", annotation_position="bottom left"); fig1.update_layout(title_text=f'<b>Comparing 95% CIs for {successes}/{n_samples} Concordant Results</b>', title_x=0.5, xaxis_title='Concordance Rate', showlegend=False, height=500, xaxis_range=[-0.05, 1.05])
    true_proportions = np.linspace(0.01, 0.99, 200); n_coverage = n_samples
    @st.cache_data
    def calculate_coverage(n_cov, p_array):
        wald_cov, wilson_cov, cp_cov = [], [], []
        for p in p_array:
            k = np.arange(0, n_cov + 1); p_k = stats.binom.pmf(k, n_cov, p)
            wald_l, wald_u = stats.norm.interval(0.95, loc=k/n_cov, scale=np.sqrt((k/n_cov)*(1-k/n_cov)/n_cov)); wald_cov.append(np.sum(p_k[(wald_l <= p) & (p <= wald_u)]))
            wilson_l, wilson_u = wilson_score_interval(k/n_cov, n_cov); wilson_cov.append(np.sum(p_k[(wilson_l <= p) & (p <= wilson_u)]))
            cp_l, cp_u = stats.beta.interval(0.95, k + 1e-9, n_cov - k + 1); cp_cov.append(np.sum(p_k[(cp_l <= p) & (p <= cp_u)]))
        return wald_cov, wilson_cov, cp_cov
    wald_coverage, wilson_coverage, cp_coverage = calculate_coverage(n_coverage, true_proportions)
    fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=true_proportions, y=wald_coverage, mode='lines', name='Wald', line=dict(color='red'))); fig2.add_trace(go.Scatter(x=true_proportions, y=wilson_coverage, mode='lines', name='Wilson Score', line=dict(color='blue'))); fig2.add_trace(go.Scatter(x=true_proportions, y=cp_coverage, mode='lines', name='Clopper-Pearson', line=dict(color='green')))
    fig2.add_hrect(y0=0, y1=0.95, fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0); fig2.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Nominal 95% Coverage")
    fig2.update_layout(title_text=f'<b>Coverage Probability for n={n_samples}</b>', title_x=0.5, xaxis_title='True Proportion', yaxis_title='Actual Coverage Probability', yaxis_range=[min(0.8, np.nanmin(wald_coverage)-0.02 if np.any(np.isfinite(wald_coverage)) else 0.8), 1.02], legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
    return fig1, fig2

@st.cache_data
def plot_bayesian(prior_type):
    n_qc, successes_qc = 20, 18; observed_rate = successes_qc / n_qc
    if prior_type == "Strong R&D Prior": prior_alpha, prior_beta = 490, 10
    elif prior_type == "Skeptical/Regulatory Prior": prior_alpha, prior_beta = 10, 10
    else: prior_alpha, prior_beta = 1, 1
    p_range = np.linspace(0.6, 1.0, 501)
    prior_dist, prior_mean = beta.pdf(p_range, prior_alpha, prior_beta), prior_alpha / (prior_alpha + prior_beta)
    likelihood = stats.binom.pmf(k=successes_qc, n=n_qc, p=p_range)
    posterior_alpha, posterior_beta = prior_alpha + successes_qc, prior_beta + (n_qc - successes_qc); posterior_dist, posterior_mean = beta.pdf(p_range, posterior_alpha, posterior_beta), posterior_alpha / (posterior_alpha + posterior_beta)
    fig = go.Figure(); max_y = np.max(posterior_dist) * 1.1
    fig.add_trace(go.Scatter(x=p_range, y=likelihood * max_y / np.max(likelihood), mode='lines', name='Likelihood (from QC Data)', line=dict(dash='dot', color='red', width=2), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)', hovertemplate="p=%{x:.3f}<br>Likelihood (scaled)<extra></extra>"))
    fig.add_trace(go.Scatter(x=p_range, y=prior_dist, mode='lines', name='Prior Belief', line=dict(dash='dash', color='green', width=3), hovertemplate="p=%{x:.3f}<br>Prior Density: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=p_range, y=posterior_dist, mode='lines', name='Posterior Belief', line=dict(color='blue', width=4), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)', hovertemplate="p=%{x:.3f}<br>Posterior Density: %{y:.2f}<extra></extra>"))
    fig.add_vline(x=prior_mean, line_dash="dash", line_color="green", annotation_text=f"Prior Mean={prior_mean:.3f}"); fig.add_vline(x=observed_rate, line_dash="dot", line_color="red", annotation_text=f"Data (MLE)={observed_rate:.3f}"); fig.add_vline(x=posterior_mean, line_dash="solid", line_color="blue", annotation_text=f"Posterior Mean={posterior_mean:.3f}", annotation_font=dict(color="blue", size=14))
    fig.update_layout(title_text='<b>Bayesian Inference: How Evidence Updates Belief</b>', title_x=0.5, xaxis_title='Assay Pass Rate (Concordance)', yaxis_title='Probability Density / Scaled Likelihood', height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig, prior_mean, observed_rate, posterior_mean

@st.cache_data
def plot_ci_concept(n=30):
    np.random.seed(123); pop_mean, pop_std = 100, 15; n_sims = 100; population = np.random.normal(pop_mean, pop_std, 10000); sample_means = [np.mean(np.random.normal(pop_mean, pop_std, n)) for _ in range(1000)]
    fig1 = go.Figure()
    kde_pop = stats.gaussian_kde(population); x_range_pop = np.linspace(population.min(), population.max(), 500); fig1.add_trace(go.Scatter(x=x_range_pop, y=kde_pop(x_range_pop), fill='tozeroy', name='True Population Distribution', marker_color='skyblue', opacity=0.6, hoverinfo='none'))
    kde_means = stats.gaussian_kde(sample_means); x_range_means = np.linspace(min(sample_means), max(sample_means), 500); fig1.add_trace(go.Scatter(x=x_range_means, y=kde_means(x_range_means), fill='tozeroy', name=f'Distribution of Sample Means (n={n})', marker_color='darkorange', opacity=0.6, hoverinfo='none'))
    our_sample_mean = sample_means[0]; fig1.add_trace(go.Scatter(x=[our_sample_mean], y=[0], mode='markers', name='Our One Sample Mean', marker=dict(color='black', size=12, symbol='x'), hovertemplate=f"Our Sample Mean: {our_sample_mean:.2f}<extra></extra>"))
    fig1.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}"); fig1.update_layout(title_text=f"<b>The Theoretical Universe (Sample Size n={n})</b>", title_x=0.5, yaxis_title="Density", xaxis_title="Value", showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig2 = go.Figure(); capture_count, total_width = 0, 0
    for i in range(n_sims):
        sample = np.random.normal(pop_mean, pop_std, n); sample_mean = np.mean(sample); margin_of_error = 1.96 * (pop_std / np.sqrt(n)); ci_lower, ci_upper = sample_mean - margin_of_error, sample_mean + margin_of_error; total_width += (ci_upper - ci_lower)
        color = 'cornflowerblue' if ci_lower <= pop_mean <= ci_upper else 'red';
        if color == 'cornflowerblue': capture_count += 1
        status = "Capture" if color == 'cornflowerblue' else "Miss"
        fig2.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[i, i], mode='lines', line=dict(color=color, width=3), hovertemplate=f"<b>Run {i+1} (n={n})</b><br>Status: {status}<br>Interval: [{ci_lower:.2f}, {ci_upper:.2f}]<extra></extra>"))
        fig2.add_trace(go.Scatter(x=[sample_mean], y=[i], mode='markers', marker=dict(color='black', size=5, symbol='line-ns-open'), hovertemplate=f"<b>Run {i+1} (n={n})</b><br>Sample Mean: {sample_mean:.2f}<extra></extra>"))
    avg_width = total_width / n_sims; fig2.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}"); fig2.update_layout(title_text=f"<b>The Practical Result: 100 Simulated CIs (Sample Size n={n})</b>", title_x=0.5, yaxis_title="Simulation Run", xaxis_title="Value", showlegend=False, yaxis_range=[-2, n_sims+2])
    return fig1, fig2, capture_count, n_sims, avg_width
# ==============================================================================
# POWERPOINT REPORT GENERATION FUNCTION
# ==============================================================================
def generate_ppt_report(kpi_data, spc_fig):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Validation & Tech Transfer Analytics Summary"
    subtitle.text = f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}"

    # KPI Dashboard Slide
    kpi_slide_layout = prs.slide_layouts[5] # Blank layout
    slide = prs.slides.add_slide(kpi_slide_layout)
    slide.shapes.title.text = "Key Validation Performance Indicators (KPIs)"
    
    positions = [(Inches(1), Inches(1.5)), (Inches(6), Inches(1.5)), (Inches(11), Inches(1.5))]
    for i, (kpi_title, kpi_val, kpi_desc) in enumerate(kpi_data):
        txBox = slide.shapes.add_textbox(positions[i][0], positions[i][1], Inches(4), Inches(2.5))
        tf = txBox.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = kpi_title; p1.font.bold = True; p1.font.size = Pt(24)
        p2 = tf.add_paragraph(); p2.text = str(kpi_val); p2.font.size = Pt(48); p2.font.bold = True
        p3 = tf.add_paragraph(); p3.text = str(kpi_desc); p3.font.size = Pt(16)

    # Chart Slide
    chart_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(chart_slide_layout)
    slide.shapes.title.text = "Process Stability (Shewhart Control Chart)"
    try:
        image_stream = io.BytesIO()
        spc_fig.write_image(image_stream, format='png', scale=2, width=1200, height=600)
        image_stream.seek(0)
        slide.shapes.add_picture(image_stream, Inches(1), Inches(1.75), width=Inches(14))
    except Exception as e:
        txBox = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(14), Inches(4))
        p = txBox.text_frame.add_paragraph()
        p.text = f"Chart Generation Failed: {e}\nEnsure 'kaleido' is installed."; p.font.color.rgb = RGBColor(255, 0, 0)

    # Save to a byte stream
    ppt_stream = io.BytesIO()
    prs.save(ppt_stream)
    ppt_stream.seek(0)
    return ppt_stream
# ==============================================================================
# UI RENDERING FUNCTIONS
# ==============================================================================

def render_gage_rr():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To rigorously quantify the inherent variability (error) of a measurement system and decompose it from the true, underlying variation of the process or product. A Gage R&R study is the definitive method for assessing the **metrological fitness-for-purpose** of any analytical method or instrument. It answers the fundamental question: "Is my measurement system a precision instrument, or a random number generator?"
    
    **Strategic Application:** This is the non-negotiable **foundational checkpoint** in any technology transfer, process validation, or serious quality improvement initiative. Attempting to characterize a process with an uncharacterized measurement system is scientifically invalid. An unreliable measurement system creates a "fog of uncertainty," injecting noise that can lead to two costly errors:
    1.  **Type I Error (False Alarm):** The measurement system's noise makes a good batch appear out-of-spec, leading to unnecessary investigations and rejection of good product.
    2.  **Type II Error (Missed Signal):** The noise masks a real process drift or shift, allowing a bad batch to be released, potentially leading to catastrophic field failures.

    By partitioning the total observed variation into its distinct components—**Repeatability** (within-system consistency), **Reproducibility** (between-system consistency), and **Part-to-Part** variation (the "voice of the process")—this analysis provides the objective, statistical evidence needed to trust your data.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, pct_rr, pct_part = plot_gage_rr()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse", help="This represents the percentage of the total observed variation that is consumed by measurement error.")
            st.metric(label="💡 KPI: Number of Distinct Categories (ndc)", value=f"{int(1.41 * (pct_part / pct_rr)**0.5) if pct_rr > 0 else '>10'}", help="An estimate of how many distinct groups the measurement system can discern in the process data. A value < 5 is problematic.")

            st.markdown("""
            - **Variation by Part & Operator (Main Plot):** The diagnostic heart of the study.
                - *High Repeatability Error:* Wide boxes for a given operator, indicating the instrument/assay has poor precision. This is often a hardware or chemistry problem.
                - *High Reproducibility Error:* The colored lines (operator means) are not parallel or are vertically offset. This is often a human factor or training issue.
                - ***The Interaction Term:*** A significant Operator-by-Part interaction is the most insidious problem. It means operators are not just biased, but *inconsistently* biased. Operator A measures Part 1 high and Part 5 low, while Operator B does the opposite. This points to ambiguous instructions or a flawed measurement technique.

            - **The Core Strategic Insight:** A low % Gage R&R validates your measurement system as a trustworthy "ruler," confirming that the variation you observe reflects genuine process dynamics, not measurement noise. A high value means your ruler is "spongy," making any conclusions about your process's health statistically indefensible. You cannot manage what you cannot reliably measure.
            
            - **The "Number of Distinct Categories" (ndc):** This is a powerful, less-known metric that translates the %R&R into practical terms. It estimates how many non-overlapping groups your measurement system can reliably distinguish within your process's variation.
                - `ndc = 1`: The system is useless; it cannot even tell the difference between a high part and a low part.
                - `ndc = 2-4`: The system can only perform crude screening (e.g., pass/fail).
                - `ndc ≥ 5`: The system is considered adequate for process control. This is a much more intuitive target than "10% R&R."
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria are risk-based and derived from the **AIAG's Measurement Systems Analysis (MSA)** manual, the de facto global standard. The percentage is calculated against the **total study variation**.")
            st.markdown("- **< 10% Gage R&R:** The system is **acceptable**. The 'fog of uncertainty' is minimal. The system can reliably detect process shifts and can be used for SPC and capability analysis.")
            st.markdown("- **10% - 30% Gage R&R:** The system is **conditionally acceptable or marginal**. This is a gray area. Its use may be approved for less critical characteristics, but it is likely unsuitable for controlling a critical-to-quality parameter. This result should trigger a mandatory improvement project for the measurement method.")
            st.markdown("- **> 30% Gage R&R:** The system is **unacceptable and must be rejected**. Data generated by this system is untrustworthy. Using this system for process decisions is equivalent to making decisions by flipping a coin. The method must be fundamentally improved (e.g., new instrument, revised SOP, extensive operator retraining) and the Gage R&R study repeated. ")
            st.info("""
            **Beyond the Numbers: The Part Selection Strategy**
            The most common failure mode of a Gage R&R study is not the math, but the study design. The parts selected for the study **must span the full expected range of process variation**. If you only select parts from the middle of the distribution, your Part-to-Part variation will be artificially low, which will mathematically inflate your % Gage R&R and cause a good system to fail. A robust study includes parts from near the Lower and Upper Specification Limits.
            """)
            
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            While the concepts are old, their modern, rigorous application was born out of the quality crisis in the American automotive industry in the late 1970s and early 1980s. Faced with superior Japanese quality, US manufacturers, guided by luminaries like **W. Edwards Deming**, realized they were often "tampering" with their processes—adjusting a stable process based on faulty measurement data, thereby *increasing* variation.
            
            The **AIAG** codified the solution in the first MSA manual. The critical evolution was the move from the simple **Average and Range (X-bar & R) method** to the **ANOVA method**. The X-bar & R method is computationally simpler but has a critical flaw: it confounds the operator-part interaction with reproducibility. The **ANOVA method**, pioneered for agriculture by the legendary geneticist and statistician **Sir Ronald A. Fisher**, became the gold standard because of its unique ability to cleanly partition and test the significance of each variance component, including the crucial interaction term.
            
            #### Mathematical Basis
            The ANOVA method is founded on the elegant principle of partitioning the total sum of squared deviations from the mean ($SS_T$) into components attributable to each factor in the experiment.
            """)
            st.latex(r"SS_{Total} = SS_{Part} + SS_{Operator} + SS_{Part \times Operator} + SS_{Error}")
            st.markdown("""
            These sums of squares are then converted to Mean Squares (MS) by dividing by their respective degrees of freedom (df). The variance components ($\hat{\sigma}^2$) for each source of variation are then estimated from these MS values.
            - **Repeatability (Equipment Variation, EV):** This is the irreducible, inherent random error of the measurement process under fixed conditions. It is the variation observed when the same operator measures the same part multiple times. It is estimated directly by the Mean Square Error, which represents the "unexplained" variance.
            """)
            st.latex(r"\hat{\sigma}^2_{Repeatability} = \hat{\sigma}^2_{EV} = MS_{Error}")
            st.markdown("- **Reproducibility (Appraiser Variation, AV):** This is the variation introduced when different operators, systems, or labs measure the same parts. It's a measure of systematic bias between appraisers. Crucially, it is composed of two sub-components: the 'pure' operator effect and the operator-part interaction effect.")
            st.latex(r"\hat{\sigma}^2_{Reproducibility} = \hat{\sigma}^2_{Operator} + \hat{\sigma}^2_{Interaction}")
            st.latex(r"\text{where } \hat{\sigma}^2_{Operator} = \frac{MS_{Operator} - MS_{Interaction}}{n_{parts} \cdot n_{replicates}} \text{ and } \hat{\sigma}^2_{Interaction} = \frac{MS_{Interaction} - MS_{Error}}{n_{replicates}}")
            st.warning("**Negative Variance Components:** It is mathematically possible for the formulas above to yield a negative variance for the Operator or Interaction term (if, for example, MS_Interaction > MS_Operator). This is a statistical artifact. The correct interpretation is that the true variance component is zero, and it should be set to zero for calculating the final %R&R.")

def render_linearity():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify and validate that an assay's response is directly and predictably proportional to the known concentration or quantity of the analyte across its entire intended operational range. This establishes the fundamental relationship between a measured signal and a physical quantity.
    
    **Strategic Application:** This is a cornerstone of quantitative assay validation, mandated by every major regulatory body (FDA, EMA, ICH). It provides the statistical evidence that the assay is not just precise, but **globally accurate** across its reportable range. A method exhibiting non-linearity might be perfectly accurate at a central control point but dangerously inaccurate at the upper or lower specification limits. This can lead to incorrect batch disposition decisions or, in a clinical setting, misdiagnosis. This study is therefore critical for ensuring the **interchangeability of results** regardless of where they fall within the range.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, model = plot_linearity()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}", help="Indicates the proportion of variance in the measured values explained by the nominal values. A necessary, but not sufficient, criterion.")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}", help="Ideal = 1.0. A slope < 1 indicates signal compression at high concentrations; > 1 indicates signal expansion.")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}", help="Ideal = 0.0. A non-zero intercept indicates a constant systematic error, or background bias.")
            st.markdown("""
            - **Linearity Plot:** A primary visual check. The data should cluster tightly around the Line of Identity (y=x). Any systematic deviation (e.g., a gentle 'S' curve) suggests non-linearity that R² alone might miss.
            - **Residual Plot:** The single most powerful diagnostic for linearity. A perfect model shows a random, "shotgun blast" pattern of points centered on zero.
                - **A curved (U or ∩) pattern** is the classic sign of non-linearity, indicating the straight-line model is inappropriate. This is often due to detector saturation at high concentrations or complex binding kinetics at low concentrations.
                - **A funnel or cone shape (heteroscedasticity)** indicates that the absolute error of the measurement increases with concentration. This is extremely common in analytical chemistry and violates a key assumption of OLS. Forcing an OLS fit on heteroscedastic data gives undue weight to the high-concentration points and can lead to poor accuracy at the low end. The proper technique is **Weighted Least Squares (WLS) Regression.**
            - **Recovery Plot:** The practical business-end of the analysis. It translates statistical error into analytical accuracy. It directly answers the question: "At a given true concentration, what result does my assay report, and by how much is it off?"
            
            **The Core Strategic Insight:** A high R², a slope of 1, an intercept of 0, randomly scattered residuals, and recovery within tight limits collectively provide a **verifiable chain of evidence** that the assay is a trustworthy quantitative tool across its entire defined range. This builds the fundamental trust required for product release or clinical decisions.
            """)

        with tabs[1]:
            st.markdown("These criteria are defined in the validation protocol and must be met to declare the method linear. They are often tiered based on the assay type.")
            st.markdown("- **R-squared (R²):** While common, it is a weak criterion alone. An R² > **0.995** is a typical starting point, but for chromatography (HPLC, GC), R² > **0.999** is often required.")
            st.markdown("- **Slope:** The 95% confidence interval for the slope must contain 1.0. A common acceptance range for the point estimate is **0.95 to 1.05** for immunoassays, but may be tightened to **0.98 to 1.02** for more precise methods.")
            st.markdown("- **Y-Intercept:** The 95% confidence interval for the intercept must contain 0. This statistically proves the absence of a significant constant bias.")
            st.markdown("- **Residuals:** There should be no obvious pattern or trend in the residual plot. Formal statistical tests like the **Lack-of-Fit test** can be used to objectively prove linearity. This requires designing the experiment with true replicates at each concentration level.")
            st.markdown("- **Recovery:** The percent recovery at each concentration level must fall within a pre-defined range. This is often the most important criterion. For bioassays, this might be **80% to 120%**, but for drug substance purity assays, it could be as strict as **99.0% to 101.0%**.")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The mathematical engine is **Ordinary Least Squares (OLS) Regression**, a cornerstone of statistics developed by **Adrien-Marie Legendre (1805)** and, more famously, **Carl Friedrich Gauss (1809)**. Gauss, the "Prince of Mathematicians," used it to solve a pressing astronomical problem: rediscovering the dwarf planet Ceres after it was lost behind the sun. With only a few data points on its early trajectory, he calculated a best-fit orbit that allowed astronomers to find it again exactly where he predicted.
            
            The genius of OLS lies in its objective function: to find the line that **minimizes the sum of the squared vertical distances (the "residuals")** between the observed data and the fitted line. This principle is not arbitrary; under the assumption of normally distributed errors, the OLS estimates are the **Maximum Likelihood Estimates (MLE)**, meaning they are the parameter values that make the observed data most probable. This provides a deep theoretical justification for what seems like a simple curve-fitting exercise. In validation, we use Gauss's powerful tool to confirm a simple, but critical, physical reality for our assay.

            #### Mathematical Basis
            The goal is to fit a simple linear model to the calibration data, which links the true concentration ($x$) to the measured response ($y$).
            """)
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - $y$: The measured concentration or instrument signal.
            - $x$: The nominal (true) concentration of the reference standard.
            - $\\beta_0$ (Intercept): Represents the assay's **constant systematic error**. This is the signal you'd expect at zero concentration.
            - $\\beta_1$ (Slope): Represents the assay's **proportional systematic error**. This is the sensitivity of the assay.
            - $\\epsilon$: The random, unpredictable measurement error, assumed to be normally distributed with a mean of zero and constant variance.

            The validation hinges on formal statistical tests of the estimated coefficients ($\hat{\beta}_0, \hat{\beta}_1$):
            - **Hypothesis Test for Slope:** $H_0: \\beta_1 = 1$ (no proportional bias) vs. $H_a: \\beta_1 \\neq 1$.
            - **Hypothesis Test for Intercept:** $H_0: \\beta_0 = 0$ (no constant bias) vs. $H_a: \\beta_0 \\neq 0$.
            A p-value > 0.05 for these tests supports the claim of linearity and no bias.
            """)

def render_lod_loq():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally establish the absolute lower performance boundaries of a quantitative assay. It determines the lowest analyte concentration an assay can reliably **detect (LOD)** and the lowest concentration it can reliably and accurately **quantify (LOQ)**.
    
    **Strategic Application:** This is a mission-critical parameter for any assay used to measure trace components. The validity of entire programs can hinge on these values. Examples include:
    - **Host Cell Protein / Impurity Testing:** The LOQ *must* be demonstrably below the specification limit for a potentially harmful impurity in a final drug product. A method with an LOQ higher than the spec is not fit for purpose.
    - **Early-Stage Disease Diagnosis:** The LOD/LOQ for a cancer biomarker must be low enough to detect the disease at its earliest, most treatable stage, when the biomarker concentration is vanishingly small.
    - **Pharmacokinetics (PK):** To properly characterize a drug's elimination phase, the assay LOQ must be low enough to measure the final few datapoints in the concentration-time curve.

    The **Limit of Detection (LOD)** is a qualitative threshold based on hypothesis testing, answering "Is the analyte present or not?" It controls the risk of a false positive. The **Limit of Quantitation (LOQ)** is a much higher and more stringent quantitative bar, answering "What is the concentration, and can I trust the numerical value?" It is fundamentally about ensuring the measurement uncertainty at that level is acceptably low. The LOQ typically defines the lower end of the assay's reportable range.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, lod_val, loq_val = plot_lod_loq()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} ng/mL", help="The lowest concentration you can report with confidence in the numerical value.")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} ng/mL", help="The lowest concentration you can reliably claim is 'present'.")
            st.markdown("""
            - **Signal Distribution (Violin Plot):** This is a critical first-pass check. The distribution of signals from the 'Blank' samples (the noise) must be clearly separated from the distribution of signals from the 'Low Concentration' samples. Significant overlap indicates the assay lacks the fundamental sensitivity required.
            - **Low-Level Calibration Curve (Regression Plot):** This plot models the signal-to-concentration relationship at the low end. The LOD and LOQ are not arbitrary numbers; they are derived directly from two key parameters of this model:
                1.  **The Slope (S):** The assay's sensitivity. A steeper slope is better, as it means a small change in concentration produces a large change in signal.
                2.  **The Residual Standard Error (σ):** The inherent noise or imprecision of the assay at the low end. A smaller σ is better.

            **The Core Strategic Insight:** This analysis defines the **absolute floor of your assay's validated capability**. It provides the statistical evidence to defend every low-level result you report. Claiming a quantitative result below the validated LOQ is scientifically and regulatorily indefensible. It's the difference between seeing a faint star and being able to measure its brightness.
            """)

        with tabs[1]:
            st.markdown("Acceptance criteria are absolute and defined by the assay's intended use.")
            st.markdown("- The primary, non-negotiable criterion is that the experimentally determined **LOQ must be ≤ the lowest concentration that the assay is required to measure** for its specific application (e.g., a release specification for an impurity). If LOQ > specification, the method is not fit for purpose and must be improved.")
            st.markdown("- For a concentration to be formally declared the LOQ, it must be experimentally confirmed. This typically involves preparing and analyzing 5-6 independent samples at the claimed LOQ concentration and demonstrating that they meet pre-defined acceptance criteria for precision and accuracy (e.g., **%CV < 20% and %Recovery between 80-120%** for a bioassay).")
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
            The need to define analytical sensitivity is as old as chemistry itself, but for decades, definitions were inconsistent and often scientifically unsound. The breakthrough came with the **International Council for Harmonisation (ICH)**, a global body that brings together regulatory authorities and the pharmaceutical industry. Their landmark guideline **ICH Q2(R1) "Validation of Analytical Procedures"** harmonized the definitions and methodologies for LOD and LOQ, creating a global standard.
            
            This work was heavily influenced by the rigorous statistical framework established by **Lloyd Currie at the National Institute of Standards and Technology (NIST)**. Currie's 1968 paper, "Limits for Qualitative Detection and Quantitative Determination," established a clear, hypothesis-testing framework that distinguished between three levels of analytical decisions: "Is something detected?", "Is the substance present?", and "How much is present?". This forms the statistical foundation for the modern LOB/LOD/LOQ hierarchy.

            #### Mathematical Basis
            This method is built on the relationship between the assay's signal, its sensitivity (Slope, S), and its noise (standard deviation, σ). The standard deviation σ can be estimated in a few ways, but the most robust is using the **residual standard error** from a regression model fit to low-concentration data, as it pools variability information across multiple levels.

            - **Limit of Detection (LOD):** The formula is designed to control the risk of false positives and false negatives. The factor 3.3 is an approximation related to the Student's t-distribution that corresponds to a high level of confidence (typically >95%) that a signal at this level is not a random fluctuation of the blank. It is fundamentally about making a **decision** (present/absent).
            """)
            st.latex(r"LOD \approx \frac{3.3 \times \sigma}{S}")
            st.markdown("""
            - **Limit of Quantitation (LOQ):** This is not about detection, but about **measurement quality**. It demands a much higher signal-to-noise ratio to ensure that the measurement is not only detectable but also has an acceptable level of uncertainty (precision and accuracy). The factor of 10 is the standard convention that typically yields a precision of roughly 10% CV for a well-behaved assay.
            """)
            st.latex(r"LOQ \approx \frac{10 \times \sigma}{S}")
def render_method_comparison():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally assess and quantify the degree of agreement and systemic bias between two different measurement methods intended to measure the same quantity. This analysis moves beyond simple correlation to determine if the two methods can be used **interchangeably** in practice.
    
    **Strategic Application:** This study is the "crucible" of method transfer, validation, or replacement. It's the moment of truth where a new method is judged against an established one. Key scenarios include:
    - **Tech Transfer:** Proving the QC lab's implementation of an assay is equivalent to the original, highly-characterized R&D method.
    - **Method Modernization:** Demonstrating a new, faster, or cheaper assay (e.g., a digital PCR) yields clinically equivalent results to an older gold standard (e.g., a cell-based assay).
    - **Cross-Site/Instrument Harmonization:** Ensuring that results from an instrument in a facility in North Carolina are directly comparable to one in Ireland.

    A failed comparison study can halt a tech transfer, delay a product launch, or invalidate a clinical study, making this analysis a high-stakes gatekeeper. It answers the critical business and regulatory question: “Do these two methods produce the same result, for the same sample, within medically or technically acceptable limits?”
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, slope, intercept, bias, ua, la = plot_method_comparison()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Mean Bias (Bland-Altman)", value=f"{bias:.2f} units", help="The average systematic difference between the Test and Reference methods. A positive value means the Test method measures higher on average.")
            st.metric(label="💡 Metric: Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0. Measures proportional bias, which is often concentration-dependent.")
            st.metric(label="💡 Metric: Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0. Measures constant bias, a fixed offset across the entire range.")
            st.markdown("""
            - **Deming Regression:** This is the correct regression for method comparison. Unlike standard OLS, it accounts for measurement error in *both* methods, providing an unbiased estimate of slope (proportional bias) and intercept (constant bias). The goal is to see the red Deming line perfectly overlay the black Line of Identity.
            - **Bland-Altman Plot:** This plot transforms the question from "are they correlated?" to "how much do they differ?". It visualizes the random error and quantifies the **95% Limits of Agreement (LoA)**. This is the expected range of disagreement for 95% of future measurements. A "smile" or "frown" pattern in the points indicates the variance of the difference changes with concentration.
            - **% Bias Plot:** This plot assesses **practical significance**. It shows if the bias at any specific concentration exceeds a pre-defined acceptable limit (e.g., ±15%). A method can have a small average bias but a large, unacceptable bias at the low end of the range, which this plot will reveal.

            **The Core Strategic Insight:** This dashboard provides a multi-faceted verdict on method interchangeability. Deming regression diagnoses the *type* of bias (constant vs. proportional), the Bland-Altman plot quantifies the *magnitude* of expected random disagreement, and the % Bias plot confirms *local* acceptability. A successful comparison requires passing all three checks.
            """)
        with tabs[1]:
            st.markdown("Acceptance criteria must be pre-defined in the validation protocol and be clinically or technically justified.")
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope must contain 1.0**, and the 95% CI for the **intercept must contain 0**. This provides statistical proof of no systematic bias.")
            st.markdown(f"- **Bland-Altman:** The primary criterion is that the **95% Limits of Agreement (`{la:.2f}` to `{ua:.2f}`) must be clinically or technically acceptable**. This is a judgment call. A 20-unit LoA might be acceptable for a glucose monitor but catastrophic for a cancer biomarker. Furthermore, at least 95% of the data points must fall within these calculated limits.")
            st.markdown("- **Total Analytical Error (TAE):** An advanced approach combines bias and imprecision into a single metric. The model can be accepted if, across the entire range, `|Bias| + 1.96 * SD_of_difference` is less than a predefined Total Allowable Error (TEa).")
            st.error("**The Correlation Catastrophe:** Do not, under any circumstances, use the correlation coefficient (R or R²) as a measure of agreement. Two methods can be perfectly correlated (R=1.0) but have a huge bias (e.g., one method always reads exactly twice as high as the other). A high correlation is a prerequisite for agreement, but it is not evidence of agreement. This is one of the most common and severe statistical errors in scientific literature.")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            For decades, the scientific community committed a cardinal statistical sin: using **Ordinary Least Squares (OLS) regression** and the **correlation coefficient (r)** to compare methods. This is fundamentally flawed because OLS assumes the x-axis (reference method) is measured without error, a clear impossibility.
            
            - **Deming's Correction:** While known to statisticians for a century (as errors-in-variables models), it was **W. Edwards Deming** who championed and popularized this type of regression in the 1940s. It correctly assumes both methods have measurement error, providing an unbiased estimate of the true relationship. The most common form, **Deming Regression**, assumes the ratio of the error variances is known, while **Passing-Bablok regression** is a non-parametric alternative that is robust to outliers.
            
            - **The Bland-Altman Revolution:** The bigger conceptual leap came in a landmark 1986 paper in *The Lancet* by **J. Martin Bland and Douglas G. Altman**. They ruthlessly exposed the misuse of correlation and proposed their brilliantly simple alternative. Instead of plotting Y vs. X, they plotted the **Difference (Y-X) vs. the Average ((Y+X)/2)**. This simple change of coordinates directly visualizes what scientists actually care about: the magnitude and patterns of disagreement. The Bland-Altman plot is now the undisputed gold standard for method comparison studies in the medical and biological sciences.
            
            #### Mathematical Basis
            **Deming Regression:** OLS minimizes the sum of squared vertical distances. Deming regression minimizes the sum of squared distances from the points to the line, weighted by the ratio of the error variances of the two methods ($\lambda = \sigma^2_{error,y} / \sigma^2_{error,x}$). If the methods have equal error ($\lambda=1$), it minimizes the sum of squared perpendicular distances.
            
            **Bland-Altman Plot:** This is a graphical analysis, not a statistical test. The key metrics are the **mean difference (bias)**, $\bar{d}$, and the **standard deviation of the differences**, $s_d$. The 95% Limits of Agreement (LoA) are calculated assuming the differences are approximately normally distributed:
            """)
            st.latex(r"LoA = \bar{d} \pm 1.96 \cdot s_d")
            st.markdown("This interval provides a predictive range: for any future sample, we can be 95% confident that the difference between the two methods will fall within these limits. It is the 'margin of error' for method disagreement.")

def render_robustness_rsm():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To systematically and efficiently identify the "vital few" process parameters that significantly impact an assay's performance and to characterize their interactions. This allows for the creation of a **"Design Space"**—a proven region of operation where the method is resilient to the unavoidable, small variations of routine use.
    
    **Strategic Application:** This is the essence of modern **Quality by Design (QbD)** and a cornerstone of advanced process validation (ICH Q8). Instead of discovering process weaknesses through painful, expensive failures in routine production, this study proactively "stress tests" the method in a controlled, multi-factorial experiment.
    - **Screening (DOE):** Used early in development with fractional factorial designs to quickly identify which of many potential factors (e.g., reagent lot, incubation time, temperature, pH, mixing speed) are actually important. This focuses future efforts on what matters.
    - **Optimization (RSM):** Once the critical factors are known, Response Surface Methodology is used to create a detailed, predictive mathematical map of their effects. This allows scientists to find the "sweet spot" that maximizes performance (e.g., signal-to-noise ratio) while simultaneously minimizing variability (i.e., finding a flat plateau on the response surface). The result is a scientifically justified **Normal Operating Range (NOR)** that guarantees robustness.
    """)
    vis_type = st.radio("Select Analysis Stage:", ["📊 **Stage 1: Factor Screening (Pareto Plot)**", "📈 **Stage 2: Process Optimization (2D Contour)**", "🧊 **Stage 2: Process Optimization (3D Surface)**"], horizontal=True)
    fig_pareto, fig_contour, fig_surface, effects = plot_robustness_rsm()
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        if "Screening" in vis_type: st.plotly_chart(fig_pareto, use_container_width=True)
        elif "2D Contour" in vis_type: st.plotly_chart(fig_contour, use_container_width=True)
        else: st.plotly_chart(fig_surface, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Most Significant Factor", value=f"{effects.index[0]}", help="The factor with the largest standardized effect on the response.")
            st.metric(label="💡 Effect Magnitude", value=f"{effects.values[0]:.2f}", help="A value of -5.0 means moving from the low to high setting of this factor decreases the response by 5 units on average.")
            st.markdown("""
            - **Screening (Pareto):** A visual application of the Pareto Principle (80/20 rule). It instantly separates the "vital few" from the "trivial many." Any bar that crosses the red significance line represents a factor or interaction that *must* be tightly controlled. The most dangerous finding is often a large **interaction effect** (like `Temp:pH`), which means the effect of Temperature *depends on the level of pH*. This kind of complexity is impossible to find with one-factor-at-a-time (OFAT) experiments.
            - **Optimization (Contour/Surface):** These are topographical maps of your process. The goal is not just to find the peak of the mountain (optimal response), but to find a large, flat plateau. Operating on a sharp peak is risky; operating on a plateau ensures that small, unavoidable variations in input parameters (a little temperature drift, a slight pH error) have minimal impact on the final result. This is the definition of a **robust process**.

            **The Core Strategic Insight:** This analysis builds a predictive model of your process. It provides the "operating manual" that allows you to set scientifically-backed control limits, predict the impact of deviations, and design a process that is inherently resilient to the noise of the real world. It transforms process understanding from tribal knowledge and guesswork into predictive science.
            """)
        with tabs[1]:
            st.markdown("- **Screening:** Any factor or interaction with a p-value **< 0.05** is deemed statistically significant. These factors must be identified in the validation report as **Critical Process Parameters (CPPs)** and require defined control limits in the SOP.")
            st.markdown("- **Optimization & Design Space:** The acceptance criterion is the successful definition of a **Design Space**. Per ICH Q8, the Design Space is the 'multidimensional combination and interaction of input variables and process parameters that have been demonstrated to provide assurance of quality.' Movement within the Design Space is not considered a change and does not require a new regulatory filing, providing enormous operational flexibility.")
            st.success("""
            **The Hierarchy of Control:**
            - **Design Space:** The entire multidimensional space where quality is assured.
            - **Proven Acceptable Range (PAR):** The edges of the operating ranges tested during validation.
            - **Normal Operating Range (NOR):** The tighter, target range for routine production, set comfortably within the PAR.
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **DOE:** The entire field of modern experimental design was invented by one man: **Sir Ronald A. Fisher**, working at the Rothamsted Agricultural Experimental Station in the 1920s. Before Fisher, scientists used the inefficient and often misleading one-factor-at-a-time (OFAT) approach. Fisher's revolutionary insight was **factorial design**: varying all factors simultaneously in a structured way. This was orders of magnitude more efficient and, crucially, was the only way to estimate **interactions**, the synergistic or antagonistic effects between factors.
            
            **RSM:** In the 1950s, the brilliant statistician **George E. P. Box** (a Fisher disciple) and K. B. Wilson extended this work for the chemical industry. They weren't just interested in which factors were significant, but in finding their *optimal settings*. They developed **Response Surface Methodology (RSM)**, which uses more sophisticated designs (like the Central Composite Design shown here with its 'star points') to fit quadratic models. This allows for modeling the curvature of the response and mathematically finding the "peak of the mountain" or the "bottom of the valley." This work is the direct intellectual parent of the modern QbD movement.
            
            #### Mathematical Basis
            **DOE (Screening):** Uses a simple 2-level factorial design to fit a linear model with interaction terms. The coefficients ($\beta$) represent the standardized effect of each factor.
            """)
            st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{12} X_1 X_2 + \epsilon")
            st.markdown("""
            **RSM (Optimization):** To model curvature, RSM requires a more complex, second-order polynomial model containing quadratic terms ($X_i^2$).
            """)
            st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{11} X_1^2 + \beta_{22} X_2^2 + \beta_{12} X_1 X_2 + \epsilon")
            st.markdown("Finding the optimum involves taking the partial derivatives of this equation with respect to each factor, setting them to zero, and solving the resulting system of equations—a task easily handled by modern software.")

def render_shewhart():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To determine if a process is in a state of **statistical control**. A process in control is stable and predictable, with its variation arising only from inherent, random "common causes." This analysis distinguishes this natural process "noise" from "special cause" variation, which signals a fundamental and often undesirable change in the process.
    
    **Strategic Application:** This is the foundational tool of **Statistical Process Control (SPC)** and the **absolute prerequisite** for any meaningful analysis of process capability (Cpk). A process must first be brought into a state of statistical control before its ability to meet specifications can be evaluated. An out-of-control process is unpredictable; its mean and standard deviation are unstable, making any calculation of Cpk a meaningless snapshot in time. Establishing control is the first victory in demonstrating that a process is understood and managed, not just a series of random events. It is the real-time "voice of the process."
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_shewhart(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Process Stability", value="Signal Detected", delta="Action Required", delta_color="inverse", help="The process has shown a 'special cause' variation that requires investigation.")
            st.markdown("""
            - **I-Chart (Individuals Chart):** Monitors the process center (location or accuracy) over time. It is highly sensitive to shifts in the process mean. The zones (1, 2, and 3-sigma) help in applying advanced detection rules (like Westgard or Nelson rules).
            - **MR-Chart (Moving Range Chart):** Monitors the short-term, run-to-run variability (spread or precision). An out-of-control signal on the MR-chart is often more serious than on the I-chart, as it indicates the process has become fundamentally unstable and inconsistent. **The MR-chart must be in control before the I-chart can be properly interpreted.**
            
            **The Core Strategic Insight:** These charts are the EKG of your process. In this example, the process was stable and predictable for 15 runs. The introduction of a new reagent lot created a "special cause" - a shock to the system that shifted the mean upwards, driving the process out of statistical control. An engineer's job is to identify and eliminate special causes to return the process to its predictable state. Only then can they work on improving the stable process itself (e.g., reducing its common cause variation). This chart separates the signal from the noise.
            """)
        with tabs[1]:
            st.markdown("- A process is deemed to be in a state of statistical control only after a baseline period (typically **25-30 consecutive subgroups or individual measurements**) shows **no special cause signals** on *both* the I-chart and the MR-chart.")
            st.markdown("- Any point violating a selected control rule (e.g., a point outside the ±3σ limits) signals the presence of a special cause. This mandates a formal **Out-of-Control Action Plan (OCAP)**, which involves:
                1.  Halting the process.
                2.  Isolating potentially affected product.
                3.  Conducting a root cause investigation (e.g., using an Ishikawa / Fishbone diagram).
                4.  Implementing corrective and preventive actions (CAPA).
                5.  Only restarting the process after the special cause is eliminated.")
            st.error("""
            **The Central Fallacy of Management:**
            A common and disastrous mistake is to treat common cause variation as if it were a special cause. For example, if a point is high but still within the control limits, and a manager demands an "adjustment," this is called **tampering**. As Deming proved, tampering with a stable process *always increases* its variation and makes it worse. Control charts prevent tampering by defining when to act and when to leave the process alone.
            """)
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            Control charts were invented by the American physicist and statistician **Dr. Walter A. Shewhart** at Bell Telephone Laboratories in the 1920s. His work was a profound intellectual breakthrough that created the entire field of statistical process control. Shewhart's genius was to recognize that variation in any process comes from two sources:
            1.  **Common Cause (or Chance) Variation:** The natural, inherent "noise" of a stable, well-designed process. It's the cumulative effect of many small, unavoidable factors. This variation is predictable within a range.
            2.  **Special Cause (or Assignable) Variation:** The result of a specific, identifiable event external to the usual process, such as a machine malfunction, a bad batch of raw material, or an untrained operator. This variation is unpredictable.

            Shewhart's chart provided, for the first time, an **objective, operational method** to distinguish between these two. Its purpose is to tell managers when to act on the process (a special cause) and, just as importantly, **when to leave the process alone** (only common cause variation present). As his student W. Edwards Deming would later teach, reacting to common cause variation as if it were a special cause ("tampering") is a fundamental management error that *increases* overall variation.
            
            #### Mathematical Basis
            The control limits are not specification limits. They are the **"voice of the process."** They are calculated from the process data itself to reflect its natural, inherent variation. For an I-MR chart, the process standard deviation ($\hat{\sigma}$) is estimated from the short-term variability seen in the moving range.
            """)
            st.latex(r"\hat{\sigma} = \frac{\overline{MR}}{d_2}")
            st.markdown(r"""
            where $\overline{MR}$ is the average moving range between consecutive points and $d_2$ is a statistical constant that corrects for bias based on the subgroup size (for a moving range of 2, $d_2 \approx 1.128$).
            - **I-Chart Limits:** The famous "3-sigma" limits represent the range where 99.73% of points should fall if the process is stable and only common cause variation is present. A point outside this range is an extremely unlikely event, hence a signal of a special cause.
            """)
            st.latex(r"UCL_I / LCL_I = \bar{x} \pm 3\hat{\sigma} = \bar{x} \pm 3 \left(\frac{\overline{MR}}{d_2}\right)")
            st.markdown("- **MR-Chart Limits:** The limits for the moving range chart are also derived from $\overline{MR}$ and statistical constants to detect unusual spikes in short-term volatility.")
            st.latex(r"UCL_{MR} = D_4 \overline{MR} \quad (\text{where } D_4 \approx 3.267)")
            st.latex(r"LCL_{MR} = D_3 \overline{MR} \quad (\text{where } D_3 = 0 \text{ for n < 7})")
def render_ewma_cusum():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To deploy sensitive control charts designed to rapidly detect **small, sustained shifts** in the process mean, which a standard Shewhart chart, by design, is poor at detecting. These charts achieve their power by incorporating "memory" of past data points.
    
    **Strategic Application:** These are advanced, second-generation SPC tools, best suited for mature, stable processes where large, dramatic failures have been eliminated. They act as a high-sensitivity early warning system for the subtle, insidious process drifts that can occur over time due to factors like equipment wear, reagent degradation, or slow environmental changes.
    - **EWMA (Exponentially Weighted Moving Average):** This chart is an excellent all-around performer, effective at detecting small shifts and gradual drifts. It is particularly useful for monitoring continuous processes where you want to smooth out high-frequency noise to see the underlying signal, such as monitoring bioreactor pH or temperature.
    - **CUSUM (Cumulative Sum):** This chart is the **most statistically powerful** tool for detecting a specific, small, abrupt, and sustained shift in the mean. It is the optimal tool when you have a specific failure mode in mind (e.g., "I need to detect a 0.5 sigma shift within 3 runs"). It is essentially a continuous, sequential hypothesis test running in real-time.
    """)
    chart_type = st.sidebar.radio("Select Chart Type:", ('EWMA', 'CUSUM'))
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        if chart_type == 'EWMA': 
            lmbda = st.sidebar.slider("EWMA Lambda (λ)", 0.05, 1.0, 0.2, 0.05, help="Controls the chart's memory. A small λ gives more weight to past data (more memory, better for smaller shifts). A λ of 1.0 makes it a Shewhart chart.")
            st.plotly_chart(plot_ewma_cusum(chart_type, lmbda, 0, 0), use_container_width=True)
        else: 
            k_sigma = st.sidebar.slider("CUSUM Slack (k, in σ)", 0.25, 1.5, 0.5, 0.25, help="The 'slack' in the system, typically set to half the size of the shift you want to detect (in standard deviations).")
            H_sigma = st.sidebar.slider("CUSUM Limit (H, in σ)", 2.0, 8.0, 5.0, 0.5, help="The decision interval or control limit (in standard deviations). A smaller H leads to faster detection but more false alarms.")
            st.plotly_chart(plot_ewma_cusum(chart_type, 0, k_sigma, H_sigma), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Shift Detection Speed (ARL)", value="Signal Detected", delta="Action Required", delta_color="inverse", help="These charts are optimized to have a low Average Run Length (ARL) to detect small shifts.")
            st.markdown("""
            - **Top Plot (Raw Data):** The 1.25σ shift introduced after run 25 is nearly invisible to the naked eye and would likely not trigger a standard Shewhart chart for many runs.
            - **Bottom Plot (EWMA/CUSUM):** This chart acts as a signal accumulator. It makes the invisible visible.
                - **EWMA:** The purple line is a smoothed version of the raw data. It clearly trends upwards after the shift, eventually crossing the tightening red control limit.
                - **CUSUM:** The chart accumulates deviations from the target. Before the shift, the sums hover around zero. After the shift, the high-side sum (SH) begins to climb relentlessly, like a rising tide, until it breaches the decision limit H.

            **The Core Strategic Insight:** These charts fundamentally change the SPC paradigm from "is it out?" to "where is it heading?". They provide a crucial early warning, allowing for proactive intervention (e.g., scheduling maintenance, changing a reagent column) *before* a specification limit is ever breached, thus preventing scrap and deviation investigations. They enable management *by exception* for highly stable processes.
            """)
        with tabs[1]:
            st.markdown("- **Tuning is Key:** The performance of these charts depends critically on their tuning parameters, which are chosen based on the desired **Average Run Length (ARL)**. The ARL is the average number of points that will be plotted before a signal is given.")
            st.markdown("  - **ARL₀:** The in-control ARL (should be high, e.g., >500, to minimize false alarms).")
            st.markdown("  - **ARL₁:** The out-of-control ARL (should be low, for rapid detection of a given shift).")
            st.markdown("- **EWMA Tuning:** For detecting small shifts (0.5σ to 1.0σ), a `λ` between **0.05 and 0.25** is typically optimal. The control limit width `L` is usually set to 3.0.")
            st.markdown("- **CUSUM Tuning:** The chart is tuned to optimally detect a shift of size $\delta$ (in units of σ). The standard design sets the reference value (or slack parameter) `k = δ / 2` and the decision interval `H = 5σ`. For example, to best detect a 1σ shift, set k=0.5σ and H=5σ.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            Both charts were developed in the UK in the 1950s as a direct response to the primary weakness of the Shewhart chart: its insensitivity to small, sustained shifts (a consequence of it being "memoryless").
            - **CUSUM:** Developed by the British statistician **E. S. Page in 1954**, the CUSUM chart is a direct application of the **Sequential Probability Ratio Test (SPRT)**. The SPRT was a classified statistical method developed by Abraham Wald at Columbia University during WWII for efficiently testing munitions lots (i.e., determining if a batch was good or bad with the minimum number of destructive tests). Page's genius was to turn this batch-level test into a continuous process monitoring scheme.
            - **EWMA:** Proposed by **S. W. Roberts in 1959**, the EWMA chart comes from the world of time series forecasting. It uses the same logic as exponential smoothing to create a smoothed forecast of the next data point. The control chart then simply plots the difference between the actual point and its forecast. Its performance is nearly as good as CUSUM for specific shifts, but it is more robust if the true shift size is unknown.

            #### Mathematical Basis
            **EWMA:** The statistic $z_i$ is a weighted average that gives exponentially decreasing weight to older observations. The "memory" is controlled by $\lambda$.
            """)
            st.latex(r"z_i = \lambda x_i + (1-\lambda)z_{i-1}, \quad \text{where } z_0 = \mu_0")
            st.markdown("""
            The control limits for an EWMA chart are not constant; they widen at the beginning of the chart and then approach a steady-state value. This accounts for the higher uncertainty in the EWMA estimate when there are few data points.

            **CUSUM:** This chart uses two one-sided sums, one for upward shifts ($SH$) and one for downward shifts ($SL$). The "slack" or "reference" value `k` is incorporated at each step. If the process is on target, the deviations are smaller than `k`, so the sum doesn't grow. If a shift occurs, the deviations consistently overcome `k` and the sum trends upwards.
            """)
            st.latex(r"SH_i = \max(0, SH_{i-1} + (x_i - \mu_0) - k)")
            st.latex(r"SL_i = \max(0, SL_{i-1} + (\mu_0 - x_i) - k)")
            st.markdown("A signal is triggered if either $SH_i$ or $SL_i$ exceeds the decision interval $H$.")
            
def render_multi_rule():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To establish and operationalize an objective, statistically-driven system for the real-time acceptance or rejection of individual analytical runs. This is achieved by evaluating the performance of Quality Control (QC) samples against a set of predefined statistical rules that are sensitive to different error patterns.
    
    **Strategic Application:** This is the vigilant gatekeeper of daily laboratory operations. It is the practical application of SPC for run-level decision-making. A well-designed multi-rule system provides a robust defense against releasing invalid data, safeguarding against both large, random errors (e.g., a major pipetting blunder) and smaller, insidious systematic errors (e.g., a slow degradation of a reagent). It is a core requirement for any laboratory operating under regulatory scrutiny (CLIA, CAP, ISO 15189, GxP) as it provides a documented, objective basis for every run disposition decision.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_multi_rule(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Run Status", value="Violations Detected", delta="Action Required", delta_color="inverse", help="One or more QC rules have been violated, flagging the run for review or rejection.")
            st.markdown("""
            - **Levey-Jennings Chart:** This is a specialized Shewhart chart for QC data. The annotations flagging specific rule violations are the key output, converting raw data into actionable information.
            - **Distribution Plot:** Provides a long-term view of QC performance. A healthy process will show a symmetric, bell-shaped distribution centered on the target mean. Skewness or multiple peaks (bimodality) can indicate persistent, unresolved systematic issues.
            
            **The Core Strategic Insight:** Different rules detect different types of errors. This is the power of the multi-rule approach. The type of rule violated provides a critical clue for the subsequent root cause investigation, making troubleshooting faster and more effective.
            - **Random Error Rules (e.g., `1_3s`, `R_4s`):** These rules are sensitive to single, sporadic events that indicate a loss of precision. An `R_4s` violation (one point at +2s, the next at -2s) is a classic sign of high imprecision or random blunders.
            - **Systematic Error Rules (e.g., `2_2s`, `4_1s`, `10x`):** These rules are sensitive to sustained shifts or trends that indicate a loss of accuracy. A `4_1s` violation is an early warning of a small but real shift in the mean. A `10x` violation is a strong signal of a significant, persistent bias.
            """)
        with tabs[1]:
            st.markdown("""
            #### Historical Context & Origin
            The **Levey-Jennings chart**, developed by S. Levey and E. R. Jennings in 1950, was a direct adaptation of industrial Shewhart charts for the unique needs of the clinical laboratory. They plotted individual QC sample results over time, which was a major step forward.
            
            However, using only the simple ±2σ (warning) and ±3σ (rejection) limits was a blunt instrument, leading to an undesirable trade-off: high sensitivity to errors came at the cost of too many false rejections, while reducing false rejections meant missing real errors. The definitive solution came in a landmark 1981 paper by **Dr. James O. Westgard**. He logically combined several control rules into a hierarchical algorithm, creating a system with a very low false rejection rate (<1%) but a very high probability of detecting true errors (>90%). This "multi-rule" system, now famously known as the **Westgard Rules**, became the global standard for quality control in clinical laboratories and is essential for meeting the stringent requirements of regulatory and accreditation bodies like CLIA, CAP, and ISO 15189.
            """)
    st.subheader("Standard Industry Rule Sets: A Comparative Guide")
    tab_w, tab_n, tab_we = st.tabs(["✅ Westgard Rules (Clinical/Medical Labs)", "✅ Nelson Rules (Industrial/Manufacturing)", "✅ Western Electric Rules (The Foundation)"])
    with tab_w:
        st.markdown("""Optimized for clinical lab QC, balancing error detection and false rejection. Essential for CLIA, CAP, ISO 15189 compliance. A run is rejected if any of the mandatory rejection rules are violated.
| Rule | Interpretation | Type of Error Detected |
|---|---|---|
| **1_2s** | **Warning Rule:** One control measurement exceeds the mean ± 2s. Triggers inspection of other rules, does not cause rejection by itself. | Serves as a screen. |
| **1_3s** | **Rejection Rule:** One control measurement exceeds the mean ± 3s. | Large Random Error |
| **2_2s** | **Rejection Rule:** Two *consecutive* control measurements exceed the same mean ± 2s limit (e.g., both > +2s). | Systematic Error (Bias) |
| **R_4s** | **Rejection Rule:** One control is > +2s and the next is < -2s (or vice-versa) within a single run. The range between them exceeds 4s. | Random Error (Imprecision) |
| **4_1s** | **Rejection Rule:** Four *consecutive* control measurements exceed the same mean ± 1s limit. | Small, persistent Systematic Error |
| **10x** | **Rejection Rule:** Ten *consecutive* control measurements fall on the same side of the mean. | Significant Systematic Error (Bias) |""")
    with tab_n:
        st.markdown("""A comprehensive set of rules developed by Lloyd Nelson at General Electric in the 1980s, excellent for catching a wide variety of non-random patterns in industrial process data. Any rule violation signals a special cause.
| Rule | What It Flags | Common Cause |
|---|---|---|
| 1. One point > 3σ from the mean | A single large deviation, an outlier. | Gross error, measurement blunder. |
| 2. Nine points in a row on same side of mean | A sustained shift in the process mean. | New raw material, machine setting drift. |
| 3. Six points in a row, all increasing or decreasing | A process trend or drift. | Tool wear, reagent degradation, operator fatigue. |
| 4. Fourteen points in a row, alternating up and down | Systematic oscillation, often from over-control. | Two alternating suppliers, "tampering" by operators. |
| 5. Two out of three points > 2σ (same side) | A moderate shift in the process mean. | A less obvious shift in conditions. |
| 6. Four out of five points > 1σ (same side) | A small but persistent shift. | An even smaller, but real, change in the mean. |
| 7. Fifteen points in a row within ±1σ | Stratification; reduced variation (hugging the mean). | Improper sampling (e.g., not sampling the whole process). |
| 8. Eight points in a row outside ±1σ | Increased variation or bimodal distribution. | Two different processes are being measured as one. |""")
    with tab_we:
        st.markdown("""The foundational rules developed at Western Electric (part of Bell Labs) in the 1950s, documented in their influential Statistical Quality Control Handbook. They are the basis from which the Westgard and Nelson rules were derived. Simpler but still powerful.
| Rule | Interpretation | Error Type |
|---|---|---|
| **Rule 1** | One point falls outside the ±3σ limits. | Large Random or Systematic Error |
| **Rule 2** | Two out of three consecutive points fall beyond the ±2σ limit on the same side of the mean. | Systematic Error (Bias) |
| **Rule 3** | Four out of five consecutive points fall beyond the ±1σ limit on the same side of the mean. | Small Systematic Error |
| **Rule 4** | Eight (or nine) consecutive points fall on the same side of the mean. | Significant Systematic Error |""")


def render_capability():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To quantitatively determine if a process, once proven to be in a state of statistical control, is **capable** of consistently producing output that meets pre-defined specification limits (USL/LSL).
    
    **Strategic Application:** This is the ultimate verdict on process performance, often the final gate in a process validation or technology transfer. It directly answers the critical business question: "Is our process good enough to reliably meet customer or regulatory requirements with a high degree of confidence?" 
    - A high capability index (Cpk) provides objective, statistical evidence that the process is robust, predictable, and delivers high quality. It is a key metric for supplier qualification, internal performance tracking, and regulatory submissions.
    - A low Cpk is a clear signal that the process requires fundamental improvement, either by **re-centering the process mean** to better align with the target, or by **reducing the process variation (standard deviation)**.
    In many ways, achieving a high Cpk is the statistical equivalent of "mission accomplished" for a process development or transfer team.
    """)
    scenario = st.sidebar.radio("Select Process Scenario:", ('Ideal', 'Shifted', 'Variable', 'Out of Control'))
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, cpk_val, scn = plot_capability(scenario)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scn != 'Out of Control' else "INVALID", help="Measures how well the process fits within the spec limits, accounting for centering. Higher is better.")
            st.markdown("""
            - **The Mantra: Control Before Capability.** The control chart (top plot) is a prerequisite. The Cpk metric is only statistically valid and meaningful if the process is stable and in-control. The 'Out of Control' scenario yields an **INVALID** Cpk because an unstable process has no single, predictable "voice" to measure. Its future performance is unknown.
            - **The Key Insight: Control ≠ Capability.** A process can be perfectly in-control (predictable) but not capable (producing bad product). 
                - The **'Shifted'** scenario shows a process that is precise but inaccurate.
                - The **'Variable'** scenario shows a process that is centered but imprecise.
            Both are in control, but both have a poor Cpk. This demonstrates why you need both SPC (for control) and Capability Analysis (for quality).
            """)
        with tabs[1]:
            st.markdown("These are industry-standard benchmarks, often required by customers, especially in automotive and aerospace. For pharmaceuticals, a high Cpk in validation provides strong assurance of lifecycle performance.")
            st.markdown("- `Cpk < 1.00`: Process is **not capable**. The "voice of the process" is wider than the "voice of the customer." A significant portion of output will not meet specifications.")
            st.markdown("- `1.00 ≤ Cpk < 1.33`: Process is **marginally capable**. It requires tight control and monitoring, as small shifts can lead to non-conforming product.")
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable**. This is a common minimum target for many industries, corresponding to a "4-sigma" quality level and a theoretical defect rate of ~63 parts per million (PPM).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** and is approaching **Six Sigma** quality. This corresponds to a "5-sigma" level and a theoretical defect rate of ~0.6 PPM.")
            st.markdown("- `Cpk ≥ 2.00`: Process has achieved **Six Sigma capability** (assuming no long-term shift). This represents world-class performance with a theoretical defect rate of just 2 parts per *billion*. ")

        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of comparing process output to specification limits is old, but the formalization into capability indices originated in the Japanese manufacturing industry in the 1970s as a core part of the Total Quality Management (TQM) movement. These tools allowed engineers to express process quality with a single, universal number.
            
            However, it was the **Six Sigma** initiative, pioneered by engineer Bill Smith at **Motorola in the 1980s**, that catapulted Cpk to global prominence. Motorola was suffering from high warranty costs and realized they needed a much higher standard of quality. The "Six Sigma" concept was born: a process so capable that the nearest specification limit is at least six standard deviations away from the process mean. This translates to a defect rate of just 3.4 parts per million (which famously accounts for a hypothetical 1.5 sigma long-term drift of the process mean). Cpk became the standard metric for measuring progress toward this ambitious goal.
            
            #### Mathematical Basis
            Capability analysis is a direct comparison between the **"Voice of the Customer"** (the allowable spread defined by the specification limits, USL - LSL) and the **"Voice of the Process"** (the actual, natural spread of the data, conventionally defined as a 6σ spread).

            - **Cp (Potential Capability):** This metric measures if the process is narrow enough, but it ignores centering. It's the best the process *could* be if it were perfectly centered. It answers the question: "Is our process fundamentally precise enough?"
            """)
            st.latex(r"C_p = \frac{\text{Tolerance Width}}{\text{Process Width}} = \frac{USL - LSL}{6\hat{\sigma}}")
            st.markdown("- **Cpk (Actual Capability):** This is the more important metric as it accounts for process centering. It is the lesser of the upper and lower capability indices, effectively measuring the distance from the process mean to the *nearest* specification limit. It is the 'worst-case scenario' and answers: "How is our process *actually* performing right now?")
            st.latex(r"C_{pk} = \min(C_{pu}, C_{pl}) = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right)")
            st.markdown("A Cpk of 1.33 means that the process distribution could fit between the mean and the nearest spec limit 1.33 times. This provides a 'buffer' zone to absorb small process shifts without producing defects.")

def render_anomaly_detection():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To leverage unsupervised machine learning to detect novel, complex, and multivariate anomalies that traditional, rule-based univariate control charts are fundamentally blind to.
    
    **Strategic Application:** This is a crucial tool for **proactive process monitoring and deviation investigation** in complex systems like bioreactors, multi-step chemical syntheses, or mass spectrometry analyses. An operator might confirm that every individual parameter (temperature, pH, pressure, flow rate) is within its established univariate control limits, yet the process could still be in an anomalous state due to a previously unseen *combination* of these parameters. This ML model acts as a "digital Subject Matter Expert," flagging these subtle, combinatorial deviations. It's a powerful tool for:
    - **Detecting emerging, unknown failure modes ("unknown unknowns").**
    - **Identifying the multivariate "golden batch" profile.**
    - **Accelerating root cause analysis** for unexplained process failures by pinpointing exactly when the process began to deviate from its normal multi-dimensional "fingerprint."
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_anomaly_detection(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Rules", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Anomalies Detected", value="3", help="Number of data points flagged by the model as being statistically unusual relative to the historical norm.")
            st.markdown("""
            - **The Plot:** The blue shaded area represents the model's learned definition of 'normal' multidimensional operating space. Points falling outside this boundary are flagged as anomalies (red 'X'). The contour lines show the gradient of the anomaly score—the further a point is from the dense blue region, the more anomalous it is.
            - **The Key Insight:** The power of this method is that the anomalous points are not necessarily extreme on any single axis. For example, the anomaly at `(95, 25)` is not the lowest response or the highest time, but its *combination* is unusual and falls in a low-density "valley" of the data cloud. A traditional control chart for 'Response' and 'Time' would not have flagged this point.

            **The Core Strategic Insight:** This approach moves monitoring from a one-dimensional, rule-based system to a holistic, multi-dimensional assessment of process health. It is an "unsupervised" method, meaning it learns what is normal directly from your historical data without needing pre-labeled examples of failures. This allows it to discover **previously unknown failure modes**, acting as a powerful safety net and an early-warning system for problems that have no precedent.
            """)
        with tabs[1]:
            st.markdown("- **This is an exploratory and monitoring tool, not a pass/fail criterion for product release.** It is a core component of a modern Process Analytical Technology (PAT) or Continued Process Verification (CPV) program under FDA guidance.")
            st.markdown("- The primary rule is that any point flagged as an **anomaly must trigger an automated alert and a formal SME investigation**. The goal is to determine the root cause of the anomaly and assess its potential impact on product quality. It is an input to the quality system, not a replacement for it.")
            st.markdown("- **Model Tuning:** The key parameter is `contamination`, which is an estimate of the proportion of anomalies in the training data. This sets the sensitivity of the model. It should be set to a low value (e.g., 0.01 to 0.05) and justified based on historical process knowledge.")
            st.info("**Deployment Strategy:** In a validated environment, the model is first trained on a large dataset of "good" historical batches. It is then deployed in a "read-only" or "monitoring" mode. When it flags an anomaly, that batch is subjected to extra scrutiny and testing. Over time, as certain types of anomalies are confirmed to be linked to failures, the model's outputs can gain more weight in decision-making.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The **Isolation Forest** algorithm, proposed by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou in a groundbreaking 2008 ICDM paper, represented a paradigm shift in anomaly detection. Previous methods were often "density-based" (like DBSCAN) or "distance-based" (like k-Nearest Neighbors), which involved complex calculations to define what a "normal" region looks like. These methods often fail in high-dimensional spaces due to the **"curse of dimensionality,"** where distances between all points start to look similar.
            
            The authors of Isolation Forest flipped the problem on its head with a simple but profound observation: **anomalies are "few and different."** This means they should be easier to *isolate* from the rest of the data points than normal points are. This elegant, counter-intuitive approach proved to be surprisingly effective, computationally efficient, and less sensitive to the curse of dimensionality, making it a go-to algorithm for modern anomaly detection.
            
            #### Mathematical Basis
            The core mechanism is brilliantly simple. The algorithm builds an ensemble of "Isolation Trees" (iTrees). Each tree is built by recursively partitioning the data with random splits on random features.
            - A **normal point**, being in a dense region with many similar points, will require many random splits to be isolated into its own leaf node. It will have a **long average path length** across all trees in the forest.
            - An **anomalous point**, being "different" and in a sparse region, will be isolated very quickly with few splits. It will have a **short average path length**.

            The anomaly score $s(x, n)$ for a point $x$ from a sample of size $n$ is derived from its average path length $E(h(x))$ across all the iTrees in the forest.
            """)
            st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")
            st.markdown("Where $c(n)$ is a normalization factor representing the average path length of an unsuccessful search in a Binary Search Tree. A score close to 1 indicates a definite anomaly, while a score much smaller than 0.5 indicates a normal point.")

def render_predictive_qc():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To build a supervised machine learning model that moves quality control from being *reactive* (detecting a failure after it has occurred) to being **proactive and predictive** (forecasting the probability of failure *before* a run is started).
    
    **Strategic Application:** This is a high-value application of machine learning in operations, directly impacting efficiency, cost, and "Right First Time" (RFT) rates. Before committing expensive, single-use reagents and valuable instrument time, this model acts as a "pre-flight check." It evaluates key initial process parameters (e.g., reagent age, raw material lot purity, operator certification, ambient temperature) to predict the likelihood of the run failing to meet its quality specifications.
    - **High-Risk Alert:** If the model predicts a high probability of failure, it can trigger a preemptive alert, allowing the operator to take corrective action (e.g., use a newer reagent, recalibrate the instrument) *before* starting the run. This prevents a costly Out-of-Specification (OOS) event.
    - **Resource Optimization:** For runs with a very low predicted probability of failure, quality control testing could potentially be reduced (e.g., via a skip-lot testing program), freeing up valuable lab capacity. This is a key concept in Real-Time Release Testing (RTRT).
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_predictive_qc(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Rules", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Predictive Risk Profiling", value="Enabled", help="The model provides a risk score for each potential run based on its inputs.")
            st.markdown("""
            - **Decision Boundary Risk Map (left):** This is the model's 'risk map' or 'operating envelope'. The color gradient shows the predicted probability of failure for any combination of the input parameters. The model has learned that runs with old reagents *and* high temperatures are in the high-risk 'red zone'.
            - **Probability Distribution Plot (right):** This is the model's performance report card. It shows the model's predicted failure probabilities for runs that we know (from historical data) actually passed versus those that actually failed. **A clear separation between the green (Pass) and red (Fail) distributions is the goal.** It proves that the model can reliably distinguish between the characteristics of a good run and a bad one. Overlap between the distributions represents the model's uncertainty or error rate.

            **The Core Strategic Insight:** This model digitizes and automates the intuition of an experienced senior scientist. It learns the subtle, hidden patterns in the input parameters that lead to failure. It allows an organization to move from forensics (figuring out why a run failed) to prognostics (preventing the failure from ever happening).
            """)
        with tabs[1]:
            st.markdown("- **Risk Threshold:** A risk threshold must be established based on a cost-benefit analysis and the risk tolerance of the organization (e.g., "If Predicted P(Fail) > 30%, flag the run for mandatory SME review before proceeding"). This threshold directly controls the trade-off between false alarms and missed failures.")
            st.markdown("- **Formal Model Validation (per GAMP 5):** Before use in a GxP environment, the ML model must be rigorously validated like any other piece of software or equipment. This involves documenting its performance using metrics derived from a dedicated test dataset:")
            st.markdown("  - **Confusion Matrix:** Shows the raw counts of True Positives, True Negatives, False Positives (Type I Error), and False Negatives (Type II Error).")
            st.markdown("  - **ROC Curve & AUC:** The Receiver Operating Characteristic (ROC) curve plots the model's sensitivity vs. its false positive rate. The Area Under the Curve (AUC) provides a single metric of overall performance (AUC > 0.8 is often considered good; > 0.9 is excellent).")
            st.markdown("- **Model Lifecycle Management:** The model is not static. It must be periodically monitored for **model drift** and retrained as new process data becomes available and the process itself evolves. A formal plan for model maintenance is a regulatory expectation.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **Logistic Regression** is a workhorse statistical model developed by the brilliant British statistician **Sir David Cox in 1958**. It was created to solve a common problem: how to predict the probability of a binary (Yes/No, Pass/Fail) outcome. Standard linear regression is unsuitable because its output is unbounded and can produce nonsensical probabilities like -20% or 150%.
            
            Cox's elegant solution was to use the **logistic function (also known as the sigmoid function)** to "squash" the output of a standard linear equation into the required [0, 1] probability range. The model's power comes from its simplicity and, most importantly, its **interpretability**. The coefficients of a logistic regression model directly tell you the effect of each input variable on the **log-odds** of the outcome, making it a transparent "white-box" model, which is highly desirable in regulated industries where model explainability is paramount.
            
            #### Mathematical Basis
            The model first creates a linear combination of the input features ($x_i$), which is called the **log-odds** or **logit** ($z$). This is the "linear regression" part.
            """)
            st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
            st.markdown("""
            This unbounded logit value is then transformed into a probability $P$ using the standard logistic (sigmoid) function, $\sigma(z)$.
            """)
            st.latex(r"P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}")
            st.markdown("The coefficients ($\beta_i$) are found by a process called **Maximum Likelihood Estimation**, which finds the coefficient values that maximize the probability of observing the actual outcomes in the training data. The decision boundary shown on the plot is the line or surface where the predicted probability is exactly 0.5, which corresponds to where the logit $z=0$.")
def render_forecasting():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To leverage a modern time series model to forecast the future performance and behavior of key process parameters or quality controls, enabling proactive and strategic management instead of reactive problem-solving.
    
    **Strategic Application:** This AI-powered tool transforms quality and operations management from being reactive or calendar-based to being **predictive and condition-based**. By providing a statistically sound forecast of a control's trajectory, the system can identify future problems before they occur. This enables:
    - **Proactive Maintenance & Calibration:** "The model predicts the instrument's calibration control will drift out of spec in 4 weeks. Let's schedule maintenance for week 3." This avoids an OOS investigation and instrument downtime.
    - **Intelligent Inventory Management:** "This reagent lot's performance is forecasted to decline rapidly. Let's order a new lot now and plan the crossover validation study to ensure a seamless transition."
    - **Improved Planning & Root Cause Analysis:** "The model shows a strong seasonal effect in the summer months. Let's allocate extra engineering support during that period." The changepoint detection can also pinpoint the exact date a process problem started, dramatically speeding up an investigation.
    This approach minimizes downtime, reduces fire-fighting, and optimizes resource allocation.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_fc, fig2_fc, fig3_fc = plot_forecasting()
        st.plotly_chart(fig1_fc, use_container_width=True)
        st.plotly_chart(fig2_fc, use_container_width=True)
        st.plotly_chart(fig3_fc, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Rules", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Forecast Status", value="Future Breach Predicted", help="The model's forecast interval crosses a known specification limit within the forecast horizon.")
            st.markdown("""
            - **Forecast Plot (Top):** Shows the historical data (black dots), the model's one-step-ahead fit (dark blue line), and the future forecast with its uncertainty interval (light blue band). The key output is the diamond marker, indicating the point where the upper uncertainty bound is predicted to breach the specification limit.
            - **Trend & Changepoints (Middle):** This is the most powerful diagnostic plot for understanding the process's long-term behavior. It extracts the underlying trend, filtering out seasonality and noise. The red dashed lines are **changepoints** automatically detected by the model, indicating dates where the growth rate of the trend fundamentally changed. These are critical clues for process investigation.
            - **Seasonality (Bottom):** This plot extracts the predictable, repeating patterns within the data (e.g., a yearly cycle). Understanding seasonality can help distinguish true process drifts from normal, expected fluctuations.

            **The Core Strategic Insight:** This analysis decomposes your complex time series into its fundamental, interpretable components: Trend, Seasonality, and Events. This provides a roadmap for the future, telling you not just *what* might happen, but giving you strong clues as to *why*. The detected changepoint around mid-2022 is a prime example, pointing to a specific time when the process fundamentally changed for the better or worse, such as when a new piece of equipment was installed or a new SOP was implemented.
            """)
        with tabs[1]:
            st.markdown("- **Proactive Alert Threshold:** An alert should be triggered if the **80% or 95% forecast confidence interval** (`yhat_upper` or `yhat_lower`) is predicted to cross a specification limit within a defined, actionable forecast horizon (e.g., the next 4-8 weeks).")
            st.markdown("- **Changepoint Investigation:** Any automatically detected changepoint must be treated as a significant event. A formal investigation should be launched to correlate the date of the changepoint with historical batch records, maintenance logs, or personnel changes to identify the root cause. This is a powerful tool for Continued Process Verification (CPV).")
            st.markdown("- **Model Monitoring:** The model's forecast accuracy (e.g., using Mean Absolute Percentage Error - MAPE on a hold-out test set) should be tracked over time. A significant degradation in accuracy indicates the process dynamics have changed, and the model needs to be retrained.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **Prophet**, developed and open-sourced by **Facebook's Core Data Science team in 2017**, was a direct response to the challenges of producing high-quality business forecasts at scale. Traditional statistical methods like **ARIMA** (AutoRegressive Integrated Moving Average) or **Exponential Smoothing** are powerful but often require significant manual effort, deep statistical expertise for tuning (e.g., interpreting ACF/PACF plots), and struggle with common business data features like multiple seasonalities, shifting trends, and missing data.
            
            Prophet's developers took a different approach, framing forecasting as a **generalized additive model (GAM)** curve-fitting problem, which is much more intuitive and automatic. It was designed from the ground up to be robust, easy to use, and to handle the messy realities of real-world business time series data, democratizing the ability to create reliable forecasts without requiring a PhD in statistics.
            
            #### Mathematical Basis
            Prophet is a **decomposable time series model**, which models the time series $y(t)$ as a sum of three main components plus an error term:
            """)
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("""
            - **$g(t)$ is the trend function.** Prophet models this with a piecewise linear or logistic growth curve. It automatically selects potential changepoints and uses a sparse prior on the rate changes, which prevents overfitting while allowing flexibility.
            - **$s(t)$ is the seasonality function.** This component models periodic changes (e.g., weekly, yearly). Prophet cleverly uses a flexible **Fourier series** to model seasonality. For a yearly seasonality, it's a sum of sines and cosines: $s(t) = \sum_{n=1}^{N}(a_n \cos(\frac{2\pi n t}{365.25}) + b_n \sin(\frac{2\pi n t}{365.25}))$.
            - **$h(t)$ is the holidays/events function.** This allows the user to provide a custom list of special events that are modeled as additional regressors.
            - **$\epsilon_t$ is the error term,** assumed to be normally distributed noise.

            The entire model is fit within a Bayesian framework using Stan (a probabilistic programming language), which allows it to produce realistic uncertainty intervals for the forecast and all of its components.
            """)
        
def render_pass_fail():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately calculate and critically compare confidence intervals for a binomial proportion, which is the underlying statistic for any pass/fail, present/absent, or concordant/discordant outcome.
    
    **Strategic Application:** This is essential for the validation of **qualitative assays** or for agreement studies in method transfers. The goal is to prove, with a high degree of statistical confidence, that the assay's success rate (e.g., >95% concordance with a reference method, or >99% sensitivity/specificity) is above a required performance threshold. 
    
    The critical challenge, especially with the small sample sizes typical in validation studies (n=30 is common), is that the simple, textbook methods for calculating confidence intervals (the 'Wald' interval) are dangerously inaccurate and misleading. Choosing the wrong statistical method can lead to falsely concluding a method is acceptable when it is not, a major regulatory and quality risk. This tool demonstrates why a rigorous approach is non-negotiable.
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
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}", help="The point estimate of the success rate. This value alone is insufficient without a confidence interval.")
            st.markdown("""
            - **CI Comparison (Top):** This plot reveals the dramatic differences between interval methods. Note how the 'Wald' interval is often much narrower, giving a false sense of precision. At the extremes (e.g., 30/30 successes), the Wald interval collapses to a width of zero, which is statistically indefensible.
            - **Coverage Probability (Bottom):** This is the crucial diagnostic plot. It shows the *actual* probability that an interval, calculated at a given true proportion, will contain that true proportion.
                - The **Wald interval (red)** is a disaster. Its actual coverage plummets to near zero at the extremes and is wildly erratic everywhere else. It consistently fails to meet the nominal 95% level.
                - The **Wilson and Clopper-Pearson intervals (blue/green)** are far superior. Their coverage probability is always at or above the nominal 95% level, making them reliable and conservative.

            **The Core Strategic Insight:** Never use the standard Wald (or "Normal Approximation") interval for important decisions, especially with sample sizes under 100. It is systematically wrong and will lead to overconfidence. The **Wilson Score interval** provides the best balance of accuracy and interval width for most applications. The **Clopper-Pearson** is the most conservative ("exact") choice, often preferred in regulatory submissions for its guaranteed coverage, which can make it easier to defend.
            """)
        with tabs[1]:
            st.markdown("- **The Golden Rule of Binomial Acceptance:** The acceptance criterion must **always be based on the lower bound of the confidence interval**, never on the point estimate.")
            st.markdown("- **Example Criterion:** 'The lower bound of the 95% **Wilson Score** (or Clopper-Pearson) confidence interval for the concordance rate must be greater than or equal to the target of 90%.'")
            st.markdown("- **Sample Size Implication:** This interactive tool powerfully demonstrates why larger sample sizes are needed for high-confidence claims. With a small `n`, even a perfect result (e.g., 20/20 successes) may have a lower confidence bound that fails to meet a high target (like 95%), forcing the study to be repeated with more samples. This tool can be used to plan the required `n` to achieve a desired lower bound.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            For much of the 20th century, the simple **Wald interval** (named after Abraham Wald) was taught in introductory statistics classes due to its simplicity. However, its poor performance was well-known to statisticians. A famous 1998 paper by Brown, Cai, and DasGupta, titled "Interval Estimation for a Binomial Proportion," comprehensively documented the Wald interval's failures and strongly advocated for the use of superior alternatives. This paper is a classic in the field and effectively killed the Wald interval's credibility for serious work.
            
            The **Wilson Score Interval**, developed by Edwin Bidwell Wilson in 1927, and the **Clopper-Pearson Interval**, developed by C. J. Clopper and Egon Pearson in 1934, were created to solve this problem.
            - The **Clopper-Pearson** interval is called an "exact" method because it is derived directly from the quantiles of the binomial distribution. It guarantees that the coverage probability will never be less than the nominal level (e.g., 95%). This guarantee makes it conservative (i.e., wider than necessary on average).
            - The **Wilson Score** interval is derived by inverting the score test, a more sophisticated statistical test. It does not have the strict guarantee of the Clopper-Pearson, but its average coverage probability is much closer to the nominal 95% level, making it more accurate and less conservative in practice.
            
            #### Mathematical Basis
            The Wald interval is simply $\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$. The Wilson Score interval, however, is the solution to a quadratic equation derived from inverting the score test for a proportion. This results in its more complex, but far superior, formula:
            """)
            st.latex(r"CI_{Wilson} = \frac{1}{1 + z^2/n} \left( \hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \right)")
            st.markdown("Notice the 'smoothing' effect: it pulls the center of the interval away from 0 or 1 and towards 0.5 by adding pseudo-successes and failures ($z^2/2$). This is what gives it such good performance near the boundaries where the Wald interval fails catastrophically.")
            
def render_bayesian():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To employ Bayesian inference to formally and quantitatively synthesize existing knowledge (a **Prior** belief) with new experimental data (the **Likelihood**) to arrive at an updated, more robust conclusion (the **Posterior** belief).
    
    **Strategic Application:** This is a paradigm-shifting tool for driving efficient, knowledge-based validation and decision-making. In a traditional (Frequentist) world, every study starts from a blank slate of ignorance. In the Bayesian world, we can formally and transparently leverage what we already know. This is particularly powerful for:
    - **Accelerating Tech Transfer:** We can use the extensive data from an R&D validation study to form a **strong, informative prior**. This allows the receiving QC lab to demonstrate success with a smaller, more targeted confirmation study, saving significant time and resources.
    - **Adaptive Clinical Trials:** Data from an interim analysis can serve as a prior for the final analysis, allowing trials to be stopped early for success or futility.
    - **Quantifying Belief & Risk:** It provides a natural framework to answer the question business leaders and scientists actually ask: "Given what we already knew, and what this new data shows, what is the probability that the pass rate is actually above 95%?" This is a question a frequentist confidence interval cannot answer.
    """)
    prior_type_bayes = st.sidebar.radio("Select Prior Belief:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, prior_mean, mle, posterior_mean = plot_bayesian(prior_type_bayes)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}", help="The final, data-informed belief; a weighted average of the prior and the data.")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}", help="The initial belief *before* seeing the new QC data.")
            st.metric(label="💡 Data-only Estimate (MLE)", value=f"{mle:.3f}", help="The evidence from the new QC data alone (the frequentist result).")
            st.markdown("""
            - **Prior (Green Dashed):** Our initial belief about the pass rate. A **Strong Prior** is tall and narrow, representing high confidence. A **Skeptical Prior** is wide and flat, representing uncertainty. The **No Prior** case uses a uniform "uninformative" prior, letting the data speak for itself.
            - **Likelihood (Red Dotted):** The "voice of the new data." This is the evidence from our 20 QC runs. Note that it is not a probability distribution.
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
            The underlying theorem was conceived by the Reverend **Thomas Bayes** in the 1740s and published posthumously by Richard Price. However, for nearly 200 years, Bayesian inference remained a philosophical curiosity, largely overshadowed by the "Frequentist" school of thought championed by figures like R.A. Fisher and Jerzy Neyman. There were two main reasons for this: philosophical objections to the subjective nature of priors, and, more practically, the computational difficulty of calculating the posterior distribution for all but the simplest problems.
            
            The **"Bayesian Revolution"** began in the late 20th century, driven by the rise of powerful computers and the development of sophisticated simulation algorithms like **Markov Chain Monte Carlo (MCMC)**. These methods allowed scientists to approximate the posterior distribution for incredibly complex models, making Bayesian methods practical for the first time. It is now a co-equal paradigm in statistics and the dominant approach in many fields of machine learning.
            
            #### Mathematical Basis
            Bayes' Theorem is elegantly simple:
            """)
            st.latex(r"P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}")
            st.markdown(r"In words: **Posterior Probability = (Likelihood × Prior Probability) / Evidence**")
            st.markdown(r"""
            - $P(\theta|D)$ (Posterior): The probability of our parameter $\theta$ (e.g., the true pass rate) given the new Data D. This is what we want to compute.
            - $P(D|\theta)$ (Likelihood): The probability of observing our Data D, given a specific value of the parameter $\theta$.
            - $P(\theta)$ (Prior): Our initial belief about the distribution of the parameter $\theta$ before seeing the data.
            
            For binomial data (pass/fail), there is a beautiful mathematical shortcut. The **Beta distribution** is a **conjugate prior** for the binomial likelihood. Conjugacy means that if you start with a prior from one family of distributions (Beta) and your likelihood is from another family (Binomial), your posterior will also be in the first family (Beta). This avoids the need for complex MCMC simulations.
            - If Prior is Beta($\alpha_{prior}, \beta_{prior}$)
            - And Data is $k$ successes in $n$ trials:
            - Then the Posterior is simply Beta($\alpha_{prior} + k, \beta_{prior} + n - k$).
            The $\alpha$ and $\beta$ parameters can be thought of as "pseudo-counts" of prior successes and failures, which are simply added to the new observed counts.
            """)

def render_ci_concept():
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
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ The Golden Rule", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label=f"📈 KPI: Average CI Width (Precision) at n={n_slider}", value=f"{avg_width:.2f} units", help="A smaller width indicates higher precision. This is inversely proportional to the square root of n.")
            st.metric(label="💡 Empirical Coverage Rate", value=f"{(capture_count/n_sims):.0%}", help=f"The % of our {n_sims} simulated CIs that successfully 'captured' the true population mean. Should be close to 95%.")
            st.markdown("""
            - **Theoretical Universe (Top):**
                - The wide, light blue curve is the **true population distribution**. In real life, we never see this. It represents every possible measurement.
                - The narrow, orange curve is the **sampling distribution of the mean**. This is a theoretical distribution of *all possible sample means* of size `n`. Its narrowness, guaranteed by the **Central Limit Theorem**, is the miracle that makes statistical inference possible.
            - **CI Simulation (Bottom):** This plot shows the reality we live in. We only get to run *one* experiment and get *one* confidence interval (e.g., the first blue line). We don't know if ours is one of the 95 that captured the true mean or one of the 5 that missed.
            - **The n-slider is key:** As you increase `n`, the orange curve gets narrower and the CIs in the bottom plot become dramatically shorter. This shows that precision is a direct function of sample size.
            - **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This illustrates that the cost of doubling precision is a quadrupling of sample size, as precision scales with $\sqrt{n}$.

            **The Core Strategic Insight:** A confidence interval is a statement about the *procedure*, not the result. The "95% confidence" is our confidence in the *method* used to generate the interval, not in any single interval itself. We are confident that if we were to repeat our experiment 100 times, roughly 95 of the resulting CIs would contain the true, unknown mean.
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
            The concept of **confidence intervals** was introduced to the world by the brilliant Polish-American mathematician and statistician **Jerzy Neyman** in a landmark 1937 paper. Neyman was a fierce advocate for the frequentist school of statistics and sought a rigorously objective method for interval estimation that did not rely on the "subjective" priors of Bayesian inference.
            
            He was a philosophical rival of Sir R.A. Fisher, who had proposed a similar concept called a "fiducial interval," which attempted to assign a probability distribution to a fixed parameter. Neyman found this logically incoherent. His revolutionary idea was to shift the probabilistic statement away from the fixed, unknown parameter and onto the **procedure used to create the interval**. This clever reframing provided a practical and logically consistent solution that quickly became, and remains, the dominant paradigm for interval estimation in applied statistics worldwide.
            
            #### Mathematical Basis
            The general form of a two-sided confidence interval is a combination of three components:
            """)
            st.latex(r"\text{Point Estimate} \pm (\text{Margin of Error})")
            st.latex(r"\text{Margin of Error} = (\text{Critical Value} \times \text{Standard Error})")
            st.markdown("""
            - **Point Estimate:** Our best single-value guess for the population parameter (e.g., the sample mean, $\bar{x}$).
            - **Standard Error:** The standard deviation of the sampling distribution of the point estimate. For the mean, this is $\frac{s}{\sqrt{n}}$, where $s$ is the sample standard deviation. It measures the typical error in our point estimate. Note that it shrinks as the square root of the sample size, `n`.
            - **Critical Value:** A multiplier determined by our desired confidence level and the relevant statistical distribution (e.g., a z-score from the normal distribution or a t-score from the Student's t-distribution). For a 95% CI, this value is typically close to 2.
            
            For the mean, this results in the familiar formula:
            """)
            st.latex(r"CI = \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}")
            


# ==============================================================================
# MAIN APP LAYOUT & LOGIC
# ==============================================================================
st.title("🛠️ Biotech V&V Analytics Toolkit")
st.markdown("### An Interactive Guide to Assay Validation, Tech Transfer, and Lifecycle Management")
st.markdown("Welcome! This toolkit is a collection of interactive modules designed to explore the statistical and machine learning methods that form the backbone of a robust V&V, technology transfer, and process monitoring plan.")

tab_intro, tab_map, tab_journey = st.tabs(["🚀 The V&V Framework", "🗺️ Concept Map", "📖 The Scientist's Journey"])
with tab_intro:
    st.markdown('<h4 class="section-header">The V&V Model: A Strategic Framework</h4>', unsafe_allow_html=True)
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework...")
    st.plotly_chart(plot_v_model(), use_container_width=True)

with tab_map:
    st.markdown('<h4 class="section-header">Conceptual Map of V&V Tools</h4>', unsafe_allow_html=True)
    st.plotly_chart(create_conceptual_map_plotly(), use_container_width=True)
    st.markdown("This map illustrates how foundational **Academic Disciplines** give rise to **Core Domains** such as Statistical Process Control (SPC)...")

with tab_journey:
    st.header("The Scientist's/Engineer's Journey: A Three-Act Story")
    st.markdown("""In the world of quality and development, the challenges are often complex and hidden in the details...""")
    act1, act2, act3 = st.columns(3)
    with act1: st.subheader("Act I: Know Thyself (The Foundation)"); st.markdown("Before any transfer or scale-up, you must understand the capability and limits of your current measurement systems... **(Tools 1-5)**")
    with act2: st.subheader("Act II: The Transfer (The Crucible)"); st.markdown("A validated method must prove its robustness in a new environment... **(Tools 6-9)**")
    with act3: st.subheader("Act III: The Guardian (Lifecycle Management)"); st.markdown("Once the method is live, continuous monitoring is essential... **(Tools 10-15)**")
st.divider()


# --- CORRECTED SIDEBAR LOGIC ---

st.sidebar.title("🧰 Toolkit Navigation")
st.sidebar.markdown("Select a statistical method to analyze and visualize.")

# Define the menu options and icons for each act
act1_options = ["Gage R&R", "Linearity and Range", "LOD & LOQ", "Method Comparison", "Assay Robustness (DOE/RSM)"]
act2_options = ["Process Stability (Shewhart)", "Small Shift Detection", "Run Validation", "Process Capability (Cpk)"]
act3_options = ["Anomaly Detection (ML)", "Predictive QC (ML)", "Control Forecasting (AI)", "Pass/Fail Analysis", "Bayesian Inference", "Confidence Interval Concept"]

act1_icons = [ICONS.get(opt, "question-circle") for opt in act1_options]
act2_icons = [ICONS.get(opt, "question-circle") for opt in act2_options]
act3_icons = [ICONS.get(opt, "question-circle") for opt in act3_options]

# Initialize session state for the selected method if it doesn't exist
if 'method_key' not in st.session_state:
    st.session_state.method_key = act1_options[0]

# CORRECTED: Define callback functions that accept the selected option value as an argument
def update_method(selected_option):
    st.session_state.method_key = selected_option

with st.sidebar.expander("ACT I: FOUNDATION & CHARACTERIZATION", expanded=True):
    option_menu(None, act1_options, icons=act1_icons, menu_icon="cast", 
                key='act1_menu', 
                on_change=update_method,
                default_index=act1_options.index(st.session_state.method_key) if st.session_state.method_key in act1_options else 0)

with st.sidebar.expander("ACT II: TRANSFER & STABILITY", expanded=True):
    option_menu(None, act2_options, icons=act2_icons, menu_icon="cast",
                key='act2_menu',
                on_change=update_method,
                default_index=act2_options.index(st.session_state.method_key) if st.session_state.method_key in act2_options else 0)

with st.sidebar.expander("ACT III: LIFECYCLE & PREDICTIVE MGMT", expanded=True):
    option_menu(None, act3_options, icons=act3_icons, menu_icon="cast",
                key='act3_menu',
                on_change=update_method,
                default_index=act3_options.index(st.session_state.method_key) if st.session_state.method_key in act3_options else 0)

# --- PowerPoint Download Section in Sidebar ---
st.sidebar.divider()
st.sidebar.header("📊 Generate Report")
if st.sidebar.button("Generate Executive PowerPoint Summary", use_container_width=True):
    with st.spinner("Building your report..."):
        _, pct_rr, _ = plot_gage_rr()
        _, model_lin = plot_linearity()
        _, cpk_val, _ = plot_capability('Ideal')
        kpi_data = [
            ("Gage R&R", f"{pct_rr:.1f}%", "Measurement System Variation"),
            ("Linearity R²", f"{model_lin.rsquared:.4f}", "Goodness of Fit"),
            ("Process Capability (Cpk)", f"{cpk_val:.2f}", "Ability to Meet Specs")
        ]
        spc_fig = plot_shewhart()
        ppt_buffer = generate_ppt_report(kpi_data, spc_fig)
        st.sidebar.download_button(
            label="📥 Download PowerPoint Report",
            data=ppt_buffer,
            file_name=f"V&V_Analytics_Summary_{pd.Timestamp.now().strftime('%Y%m%d')}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True
        )

st.divider()

# Get the currently selected method from session state
method_key = st.session_state.method_key
method_list = act1_options + act2_options + act3_options
method_key_with_num = f"{method_list.index(method_key) + 1}. {method_key}"
st.header(method_key_with_num)

# --- Main Content Area Dispatcher ---
PAGE_DISPATCHER = {
    "Gage R&R": render_gage_rr, "Linearity and Range": render_linearity, "LOD & LOQ": render_lod_loq,
    "Method Comparison": render_method_comparison, "Assay Robustness (DOE/RSM)": render_robustness_rsm,
    "Process Stability (Shewhart)": render_shewhart, "Small Shift Detection": render_ewma_cusum,
    "Run Validation": render_multi_rule, "Process Capability (Cpk)": render_capability,
    "Anomaly Detection (ML)": render_anomaly_detection, "Predictive QC (ML)": render_predictive_qc,
    "Control Forecasting (AI)": render_forecasting, "Pass/Fail Analysis": render_pass_fail,
    "Bayesian Inference": render_bayesian, "Confidence Interval Concept": render_ci_concept
}

if method_key in PAGE_DISPATCHER:
    PAGE_DISPATCHER[method_key]()
else:
    st.error("An error occurred with the navigation. Please refresh the page.")
    st.session_state.method_key = act1_options[0]
    st.rerun()
