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

    /* Sidebar styling */
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
# UI RENDERING FUNCTIONS
# ==============================================================================

def render_gage_rr():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To rigorously quantify the inherent variability (error) of a measurement system and separate it from the true variation of the process being measured. A Gage R&R study is the definitive method for assessing the reliability of a measurement instrument or analytical method.
    
    **Application:** This study represents the foundational checkpoint in any technology transfer or process validation. Before one can claim a process is stable or capable, one must first prove that the "ruler" being used to measure it is trustworthy. An unreliable measurement system injects noise and uncertainty into the data, potentially masking real process shifts or creating false alarms. By partitioning the total observed variation into its distinct components—**Repeatability** (equipment variation), **Reproducibility** (operator variation), and **Part-to-Part** variation—this analysis provides statistical proof of the measurement system's fitness for purpose. It is a non-negotiable prerequisite for all subsequent validation activities.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, pct_rr, pct_part = plot_gage_rr()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse")
            st.metric(label="💡 KPI: % Part Variation", value=f"{pct_part:.1f}%", delta="Higher is better")
            st.markdown("- **Variation by Part & Operator (Main Plot):** This plot visualizes the core interactions. Ideally, the colored lines (operator means) should track each other closely, and the boxes (representing measurement spread) should be small and consistent across all parts.")
            st.markdown("- **Overall Variation by Operator (Top Right):** This provides a summary view. If the boxes are at different heights, it indicates a systematic bias between operators (a reproducibility issue).")
            st.markdown("**The Core Insight:** A low % Gage R&R proves that your measurement system is a reliable 'ruler' and that most of the variation you see in your process is real process variation, not measurement noise. A high value indicates that your ruler is "spongy," making it impossible to trust your measurements.")
        with tabs[1]:
            st.markdown("Acceptance criteria are typically based on guidelines from the **Automotive Industry Action Group (AIAG)**, which are considered the global standard:")
            st.markdown("- **< 10% Gage R&R:** The measurement system is **acceptable** and can be used without reservation.")
            st.markdown("- **10% - 30% Gage R&R:** The system is **conditionally acceptable**. Its use may be approved based on the importance of the application, the cost of improving the measurement system, and other factors. It signals a need for caution.")
            st.markdown("- **> 30% Gage R&R:** The system is **unacceptable**. It must be improved before it can be used for process control or validation. Data collected with such a system is considered unreliable.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concepts of Repeatability and Reproducibility have been a cornerstone of measurement science for over a century, but they were formally codified and popularized by the **Automotive Industry Action Group (AIAG)** in the 1980s. During the major quality revolution in the US auto industry, driven by competition from Japan, the AIAG created the Measurement Systems Analysis (MSA) manual. This manual established the Gage R&R study as a global standard for assessing the quality of a measurement system.
            
            The earliest methods for calculation were simple range-based approximations. However, these methods had a critical flaw: they could not separate the variation due to operator-part interaction from the variation due to the operators themselves. To solve this, the industry adopted **Analysis of Variance (ANOVA)**, a technique pioneered by Sir Ronald A. Fisher, as the preferred method. ANOVA is a powerful statistical tool that can rigorously partition the total variation into its distinct sources, providing a much more precise and insightful analysis.

            #### Mathematical Basis
            The ANOVA method is based on partitioning the total sum of squares ($SS_T$) into components attributable to each source of variation:
            """)
            st.latex(r"SS_T = SS_{Part} + SS_{Operator} + SS_{Interaction} + SS_{Error}")
            st.markdown("""
            From the Mean Squares (MS = SS/df) in the ANOVA table, we can estimate the variance components ($\hat{\sigma}^2$) for each source:
            - **Repeatability (Equipment Variation, EV):** This is the inherent, random error of the measurement system itself, estimated directly from the Mean Square Error.
            """)
            st.latex(r"\hat{\sigma}^2_{EV} = MS_{Error}")
            st.markdown("- **Reproducibility (Appraiser Variation, AV):** This is the variation introduced by different operators. It includes the main effect of the operator and the operator-part interaction.")
            st.latex(r"\hat{\sigma}^2_{AV} = \frac{MS_{Operator} - MS_{Interaction}}{n_{parts} \cdot n_{replicates}} + \frac{MS_{Interaction} - MS_{Error}}{n_{replicates}}")
            st.markdown("""
            The total **Gage R&R** variance is the sum of these two components, and the key KPI is its contribution to the total process variation.
            """)
            st.latex(r"\%R\&R = 100 \times \left( \frac{\hat{\sigma}^2_{EV} + \hat{\sigma}^2_{AV}}{\hat{\sigma}^2_{Total}} \right)")

def render_linearity():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To verify an assay's ability to provide results that are directly proportional to the concentration of the analyte across a specified, reportable range.
    
    **Application:** This study is a fundamental part of assay validation, as stipulated by regulatory bodies like the FDA and ICH. It provides statistical evidence that the assay is not only precise but also consistently accurate across its entire measurement range. An assay with non-linearity can produce dangerously misleading results at the extremes (high or low concentrations), even if it performs well in the middle. This evaluation is therefore critical for ensuring reliable data interpretation for all patient samples or product batches.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, model = plot_linearity()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}")
            st.markdown("- **Linearity Plot:** Visually confirms the straight-line relationship between nominal and measured values, ideally tracking the black 'Line of Identity'.")
            st.markdown("- **Residual Plot:** This is the most powerful diagnostic tool. A random, structureless scatter of points around zero confirms linearity. A curve or funnel shape reveals non-linearity or heteroscedasticity (non-constant variance), respectively.")
            st.markdown("- **Recovery Plot:** Directly assesses accuracy at each level. Points falling outside the 80-120% limits indicate a significant bias at those concentrations, which may require limiting the reportable range.")
            st.markdown("**The Core Insight:** A high R², a slope near 1, an intercept near 0, random residuals, and recovery within limits collectively provide statistical proof that your assay is trustworthy across its entire reportable range.")
        with tabs[1]:
            st.markdown("- **R-squared (R²):** The coefficient of determination should be very high, typically **> 0.995** for analytical methods.")
            st.markdown("- **Slope:** The slope of the regression line should be statistically indistinguishable from 1.0. A common acceptance range is **0.95 to 1.05**.")
            st.markdown("- **Y-Intercept:** The 95% confidence interval for the intercept should contain **0**, indicating no significant constant bias.")
            st.markdown("- **Recovery:** The percent recovery at each concentration level should fall within a pre-defined range, often **80% to 120%** for bioassays or **98% to 102%** for simpler chemical assays.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The mathematical foundation for this analysis is **Ordinary Least Squares (OLS) Regression**, a fundamental statistical method developed independently by Adrien-Marie Legendre (1805) and Carl Friedrich Gauss (1809). Gauss, working in astronomy, developed the method to predict the orbits of celestial bodies from a limited number of observations. He was trying to find the "best fit" curve to describe the path of the dwarf planet Ceres.

            The core principle of OLS is to find the line that minimizes the sum of the squared vertical distances (the "residuals") between the observed data points and the fitted line. This concept of minimizing squared error is one of the most powerful and widely used ideas in all of statistics and machine learning. In assay validation, we apply this centuries-old technique to answer a very modern question: "Does my instrument's response have a linear relationship with the true concentration of the substance I'm measuring?"
            
            #### Mathematical Basis
            We fit a simple linear model to the calibration data:
            """)
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - $y$ is the measured concentration (the response).
            - $x$ is the nominal (true) concentration.
            - $\\beta_0$ is the y-intercept, which represents the constant systematic bias of the assay (ideally 0).
            - $\\beta_1$ is the slope, which represents the proportional bias of the assay (ideally 1).
            - $\\epsilon$ is the random error term.

            The analysis involves statistical tests on the estimated coefficients:
            - **Hypothesis Test for Slope:** $H_0: \\beta_1 = 1$ vs. $H_a: \\beta_1 \\neq 1$
            - **Hypothesis Test for Intercept:** $H_0: \\beta_0 = 0$ vs. $H_a: \\beta_0 \\neq 0$
            """)

def render_lod_loq():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To determine the lowest concentration of an analyte that an assay can reliably detect (LOD) and accurately quantify (LOQ).
    
    **Application:** This is a critical component of assay characterization, defining the absolute lower limits of the method's capability. The **Limit of Detection (LOD)** answers the qualitative question, "Is the analyte present?" It is the lowest concentration that produces a signal distinguishable from the background noise. The **Limit of Quantitation (LOQ)** is more stringent; it is the lowest concentration that can be measured with an acceptable level of precision and accuracy and typically defines the lower boundary of the assay's reportable range. Understanding these limits is essential for applications such as impurity testing or low-level biomarker detection, where trust in sensitive measurements is paramount.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, lod_val, loq_val = plot_lod_loq()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} ng/mL")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} ng/mL")
            st.markdown("- **Signal Distribution:** The violin plot (top) visually confirms that the distribution of the low-concentration samples is clearly separated from the distribution of the blank samples. Overlap here would indicate poor sensitivity.")
            st.markdown("- **Low-Level Calibration Curve:** The regression plot (bottom) confirms the assay is linear at the low end of the range. The LOD and LOQ are derived from the variability of the residuals (residual standard error) and the slope of this line.")
            st.markdown("**The Core Insight:** This analysis defines the absolute floor of your assay's capability. It provides the statistical evidence to claim that you can trust quantitative measurements down to the LOQ, and reliably detect the presence of the analyte down to the LOD.")
        with tabs[1]:
            st.markdown("- The primary acceptance criterion is that the experimentally determined **LOQ must be less than or equal to the lowest concentration that needs to be measured** for the assay's intended use (e.g., the specification limit for a critical impurity).")
            st.markdown("- The precision (%CV) and accuracy (%Recovery) at the claimed LOQ should also meet pre-defined acceptance criteria (e.g., CV < 20%, Recovery 80-120%).")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concepts of LOD and LOQ were formalized and harmonized for the pharmaceutical industry by the **International Council for Harmonisation (ICH)** in their influential **Q2(R1) guideline on Validation of Analytical Procedures**. Before the ICH guidelines, different regulatory bodies had varying definitions and methods, leading to confusion and inconsistency. The ICH guidelines provided a scientifically sound and globally accepted framework for determining and validating these crucial performance characteristics.
            
            ICH Q2(R1) describes several methods, including visual evaluation and signal-to-noise ratio. However, the most common and statistically robust approach for quantitative assays, and the one visualized here, is **based on the standard deviation of the response and the slope of the calibration curve.**

            #### Mathematical Basis
            This method is based on the **standard deviation of the response ($\sigma$)** and the **slope of the calibration curve (S)**. The $\sigma$ can be determined from the standard deviation of blank measurements or, more robustly, from the standard deviation of the residuals from a low-level regression line.

            - **Limit of Detection (LOD):** The formula is derived to provide a high level of confidence (typically >95%) that a signal at this level is not just random noise. The factor 3.3 is a common approximation.
            """)
            st.latex(r"LOD = \frac{3.3 \times \sigma}{S}")
            st.markdown("""
            - **Limit of Quantitation (LOQ):** This requires a higher signal-to-noise ratio to ensure not just detection, but also acceptable precision and accuracy. The factor of 10 is the standard convention for this.
            """)
            st.latex(r"LOQ = \frac{10 \times \sigma}{S}")

def render_method_comparison():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally assess the agreement and bias between two different measurement methods (e.g., a new assay vs. a gold standard, or the R&D lab vs. the QC lab). This goes far beyond a simple correlation to determine if the methods can be used interchangeably.
    
    **Application:** This study is central to the “Crucible” phase of assay transfer. After developing a new assay, it is essential to demonstrate that it performs equivalently to the established method. This comparison provides definitive evidence to answer the critical question: “Do these two methods agree sufficiently well?” A successful agreement is a key milestone in ensuring a smooth and reliable assay transfer or method validation.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig, slope, intercept, bias, ua, la = plot_method_comparison()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Mean Bias (Bland-Altman)", value=f"{bias:.2f} units")
            st.metric(label="💡 Metric: Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0. Measures proportional bias.")
            st.metric(label="💡 Metric: Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0. Measures constant bias.")
            st.markdown("- **Deming Regression:** Checks for systematic constant (intercept) and proportional (slope) errors, correctly accounting for error in both methods.")
            st.markdown("- **Bland-Altman Plot:** Visualizes the random error and quantifies the expected range of disagreement via the Limits of Agreement (LoA).")
            st.markdown("- **% Bias Plot:** Directly assesses practical significance. Does the bias at any concentration exceed the pre-defined limits (e.g., ±15%)?")
            st.markdown("**The Core Insight:** Passing all three analyses proves that the receiving lab's method is statistically indistinguishable from the reference method, confirming a successful transfer.")
        with tabs[1]:
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope should contain 1.0**, and the 95% CI for the **intercept should contain 0**.")
            st.markdown(f"- **Bland-Altman:** At least 95% of the data points must fall within the Limits of Agreement (`{la:.2f}` to `{ua:.2f}`). The width of this interval must also be practically or clinically acceptable.")
            st.markdown("- **Percent Bias:** The bias at each concentration level should not exceed a pre-defined limit, often **±15%**. ")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **The Problem with Simple Regression:** For decades, scientists incorrectly used Ordinary Least Squares (OLS) regression and correlation (R²) to compare methods. This approach is fundamentally flawed because it assumes the reference method (x-axis) is measured without error, which is never true.
            
            - **Deming's Solution:** W. Edwards Deming popularized **Errors-in-Variables Regression** (Deming Regression), which acknowledges that *both* methods have inherent measurement error. It finds a line that minimizes errors in both the x and y directions, providing a much more accurate estimate of the true relationship.

            **The Problem with Correlation:** A high correlation (e.g., R² = 0.99) does not mean two methods agree. It only means they are proportional.
            - **The Bland-Altman Revolution:** In a landmark 1986 paper, **J. Martin Bland and Douglas G. Altman** addressed this widespread misuse. They proposed a simple, intuitive graphical method that directly assesses **agreement**. By plotting the *difference* between the two methods against their *average*, their plot makes it easy to visualize the mean bias and random error. It has since become the gold standard for method comparison studies.
            
            #### Mathematical Basis
            **Deming Regression:** Unlike OLS, Deming regression minimizes the sum of squared perpendicular distances from the data points to the regression line, weighted by the ratio of the error variances ($\lambda = \sigma^2_y / \sigma^2_x$).
            
            **Bland-Altman Plot:** The key metrics are the **mean bias** ($\bar{d}$) and the **Limits of Agreement (LoA)**, which define the range where 95% of future differences are expected to lie:
            """)
            st.latex(r"LoA = \bar{d} \pm 1.96 \cdot s_d")

def render_robustness_rsm():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To systematically explore how deliberate variations in assay parameters (e.g., temperature, pH, incubation time) affect the outcome, and to map the "safe operating space" for the method.
    
    **Application:** This is a proactive approach to managing variation. Rather than reacting to problems, this study identifies which parameters—the "vital few"—must be tightly controlled versus those that have minimal impact (the "trivial many"). **Design of Experiments (DOE)** is used for initial screening, while **Response Surface Methodology (RSM)** is used for optimization. This enables the design of a robust process that reliably withstands real-world variability.
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
            st.metric(label="📈 KPI: Most Significant Factor", value=f"{effects.index[0]}"); st.metric(label="💡 Effect Magnitude", value=f"{effects.values[0]:.2f}")
            st.markdown("- **Screening (Pareto):** Instantly reveals the 'vital few' parameters with significant effects (those crossing the red line). Here, `Temp` and the `Temp:pH` interaction are the most critical drivers.")
            st.markdown("- **Optimization (Contour/Surface):** These plots provide a map of the process, revealing the 'sweet spot'—the combination of settings that yields the optimal response (highest point on the surface).")
            st.markdown("**The Core Insight:** This study provides a map of your assay's operating space, allowing you to set control limits that guarantee robustness against real-world process noise.")
        with tabs[1]:
            st.markdown("- **Screening:** Any factor whose effect is statistically significant (typically p < 0.05) is considered a **critical parameter**. The SOP must include tighter controls for these parameters.")
            st.markdown("- **Optimization:** The goal is to define a **Design Space** or **Normal Operating Range (NOR)**—a region on the contour plot where the assay is proven to be robust. Final process parameters should be set well within this space.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **DOE:** The foundation of modern DOE was pioneered by **Sir Ronald A. Fisher** in the 1920s at the Rothamsted Agricultural Experimental Station. His revolutionary insight was to test multiple factors simultaneously in a structured **factorial design**, which was the only way to systematically study **interactions**.
            
            **RSM:** In the 1950s, **George E. P. Box and K. B. Wilson** built upon Fisher's work to not just identify important factors, but to find their *optimal settings*. They developed **Response Surface Methodology (RSM)**, which uses a more detailed design (like the Central Composite Design shown here) to model the curvature in the response and mathematically find the "peak of the mountain."
            
            #### Mathematical Basis
            **DOE (Screening):** A 2-level factorial design is used to fit a linear model to estimate main effects and interactions.
            """)
            st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{12} X_1 X_2 + \epsilon")
            st.markdown("""
            **RSM (Optimization):** RSM is used to model the curvature of the response surface by fitting a second-order polynomial model.
            """)
            st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{11} X_1^2 + \beta_{22} X_2^2 + \beta_{12} X_1 X_2 + \epsilon")

def render_shewhart():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To establish if a process is in a state of statistical control, meaning its variation is stable, consistent, and predictable over time. It distinguishes between inherent, random "common cause" variation and specific, assignable "special cause" variation.
    
    **Application:** This is the foundational step in process monitoring and a strict prerequisite for process capability analysis. Before assessing whether a process consistently meets specifications, it must first be shown to be stable. An out-of-control process is unpredictable, making any capability assessment invalid. Establishing control is the first critical step in demonstrating that variation is understood and managed.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_shewhart(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Process Stability", value="Signal Detected", delta="Action Required", delta_color="inverse")
            st.markdown("- **I-Chart (top):** Monitors the process center (accuracy). The single blue line shows the continuous process. Points marked with a red 'X' are out-of-control signals.")
            st.markdown("- **MR-Chart (bottom):** Monitors the short-term, run-to-run variability (precision). An out-of-control signal here would indicate the process has become inconsistent.")
            st.markdown("**The Core Insight:** These charts are the heartbeat of your process. This chart shows a stable process for the first 15 runs, after which a new reagent lot caused a special cause variation, driving the process out of control. This must be fixed before proceeding.")
        with tabs[1]:
            st.markdown("- A process is considered stable and ready for the next validation step only when **at least 20-25 consecutive points on both the I-chart and MR-chart show no out-of-control signals** according to the chosen rule set (e.g., Nelson, Westgard). Any signal requires a formal investigation and corrective action.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            Developed by the physicist **Walter A. Shewhart** at Bell Labs in the 1920s, these charts are the foundation of modern Statistical Process Control (SPC). Shewhart's breakthrough was recognizing that industrial processes contain two types of variation: **common cause** (the natural, inherent "noise" of a stable process) and **special cause** (unexpected, external events). The purpose of a Shewhart chart is not to eliminate all variation, but to provide a clear, statistical signal to distinguish between these two types, allowing engineers to fix real problems instead of chasing random noise.
            
            #### Mathematical Basis
            The key is estimating the process standard deviation ($\hat{\sigma}$) from the average moving range ($\overline{MR}$). For an I-MR chart, the formulas are:
            """)
            st.latex(r"\hat{\sigma} = \frac{\overline{MR}}{d_2} \quad \text{(where } d_2 \approx 1.128 \text{)}")
            st.markdown("**I-Chart Limits:**")
            st.latex(r"UCL/LCL = \bar{x} \pm 3\hat{\sigma}")
            st.markdown("**MR-Chart Limits:**")
            st.latex(r"UCL = D_4 \overline{MR} \quad \text{(where } D_4 \approx 3.267 \text{)}")

def render_ewma_cusum():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To implement sensitive charts that can detect small, sustained shifts in the process mean that a standard Shewhart chart, which lacks memory, would miss.
    
    **Application:** These are advanced early-warning systems for mature processes. Once major sources of variation are controlled, remaining issues are often small, gradual drifts (e.g., instrument degradation, reagent aging). **EWMA (Exponentially Weighted Moving Average)** is excellent for detecting gradual drifts, while **CUSUM (Cumulative Sum)** is the most statistically powerful tool for detecting small, abrupt, and sustained shifts of a specific magnitude.
    """)
    chart_type = st.sidebar.radio("Select Chart Type:", ('EWMA', 'CUSUM'))
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        if chart_type == 'EWMA': lmbda = st.sidebar.slider("EWMA Lambda (λ)", 0.05, 1.0, 0.2, 0.05); st.plotly_chart(plot_ewma_cusum(chart_type, lmbda, 0, 0), use_container_width=True)
        else: k_sigma, H_sigma = st.sidebar.slider("CUSUM Slack (k, in σ)", 0.25, 1.5, 0.5, 0.25), st.sidebar.slider("CUSUM Limit (H, in σ)", 2.0, 8.0, 5.0, 0.5); st.plotly_chart(plot_ewma_cusum(chart_type, 0, k_sigma, H_sigma), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Shift Detection", value="Signal Detected", delta="Action Required", delta_color="inverse")
            st.markdown("- **Top Plot (Raw Data):** The small 1.25σ shift after run 25 is almost impossible to see by eye.\n- **Bottom Plot (EWMA/CUSUM):** This chart makes the invisible visible by accumulating small deviations until they cross the red control limit, providing a clear statistical signal.")
            st.markdown("**The Core Insight:** These charts act as a magnifying glass for the process mean, allowing you to catch subtle problems early and maintain a high level of quality.")
        with tabs[1]:
            st.markdown("- **EWMA Rule:** For long-term monitoring, a `λ` between **0.1 to 0.3** is a common choice. A signal occurs when the EWMA line crosses the control limits.\n- **CUSUM Rule:** To detect a shift of size $\delta$, set the slack parameter `k` to approximately **$\delta / 2$**. A signal occurs when the CUSUM statistic crosses the decision interval `H`.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            Both EWMA (Roberts, 1959) and CUSUM (Page, 1954) were developed in the 1950s to address the Shewhart chart's insensitivity to small shifts by incorporating "memory". **EWMA** uses an exponentially weighted average to smooth data and reveal underlying trends. **CUSUM** directly accumulates deviations from a target; if the process is on target, the sum hovers around zero, but a small shift causes the sum to trend steadily until it crosses a decision threshold. CUSUM is considered the most statistically powerful method for detecting small, sustained shifts.
            
            #### Mathematical Basis
            **EWMA:** The statistic $z_i$ is a weighted average of the current observation $x_i$ and the previous EWMA value $z_{i-1}$:
            """)
            st.latex(r"z_i = \lambda x_i + (1-\lambda)z_{i-1}")
            st.markdown("""
            **CUSUM:** Uses two one-sided statistics to detect upward ($SH$) and downward ($SL$) shifts, using a slack value (k) and decision interval (H).
            """)
            st.latex(r"SH_i = \max(0, SH_{i-1} + (x_i - \mu_0) - k)")
            
def render_multi_rule():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To create an objective, statistically-driven system for accepting or rejecting each individual analytical run based on the performance of Quality Control (QC) samples.
    
    **Application:** This is the daily responsibility of the process steward. This multi-rule system serves as a vigilant gatekeeper, ensuring that only valid, trustworthy results are released. It forms the backbone of routine quality control, safeguarding the reliability and compliance of analytical outputs by detecting both large random errors and smaller systematic trends.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_multi_rule(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Run Status", value="Violations Detected", delta="Action Required", delta_color="inverse")
            st.markdown("- **Levey-Jennings Chart:** Visualizes QC data over time with 1, 2, and 3-sigma zones. Specific rule violations are automatically flagged and annotated.\n- **Distribution Plot:** Shows the overall histogram of the QC data; it should approximate a bell curve.")
            st.markdown("**The Core Insight:** The annotations on the Levey-Jennings chart provide immediate, actionable feedback, distinguishing between random errors (like the `R_4s` rule) and systematic errors (like the `2_2s` or `4_1s` rules), guiding the troubleshooting process.")
        with tabs[1]:
            st.markdown("""
            #### Historical Context & Origin
            The **Levey-Jennings chart**, an adaptation of industrial control charts for the clinical lab, was developed in the 1950s. However, using simple ±2σ or ±3σ limits was a blunt instrument. In a landmark 1981 paper, **Dr. James Westgard** proposed a "multi-rule" system that combines several different rules to create a highly sensitive yet specific quality control procedure. The Westgard Rules are now the global standard for clinical laboratory QC and are essential for meeting regulatory requirements from bodies like CLIA, CAP, and ISO 15189.
            """)
    st.subheader("Standard Industry Rule Sets")
    tab_w, tab_n, tab_we = st.tabs(["✅ Westgard Rules", "✅ Nelson Rules", "✅ Western Electric Rules"])

def render_capability():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To determine if a stable process is capable of consistently producing results that meet the required specifications.
    
    **Application:** This is often the final and most critical gate of a successful assay transfer. After demonstrating that the assay is reliable (Act I) and stable in the receiving lab environment (Act II), the final step is to provide statistical evidence that the process can consistently meet stringent quality targets. A high Cpk (process capability index) serves as the quantitative proof that the method performs within specification—delivering reproducible, trustworthy results under routine conditions. In many ways, it is the statistical equivalent of "mission accomplished."
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
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scn != 'Out of Control' else "INVALID")
            st.markdown("- **The Mantra:** Control before Capability. Cpk is only meaningful for a stable, in-control process (see I-Chart in the plot). The 'Out of Control' scenario yields an invalid Cpk because the process is unpredictable.")
            st.markdown("- **The Key Insight:** A process can be perfectly **in control but not capable** (the 'Shifted' and 'Variable' scenarios). The control chart looks fine, but the process is producing out-of-spec results. This is why you need both tools.")
        with tabs[1]:
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable** (a common minimum target, corresponding to a 4-sigma process).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** (a common Six Sigma target).")
            st.markdown("- `Cpk < 1.0`: Process is **not capable** of meeting specifications and requires improvement (either recentering or variance reduction).")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of process capability indices originated in the manufacturing industry, particularly in Japan in the 1970s, as part of the Total Quality Management (TQM) revolution. However, it was the rise of **Six Sigma** at Motorola in the 1980s that truly popularized Cpk and made it a global standard for quality. The core idea of Six Sigma is to reduce process variation so that the nearest specification limit is at least six standard deviations away from the process mean. A Cpk of 2.0 corresponds to a true Six Sigma process.
            
            #### Mathematical Basis
            Capability analysis compares the **"Voice of the Process"** (the actual spread of the data, typically a 6σ spread) to the **"Voice of the Customer"** (the allowable spread defined by the specification limits).

            - **Cpk (Actual Capability):** This metric measures if the process is narrow enough *and* well-centered. It is the lesser of the upper and lower capability indices, effectively measuring the distance from the process mean to the *nearest* specification limit.
            """)
            st.latex(r"C_{pk} = \min(C_{pu}, C_{pl}) = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right)")
            st.markdown("A Cpk of 1.33 means there is a 'buffer' equivalent to one standard deviation between the process edge and the nearest specification limit.")

def render_anomaly_detection():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To leverage machine learning to detect complex, multivariate anomalies that traditional univariate control charts would miss.
    
    **Application:** This is a critical tool for uncovering the “ghost in the machine.” An operator might confirm that every individual parameter is within specification, yet an ML model can flag a run as anomalous due to an unusual combination of otherwise acceptable inputs. This capability is essential for detecting subtle, emerging failure modes that traditional rule-based systems might overlook—enhancing process awareness and proactive quality control.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_anomaly_detection(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Rules", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Anomalies Detected", value="3", help="Number of points flagged by the model.")
            st.markdown("- **The Plot:** The blue shaded area represents the model's learned 'normal' operating space. Points outside this area are flagged as anomalies (red).")
            st.markdown("- **The Key Insight:** The anomalous points are not necessarily extreme on any single axis. Their *combination* is what makes them unusual, a pattern that is nearly impossible for a human or a simple control chart to detect.")
            st.markdown("**The Core Insight:** This is a proactive monitoring tool that moves beyond simple rule-based alarms to a holistic assessment of process health, enabling the detection of previously unknown problems.")
        with tabs[1]:
            st.markdown("- This is an exploratory and monitoring tool. There is no hard 'pass/fail' rule during validation.")
            st.markdown("- The primary rule is that any point flagged as an **anomaly must be investigated** by Subject Matter Experts (SMEs) to determine the root cause and assess its impact on product quality. It serves as an input to a deviation or non-conformance investigation.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The **Isolation Forest** algorithm was proposed by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou in a groundbreaking 2008 paper. It represented a fundamental shift in how to think about anomaly detection. Previous methods were often "density-based," trying to define what a "normal" region looks like. The authors of Isolation Forest flipped the problem on its head with a simple but powerful observation: **anomalies are "few and different."** Because they are different, they should be easier to *isolate*. This counter-intuitive approach proved to be both highly effective and computationally efficient.
            
            #### Mathematical Basis
            The core idea is that if you randomly partition a dataset, anomalies will be isolated in fewer steps than normal points. The algorithm builds an ensemble of "Isolation Trees" (iTrees) by recursively partitioning the data with random splits. Anomalous points, being different, will require fewer partitions and will therefore have a much shorter average path length from the root to the leaf. The anomaly score is derived from this average path length.
            """)
            st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")

def render_predictive_qc():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To move from *reactive* quality control (detecting a failure after it happens) to *proactive* failure prevention using supervised machine learning.
    
    **Application:** This is a predictive decision-support tool designed to reduce waste and improve right-first-time rates. Before committing costly reagents and valuable instrument time, the model evaluates key starting conditions—such as reagent age, instrument warm-up time, and environmental factors—to estimate the likelihood of a successful run. If the model detects a high probability of failure, it can trigger a preemptive alert, enabling the operator to take corrective action before the run proceeds.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_predictive_qc(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Rules", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Predictive Risk Profiling", value="Enabled")
            st.markdown("- **Decision Boundary Plot (left):** This is the model's 'risk map.' The color gradient shows the predicted probability of failure, from low (green) to high (red).")
            st.markdown("- **Probability Distribution Plot (right):** This is the model's report card. It shows the predicted failure probabilities for runs that actually passed (green distribution) versus runs that actually failed (red distribution).")
            st.markdown("**The Core Insight:** A clear separation between the green and red distributions proves that the model has learned the hidden patterns that lead to failure and can reliably distinguish a good run from a bad one before it's too late.")
        with tabs[1]:
            st.markdown("- A risk threshold is established based on the model and business needs (e.g., 'If P(Fail) > 20%, flag run for review').")
            st.markdown("- The model's performance (e.g., accuracy, sensitivity, specificity) must be formally validated and documented via a confusion matrix and ROC analysis before use in a regulated environment.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **Logistic regression** is a statistical model developed by the brilliant British statistician **Sir David Cox in 1958**. Its origins lie in the need to model the probability of a binary event. While linear regression predicts a continuous value, this doesn't make sense for probabilities, which must be constrained between 0 and 1. Cox's breakthrough was to use the **logistic function (or sigmoid function)** to "squash" the output of a linear equation into this [0, 1] range. Its power lies in its **interpretability**, making it a foundational and still widely used algorithm.
            
            #### Mathematical Basis
            The model creates a linear combination of the input features ($x$), called the **log-odds** or **logit** ($z$), and passes it through the sigmoid function, $\sigma(z)$, to transform it into a probability.
            """)
            st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 \quad , \quad P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}")
            st.markdown("The decision boundary is the line where the predicted probability is exactly 0.5 (i.e., where $z=0$).")

def render_forecasting():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To use a time series model to forecast the future performance of assay controls, enabling proactive management instead of reactive problem-solving.
    
    **Application:** This is a powerful forecasting tool that enables proactive quality and operations management. By predicting the future trajectory of key controls, the system can identify issues before they occur—such as signaling that a reagent lot may begin to fail in three weeks, or that an instrument is likely to require recalibration next month. This transforms maintenance, inventory planning, and quality oversight from reactive or scheduled tasks into intelligent, data-driven strategies.
    """)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_fc, fig2_fc, fig3_fc = plot_forecasting()
        st.plotly_chart(fig1_fc, use_container_width=True); st.plotly_chart(fig2_fc, use_container_width=True); st.plotly_chart(fig3_fc, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Rules", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Forecast Status", value="Future Breach Predicted", help="The model predicts the spec limit will be crossed.")
            st.markdown("- **Forecast Plot (Top):** Shows the historical data (dots), the model's prediction (blue line), and the confidence interval (blue band).\n- **Trend & Changepoints (Middle):** This is the most powerful diagnostic plot. It shows the underlying long-term trend and red dashed lines where the model detected a significant shift.\n- **Seasonality (Bottom):** Shows predictable yearly patterns.")
            st.markdown("**The Core Insight:** This analysis provides a roadmap for the future, telling you *when* a problem is likely to occur and *why* (is it a long-term trend or a seasonal effect?).")
        with tabs[1]:
            st.markdown("- A **'Proactive Alert'** should be triggered if the 80% forecast interval (`yhat_upper`) is predicted to cross a specification limit within the defined forecast horizon (e.g., the next 4-6 weeks).\n- Any automatically detected **changepoint** should be investigated and correlated with historical batch records or lab events to understand its root cause.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            **Prophet**, developed by **Facebook's Core Data Science team in 2017**, was created to produce high-quality forecasts at scale with minimal manual effort. Traditional methods like ARIMA often require deep statistical knowledge and struggle with common business data features like multiple seasonalities and shifting trends. Prophet was designed from the ground up to handle these features automatically by framing forecasting as a curve-fitting exercise.
            
            #### Mathematical Basis
            Prophet is a **decomposable time series model** which models the time series as a combination of distinct components:
            """)
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("- **$g(t)$ is the trend function** (piecewise linear).\n- **$s(t)$ is the seasonality function** (modeled with a Fourier series).\n- **$h(t)$ is the holidays/events function.**\n The model is fit within a Bayesian framework to produce uncertainty intervals.")

def render_pass_fail():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To accurately calculate and compare confidence intervals for a binomial proportion (e.g., pass/fail, concordant/discordant).
    
    **Application:** Essential for validating qualitative assays. This proves, with a high degree of confidence, that the assay's success rate (e.g., concordance with a reference method) is above a certain threshold. Choosing the wrong statistical method here can lead to dangerously misleading conclusions, especially with the small sample sizes common in validation studies.
    """)
    n_samples_wilson = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30, key='wilson_n')
    successes_wilson = st.sidebar.slider("Concordant Results", 0, n_samples_wilson, int(n_samples_wilson * 0.95), key='wilson_s')
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_wilson, fig2_wilson = plot_wilson(successes_wilson, n_samples_wilson)
        st.plotly_chart(fig1_wilson, use_container_width=True); st.plotly_chart(fig2_wilson, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ Acceptance Criteria", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}")
            st.markdown("- **CI Comparison (Top):** Shows that the unreliable 'Wald' interval can be dangerously narrow.\n- **Coverage Probability (Bottom):** The Wald interval's coverage (red) is terrible, giving a false sense of precision. The Wilson and Clopper-Pearson intervals are much more reliable.")
            st.markdown("**The Core Insight:** Never use the standard Wald interval for important decisions. The Wilson Score interval provides the best balance of accuracy and width.")
        with tabs[1]:
            st.markdown("- **Acceptance Criterion:** 'The lower bound of the 95% **Wilson Score** (or Clopper-Pearson) confidence interval must be greater than or equal to the target concordance rate' (e.g., 90%).")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The **Wilson Score Interval (1927)** and **Clopper-Pearson Interval (1934)** were developed to fix the known poor performance of the simpler Wald interval, especially for small sample sizes or proportions near 0 or 1. The Wilson interval inverts the score test, while the Clopper-Pearson is an "exact" method based on the binomial distribution, making it more conservative.
            
            #### Mathematical Basis
            The Wilson Score interval is the solution to a quadratic equation that results from inverting the score test, which is why it is more complex but far superior to the simple Wald formula.
            """)
            st.latex(r"\frac{1}{1 + z^2/n} \left( \hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \right)")

def render_bayesian():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To formally combine existing knowledge (the 'Prior') with new experimental data (the 'Likelihood') to arrive at an updated, more robust conclusion (the 'Posterior').
    
    **Application:** This is a key tool for driving efficient, knowledge-informed validation. It allows teams to leverage prior data from R&D to design smaller, more targeted validation studies at the QC site, reducing redundancy and accelerating tech transfer. It answers: "Given what we already knew, what does this new data tell us now?"
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
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}", help="The final, data-informed belief.")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}", help="The initial belief before seeing the new data.")
            st.metric(label="💡 Data-only Estimate (MLE)", value=f"{mle:.3f}", help="The evidence from the new data alone.")
            st.markdown("- **Prior (Green):** Our initial belief.\n- **Likelihood (Red):** The 'voice of the data.'\n- **Posterior (Blue):** The final, updated belief—a weighted compromise between the Prior and the Likelihood.")
            st.markdown("**The Key Insight:** With a strong prior, our belief barely moves. With a skeptical prior, the new data almost completely dictates our final belief. This is Bayesian updating in action.")
        with tabs[1]:
            st.markdown("- The **95% credible interval** (the central 95% of the blue posterior distribution) must be entirely above the target (e.g., 90%).\n- This approach allows for demonstrating success with smaller sample sizes if a strong, justifiable prior is used.")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            Based on **Bayes' Theorem** (conceived by Rev. Thomas Bayes in the 1740s), this framework remained a theoretical curiosity for centuries due to computational complexity. The "Bayesian revolution" began in the late 20th century with the development of computational techniques like **Markov Chain Monte Carlo (MCMC)**, which allow computers to approximate the posterior distribution for complex models.
            
            #### Mathematical Basis
            For binomial data, the **Beta distribution** is a **conjugate prior** for the binomial likelihood. This means if you start with a Beta prior and get binomial data, your posterior is also a Beta distribution, making the math simple.
            - If Prior is Beta($\\alpha, \\beta$) and Data is $k$ successes in $n$ trials:
            - The Posterior is Beta($\\alpha + k, \\beta + n - k$).
            """)

def render_ci_concept():
    st.markdown("""
    #### Purpose & Application
    **Purpose:** To understand the fundamental concept and correct interpretation of frequentist confidence intervals.
    
    **Application:** This is a foundational concept that directly impacts resource planning. This interactive simulation allows users to explore the trade-offs between sample size, cost, and statistical precision, enabling data-driven decisions on how many samples are needed to achieve a reliable result.
    """)
    n_slider = st.sidebar.slider("Select Sample Size (n) for Simulation:", 5, 100, 30, 5)
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        fig1_ci, fig2_ci, capture_count, n_sims, avg_width = plot_ci_concept(n=n_slider)
        st.plotly_chart(fig1_ci, use_container_width=True); st.plotly_chart(fig2_ci, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tabs = st.tabs(["💡 Key Insights & Interpretation", "✅ The Golden Rule", "📖 Method Theory & History"])
        with tabs[0]:
            st.metric(label=f"📈 KPI: Average CI Width (n={n_slider})", value=f"{avg_width:.2f} units")
            st.metric(label="💡 Empirical Coverage", value=f"{(capture_count/n_sims):.0%}", help="The % of simulated CIs that captured the true mean.")
            st.markdown("- **Theoretical Universe (Top):** The wide blue curve is the population. The narrow orange curve is the distribution of *all possible sample means*. Because it's so narrow, any single sample is likely to be close to the true mean.\n- **CI Simulation (Bottom):** As you increase `n`, the CIs become dramatically shorter.\n- **Diminishing Returns:** The gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller.")
        with tabs[1]:
            st.error("🔴 **Incorrect:** 'There is a 95% probability that the true mean is in this interval.'")
            st.success("🟢 **Correct:** 'We are 95% confident that this interval contains the true mean.' (This interval was constructed using a procedure that, in the long run, captures the true mean 95% of the time.)")
        with tabs[2]:
            st.markdown("""
            #### Historical Context & Origin
            The concept of **confidence intervals** was introduced by **Jerzy Neyman** in a 1937 paper. He sought a rigorously frequentist method that provided a practical range of plausible values. His revolutionary idea was to make a probabilistic statement about the *procedure* used to create the interval, not about the fixed, unknown parameter itself. This elegant and practical solution quickly became the dominant paradigm in applied statistics.
            
            #### Mathematical Basis
            The general form is: `Point Estimate ± (Critical Value × Standard Error)`. For the mean, this becomes:
            """)
            st.latex(r"\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}")

# ==============================================================================
# MAIN APP LAYOUT & LOGIC
# ==============================================================================
st.title("🛠️ Biotech V&V Analytics Toolkit")
st.markdown("### An Interactive Guide to Assay Validation, Tech Transfer, and Lifecycle Management")
st.markdown("Welcome! This toolkit is a collection of interactive modules designed to explore the statistical and machine learning methods that form the backbone of a robust V&V, technology transfer, and process monitoring plan.")

tab_intro, tab_map, tab_journey = st.tabs(["🚀 The V&V Framework", "🗺️ Concept Map", "📖 The Scientist's Journey"])
with tab_intro:
    st.markdown('<h4 class="section-header">The V&V Model: A Strategic Framework</h4>', unsafe_allow_html=True)
    st.markdown("The **Verification & Validation (V&V) Model**, shown below, provides a structured, widely accepted framework for technology transfer...")
    st.plotly_chart(plot_v_model(), use_container_width=True)

with tab_map:
    st.markdown('<h4 class="section-header">Conceptual Map of V&V Tools</h4>', unsafe_allow_html=True)
    st.plotly_chart(create_conceptual_map_plotly(), use_container_width=True)
    st.markdown("This map illustrates how foundational **Academic Disciplines** give rise to **Core Domains** such as Statistical Process Control (SPC)...")

with tab_journey:
    st.header("The Scientist's/Engineer's Journey: A Three-Act Story")
    st.markdown("""In the world of quality and development, the challenges are often complex and hidden in the details. This toolkit is structured as a three-act journey to navigate these phases with clarity and confidence.""")
    act1, act2, act3 = st.columns(3)
    with act1: st.subheader("Act I: Know Thyself (The Foundation)"); st.markdown("Before any transfer or scale-up, you must understand the capability and limits of your current measurement systems. This phase focuses on characterization and validation—building a strong foundation of data integrity. **(Tools 1-5)**")
    with act2: st.subheader("Act II: The Transfer (The Crucible)"); st.markdown("A validated method must prove its robustness in a new environment. This is the moment to assess transferability, stability, and comparability under operational conditions. **(Tools 6-9)**")
    with act3: st.subheader("Act III: The Guardian (Lifecycle Management)"); st.markdown("Once the method is live, continuous monitoring is essential. This act focuses on using advanced tools to detect signals, predict deviations, and maintain process control over time. **(Tools 10-15)**")
st.divider()

st.sidebar.title("🧰 Toolkit Navigation")
st.sidebar.markdown("Select a statistical method to analyze and visualize.")

act1_options = ["Gage R&R", "Linearity and Range", "LOD & LOQ", "Method Comparison", "Assay Robustness (DOE/RSM)"]
act2_options = ["Process Stability (Shewhart)", "Small Shift Detection", "Run Validation", "Process Capability (Cpk)"]
act3_options = ["Anomaly Detection (ML)", "Predictive QC (ML)", "Control Forecasting (AI)", "Pass/Fail Analysis", "Bayesian Inference", "Confidence Interval Concept"]

act1_icons = [ICONS.get(opt.split(". ")[-1], "question-circle") for opt in act1_options]
act2_icons = [ICONS.get(opt.split(". ")[-1], "question-circle") for opt in act2_options]
act3_icons = [ICONS.get(opt.split(". ")[-1], "question-circle") for opt in act3_options]

# Determine which option is currently selected to set the correct default index
if 'method_key' not in st.session_state:
    st.session_state.method_key = act1_options[0]

with st.sidebar.expander("ACT I: FOUNDATION & CHARACTERIZATION", expanded=True):
    selected_act1 = option_menu(None, act1_options, icons=act1_icons, menu_icon="cast", 
                                key='act1_menu', 
                                default_index=act1_options.index(st.session_state.method_key) if st.session_state.method_key in act1_options else 0)

with st.sidebar.expander("ACT II: TRANSFER & STABILITY", expanded=True):
    selected_act2 = option_menu(None, act2_options, icons=act2_icons, menu_icon="cast",
                                key='act2_menu',
                                default_index=act2_options.index(st.session_state.method_key) if st.session_state.method_key in act2_options else 0)

with st.sidebar.expander("ACT III: LIFECYCLE & PREDICTIVE MGMT", expanded=True):
    selected_act3 = option_menu(None, act3_options, icons=act3_icons, menu_icon="cast",
                                key='act3_menu',
                                default_index=act3_options.index(st.session_state.method_key) if st.session_state.method_key in act3_options else 0)

if selected_act1 != st.session_state.method_key and selected_act1 in act1_options:
    st.session_state.method_key = selected_act1
    st.experimental_rerun()
if selected_act2 != st.session_state.method_key and selected_act2 in act2_options:
    st.session_state.method_key = selected_act2
    st.experimental_rerun()
if selected_act3 != st.session_state.method_key and selected_act3 in act3_options:
    st.session_state.method_key = selected_act3
    st.experimental_rerun()
# --- PowerPoint Download Section in Sidebar ---
st.sidebar.divider()
st.sidebar.header("📊 Generate Report")
if st.sidebar.button("Generate Executive PowerPoint Summary", use_container_width=True):
    # ... (PPTX generation logic) ...
    st.sidebar.download_button(...)

st.divider()

# Add the number back to the header for clarity
method_key_with_num = f"{list(ICONS.keys()).index(st.session_state.method_key) + 1}. {st.session_state.method_key}"
st.header(method_key_with_num)

PAGE_DISPATCHER = {
    "1. Gage R&R": render_gage_rr, "2. Linearity and Range": render_linearity, "3. LOD & LOQ": render_lod_loq,
    "4. Method Comparison": render_method_comparison, "5. Assay Robustness (DOE/RSM)": render_robustness_rsm,
    "6. Process Stability (Shewhart)": render_shewhart, "7. Small Shift Detection": render_ewma_cusum,
    "8. Run Validation": render_multi_rule, "9. Process Capability (Cpk)": render_capability,
    "10. Anomaly Detection (ML)": render_anomaly_detection, "11. Predictive QC (ML)": render_predictive_qc,
    "12. Control Forecasting (AI)": render_forecasting, "13. Pass/Fail Analysis": render_pass_fail,
    "14. Bayesian Inference": render_bayesian, "15. Confidence Interval Concept": render_ci_concept
}

PAGE_DISPATCHER[method_key]()
