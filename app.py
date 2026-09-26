# app2_updated.py
# Kaggle 2018 DS Survey EDA Dashboard (Streamlit)

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time
from ml_predictor_tab import load_ml_model, render_ml_predictor_tab

# -------------------------------------------------------------------
# BASIC ENVIRONMENT CHECKS (similar idea to your old app)
# -------------------------------------------------------------------
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9+ is required to run this app.")

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# SIMPLE LOGIN SYSTEM (adapted from your previous app)
# -------------------------------------------------------------------
# Login credentials live in encrypted Streamlit app secrets.
try:
    auth_secrets = st.secrets["auth"]
    USERS = {auth_secrets["username"]: auth_secrets["password"]}
except Exception:
    st.error("Login is not configured. Set [auth] username and password in Streamlit app secrets.")
    st.stop()

# Toggle authentication during development. Set to True to re-enable login.
AUTH_ENABLED = True

if "authenticated" not in st.session_state:
    # If auth is disabled, mark the session authenticated by default.
    st.session_state["authenticated"] = False if AUTH_ENABLED else True


def login():
    """Professional dark-themed login page with background image"""
    st.set_page_config(
        page_title="Kaggle 2018 DS Survey – Login",
        page_icon="📊",
        layout="centered",
    )
    
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&h=900&fit=crop');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            position: relative;
        }
        
        [data-testid="stAppViewContainer"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(15, 20, 25, 0.85);
            z-index: 1;
        }
        
        [data-testid="stMainBlockContainer"] {
            position: relative;
            z-index: 2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%); 
                        border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.5); border: 2px solid #e94560;">
                <h1 style="color: #e94560; font-size: 2.5rem; margin-bottom: 0.5rem;">📊</h1>
                <h2 style="color: white; margin-bottom: 1.5rem;">Kaggle 2018 Survey</h2>
                <p style="color: #a0aec0; margin-bottom: 1rem;">Data Science EDA Dashboard</p>
                <p style="color: #718096; font-size: 0.9rem; margin: 0;">Interactive Analytics Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        username = st.text_input(
            "Username",
            placeholder="Enter username",
            key="login_username",
            help="Enter the credentials configured for this dashboard."
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            key="login_password"
        )
        
        if st.button("🔓 Login", use_container_width=True):
            if USERS.get(username) == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("✅ Login successful!")
                safe_rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")
        
        st.markdown(
            """
            <hr style="border: 1px solid #2d3748; margin: 2rem 0;">
            <p style="text-align: center; color: #718096; font-size: 0.85rem;">
                🔒 Secure Dashboard • Kaggle 2018 Data Science Survey
            </p>
            """,
            unsafe_allow_html=True,
        )


def logout():
    """Display logout button in sidebar with proper styling."""
    with st.sidebar:
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state["authenticated"] = False
                safe_rerun()


# Helper: safe rerun that works across Streamlit versions
def safe_rerun():
    """Attempt to rerun the Streamlit script.

    Some Streamlit installs may not expose `st.experimental_rerun`.
    In that case we mutate the query params (a supported way to trigger
    a rerun) as a fallback.
    """
    try:
        # Preferred if available
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass

    try:
        # Fallback: change query params to force a rerun. Use the
        # non-experimental `st.query_params` property where available.
        if hasattr(st, 'query_params'):
            params = dict(st.query_params)
            params['_rerun'] = int(time.time())
            st.query_params = params
        else:
            st.experimental_set_query_params(_rerun=int(time.time()))
    except Exception:
        # Last resort: no-op (avoid crashing the app)
        return


if AUTH_ENABLED:
    if not st.session_state.get("authenticated", False):
        # Show login page only if not authenticated
        login()
        st.stop()
    # If we reach here, user is authenticated - proceed to dashboard

# -------------------------------------------------------------------
# PAGE CONFIG + SIMPLE STYLING (inspired by your website dashboard)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Kaggle 2018 DS Survey – EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ============================================================
                      DARK PROFESSIONAL THEME
                  Modern SaaS Dashboard Styling
    ============================================================ */
    
    /* GLOBAL STYLES */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        color: #e0e6ed;
    }
    
    /* MAIN CONTAINER */
    .main {
        padding: 0 !important;
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 2rem 3rem !important;
        background: transparent;
        max-width: 100%;
    }
    
    /* HERO SECTION */
    .hero {
        background: linear-gradient(135deg, #e94560 0%, #c9184a 100%);
        color: white;
        padding: 4rem 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin: 0;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        color: white !important;
        margin-bottom: 1rem !important;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .hero p {
        font-size: 1.3rem !important;
        color: rgba(255,255,255,0.95) !important;
        margin: 0 !important;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* TABS */
    [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-bottom: 2px solid #e94560;
        padding: 0 3rem;
        margin: 0 -3rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        display: flex;
        gap: 2rem;
    }
    
    [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #a0aec0 !important;
        padding: 1.5rem 0 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        margin: 0 !important;
        transition: all 0.3s ease !important;
        text-transform: capitalize;
    }
    
    [data-baseweb="tab"]:hover {
        color: #e94560 !important;
    }
    
    [aria-selected="true"] [data-baseweb="tab"] {
        background: transparent !important;
        color: #e94560 !important;
        border-bottom-color: #e94560 !important;
    }
    
    /* TYPOGRAPHY */
    h1 {
        color: #ffffff !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin: 2rem 0 1rem !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #e0e6ed !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 2rem 0 1.2rem !important;
        padding-bottom: 1rem !important;
        border-bottom: 3px solid #e94560 !important;
    }
    
    h3 {
        color: #cbd5e0 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 1rem !important;
    }
    
    h4, h5, h6 {
        color: #a0aec0 !important;
        font-weight: 700 !important;
    }
    
    p {
        line-height: 1.6;
        color: #a0aec0;
    }
    
    /* BUTTONS */
    .stButton > button {
        padding: 1rem 2.5rem !important;
        border-radius: 8px !important;
        height: auto !important;
        background: linear-gradient(135deg, #e94560 0%, #c9184a 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.5) !important;
    }
    
    /* METRIC CARDS */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        padding: 2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
        border: 1px solid #2d3748 !important;
        border-top: 4px solid #e94560 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 30px rgba(233, 69, 96, 0.2) !important;
    }
    
    [data-testid="metric-container"]:nth-child(2n) {
        border-top-color: #f39c12 !important;
    }
    
    [data-testid="metric-container"]:nth-child(3n) {
        border-top-color: #1abc9c !important;
    }
    
    [data-testid="metric-container"]:nth-child(4n) {
        border-top-color: #3498db !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #718096 !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #e0e6ed !important;
        margin-top: 0.5rem !important;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #1a1f2e !important;
        border-right: 2px solid #2d3748 !important;
    }
    
    [data-testid="stSidebar"] {
        background: #1a1f2e !important;
        padding-top: 2rem !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #cbd5e0 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e6ed !important;
    }
    
    /* INPUT WIDGETS */
    [data-baseweb="select"], [data-baseweb="base-input"], [data-baseweb="input"] {
        border-radius: 8px !important;
        border: 2px solid #2d3748 !important;
        background: #111827 !important;
        color: #e0e6ed !important;
    }
    
    [data-baseweb="select"]:hover, [data-baseweb="base-input"]:hover, [data-baseweb="input"]:hover {
        border-color: #e94560 !important;
        box-shadow: 0 0 0 3px rgba(233, 69, 96, 0.2) !important;
    }
    
    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
        overflow: hidden !important;
        border: 1px solid #2d3748 !important;
    }
    
    [data-testid="stDataFrame"] thead {
        background: linear-gradient(135deg, #e94560 0%, #c9184a 100%);
        color: white;
    }
    
    [data-testid="stDataFrame"] thead th {
        padding: 1.2rem !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stDataFrame"] tbody {
        background: #1a1f2e;
    }
    
    [data-testid="stDataFrame"] tbody tr {
        border-bottom: 1px solid #2d3748;
    }
    
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #252d3d !important;
    }
    
    [data-testid="stDataFrame"] tbody td {
        padding: 1rem 1.2rem !important;
        color: #cbd5e0;
    }
    
    /* PLOTLY CHARTS */
    [data-testid="plotly.src.components.plotly.Plotly"] {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
        overflow: hidden !important;
        border: 1px solid #2d3748 !important;
    }
    
    /* ALERTS */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border-left: 5px solid #e94560 !important;
        background: rgba(233, 69, 96, 0.1) !important;
        border: 1px solid rgba(233, 69, 96, 0.3) !important;
    }
    
    [data-testid="stAlert"] p {
        color: #cbd5e0 !important;
    }
    
    /* EXPANDERS */
    [data-testid="stExpander"] {
        background: #1a1f2e !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: #cbd5e0 !important;
        padding: 1rem !important;
    }
    
    /* HR */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 1px;
        background: #2d3748 !important;
    }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        h1 { font-size: 2.2rem !important; }
        h2 { font-size: 1.5rem !important; }
        .hero { padding: 2.5rem 1.5rem; }
        .hero h1 { font-size: 2.2rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# DATA LOADING (from your prepared Kaggle 2018 tables)
# -------------------------------------------------------------------
@st.cache_data
def load_kaggle_data():
    """
    Load the prepared dimension tables you used in the notebook:

    - dim_demographicsprepared.csv
    - dim_toolsprepared.csv
    - dim_mlframeworksprepared.csv
    - dim_skills_experienceprepare.csv
    - fact_respondent_clean.csv  (optional, if you want it later)
    """
    base = Path(".")

    # Helper: try glob patterns to be tolerant of small filename differences
    def _read_first_match(pattern):
        matches = list(base.glob(pattern))
        if matches:
            try:
                return pd.read_csv(matches[0])
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    demo = _read_first_match("dim*demograph*.csv")
    tools = _read_first_match("dim*tool*.csv")
    mlframeworks = _read_first_match("dim*ml*framework*.csv")
    if mlframeworks.empty:
        mlframeworks = _read_first_match("dim*ml*framwork*.csv")
    skills = _read_first_match("dim*skill*.csv")

    # Normalize `demo` column names so the rest of the app (which expects
    # names like `currentrole`, `ageyears`, etc.) can use them.
    if not demo.empty:
        # map common snake_case names from your CSV to the app's expected names
        col_map = {
            "age_years": "ageyears",
            "years_experience_current_role": "yearsexperiencecurrentrole",
            "current_role": "currentrole",
            "education_level": "educationlevel",
            "yearly_compensation_usd": "yearlycompensationusd",
            "undergraduate_major": "undergraduatemajor",
            "current_industry": "currentindustry",
            "respondent_id": "respondentid",
        }
        for src, dst in col_map.items():
            if src in demo.columns and dst not in demo.columns:
                demo[dst] = demo[src]

    # Normalize `tools` columns to what the app expects (e.g. primaryanalysistool)
    if not tools.empty:
        tools_col_map = {
            "primary_analysis_tool": "primaryanalysistool",
            "models_as_blackbox": "modelsasblackbox",
            "respondent_id": "respondentid",
        }
        for src, dst in tools_col_map.items():
            if src in tools.columns and dst not in tools.columns:
                tools[dst] = tools[src]

    # Normalize `mlframeworks` columns (best-effort)
    if not mlframeworks.empty:
        ml_col_map = {
            "most_used_ml_framework": "mostusedmlframework",
            "ml_frameworks_used": "mlframeworksused",
            "respondent_id": "respondentid",
        }
        for src, dst in ml_col_map.items():
            if src in mlframeworks.columns and dst not in mlframeworks.columns:
                mlframeworks[dst] = mlframeworks[src]

    # Normalize `skills` columns: convert pct_time_* -> pcttime_* (app expects pcttime*)
    if not skills.empty:
        skills_col_map = {}
        for c in skills.columns:
            if c.startswith("pct_time_"):
                newc = c.replace("pct_time_", "pcttime_")
                skills_col_map[c] = newc
        # Special-case actively coding column name used in the notebook/app
        if "pct_time_actively_coding_at_home_or_work" in skills.columns:
            skills_col_map["pct_time_actively_coding_at_home_or_work"] = "pcttime_activelycoding"

        for src, dst in skills_col_map.items():
            if src in skills.columns and dst not in skills.columns:
                skills[dst] = skills[src]

    # Optional: fact table (not strictly needed for the notebook-style EDA)
    fact_matches = list(base.glob("fact*respondent*clean*.csv"))
    fact = pd.read_csv(fact_matches[0]) if fact_matches else None

    return demo, tools, mlframeworks, skills, fact


# Load data so variables exist for the rest of the script
demo, tools, mlframeworks, skills, fact = load_kaggle_data()


# -------------------------------------------------------------------
# HELPER FUNCTIONS RECREATING YOUR NOTEBOOK EDA LOGIC
# -------------------------------------------------------------------


def pct(num, den):
    """Percent helper, like in your notebook."""
    return round(num / den * 100, 1) if den else np.nan


def compute_phase1_demographics_kpis(demo: pd.DataFrame):
    """
    Recreate the Phase 1 KPIs you built: respondents, countries, gender shares,
    age, roles, education, experience, comp band, etc.
    """
    demo = demo.copy()

    # Coerce numeric like in notebook
    demo["ageyears"] = pd.to_numeric(demo["ageyears"], errors="coerce")
    demo["yearsexperiencecurrentrole"] = pd.to_numeric(
        demo["yearsexperiencecurrentrole"], errors="coerce"
    )

    ntotal = len(demo)

    # Professionals = exclude Student & Not employed (same as in the notebook)
    prof = demo[~demo["currentrole"].isin(["Student", "Not employed"])].copy()

    # Geography: country concentration
    country_vc = demo["country"].value_counts(dropna=True)
    top2_country_share = (
        round(country_vc.head(2).sum() / country_vc.sum() * 100, 1)
        if len(country_vc)
        else np.nan
    )

    # Gender shares
    male_n = (demo["gender"] == "Male").sum()
    female_n = (demo["gender"] == "Female").sum()

    # Age
    age = demo["ageyears"].dropna()
    if len(age):
        age_median = float(age.median())
        age_iqr = (float(age.quantile(0.25)), float(age.quantile(0.75)))
    else:
        age_median = np.nan
        age_iqr = (np.nan, np.nan)

    # Roles
    role_vc = demo["currentrole"].value_counts(dropna=True)
    if len(role_vc):
        top_role = role_vc.index[0]
        top_role_share = pct(role_vc.iloc[0], ntotal)
    else:
        top_role = np.nan
        top_role_share = np.nan

    # NEW: top employed role excl Student/Not employed (like your code)
    employed_only = demo[~demo["currentrole"].isin(["Student", "Not employed"])].copy()
    emp_vc = employed_only["currentrole"].value_counts(dropna=True)
    if len(emp_vc):
        top_employed_role = emp_vc.index[0]
        top_employed_share_total = pct(emp_vc.iloc[0], ntotal)
        top_employed_share_employed = pct(emp_vc.iloc[0], len(employed_only))
    else:
        top_employed_role = np.nan
        top_employed_share_total = np.nan
        top_employed_share_employed = np.nan

    # Education intensity: Masters + Doctoral
    edu = demo["educationlevel"].fillna("Missing")
    masters_n = (edu == "Masters degree").sum()
    doctoral_n = (edu == "Doctoral degree").sum()
    masters_doc_share = pct(masters_n + doctoral_n, ntotal)

    # Workforce maturity proxy: median years in current role
    exp = demo["yearsexperiencecurrentrole"].dropna()
    exp_median = float(exp.median()) if len(exp) else np.nan

    # Compensation headline: most common comp band among professionals only
    comp_vc = (
        prof["yearlycompensationusd"]
        .dropna()
        .astype(str)
        .value_counts()
    )
    comp_mode_band = comp_vc.index[0] if len(comp_vc) else np.nan

    # Put into table like kpitable
    kpi_rows = [
        ("Respondents N", ntotal),
        ("Countries represented", int(demo["country"].nunique(dropna=True))),
        ("Top 2 countries share", top2_country_share),
        ("Gender Female share", pct(female_n, ntotal)),
        ("Gender Male share", pct(male_n, ntotal)),
        ("Age Median years", age_median),
        (
            "Age IQR 25–75",
            f"{age_iqr[0]:.0f}–{age_iqr[1]:.0f}" if not np.isnan(age_iqr[0]) else np.nan,
        ),
        ("Top role overall", top_role),
        ("Top role share overall", top_role_share),
        ("Top employed role excl. Student/Unemployed", top_employed_role),
        ("Top employed role share of employed", top_employed_share_employed),
        ("Top employed role share of total", top_employed_share_total),
        ("Professionals excl. Student/Unemployed (N)", len(prof)),
        ("Professionals share", pct(len(prof), ntotal)),
        ("Education Masters+Doctoral share", masters_doc_share),
        ("Experience median years in current role", exp_median),
        ("Most common comp band (professionals)", comp_mode_band),
    ]

    kpitable = pd.DataFrame(kpi_rows, columns=["KPI", "Value"])

    # Also return numeric dict for metrics
    kpi_dict = {
        "ntotal": ntotal,
        "countries_represented": int(demo["country"].nunique(dropna=True)),
        "top2_country_share": top2_country_share,
        "female_share": pct(female_n, ntotal),
        "male_share": pct(male_n, ntotal),
        "age_median": age_median,
        "age_iqr": age_iqr,
        "top_role": top_role,
        "top_role_share": top_role_share,
        "top_employed_role": top_employed_role,
        "top_employed_share_total": top_employed_share_total,
        "top_employed_share_employed": top_employed_share_employed,
        "professionals_n": len(prof),
        "professionals_share": pct(len(prof), ntotal),
        "masters_doc_share": masters_doc_share,
        "exp_median": exp_median,
        "comp_mode_band": comp_mode_band,
    }

    return kpitable, kpi_dict


def compute_country_summary(demo: pd.DataFrame, top_n: int = 20):
    """
    Recreate your 'countrysummary' table: top N countries by respondents + %
    """
    demo = demo.copy()
    ntotal = len(demo)
    country_counts = demo["country"].value_counts(dropna=False).head(top_n)
    country_pct = (country_counts / ntotal * 100).round(2)

    summary = pd.DataFrame(
        {
            "country": country_counts.index.astype(str),
            "respondents": country_counts.values,
            "percent": country_pct.values,
        }
    )
    return summary


def compute_task_time_allocation(skills: pd.DataFrame):
    """
    Recreate your avg time per 'pcttime*' task bar chart
    (excluding 'activelycoding', which is separate).
    """
    skills = skills.copy()
    pct_cols = [
        c for c in skills.columns
        if c.startswith("pcttime") and "activelycoding" not in c
    ]

    if not pct_cols:
        return None

    pct_df = skills[pct_cols].apply(pd.to_numeric, errors="coerce")
    avg_time = pct_df.mean().reset_index()
    avg_time.columns = ["task", "avg_pct"]

    # Clean up labels like in the notebook
    avg_time["task"] = (
        avg_time["task"]
        .str.replace("pcttime", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    # Sort ascending for horizontal bar
    avg_time = avg_time.sort_values("avg_pct", ascending=True)
    avg_time["label"] = avg_time["avg_pct"].apply(lambda x: f"{x:.1f}%")
    return avg_time


def plot_task_time_allocation(avg_time: pd.DataFrame):
    if avg_time is None or avg_time.empty:
        return None

    fig = px.bar(
        avg_time,
        x="avg_pct",
        y="task",
        orientation="h",
        text="label",
        title=(
            "% of Time per Task During a Data Science Project"
            "<br><sup>Average time per task values sum to ~100% (Kaggle DS Survey 2018)</sup>"
        ),
        labels={"avg_pct": "Avg time (%)", "task": "Task"},
        color="avg_pct",
        color_continuous_scale="Teal",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=200, r=40, t=80, b=40),
        height=430,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_country_distribution(countrysummary: pd.DataFrame):
    fig = px.bar(
        countrysummary,
        x="country",
        y="respondents",
        text="percent",
        labels={
            "country": "Country",
            "respondents": "Respondents",
            "percent": "% of total",
        },
        title="Top Countries by Number of Respondents",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        xaxis_tickangle=45,
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_models_black_box(tools: pd.DataFrame):
    """
    Simple bar chart for 'modelsasblackbox' question (from your tools table).
    """
    if "modelsasblackbox" not in tools.columns:
        return None

    vc = tools["modelsasblackbox"].value_counts(dropna=False)
    df = (
        vc.rename_axis("answer")
        .reset_index(name="count")
    )
    df["answer"] = df["answer"].astype(str)

    fig = px.bar(
        df,
        x="answer",
        y="count",
        text="count",
        title="Perception of ML Models as Black Boxes",
        labels={"answer": "Answer", "count": "Respondents"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=20,
        height=500,
    )
    return fig


def plot_primary_analysis_tools(tools: pd.DataFrame, top_n: int = 15):
    """
    Simple bar chart of primary analysis tools (Python, R, etc.).
    """
    col = "primaryanalysistool"
    if col not in tools.columns:
        return None

    vc = tools[col].value_counts(dropna=False).head(top_n)
    df = vc.rename_axis("tool").reset_index(name="count")
    df["tool"] = df["tool"].astype(str)

    fig = px.bar(
        df,
        x="tool",
        y="count",
        text="count",
        title=f"Top {top_n} Primary Analysis Tools",
        labels={"tool": "Tool", "count": "Respondents"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=30,
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


# Additional visuals imported from the notebook

def plot_country_hbar(country_summary: pd.DataFrame):
    if country_summary is None or country_summary.empty:
        return None
    df = country_summary.sort_values("respondents", ascending=True)
    fig = px.bar(
        df,
        x="respondents",
        y="country",
        orientation="h",
        text="respondents",
        title="Respondents by country (Top 20)",
        labels={"respondents": "Respondents", "country": "Country"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=140, r=40, t=70, b=40),
        height=600,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_recommended_languages(skills: pd.DataFrame, top_n: int = 15):
    if skills is None or skills.empty:
        return None
    col = "recommended_first_language"
    if col not in skills.columns:
        return None
    vc = skills[col].fillna("Missing").value_counts().head(top_n)
    df = pd.DataFrame({"language": vc.index.astype(str), "respondents": vc.values}).sort_values("respondents", ascending=True)
    fig = px.bar(
        df,
        x="respondents",
        y="language",
        orientation="h",
        text="respondents",
        title=f"Recommended first programming language (Top {top_n})",
        labels={"respondents": "Respondents", "language": "Language"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=120, r=40, t=70, b=40),
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_gender_donut(demo: pd.DataFrame):
    if demo is None or demo.empty:
        return None
    df = demo["gender"].fillna("Missing").value_counts().reset_index()
    df.columns = ["gender", "respondents"]
    fig = px.pie(df, names="gender", values="respondents", hole=0.45, title="Gender distribution (donut)")
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(
        margin=dict(l=40, r=40, t=70, b=40),
        height=400,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_role_gender_stacked(demo: pd.DataFrame):
    if demo is None or demo.empty:
        return None
    d = demo.copy()
    def role_group(role):
        if pd.isna(role):
            return "Missing"
        r = str(role).strip().lower()
        if r == "student":
            return "Student"
        if r == "not employed":
            return "Not employed"
        return "Employed/Other"
    d["role_group"] = d["currentrole"].apply(role_group)
    d["gender"] = d["gender"].fillna("Missing")
    gb = d.groupby(["role_group", "gender"]).size().reset_index(name="respondents")
    gb["percent"] = gb["respondents"] / gb.groupby("role_group")["respondents"].transform("sum") * 100
    gb["percent_label"] = gb["percent"].round(1).astype(str) + "%"
    fig = px.bar(
        gb,
        x="role_group",
        y="percent",
        color="gender",
        text="percent_label",
        title="Gender share by role group (100% stacked)",
        labels={"percent": "Percent", "role_group": "Role group", "gender": "Gender"},
    )
    fig.update_yaxes(range=[0, 100])
    fig.update_traces(textposition="inside", insidetextanchor="middle", cliponaxis=False)
    fig.update_layout(
        barmode="stack",
        margin=dict(l=40, r=40, t=70, b=40),
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_age_histogram(demo: pd.DataFrame):
    if demo is None or demo.empty:
        return None
    d = demo.copy()
    if "ageyears" not in d.columns:
        # Try alternate name
        if "age_years" in d.columns:
            d["ageyears"] = pd.to_numeric(d["age_years"], errors="coerce")
        else:
            return None
    d = d.dropna(subset=["ageyears"])
    med = d["ageyears"].median()
    mean = d["ageyears"].mean()
    fig = px.histogram(d, x="ageyears", nbins=26, title="Age distribution (Kaggle DS Survey 2018)", labels={"ageyears": "Age (yrs)"})
    fig.update_xaxes(title_text="Age (yrs)")
    fig.update_yaxes(title_text="Respondents")
    fig.update_layout(
        title={"text": f"Age distribution (Kaggle DS Survey 2018)\nMedian {med:.0f}, Mean {mean:.1f}"},
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    fig.add_vline(x=med, line_width=2, line_dash="dash")
    fig.add_vline(x=mean, line_width=2, line_dash="dot")
    return fig


def topn_other(series, n=15, other_label="Other (rest)"):
    """Helper: group top N + rest."""
    vc = series.fillna("Missing").value_counts()
    top = vc.head(n)
    rest = vc.iloc[n:].sum() if len(vc) > n else 0
    out = top.reset_index()
    out.columns = ["category", "count"]
    if rest > 0:
        out.loc[len(out)] = [other_label, rest]
    out["percent"] = (out["count"] / out["count"].sum() * 100).round(1)
    out["label"] = out.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    return out


def plot_age_band(demo: pd.DataFrame):
    """Age band bar chart (e.g., 18-24, 25-34, etc.)."""
    if demo is None or demo.empty:
        return None
    d = demo.copy()
    if "ageyears" not in d.columns and "age_years" in d.columns:
        d["ageyears"] = pd.to_numeric(d["age_years"], errors="coerce")
    if "ageyears" not in d.columns:
        return None
    bins = [17, 24, 34, 44, 54, 64, 100]
    labels = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
    d["age_band"] = pd.cut(d["ageyears"], bins=bins, labels=labels)
    age_band = d["age_band"].value_counts().reindex(labels).fillna(0).astype(int).reset_index()
    age_band.columns = ["age_band", "respondents"]
    age_band["percent"] = (age_band["respondents"] / age_band["respondents"].sum() * 100).round(1)
    age_band["label"] = age_band.apply(lambda r: f"{int(r['respondents']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(age_band, x="age_band", y="respondents", text="label", title="Respondents by age band (Kaggle DS Survey 2018)", labels={"age_band": "Age band", "respondents": "Respondents"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_age_by_role_box(demo: pd.DataFrame):
    """Box plot: age by role group."""
    if demo is None or demo.empty:
        return None
    d = demo.copy()
    if "ageyears" not in d.columns and "age_years" in d.columns:
        d["ageyears"] = pd.to_numeric(d["age_years"], errors="coerce")
    if "ageyears" not in d.columns:
        return None
    def role_group(role):
        if pd.isna(role):
            return "Missing"
        r = str(role).strip().lower()
        if r == "student":
            return "Student"
        if r == "not employed":
            return "Not employed"
        return "Employed/Other"
    d["role_group"] = d["currentrole"].apply(role_group)
    d = d.dropna(subset=["ageyears"])
    fig = px.box(d, x="role_group", y="ageyears", points="outliers", title="Age by role group", labels={"role_group": "Role group", "ageyears": "Age (yrs)"})
    fig.update_layout(
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_roles_hbar(demo: pd.DataFrame, top_n: int = 15):
    """Horizontal bar chart: top roles."""
    if demo is None or demo.empty:
        return None
    roles_df = topn_other(demo["currentrole"], n=top_n).sort_values("count", ascending=True)
    fig = px.bar(roles_df, x="count", y="category", orientation="h", text="label", title=f"Top {top_n} roles by respondents", labels={"count": "Respondents", "category": "Role"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=200, r=40, t=70, b=40),
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_industries_hbar(demo: pd.DataFrame, top_n: int = 15):
    """Horizontal bar chart: top industries."""
    if demo is None or demo.empty:
        return None
    if "current_industry" not in demo.columns and "currentindustry" not in demo.columns:
        return None
    col = "current_industry" if "current_industry" in demo.columns else "currentindustry"
    ind_df = topn_other(demo[col], n=top_n).sort_values("count", ascending=True)
    fig = px.bar(ind_df, x="count", y="category", orientation="h", text="label", title=f"Top {top_n} industries by respondents", labels={"count": "Respondents", "category": "Industry"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=200, r=40, t=70, b=40),
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_education_hbar(demo: pd.DataFrame, top_n: int = 10):
    """Horizontal bar chart: education levels."""
    if demo is None or demo.empty:
        return None
    edu_df = topn_other(demo["educationlevel"], n=top_n).sort_values("count", ascending=True)
    fig = px.bar(edu_df, x="count", y="category", orientation="h", text="label", title="Education levels of respondents", labels={"count": "Respondents", "category": "Education"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=200, r=40, t=70, b=40),
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_compensation_hbar(demo: pd.DataFrame):
    """Horizontal bar chart: compensation bands (professionals only)."""
    if demo is None or demo.empty:
        return None
    prof = demo[~demo["currentrole"].isin(["Student", "Not employed"])].copy()
    if prof.empty:
        return None
    comp_vc = prof["yearlycompensationusd"].dropna().astype(str).value_counts()
    if comp_vc.empty:
        return None
    comp_df = comp_vc.reset_index()
    comp_df.columns = ["band", "respondents"]
    comp_df["percent"] = (comp_df["respondents"] / comp_df["respondents"].sum() * 100).round(1)
    comp_df["label"] = comp_df.apply(lambda r: f"{int(r['respondents']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(comp_df, x="respondents", y="band", orientation="h", text="label", title="Compensation bands (professionals only)", labels={"respondents": "Respondents", "band": "Comp band"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=150, r=40, t=70, b=40),
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def explode_multiselect(series, sep=" | "):
    """Explode pipe-separated values and clean up formatting."""
    return (
        series.dropna()
        .astype(str)
        .str.replace(r'\s*\|\s*', '|', regex=True)  # Normalize: convert ' | ' and '|' to single '|'
        .str.split('|')  # Split on normalized pipe
        .explode()  # Explode to individual rows
        .str.strip()  # Strip whitespace from each item
        .loc[lambda x: x != ""]  # Remove empty strings only
    )


def plot_programming_languages(skills: pd.DataFrame, top_n: int = 15):
    """Bar chart: top programming languages used."""
    if skills is None or skills.empty:
        return None
    if "programming_languages_used" not in skills.columns:
        return None
    langs = explode_multiselect(skills["programming_languages_used"])
    langs = langs[~langs.isin(["|", " | ", "", " ", "nan"])]
    if langs.empty:
        return None
    vc = langs.value_counts().head(top_n)
    df = vc.reset_index()
    df.columns = ["language", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="language", orientation="h", text="label", title=f"Top {top_n} programming languages used (multi-select)", labels={"count": "Mentions", "language": "Language"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=150, r=40, t=70, b=40),
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_ml_frameworks(mlframeworks: pd.DataFrame, top_n: int = 12):
    """Bar chart: top ML frameworks."""
    if mlframeworks is None or mlframeworks.empty:
        return None
    col = "ml_frameworks_used" if "ml_frameworks_used" in mlframeworks.columns else "mlframeworksused"
    if col not in mlframeworks.columns:
        return None
    frameworks = explode_multiselect(mlframeworks[col])
    frameworks = frameworks[~frameworks.isin(["|", " | ", "", " ", "nan"])]
    if frameworks.empty:
        return None
    vc = frameworks.value_counts().head(top_n)
    df = vc.reset_index()
    df.columns = ["framework", "count"]
    df = df.sort_values("count", ascending=False)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="framework", y="count", text="label", title=f"Top {top_n} ML frameworks used (multi-select)", labels={"count": "Mentions", "framework": "Framework"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_tickangle=-30,
        height=450,
        margin=dict(b=100),
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_work_activities(skills: pd.DataFrame):
    """Bar chart: work activities (multi-select)."""
    if skills is None or skills.empty:
        return None
    if "work_activities" not in skills.columns:
        return None
    wa = skills["work_activities"].fillna("").astype(str)
    activities = [
        "Analyze and understand data to influence product or business decisions",
        "Build and/or run a machine learning service that operationally improves my product or workflows",
        "Build and/or run the data infrastructure",
        "Build prototypes to explore applying machine learning to new areas",
        "Do research that advances the state of the art of machine learning",
        "None of these activities are an important part of my role at work",
        "Other"
    ]
    result = []
    for val in wa:
        if pd.isna(val) or val.strip() == "":
            continue
        found = False
        for act in activities:
            if act[:30] in str(val):
                result.append(act[:50] + ("..." if len(act) > 50 else ""))
                found = True
                break
        if not found and val.strip():
            result.append("Other")
    if not result:
        return None
    acts_series = pd.Series(result)
    vc = acts_series.value_counts()
    df = vc.reset_index()
    df.columns = ["activity", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="activity", orientation="h", text="label", title="Top work activities (multi-select)", labels={"count": "Respondents", "activity": "Activity"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=300, r=40, t=70, b=40),
        height=400,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_task_breakdown(skills: pd.DataFrame):
    """Grouped bar: data prep vs model+prod vs insight."""
    if skills is None or skills.empty:
        return None
    pct_cols = [c for c in skills.columns if c.startswith("pcttime_") and "activelycoding" not in c]
    if not pct_cols:
        return None
    pct_df = skills[pct_cols].apply(pd.to_numeric, errors="coerce")
    data_prep = pct_df[["pcttime_gathering_data", "pcttime_cleaning_data"]].sum(axis=1, min_count=1).mean() if all(c in pct_df.columns for c in ["pcttime_gathering_data", "pcttime_cleaning_data"]) else 0
    model_prod = pct_df[["pcttime_model_building", "pcttime_putting_model_in_prod"]].sum(axis=1, min_count=1).mean() if all(c in pct_df.columns for c in ["pcttime_model_building", "pcttime_putting_model_in_prod"]) else 0
    insight_viz = pct_df[["pcttime_visualizing_data", "pcttime_finding_insights"]].sum(axis=1, min_count=1).mean() if all(c in pct_df.columns for c in ["pcttime_visualizing_data", "pcttime_finding_insights"]) else 0
    other = pct_df[["pcttime_other_project_tasks"]].sum(axis=1, min_count=1).mean() if "pcttime_other_project_tasks" in pct_df.columns else 0
    summary_df = pd.DataFrame({
        "category": ["Data Prep\n(Gather+Clean)", "Model+Prod\n(Build+Deploy)", "Insight+Viz", "Other"],
        "avg_pct": [round(data_prep, 1), round(model_prod, 1), round(insight_viz, 1), round(other, 1)]
    })
    summary_df["label"] = summary_df["avg_pct"].apply(lambda x: f"{x:.1f}%")
    summary_df = summary_df.sort_values("avg_pct", ascending=False)
    fig = px.bar(summary_df, x="category", y="avg_pct", text="label", title="Where effort really goes: Task group breakdown", labels={"avg_pct": "Avg % time", "category": ""})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        height=450,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_online_platforms(fact: pd.DataFrame = None, top_n: int = 12):
    """Bar chart: online learning platforms used (multi-select)."""
    if fact is None or fact.empty:
        return None
    col = "online_platforms_used"
    if col not in fact.columns:
        return None
    platforms = explode_multiselect(fact[col])
    platforms = platforms[~platforms.isin(["|", " | ", "", " ", "nan"])]
    if platforms.empty:
        return None
    vc = platforms.value_counts().head(top_n)
    df = vc.reset_index()
    df.columns = ["platform", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="platform", orientation="h", text="label", title=f"Top {top_n} Online Learning Platforms Used (multi-select)", labels={"count": "Mentions", "platform": "Platform"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=250, r=40, t=70, b=40),
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_visualization_tools(fact: pd.DataFrame = None, top_n: int = 12):
    """Bar chart: visualization tools used (multi-select)."""
    if fact is None or fact.empty:
        return None
    col = "visualization_tools_used"
    if col not in fact.columns:
        return None
    viz_tools = explode_multiselect(fact[col])
    viz_tools = viz_tools[~viz_tools.isin(["|", " | ", "", " ", "nan"])]
    if viz_tools.empty:
        return None
    vc = viz_tools.value_counts().head(top_n)
    df = vc.reset_index()
    df.columns = ["tool", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="tool", orientation="h", text="label", title=f"Top {top_n} Visualization Tools Used (multi-select)", labels={"count": "Mentions", "tool": "Tool"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=200, r=40, t=70, b=40),
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="#1a1f2e",
    )
    return fig


def plot_cloud_services(fact: pd.DataFrame = None):
    """Donut chart: individual cloud provider adoption."""
    if fact is None or fact.empty:
        return None
    col = "cloud_services_used"
    if col not in fact.columns:
        return None
    # Use explode_multiselect to split pipe-separated providers
    providers = explode_multiselect(fact[col], sep=" | ")
    if providers.empty:
        return None
    vc = providers.value_counts()
    df = vc.reset_index()
    df.columns = ["provider", "count"]
    df = df.sort_values("count", ascending=False)
    df["percent"] = (df["count"] / df["count"].sum() * 100).round(1)
    fig = px.pie(df, names="provider", values="count", hole=0.4, title="Cloud Provider Adoption", labels={"count": "Respondents", "provider": "Provider"})
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(margin=dict(l=40, r=40, t=70, b=40), height=500)
    return fig


def plot_reproducibility_barriers(fact: pd.DataFrame = None, top_n: int = 10):
    """Bar chart: top reproducibility barriers (multi-select)."""
    if fact is None or fact.empty:
        return None
    col = "reproducibility_barriers"
    if col not in fact.columns:
        return None
    barriers = explode_multiselect(fact[col])
    barriers = barriers[~barriers.isin(["|", " | ", "", " ", "nan", "None of these reasons apply to me"])]
    if barriers.empty:
        return None
    vc = barriers.value_counts().head(top_n)
    df = vc.reset_index()
    df.columns = ["barrier", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="barrier", orientation="h", text="label", title=f"Top {top_n} Reproducibility Barriers", labels={"count": "Respondents", "barrier": "Barrier"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=350, r=40, t=70, b=40), height=500)
    return fig


def plot_ml_products(fact: pd.DataFrame = None, top_n: int = 12):
    """Bar chart: ML products/cloud services used (multi-select)."""
    if fact is None or fact.empty:
        return None
    col = "ml_products_used"
    if col not in fact.columns:
        return None
    products = explode_multiselect(fact[col])
    products = products[~products.isin(["|", " | ", "", " ", "nan"])]
    if products.empty:
        return None
    vc = products.value_counts().head(top_n)
    df = vc.reset_index()
    df.columns = ["product", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / vc.sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="product", orientation="h", text="label", title=f"Top {top_n} ML Products/Cloud Services Used", labels={"count": "Mentions", "product": "Product"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=250, r=40, t=70, b=40), height=500)
    return fig


def plot_primary_vs_recommended_lang(fact: pd.DataFrame = None):
    """Side-by-side: primary language vs. recommended language (top 8)."""
    if fact is None or fact.empty:
        return None
    cols_needed = ["primary_programming_language", "recommended_first_language"]
    if not all(c in fact.columns for c in cols_needed):
        return None
    primary = fact["primary_programming_language"].value_counts(dropna=True).head(8)
    recommended = fact["recommended_first_language"].value_counts(dropna=True).head(8)
    all_langs = sorted(set(primary.index) | set(recommended.index))
    comparison_df = pd.DataFrame({
        "language": all_langs,
        "primary": [primary.get(lang, 0) for lang in all_langs],
        "recommended": [recommended.get(lang, 0) for lang in all_langs],
    }).sort_values("primary", ascending=True)
    fig = go.Figure(data=[
        go.Bar(name="Primary Language Used", x=comparison_df["primary"], y=comparison_df["language"], orientation="h", marker_color="#1f77b4"),
        go.Bar(name="Recommended First Language", x=comparison_df["recommended"], y=comparison_df["language"], orientation="h", marker_color="#ff7f0e"),
    ])
    fig.update_layout(title="Primary Language Used vs. Recommended First Language", barmode="group", xaxis_title="Respondents", yaxis_title="Language", margin=dict(l=150, r=40, t=70, b=40), height=500)
    return fig


def plot_coding_experience_distribution(fact: pd.DataFrame = None):
    """Histogram: years coding for data science."""
    if fact is None or fact.empty:
        return None
    col = "years_coding_for_data"
    if col not in fact.columns:
        return None
    years = pd.to_numeric(fact[col], errors="coerce").dropna()
    if years.empty:
        return None
    fig = px.histogram(x=years, nbins=20, title="Years of Coding Experience (for Data Science)", labels={"x": "Years", "count": "Respondents"})
    med = years.median()
    fig.add_vline(x=med, line_width=2, line_dash="dash", annotation_text=f"Median: {med:.1f}")
    fig.update_layout(height=450)
    return fig


def plot_ml_experience_distribution(fact: pd.DataFrame = None):
    """Histogram: years using ML."""
    if fact is None or fact.empty:
        return None
    col = "years_using_ml"
    if col not in fact.columns:
        return None
    years = pd.to_numeric(fact[col], errors="coerce").dropna()
    if years.empty:
        return None
    fig = px.histogram(x=years, nbins=20, title="Years of ML Experience", labels={"x": "Years", "count": "Respondents"})
    med = years.median()
    fig.add_vline(x=med, line_width=2, line_dash="dash", annotation_text=f"Median: {med:.1f}")
    fig.update_layout(height=450)
    return fig


def plot_data_scientist_identity(fact: pd.DataFrame = None):
    """Donut: self-identification as data scientist."""
    if fact is None or fact.empty:
        return None
    col = "considers_self_data_scientist"
    if col not in fact.columns:
        return None
    identity = fact[col].fillna("Not specified").astype(str)
    identity = identity[identity.str.strip().ne("")]
    if identity.empty:
        return None
    vc = identity.value_counts()
    df = vc.reset_index()
    df.columns = ["response", "count"]
    fig = px.pie(df, names="response", values="count", hole=0.4, title="Do you consider yourself a data scientist?")
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(margin=dict(l=40, r=40, t=70, b=40), height=450)
    return fig


def plot_model_success_metrics(fact: pd.DataFrame = None, top_n: int = 12):
    """Bar chart: model success metrics used (multi-select)."""
    if fact is None or fact.empty:
        return None
    col = "model_success_metrics"
    if col not in fact.columns:
        return None
    # Explode the pipe-separated values
    metrics = explode_multiselect(fact[col], sep=" | ")
    # Remove null values and filtering artifacts
    metrics = metrics[metrics.str.len() > 0]
    metrics = metrics[~metrics.str.lower().isin(["nan", "none", "not applicable"])]
    if metrics.empty:
        return None
    vc = metrics.value_counts().head(top_n)
    if vc.empty:
        return None
    df = vc.reset_index()
    df.columns = ["metric", "count"]
    df = df.sort_values("count", ascending=True)
    df["percent"] = (df["count"] / df["count"].sum() * 100).round(1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['percent']}%)", axis=1)
    fig = px.bar(df, x="count", y="metric", orientation="h", text="label", title=f"Top {top_n} Model Success Metrics Used", labels={"count": "Respondents", "metric": "Metric"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=350, r=40, t=70, b=40), height=500, hovermode="y unified")
    return fig


def plot_training_sources(fact: pd.DataFrame = None):
    """Horizontal bar: primary training sources (mean % time)."""
    if fact is None or fact.empty:
        return None
    training_cols = {"Self-Taught": "pct_training_selftaught", "Online Courses": "pct_training_online_courses", "Work": "pct_training_work", "University": "pct_training_university", "Kaggle": "pct_training_kaggle", "Other": "pct_training_other"}
    means = {}
    for label, col in training_cols.items():
        if col in fact.columns:
            means[label] = pd.to_numeric(fact[col], errors="coerce").mean()
    if not means:
        return None
    df = pd.DataFrame(list(means.items()), columns=["source", "avg_pct"]).sort_values("avg_pct", ascending=True)
    df["label"] = df["avg_pct"].apply(lambda x: f"{x:.1f}%")
    fig = px.bar(df, x="avg_pct", y="source", orientation="h", text="label", title="Primary Training Sources (Mean % Time)", labels={"avg_pct": "Avg % Time", "source": "Source"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=150, r=40, t=70, b=40), height=400)
    return fig


def plot_actively_coding_time(fact: pd.DataFrame = None):
    """Donut: % time actively coding at home/work."""
    if fact is None or fact.empty:
        return None
    col = "pct_time_actively_coding_at_home_or_work"
    if col not in fact.columns:
        return None
    coding_time = pd.to_numeric(fact[col], errors="coerce").dropna()
    if coding_time.empty:
        return None
    bins = [0, 25, 50, 75, 100, 125]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%", "100%+"]
    coding_band = pd.cut(coding_time, bins=bins, labels=labels)
    vc = coding_band.value_counts()
    df = vc.reset_index()
    df.columns = ["band", "count"]
    fig = px.pie(df, names="band", values="count", hole=0.4, title="% Time Actively Coding (home or work)")
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(margin=dict(l=40, r=40, t=70, b=40), height=450)
    return fig


# -------------------------------------------------------------------
# SIDEBAR FILTERS (demographics, like your website filters but adapted)
# -------------------------------------------------------------------
st.sidebar.title("Filters")

# Initialize session state for filters if not exists
if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = []
if "selected_roles" not in st.session_state:
    st.session_state.selected_roles = []
if "selected_genders" not in st.session_state:
    st.session_state.selected_genders = []
if "age_range" not in st.session_state:
    st.session_state.age_range = (int(demo["ageyears"].min()), int(demo["ageyears"].max()))
if "selected_slicers" not in st.session_state:
    st.session_state.selected_slicers = ["Country", "Current Role", "Gender", "Age Range"]

# Comprehensive slicer options grouped by category
slicer_categories = {
    "📍 Demographics": [
        "Country",
        "Current Role",
        "Current Industry",
        "Gender",
        "Age Range",
        "Education Level",
        "Undergraduate Major",
        "Years in Current Role"
    ],
    "💰 Compensation & Career": [
        "Yearly Compensation",
        "Years Coding for Data",
        "Years Using ML",
        "Self-identify as Data Scientist"
    ],
    "🛠️ Tools & Technologies": [
        "Primary Analysis Tool",
        "Most Used ML Framework",
        "Most Used Visualization Tool",
        "Cloud Services Used",
        "Primary Data Type",
        "Relational Databases Used"
    ],
    "📚 Education & Training": [
        "Quality of Online Learning",
        "Quality of Bootcamps",
        "Primary Learning Platform",
        "Academics vs Projects View"
    ],
    "🤖 ML & Modeling": [
        "Models as Black Box",
        "Model Success Metrics",
        "Model Insights Use Cases",
        "Perception - Fairness/Bias",
        "Perception - Interpretability",
        "Perception - Reproducibility"
    ],
    "⏱️ Time Allocation": [
        "% Time Gathering Data",
        "% Time Cleaning Data",
        "% Time Visualizing Data",
        "% Time Model Building",
        "% Time Putting Model in Prod",
        "% Time Finding Insights",
        "% Time Actively Coding"
    ]
}

# Flatten for display
all_available_slicers = [slicer for slicers in slicer_categories.values() for slicer in slicers]

# Display slicer selector with categories
st.sidebar.markdown("### 📊 Customize Slicers")
st.sidebar.markdown("""
Select which filter slicers you want to display in the sidebar.
*Slicers help you subset the data by specific dimensions.*
""")

st.session_state.selected_slicers = st.sidebar.multiselect(
    "Available Slicers",
    options=all_available_slicers,
    default=st.session_state.selected_slicers,
    help="Select one or more slicers to show filtering options"
)

st.sidebar.markdown("---")

# Basic filters on demographics
countries = sorted(demo["country"].dropna().unique().astype(str))
roles = sorted(demo["currentrole"].dropna().unique().astype(str))
genders = sorted(demo["gender"].dropna().unique().astype(str))
educations = sorted(demo["educationlevel"].dropna().unique().astype(str)) if "educationlevel" in demo.columns else []
industries = sorted(demo["currentindustry"].dropna().unique().astype(str)) if "currentindustry" in demo.columns else []
majors = sorted(demo["undergraduatemajor"].dropna().unique().astype(str)) if "undergraduatemajor" in demo.columns else []

min_age, max_age = int(demo["ageyears"].min()), int(demo["ageyears"].max())

# Dynamically show slicers based on selection
if "Country" in st.session_state.selected_slicers:
    st.session_state.selected_countries = st.sidebar.multiselect(
        "🌍 Country",
        options=countries,
        default=st.session_state.selected_countries,
        help=f"{len(countries)} countries available"
    )

if "Current Role" in st.session_state.selected_slicers:
    st.session_state.selected_roles = st.sidebar.multiselect(
        "💼 Current Role",
        options=roles,
        default=st.session_state.selected_roles,
        help=f"{len(roles)} roles available"
    )

if "Gender" in st.session_state.selected_slicers:
    st.session_state.selected_genders = st.sidebar.multiselect(
        "👥 Gender",
        options=genders,
        default=st.session_state.selected_genders,
        help=f"{len(genders)} categories available"
    )

if "Age Range" in st.session_state.selected_slicers:
    st.session_state.age_range = st.sidebar.slider(
        "📅 Age Range (years)",
        min_value=min_age,
        max_value=max_age,
        value=st.session_state.age_range,
        help=f"Range: {min_age} - {max_age} years"
    )

if "Education Level" in st.session_state.selected_slicers and educations:
    selected_education = st.sidebar.multiselect(
        "🎓 Education Level",
        options=educations,
        default=[],
        help=f"{len(educations)} education levels"
    )
else:
    selected_education = []

if "Current Industry" in st.session_state.selected_slicers and industries:
    selected_industry = st.sidebar.multiselect(
        "🏢 Current Industry",
        options=industries,
        default=[],
        help=f"{len(industries)} industries"
    )
else:
    selected_industry = []

if "Undergraduate Major" in st.session_state.selected_slicers and majors:
    selected_major = st.sidebar.multiselect(
        "🎯 Undergraduate Major",
        options=majors,
        default=[],
        help=f"{len(majors)} majors"
    )
else:
    selected_major = []

if "Years in Current Role" in st.session_state.selected_slicers:
    years_role_min = int(demo["yearsexperiencecurrentrole"].min()) if "yearsexperiencecurrentrole" in demo.columns else 0
    years_role_max = int(demo["yearsexperiencecurrentrole"].max()) if "yearsexperiencecurrentrole" in demo.columns else 50
    selected_years_role = st.sidebar.slider(
        "⏳ Years in Current Role",
        min_value=years_role_min,
        max_value=years_role_max,
        value=(years_role_min, years_role_max),
        help="Filter by years of experience in current role"
    )
else:
    selected_years_role = None

if "Yearly Compensation" in st.session_state.selected_slicers:
    compensations = sorted(demo["yearlycompensationusd"].dropna().unique().astype(str)) if "yearlycompensationusd" in demo.columns else []
    selected_compensation = st.sidebar.multiselect(
        "💵 Yearly Compensation (USD)",
        options=compensations,
        default=[],
        help=f"{len(compensations)} compensation ranges"
    )
else:
    selected_compensation = []

if "Self-identify as Data Scientist" in st.session_state.selected_slicers:
    ds_identity_options = ["Yes", "No", "Probably yes", "Probably no"]
    selected_ds_identity = st.sidebar.multiselect(
        "👨‍🔬 Self-identify as Data Scientist",
        options=ds_identity_options,
        default=[],
        help="Filter by data scientist self-identification"
    )
else:
    selected_ds_identity = []

st.sidebar.markdown("---")

# Reset Filters button - clears all filter selections
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.session_state.selected_countries = []
    st.session_state.selected_roles = []
    st.session_state.selected_genders = []
    st.session_state.age_range = (min_age, max_age)
    safe_rerun()

# Apply filters
demo_filtered = demo.copy()
fact_filtered = fact.copy() if fact is not None else None

# Apply country filter
if st.session_state.selected_countries:
    demo_filtered = demo_filtered[demo_filtered["country"].isin(st.session_state.selected_countries)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["country"].isin(st.session_state.selected_countries)]

# Apply role filter
if st.session_state.selected_roles:
    demo_filtered = demo_filtered[demo_filtered["currentrole"].isin(st.session_state.selected_roles)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["current_role"].isin(st.session_state.selected_roles)]

# Apply gender filter
if st.session_state.selected_genders:
    demo_filtered = demo_filtered[demo_filtered["gender"].isin(st.session_state.selected_genders)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["gender"].isin(st.session_state.selected_genders)]

# Apply age range filter
age_min, age_max = st.session_state.age_range
demo_filtered = demo_filtered[
    (demo_filtered["ageyears"].fillna(age_min) >= age_min)
    & (demo_filtered["ageyears"].fillna(age_max) <= age_max)
]
if fact_filtered is not None:
    fact_filtered = fact_filtered[
        (fact_filtered["age_years"].fillna(age_min) >= age_min)
        & (fact_filtered["age_years"].fillna(age_max) <= age_max)
    ]

# Apply education filter
if selected_education:
    demo_filtered = demo_filtered[demo_filtered["educationlevel"].isin(selected_education)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["education_level"].isin(selected_education)]

# Apply industry filter
if selected_industry:
    demo_filtered = demo_filtered[demo_filtered["currentindustry"].isin(selected_industry)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["current_industry"].isin(selected_industry)]

# Apply major filter
if selected_major:
    demo_filtered = demo_filtered[demo_filtered["undergraduatemajor"].isin(selected_major)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["undergraduate_major"].isin(selected_major)]

# Apply years in role filter
if selected_years_role:
    years_min, years_max = selected_years_role
    if "yearsexperiencecurrentrole" in demo.columns:
        demo_filtered = demo_filtered[
            (demo_filtered["yearsexperiencecurrentrole"].fillna(years_min) >= years_min)
            & (demo_filtered["yearsexperiencecurrentrole"].fillna(years_max) <= years_max)
        ]
    if fact_filtered is not None and "years_experience_current_role" in fact_filtered.columns:
        fact_filtered = fact_filtered[
            (fact_filtered["years_experience_current_role"].fillna(years_min) >= years_min)
            & (fact_filtered["years_experience_current_role"].fillna(years_max) <= years_max)
        ]

# Apply compensation filter
if selected_compensation:
    demo_filtered = demo_filtered[demo_filtered["yearlycompensationusd"].isin(selected_compensation)]
    if fact_filtered is not None:
        fact_filtered = fact_filtered[fact_filtered["yearly_compensation_usd"].isin(selected_compensation)]

# Apply DS identity filter
if selected_ds_identity:
    if fact_filtered is not None:
        ds_identity_lower = [x.lower() for x in selected_ds_identity]
        fact_filtered = fact_filtered[fact_filtered["considers_self_data_scientist"].str.lower().isin(ds_identity_lower)]

# -------------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------------
logout()

# Professional Header
st.markdown(
    """
    <div class="header-container">
        <h1>📊 Kaggle 2018 Data Science Survey</h1>
        <p>Interactive Exploratory Data Analysis Dashboard • 23,868 Respondents • 64 Data Points</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview KPIs", "👥 Demographics", "🛠️ Tools & Attitudes", "⏱️ Skills & Time", "🤖 ML Predictor"]
)

# -------------------------------------------------------------------
# TAB 1 – HIGH-LEVEL KPIs (Phase 1)
# -------------------------------------------------------------------
with tab1:
    st.subheader("High-Level KPIs (Demographics)")

    kpitable, kpis = compute_phase1_demographics_kpis(demo_filtered)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Respondents", f"{kpis['ntotal']:,}")
    with col2:
        st.metric("Countries", kpis["countries_represented"])
    with col3:
        st.metric("Professionals share", f"{kpis['professionals_share']:.1f}%")
    with col4:
        st.metric("Masters+Doctoral", f"{kpis['masters_doc_share']:.1f}%")
    with col5:
        st.metric("Median age", f"{kpis['age_median']:.0f}")

    st.markdown("---")
    st.markdown("#### Additional Insights")
    
    # Calculate additional KPIs from filtered fact table
    if fact_filtered is not None and not fact_filtered.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Data Scientists percentage
        ds_count = (fact_filtered["considers_self_data_scientist"].str.lower().isin(["yes", "probably yes"])).sum()
        ds_pct = (ds_count / len(fact_filtered) * 100) if len(fact_filtered) > 0 else 0
        with col1:
            st.metric("Self-identify as DS", f"{ds_pct:.1f}%")
        
        # ML Experience median
        ml_exp = pd.to_numeric(fact_filtered["years_using_ml"], errors="coerce")
        ml_median = ml_exp.median() if not ml_exp.empty else 0
        with col2:
            st.metric("Median ML Experience", f"{ml_median:.1f} yrs")
        
        # Coding Experience median
        code_exp = pd.to_numeric(fact_filtered["years_coding_for_data"], errors="coerce")
        code_median = code_exp.median() if not code_exp.empty else 0
        with col3:
            st.metric("Median Coding Experience", f"{code_median:.1f} yrs")
        
        # Cloud services adoption
        cloud_users = fact_filtered["cloud_services_used"].notna().sum()
        cloud_pct = (cloud_users / len(fact_filtered) * 100) if len(fact_filtered) > 0 else 0
        with col4:
            st.metric("Cloud Services Users", f"{cloud_pct:.1f}%")
        
        # ML Model builders
        model_builders = fact_filtered["model_success_metrics"].notna().sum()
        model_pct = (model_builders / len(fact_filtered) * 100) if len(fact_filtered) > 0 else 0
        with col5:
            st.metric("Build ML Models", f"{model_pct:.1f}%")

    st.markdown("---")
    st.markdown("#### KPI Summary Table")
    st.dataframe(kpitable, use_container_width=True)

# -------------------------------------------------------------------
# TAB 2 – DEMOGRAPHICS DETAIL (countries, age, roles)
# -------------------------------------------------------------------
with tab2:
    st.subheader("Demographics Detail")

    country_summary = compute_country_summary(demo_filtered, top_n=20)
    country_fig = plot_country_distribution(country_summary)
    country_hbar = plot_country_hbar(country_summary)

    # Top section: Country bar chart spanning full width
    st.markdown("##### Geographic Distribution - Top Countries")
    if country_hbar is not None:
        st.plotly_chart(country_hbar, use_container_width=True)
    elif country_fig is not None:
        st.plotly_chart(country_fig, use_container_width=True)
    else:
        st.info("Not enough country data to plot.")

    # Middle section: Age distribution and Gender donut (side by side)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Age Distribution")
        age_fig = plot_age_histogram(demo_filtered)
        if age_fig is not None:
            st.plotly_chart(age_fig, use_container_width=True)
        else:
            st.info("No age data available to plot.")

    with col2:
        st.markdown("##### Gender Distribution")
        gender_fig = plot_gender_donut(demo_filtered)
        if gender_fig is not None:
            st.plotly_chart(gender_fig, use_container_width=True)
        else:
            st.info("No gender data to plot.")

    # Bottom section: Country table
    st.markdown("---")
    st.markdown("##### Country Summary Table (Top 20)")
    st.dataframe(country_summary, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Role Distribution (filtered sample)")

    # Show top roles (simple bar) and a 100% stacked view by gender
    role_vc = (
        demo_filtered["currentrole"]
        .value_counts(dropna=False)
        .rename_axis("role")
        .reset_index(name="count")
    )
    role_vc["role"] = role_vc["role"].astype(str)

    if not role_vc.empty:
        fig_roles = px.bar(
            role_vc.head(25),
            x="role",
            y="count",
            text="count",
            title="Top Roles",
            labels={"role": "Role", "count": "Respondents"},
        )
        fig_roles.update_traces(textposition="outside")
        fig_roles.update_layout(xaxis_tickangle=30, height=400)
        st.plotly_chart(fig_roles, use_container_width=True)

        st.markdown("##### Gender share by role group (100% stacked)")
        stacked = plot_role_gender_stacked(demo_filtered)
        if stacked is not None:
            st.plotly_chart(stacked, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Age band distribution")
        age_band_fig = plot_age_band(demo_filtered)
        if age_band_fig is not None:
            st.plotly_chart(age_band_fig, use_container_width=True)

        st.markdown("##### Age by role group (box plot)")
        age_role_fig = plot_age_by_role_box(demo_filtered)
        if age_role_fig is not None:
            st.plotly_chart(age_role_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Top roles distribution")
        roles_fig = plot_roles_hbar(demo_filtered, top_n=15)
        if roles_fig is not None:
            st.plotly_chart(roles_fig, use_container_width=True)

        st.markdown("##### Top industries distribution")
        ind_fig = plot_industries_hbar(demo_filtered, top_n=15)
        if ind_fig is not None:
            st.plotly_chart(ind_fig, use_container_width=True)

        st.markdown("##### Education levels")
        edu_fig = plot_education_hbar(demo_filtered)
        if edu_fig is not None:
            st.plotly_chart(edu_fig, use_container_width=True)

        st.markdown("##### Compensation bands (professionals only)")
        comp_fig = plot_compensation_hbar(demo_filtered)
        if comp_fig is not None:
            st.plotly_chart(comp_fig, use_container_width=True)
    else:
        st.info("No role data after filters.")


# -------------------------------------------------------------------
# TAB 3 – TOOLS & ATTITUDES (primary tools, black-box perception)
# -------------------------------------------------------------------
with tab3:
    st.subheader("Tools & Attitudes")

    st.markdown("##### Primary Analysis Tools")
    fig_tools = plot_primary_analysis_tools(tools)
    if fig_tools is not None:
        st.plotly_chart(fig_tools, use_container_width=True)
    else:
        st.info("Column `primaryanalysistool` not found in tools table.")

    st.markdown("---")
    st.markdown("##### ML Frameworks Used (multi-select)")
    fw_fig = plot_ml_frameworks(mlframeworks, top_n=12)
    if fw_fig is not None:
        st.plotly_chart(fw_fig, use_container_width=True)
    else:
        st.info("No ML frameworks data available.")

    st.markdown("---")
    st.markdown("##### Visualization Tools Used (multi-select)")
    fig_viz_tools = plot_visualization_tools(fact_filtered, top_n=12)
    if fig_viz_tools is not None:
        st.plotly_chart(fig_viz_tools, use_container_width=True)
    else:
        st.info("No visualization tools data available.")

    st.markdown("---")
    st.markdown("##### Online Learning Platforms Used (multi-select)")
    fig_online = plot_online_platforms(fact_filtered, top_n=12)
    if fig_online is not None:
        st.plotly_chart(fig_online, use_container_width=True)
    else:
        st.info("No online platforms data available.")

    st.markdown("---")
    st.markdown("##### Cloud Provider Adoption")
    fig_cloud = plot_cloud_services(fact_filtered)
    if fig_cloud is not None:
        st.plotly_chart(fig_cloud, use_container_width=True)
        
        # Add data table underneath the chart
        if "cloud_services_used" in fact_filtered.columns:
            cloud_providers = explode_multiselect(fact_filtered["cloud_services_used"], sep=" | ")
            if not cloud_providers.empty:
                vc = cloud_providers.value_counts().reset_index()
                vc.columns = ["Provider", "Count"]
                vc["Percentage"] = (vc["Count"] / vc["Count"].sum() * 100).round(1)
                vc["Percentage_Formatted"] = vc["Percentage"].apply(lambda x: f"{x}%")
                
                # Display table with better styling
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("**Cloud Provider Adoption Summary**")
                
                # Create a more readable table display with all providers
                table_df = vc[["Provider", "Count", "Percentage_Formatted"]].copy()
                table_df.columns = ["Provider", "Count", "Percentage"]
                table_df = table_df.reset_index(drop=True)
                
                # Display as formatted table
                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Providers", len(table_df))
                with col2:
                    st.metric("Top Provider", table_df.iloc[0]["Provider"])
                with col3:
                    st.metric("Top Count", f"{table_df.iloc[0]['Count']:,}")
                with col4:
                    st.metric("Top Share", table_df.iloc[0]["Percentage"])
    else:
        st.info("No cloud provider data available.")

    st.markdown("---")
    st.markdown("##### Perception of ML Models as Black Boxes")
    fig_bb = plot_models_black_box(tools)
    if fig_bb is not None:
        st.plotly_chart(fig_bb, use_container_width=True)
    else:
        st.info("Column `modelsasblackbox` not found in tools table.")

    st.markdown("---")
    st.markdown("##### Top Reproducibility Barriers (multi-select)")
    fig_barriers = plot_reproducibility_barriers(fact_filtered, top_n=10)
    if fig_barriers is not None:
        st.plotly_chart(fig_barriers, use_container_width=True)
    else:
        st.info("No reproducibility barriers data available.")

    st.markdown("---")
    st.markdown("##### ML Products / Cloud Services Used (multi-select)")
    fig_ml_products = plot_ml_products(fact_filtered, top_n=12)
    if fig_ml_products is not None:
        st.plotly_chart(fig_ml_products, use_container_width=True)
    else:
        st.info("No ML products data available.")

# -------------------------------------------------------------------
# TAB 4 – SKILLS & TIME ALLOCATION (pcttime* questions)
# -------------------------------------------------------------------
with tab4:
    st.subheader("Skills & Time Allocation")

    st.markdown("##### % Time per Task (breakdown)")
    avg_time = compute_task_time_allocation(skills)
    fig_time = plot_task_time_allocation(avg_time)
    if fig_time is not None:
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No `pcttime*` columns (except activelycoding) found.")

    st.markdown("---")
    st.markdown("##### Task group breakdown (aggregate)")
    task_breakdown = plot_task_breakdown(skills)
    if task_breakdown is not None:
        st.plotly_chart(task_breakdown, use_container_width=True)

    st.markdown("---")
    st.markdown("##### % Time Actively Coding (home or work)")
    fig_active_coding = plot_actively_coding_time(fact_filtered)
    if fig_active_coding is not None:
        st.plotly_chart(fig_active_coding, use_container_width=True)
    else:
        st.info("No actively coding time data available.")

    st.markdown("---")
    st.markdown("##### Primary vs. Recommended Programming Language")
    fig_lang_comparison = plot_primary_vs_recommended_lang(fact_filtered)
    if fig_lang_comparison is not None:
        st.plotly_chart(fig_lang_comparison, use_container_width=True)
    else:
        st.info("No programming language comparison data available.")

    st.markdown("---")
    st.markdown("##### Programming languages used (multi-select)")
    lang_fig = plot_programming_languages(skills, top_n=15)
    if lang_fig is not None:
        st.plotly_chart(lang_fig, use_container_width=True)
    else:
        st.info("No `programming_languages_used` column found in skills table.")

    st.markdown("---")
    st.markdown("##### Years Coding for Data Science")
    fig_coding_exp = plot_coding_experience_distribution(fact_filtered)
    if fig_coding_exp is not None:
        st.plotly_chart(fig_coding_exp, use_container_width=True)
    else:
        st.info("No coding experience data available.")

    st.markdown("---")
    st.markdown("##### Years Using ML")
    fig_ml_exp = plot_ml_experience_distribution(fact_filtered)
    if fig_ml_exp is not None:
        st.plotly_chart(fig_ml_exp, use_container_width=True)
    else:
        st.info("No ML experience data available.")

    st.markdown("---")
    st.markdown("##### Do you consider yourself a Data Scientist?")
    fig_ds_identity = plot_data_scientist_identity(fact_filtered)
    if fig_ds_identity is not None:
        st.plotly_chart(fig_ds_identity, use_container_width=True)
    else:
        st.info("No data scientist identity data available.")

    st.markdown("---")
    st.markdown("##### Primary Training Sources (mean % time)")
    fig_training = plot_training_sources(fact_filtered)
    if fig_training is not None:
        st.plotly_chart(fig_training, use_container_width=True)
    else:
        st.info("No training sources data available.")

    st.markdown("---")
    st.markdown("##### Work activities (multi-select)")
    activities_fig = plot_work_activities(skills)
    if activities_fig is not None:
        st.plotly_chart(activities_fig, use_container_width=True)
    else:
        st.info("No work activities data available.")

    st.markdown("---")
    st.markdown("##### Model Success Metrics Used (multi-select)")
    fig_metrics = plot_model_success_metrics(fact_filtered, top_n=12)
    if fig_metrics is not None:
        st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("No model success metrics data available.")

# -------------------------------------------------------------------
# TAB 5 – ML PREDICTOR (Python Recommender Model)
# -------------------------------------------------------------------
with tab5:
    render_ml_predictor_tab()

# Professional Footer
st.markdown(
    """
    <div class="footer-container">
        <p style="font-size: 1.3rem; font-weight: 800; color: white;">
            📊 Kaggle 2018 Data Science Survey Dashboard
        </p>
        <p style="color: rgba(255,255,255,0.85); margin-top: 1rem;">
            Built with Streamlit • Powered by Plotly & Pandas • Interactive Data Visualization Platform
        </p>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.95rem; margin-top: 1.5rem;">
            © 2024 Data Intelligence Platform | Data-Driven Insights for Data Science Professionals
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
