"""
ui/styles.py — Thème CSS glassmorphism financier premium
=========================================================
"""

import streamlit as st

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    /* ---- Base ---- */
    .stApp {
        font-family: 'Fira Sans', sans-serif;
        background-color: #0F172A;
        color: #F8FAFC;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Fira Code', monospace !important;
        color: #F8FAFC !important;
    }
    code, .stCode { font-family: 'Fira Code', monospace !important; }

    /* ---- Global text contrast fixes ---- */
    .stApp p, .stApp li, .stApp span, .stApp div { color: #E2E8F0; }
    .stApp .stMarkdown p { color: #E2E8F0 !important; }
    .stApp .stMarkdown h3 { color: #F8FAFC !important; }
    .stApp label { color: #CBD5E1 !important; }
    .stApp .stCaption, .stApp caption, [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }

    /* Sidebar text */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: #E2E8F0; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] caption { color: #94A3B8 !important; }

    /* Alert boxes contrast */
    .stAlert p, .stAlert span, .stAlert div { color: #F1F5F9 !important; }
    [data-testid="stNotification"] p { color: #F1F5F9 !important; }
    div[data-testid="stNotificationContentInfo"] p { color: #BFDBFE !important; }
    div[data-testid="stNotificationContentWarning"] p { color: #FEF3C7 !important; }
    div[data-testid="stNotificationContentSuccess"] p { color: #D1FAE5 !important; }
    div[data-testid="stNotificationContentError"] p { color: #FEE2E2 !important; }

    /* ---- Streamlit metric override ---- */
    div[data-testid="stMetricDelta"] { color: #94A3B8 !important; }
    div[data-testid="stMetricDelta"] svg { fill: currentColor !important; }

    /* ---- SVG icon helper ---- */
    .icon-inline {
        display: inline-block;
        vertical-align: middle;
        width: 20px;
        height: 20px;
        margin-right: 6px;
    }

    /* ---- Header hero ---- */
    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #0F172A 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(245, 158, 11, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(ellipse, rgba(139, 92, 246, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(ellipse, rgba(245, 158, 11, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        background: linear-gradient(90deg, #F59E0B, #FBBF24, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .hero p {
        color: #94A3B8;
        margin: .5rem 0 0;
        font-size: .95rem;
        letter-spacing: 0.03em;
        position: relative;
        z-index: 1;
    }

    /* ---- Glass Card base ---- */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(245, 158, 11, 0.2);
        box-shadow: 0 4px 24px rgba(245, 158, 11, 0.05);
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(245, 158, 11, 0.2);
        box-shadow: 0 4px 24px rgba(245, 158, 11, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-weight: 500;
        font-family: 'Fira Sans', sans-serif !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700;
        font-family: 'Fira Code', monospace !important;
    }

    /* ---- Verdict card ---- */
    .verdict-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 1.25rem 0;
        position: relative;
        overflow: hidden;
    }
    .verdict-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #F59E0B, #8B5CF6);
        border-radius: 16px 16px 0 0;
    }
    .verdict-card h2 {
        color: #8B5CF6;
        margin: 0 0 .75rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Fira Code', monospace;
    }
    .verdict-card .strategy-name {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(90deg, #F59E0B, #FBBF24, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: .25rem 0;
        font-family: 'Fira Code', monospace;
        letter-spacing: -0.02em;
    }
    .verdict-card p {
        color: #CBD5E1;
        line-height: 1.7;
        margin: .75rem 0 0;
        font-size: 0.95rem;
    }

    /* ---- Metrics grid ---- */
    .fin-metric {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .fin-metric:hover {
        border-color: rgba(245, 158, 11, 0.2);
        box-shadow: 0 4px 24px rgba(245, 158, 11, 0.05);
    }
    .fin-metric .label {
        color: #94A3B8;
        font-size: .75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'Fira Sans', sans-serif;
    }
    .fin-metric .value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: .35rem;
        font-family: 'Fira Code', monospace;
    }
    .fin-metric .green { color: #34D399; }
    .fin-metric .red { color: #F87171; }
    .fin-metric .blue { color: #60A5FA; }
    .fin-metric .amber { color: #FBBF24; }

    /* ---- Greeks card ---- */
    .greeks-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .greeks-container { grid-template-columns: repeat(2, 1fr); }
    }
    .greek-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
        position: relative;
        cursor: default;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .greek-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.08);
    }
    .greek-card:hover .greek-hint {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
        pointer-events: auto;
    }
    .greek-symbol {
        font-family: 'Fira Code', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B5CF6;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: .25rem;
    }
    .greek-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .greek-hint {
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%) translateY(4px);
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 10px;
        padding: .75rem 1rem;
        width: 240px;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease, transform 0.2s ease;
        z-index: 100;
    }
    .greek-hint::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: rgba(139, 92, 246, 0.3);
    }
    .greek-hint-title {
        font-family: 'Fira Code', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #FBBF24;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: .3rem;
    }
    .greek-hint-text {
        font-family: 'Fira Sans', sans-serif;
        font-size: 0.78rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* ---- Dataframe ---- */
    .stDataFrame { border-radius: 14px; overflow: hidden; }

    /* ---- Section headers ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .section-header svg {
        width: 22px;
        height: 22px;
        color: #F59E0B;
        flex-shrink: 0;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #F8FAFC;
        letter-spacing: -0.01em;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #E2E8F0 !important;
        font-weight: 500;
        font-family: 'Fira Sans', sans-serif !important;
    }

    /* ---- Sidebar inputs: dark background + light text ---- */
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="control"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="valueContainer"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="Input"] {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] div[role="combobox"],
    section[data-testid="stSidebar"] [data-baseweb="select"] div[data-baseweb="select"] {
        background-color: #1E293B !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #E2E8F0 !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #94A3B8 !important;
    }
    /* Input fields (number, text) */
    section[data-testid="stSidebar"] input {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    /* Number input wrapper */
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="input"] > div {
        background-color: #1E293B !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    /* +/- buttons inside number input */
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] {
        color: #E2E8F0 !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    /* Secondary buttons (scanner) — dark bg with readable text */
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton button:not([kind="primary"]) {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    section[data-testid="stSidebar"] .stButton button:not([kind="primary"]):hover {
        background-color: #334155 !important;
        border-color: rgba(245, 158, 11, 0.3) !important;
    }

    /* ---- Dividers ---- */
    hr { border-color: rgba(255, 255, 255, 0.06) !important; }

    /* ---- Hide Streamlit chrome (keep sidebar toggle) ---- */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    /* Forcer la visibilité des boutons sidebar expand/collapse */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"],
    div[data-testid="collapsedControl"],
    div[data-testid="collapsedControl"] button {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }
</style>
"""


def inject_css():
    """Injecte le thème CSS glassmorphism dans la page Streamlit."""
    st.markdown(_CSS, unsafe_allow_html=True)
