"""
Options Trading Robo-Advisor
============================
Analyse les données de marché en temps réel et recommande la stratégie
d'options mathématiquement optimale (méthodologie Tastytrade / VRP).

Installation :
    pip install -r requirements.txt

Lancement :
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# ──────────────────────────────────────────────
# 1. CONFIGURATION & THÈME
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Options Robo-Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom — thème glassmorphism financier premium
st.markdown("""
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

    /* ---- Hide Streamlit chrome completely ---- */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. CONSTANTES & TICKERS POPULAIRES
# ──────────────────────────────────────────────

TICKER_GROUPS = {
    "🇺🇸 Index US": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
        "DIA": "Dow Jones", "VTI": "US Total Market",
        "RSP": "S&P 500 Equal Wt", "MDY": "S&P MidCap 400", "IJR": "S&P SmallCap 600",
    },
    "🌍 World": {
        "VT": "FTSE All-World", "VXUS": "International ex-US",
    },
    "🇪🇺 Europe": {
        "VGK": "FTSE Europe", "FEZ": "Euro Stoxx 50",
        "EWG": "Germany (DAX)", "EWU": "UK (FTSE 100)", "EWQ": "France (CAC 40)",
        "EWP": "Spain (IBEX)", "EWI": "Italy (FTSE MIB)", "EWL": "Switzerland (SMI)",
        "EWN": "Netherlands (AEX)", "EWD": "Sweden (OMX)",
    },
    "🌏 Asie-Pacifique": {
        "EWJ": "Japan (Nikkei)", "EWY": "South Korea (KOSPI)",
        "EWA": "Australia (ASX)", "EWH": "Hong Kong (HSI)",
        "EWT": "Taiwan (TAIEX)", "EWS": "Singapore (STI)",
        "INDA": "India (NIFTY)", "FXI": "China Large-Cap", "AAXJ": "Asia ex-Japan",
    },
    "🌎 Amériques (ex-US)": {
        "EWZ": "Brazil (Bovespa)", "EWC": "Canada (TSX)", "EWW": "Mexico (IPC)",
    },
    "🌐 Émergents": {
        "EEM": "Emerging Markets", "KWEB": "China Internet",
    },
    "💻 Tech": {
        "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
        "GOOGL": "Alphabet", "META": "Meta", "NVDA": "NVIDIA", "TSLA": "Tesla",
        "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce",
        "ADBE": "Adobe", "CSCO": "Cisco", "ACN": "Accenture", "IBM": "IBM",
    },
    "🔬 Semiconducteurs": {
        "AMD": "AMD", "INTC": "Intel", "MU": "Micron", "QCOM": "Qualcomm",
        "TSM": "TSMC", "MRVL": "Marvell", "ARM": "Arm Holdings", "SMCI": "Super Micro",
    },
    "🎬 Média": {
        "NFLX": "Netflix", "DIS": "Disney", "CMCSA": "Comcast", "WBD": "Warner Bros",
    },
    "🏦 Finance": {
        "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
        "MS": "Morgan Stanley", "WFC": "Wells Fargo", "C": "Citigroup", "SCHW": "Schwab",
        "V": "Visa", "MA": "Mastercard", "AXP": "Amex", "BLK": "BlackRock", "COF": "Capital One",
    },
    "⛽ Énergie": {
        "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
        "SLB": "Schlumberger", "OXY": "Occidental", "EOG": "EOG Resources",
    },
    "🏥 Santé / Pharma": {
        "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "PFE": "Pfizer",
        "ABBV": "AbbVie", "LLY": "Eli Lilly", "MRK": "Merck", "BMY": "Bristol-Myers",
        "AMGN": "Amgen", "GILD": "Gilead", "TMO": "Thermo Fisher",
        "ABT": "Abbott", "MDT": "Medtronic", "MRNA": "Moderna",
    },
    "🏭 Industrie": {
        "BA": "Boeing", "CAT": "Caterpillar", "DE": "Deere & Co",
        "GE": "GE Aerospace", "HON": "Honeywell", "LMT": "Lockheed Martin",
        "RTX": "RTX / Raytheon", "UPS": "UPS", "FDX": "FedEx", "UNP": "Union Pacific",
    },
    "🛒 Consommation": {
        "HD": "Home Depot", "WMT": "Walmart", "COST": "Costco",
        "TGT": "Target", "NKE": "Nike", "SBUX": "Starbucks", "MCD": "McDonald's",
        "KO": "Coca-Cola", "PEP": "PepsiCo", "PG": "Procter & Gamble",
        "LOW": "Lowe's", "BKNG": "Booking",
    },
    "📡 Télécom": {
        "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile",
    },
    "🚗 Auto & EV": {
        "F": "Ford", "GM": "General Motors", "LCID": "Lucid",
    },
    "🎰 Spéculatif / High-Vol": {
        "COIN": "Coinbase", "PLTR": "Palantir", "SOFI": "SoFi", "RIVN": "Rivian",
        "NIO": "NIO", "MARA": "Marathon Digital", "HOOD": "Robinhood",
        "SNAP": "Snap", "GME": "GameStop", "AMC": "AMC Entertainment",
        "UBER": "Uber", "LYFT": "Lyft", "SHOP": "Shopify", "ROKU": "Roku",
        "RBLX": "Roblox", "DKNG": "DraftKings", "ABNB": "Airbnb",
        "PYPL": "PayPal", "SNOW": "Snowflake", "NET": "Cloudflare",
        "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks", "ZS": "Zscaler",
    },
    "🪙 Matières Premières": {
        "GLD": "Or (Gold)", "SLV": "Argent (Silver)", "PPLT": "Platine",
        "PALL": "Palladium", "USO": "Pétrole brut (WTI)", "UNG": "Gaz naturel",
        "CPER": "Cuivre", "COPX": "Mines de cuivre", "LIT": "Lithium",
        "URA": "Uranium", "DBA": "Agriculture",
    },
    "📈 Obligations": {
        "TLT": "Treasuries 20 ans+", "HYG": "Obligations High Yield",
    },
    "📊 Secteurs ETF": {
        "XLF": "Secteur Finance", "XLE": "Secteur Énergie", "XLK": "Secteur Tech",
        "XLV": "Secteur Santé", "XLI": "Secteur Industrie",
        "XLP": "Conso. de base", "XLY": "Conso. discrétionnaire",
        "XLU": "Secteur Utilities", "XLRE": "Secteur Immobilier",
        "XLC": "Secteur Communication", "SMH": "Semiconducteurs ETF",
        "ARKK": "ARK Innovation", "SOXX": "Semiconducteurs (iShares)",
        "XBI": "Biotech ETF",
    },
}

# ── Lookup tables construits à partir des groupes ──
TICKER_LIST = []
TICKER_NAMES = {}
TICKER_CATEGORY = {}
for _cat, _tickers in TICKER_GROUPS.items():
    for _t, _name in _tickers.items():
        TICKER_LIST.append(_t)
        TICKER_NAMES[_t] = _name
        TICKER_CATEGORY[_t] = _cat

RISK_FREE_RATE = 0.05  # ~taux sans risque approximatif

# ── Mapping ticker → indice de volatilité CBOE spécifique ──
# Fallback : ^VIX si le ticker n'a pas d'indice dédié.
VOL_INDEX_MAP = {
    # S&P 500
    "SPY": "^VIX", "VOO": "^VIX", "IVV": "^VIX", "RSP": "^VIX",
    # Nasdaq 100
    "QQQ": "^VXN", "TQQQ": "^VXN", "SQQQ": "^VXN",
    # Dow Jones
    "DIA": "^VXD",
    # Pétrole / Énergie
    "USO": "^OVX", "XOM": "^OVX", "CVX": "^OVX", "COP": "^OVX",
    "SLB": "^OVX", "OXY": "^OVX", "EOG": "^OVX", "XLE": "^OVX",
    # Or
    "GLD": "^GVZ",
    # Argent
    "SLV": "^VXSLV",
    # Emerging Markets
    "EEM": "^VXEEM", "VWO": "^VXEEM", "IEMG": "^VXEEM",
    # Brésil
    "EWZ": "^VXEWZ",
    # Chine
    "FXI": "^VXFXI", "MCHI": "^VXFXI", "KWEB": "^VXFXI",
    # Europe / EAFE
    "VGK": "^VXEFA", "FEZ": "^VXEFA", "EWG": "^VXEFA", "EWU": "^VXEFA",
    "EWQ": "^VXEFA", "EWP": "^VXEFA", "EWI": "^VXEFA", "EWL": "^VXEFA",
    "EWN": "^VXEFA", "EWD": "^VXEFA", "VXUS": "^VXEFA",
    # Actions individuelles avec vol CBOE dédiée
    "AAPL": "^VXAPL",
    "AMZN": "^VXAZN",
    "GOOGL": "^VXGOG", "GOOG": "^VXGOG",
    "GS": "^VXGS",
    "IBM": "^VXIBM",
}

# Noms lisibles des indices de volatilité
VOL_INDEX_NAMES = {
    "^VIX": "VIX (S&P 500)",
    "^VXN": "VXN (Nasdaq)",
    "^VXD": "VXD (Dow Jones)",
    "^OVX": "OVX (Pétrole)",
    "^GVZ": "GVZ (Or)",
    "^VXSLV": "VXSLV (Argent)",
    "^VXEEM": "VXEEM (Émergents)",
    "^VXEWZ": "VXEWZ (Brésil)",
    "^VXFXI": "VXFXI (Chine)",
    "^VXEFA": "VXEFA (Europe)",
    "^VXAPL": "VXAPL (Apple)",
    "^VXAZN": "VXAZN (Amazon)",
    "^VXGOG": "VXGOG (Google)",
    "^VXGS": "VXGS (Goldman)",
    "^VXIBM": "VXIBM (IBM)",
}


# ──────────────────────────────────────────────
# 3. FONCTIONS BACKEND — MOTEUR DE DONNÉES
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_spot_price(ticker: str) -> float:
    """Récupère le prix actuel (Spot) du ticker."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"Aucune donnée trouvée pour le ticker « {ticker} ».")
    return float(hist["Close"].iloc[-1])


@st.cache_data(ttl=300)
def get_vol_index(ticker: str) -> tuple[float, str]:
    """
    Récupère l'indice de volatilité le plus adapté au ticker.
    Retourne (valeur, symbole_de_l_indice).
    Fallback vers ^VIX si l'indice spécifique n'est pas disponible.
    """
    vol_symbol = VOL_INDEX_MAP.get(ticker, "^VIX")

    # Essai avec l'indice spécifique
    tk = yf.Ticker(vol_symbol)
    hist = tk.history(period="5d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1]), vol_symbol

    # Fallback vers VIX si l'indice spécifique échoue
    if vol_symbol != "^VIX":
        tk = yf.Ticker("^VIX")
        hist = tk.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]), "^VIX"

    raise ValueError("Impossible de récupérer l'indice de volatilité. Le marché est peut-être fermé.")


@st.cache_data(ttl=600)
def compute_iv_rank(ticker: str) -> float:
    """
    Calcule l'IV Rank sur 252 jours.
    Utilise la volatilité historique (écart-type annualisé des rendements)
    comme proxy de l'IV si l'API ne fournit pas l'IV directement.
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y")
    if len(hist) < 30:
        raise ValueError(f"Historique insuffisant pour « {ticker} » (min 30 jours requis).")

    # Calcule la volatilité historique glissante sur 20 jours
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    rolling_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100  # annualisée en %
    rolling_vol = rolling_vol.dropna()

    if rolling_vol.empty:
        return 50.0  # valeur par défaut si calcul impossible

    iv_current = rolling_vol.iloc[-1]
    iv_min = rolling_vol.min()
    iv_max = rolling_vol.max()

    if iv_max == iv_min:
        return 50.0

    iv_rank = 100.0 * (iv_current - iv_min) / (iv_max - iv_min)
    return round(float(np.clip(iv_rank, 0, 100)), 1)


@st.cache_data(ttl=600)
def compute_historical_vol(ticker: str) -> float | None:
    """
    Calcule la volatilité historique réalisée (annualisée) sur 30 jours.
    Retourne None si données insuffisantes.
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo")
    if len(hist) < 30:
        return None
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    sigma_hist = float(log_returns.tail(30).std() * np.sqrt(252))
    return sigma_hist if sigma_hist > 0 else None


def black_scholes_delta(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str) -> float:
    """
    Calcule le Delta d'une option via le modèle de Black-Scholes.
    S = spot, K = strike, T = temps en années, r = taux sans risque,
    sigma = volatilité (décimale), option_type = 'call' ou 'put'.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def black_scholes_price(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str) -> float:
    """Prix théorique Black-Scholes d'une option européenne."""
    if T <= 0 or sigma <= 0:
        return max(0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma : taux de variation du Delta par rapport au sous-jacent."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def black_scholes_theta(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str) -> float:
    """Theta : déclin temporel journalier (en $/jour pour 1 action)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return float(theta / 365)  # par jour


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega : sensibilité à la volatilité (pour 1% de changement d'IV)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100)  # pour 1%


def compute_leg_greeks(leg: dict, S: float, T: float, sigma: float) -> dict:
    """Calcule Delta, Gamma, Theta, Vega et IV pour un leg de la stratégie."""
    K = leg["strike"]
    opt_type = leg["type"].lower()
    sign = 1 if leg["action"] == "BUY" else -1

    delta = black_scholes_delta(S, K, T, RISK_FREE_RATE, sigma, opt_type) * sign
    gamma = black_scholes_gamma(S, K, T, RISK_FREE_RATE, sigma) * sign
    theta = black_scholes_theta(S, K, T, RISK_FREE_RATE, sigma, opt_type) * sign
    vega = black_scholes_vega(S, K, T, RISK_FREE_RATE, sigma) * sign

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "iv": round(sigma * 100, 1),
    }


def simulate_pnl(legs: list, target_spot: float, days_to_target: int,
                 current_sigma: float, qty: int) -> float:
    """
    Simule le P&L théorique de la position à un prix cible et une date cible.
    Utilise Black-Scholes pour recalculer le prix de chaque leg.
    Retourne le P&L en $ (positif = profit, négatif = perte).
    """
    T_target = max(days_to_target, 1) / 365.0

    # Valeur initiale nette (coût d'ouverture)
    initial_value = 0.0
    for leg in legs:
        sign = 1 if leg["action"] == "BUY" else -1
        initial_value += sign * leg["price"]

    # Nouvelle valeur théorique au target_spot et T_target
    new_value = 0.0
    for leg in legs:
        opt_type = leg["type"].lower()
        K = leg["strike"]
        sign = 1 if leg["action"] == "BUY" else -1
        new_price = black_scholes_price(target_spot, K, T_target,
                                        RISK_FREE_RATE, current_sigma, opt_type)
        new_value += sign * new_price

    pnl = (new_value - initial_value) * 100 * qty
    return round(pnl, 2)


def estimate_take_profit_spot(legs: list, spot: float, days_to_target: int,
                              current_sigma: float, qty: int,
                              take_profit_pnl: float) -> float | None:
    """
    Estime le prix du sous-jacent nécessaire pour atteindre le Take Profit.
    Utilise une recherche par balayage puis affinage (bisection).
    Retourne le prix spot estimé ou None si introuvable.
    """
    # Chercher dans les deux directions (hausse et baisse)
    best_spot = None
    best_diff = float("inf")

    # Balayage large : de -20% à +20% par pas de 0.1%
    for pct in range(-200, 201):
        test_spot = spot * (1 + pct / 1000.0)
        pnl = simulate_pnl(legs, test_spot, days_to_target, current_sigma, qty)
        diff = abs(pnl - take_profit_pnl)
        if diff < best_diff:
            best_diff = diff
            best_spot = test_spot

    # Affinage par bisection autour du meilleur candidat
    if best_spot is not None:
        lo = best_spot * 0.995
        hi = best_spot * 1.005
        for _ in range(30):
            mid = (lo + hi) / 2
            pnl_mid = simulate_pnl(legs, mid, days_to_target, current_sigma, qty)
            if pnl_mid < take_profit_pnl:
                # Besoin d'un spot qui rapproche du profit
                pnl_lo = simulate_pnl(legs, lo, days_to_target, current_sigma, qty)
                if pnl_lo < pnl_mid:
                    lo = mid
                else:
                    hi = mid
            else:
                hi = mid
        # Vérifier que le résultat est raisonnable (dans ±20%)
        final_spot = (lo + hi) / 2
        final_pnl = simulate_pnl(legs, final_spot, days_to_target, current_sigma, qty)
        if abs(final_pnl - take_profit_pnl) < take_profit_pnl * 0.1 and abs(final_spot - spot) / spot < 0.25:
            return round(final_spot, 2)

    return None


def compute_real_probabilities(legs: list, spot: float, dte: int,
                                sigma: float, qty: int,
                                take_profit: float, max_risk: float,
                                sigma_move: float | None = None) -> dict:
    """
    Calcule les probabilités réelles via intégration numérique sur la
    distribution log-normale (GBM), en évaluant le P&L au **time-stop**
    (21 jours avant l'expiration) via Black-Scholes.

    Méthode :
      1. Le sous-jacent évolue pendant `holding_days = dte - 21` jours,
         avec `sigma_move` (vol. historique réalisée).
      2. Le P&L est évalué avec les prix BS à 21 DTE restants,
         en utilisant `sigma` (vol. implicite de la chaîne).
      3. Intégration sur 500 points z ∈ [-4σ, +4σ].

    Retourne :
      - p_take_profit : P(P&L ≥ take_profit) au time-stop
      - p_breakeven   : P(P&L ≥ 0) au time-stop
      - p_max_loss    : P(P&L ≤ -95% du max_risk) au time-stop
      - expected_pnl  : EV = ∫ P&L(S_T) × f(S_T) dS_T
    """
    if sigma_move is None:
        sigma_move = sigma  # fallback: même vol pour mouvement et pricing

    holding_days = max(1, dte - 21)
    remaining_dte = min(21, dte)
    T_holding = holding_days / 365.0

    # Paramètres GBM : mouvement du sous-jacent avec vol historique
    drift = (RISK_FREE_RATE - 0.5 * sigma_move**2) * T_holding
    vol = sigma_move * np.sqrt(T_holding)

    # Intégration numérique : 500 points sur [-4σ, +4σ] (99.99%)
    n_points = 500
    z_values = np.linspace(-4, 4, n_points)
    dz = z_values[1] - z_values[0]

    p_take_profit = 0.0
    p_breakeven = 0.0
    p_max_loss = 0.0
    expected_pnl = 0.0  # EV = ∫ P&L(S_T) × f(S_T) dS_T

    for z in z_values:
        s_t = spot * np.exp(drift + vol * z)
        prob = norm.pdf(z) * dz
        # P&L évalué avec sigma (IV) pour le pricing BS des options
        pnl = simulate_pnl(legs, s_t, remaining_dte, sigma, qty)

        expected_pnl += pnl * prob
        if pnl >= take_profit:
            p_take_profit += prob
        if pnl >= 0:
            p_breakeven += prob
        if pnl <= -max_risk * 0.95:
            p_max_loss += prob

    p_tp_pct = round(max(0.1, min(99.9, p_take_profit * 100)), 1)
    p_be_pct = round(max(0.1, min(99.9, p_breakeven * 100)), 1)
    p_ml_pct = round(max(0.1, min(99.9, p_max_loss * 100)), 1)
    p_partial_loss_pct = round(max(0.0, 100.0 - p_be_pct - p_ml_pct), 1)

    return {
        "p_take_profit": p_tp_pct,
        "p_breakeven": p_be_pct,
        "p_partial_loss": p_partial_loss_pct,
        "p_max_loss": p_ml_pct,
        "expected_pnl": round(expected_pnl, 2),
    }


@st.cache_data(ttl=300)
def get_options_chain(ticker: str):
    """
    Récupère la chaîne d'options et filtre l'expiration la plus proche
    de 45 DTE (fourchette 35-60 jours).
    Retourne (expiration_date_str, calls_df, puts_df, dte).
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        raise ValueError(f"Aucune chaîne d'options disponible pour « {ticker} ».")

    today = dt.date.today()
    best_exp = None
    best_dte = None
    best_diff = float("inf")

    for exp_str in expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        diff = abs(dte - 45)
        if 35 <= dte <= 60 and diff < best_diff:
            best_diff = diff
            best_exp = exp_str
            best_dte = dte

    # Si rien dans [35,60], prend l'expiration la plus proche de 45 DTE
    if best_exp is None:
        for exp_str in expirations:
            exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte > 0:
                diff = abs(dte - 45)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
                    best_dte = dte

    if best_exp is None:
        raise ValueError("Aucune expiration d'options valide trouvée.")

    chain = tk.option_chain(best_exp)
    return best_exp, chain.calls, chain.puts, best_dte


def get_leaps_chain(ticker: str):
    """
    Récupère la chaîne d'options LEAPS (> 200 DTE) pour les stratégies
    d'achat de temps (PMCC).
    Retourne (expiration_date_str, calls_df, puts_df, dte) ou None.
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options
    today = dt.date.today()
    best_exp = None
    best_dte = None
    best_diff = float("inf")

    for exp_str in expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte > 200:
            diff = abs(dte - 365)  # cible ~1 an
            if diff < best_diff:
                best_diff = diff
                best_exp = exp_str
                best_dte = dte

    if best_exp is None:
        return None

    chain = tk.option_chain(best_exp)
    return best_exp, chain.calls, chain.puts, best_dte


def get_short_term_chain(ticker: str):
    """
    Récupère la chaîne d'options court terme (~20 DTE)
    pour les Calendar Spreads.
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options
    today = dt.date.today()
    best_exp = None
    best_dte = None
    best_diff = float("inf")

    for exp_str in expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte > 5:
            diff = abs(dte - 20)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp_str
                best_dte = dte

    if best_exp is None:
        return None

    chain = tk.option_chain(best_exp)
    return best_exp, chain.calls, chain.puts, best_dte


# ──────────────────────────────────────────────
# 4. FONCTIONS HELPERS — SÉLECTION DE STRIKES
# ──────────────────────────────────────────────

def find_strike_by_delta(options_df: pd.DataFrame, S: float, T: float,
                         sigma: float, target_delta: float,
                         option_type: str) -> pd.Series | None:
    """
    Trouve le strike dont le Delta est le plus proche de target_delta.
    Retourne la ligne du DataFrame correspondante.
    """
    if options_df.empty:
        return None

    deltas = []
    for _, row in options_df.iterrows():
        K = float(row["strike"])
        d = black_scholes_delta(S, K, T, RISK_FREE_RATE, sigma, option_type)
        deltas.append(abs(d))

    options_df = options_df.copy()
    options_df["abs_delta"] = deltas

    target_abs = abs(target_delta)
    idx = (options_df["abs_delta"] - target_abs).abs().idxmin()
    return options_df.loc[idx]


def get_mid_price(row: pd.Series) -> float:
    """
    Retourne le prix moyen (bid+ask)/2.
    RISK MANAGER : si le carnet d'ordres est vide (bid=0 ou ask=0),
    le contrat est considéré comme illiquide/mort → prix 0.
    """
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    if bid <= 0 or ask <= 0:
        return 0.0
    return round((bid + ask) / 2, 2)


def estimate_sigma(options_df: pd.DataFrame, S: float) -> float:
    """
    Estime la volatilité implicite moyenne à partir des IV de la chaîne.
    Fallback à 0.25 si indisponible.
    """
    if "impliedVolatility" in options_df.columns:
        ivs = options_df["impliedVolatility"].dropna()
        ivs = ivs[ivs > 0]
        if not ivs.empty:
            return float(ivs.median())
    return 0.25


# ──────────────────────────────────────────────
# 5. MOTEUR DE STRATÉGIE — MATRICE DE DÉCISION
# ──────────────────────────────────────────────

def filter_liquid_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les options illiquides de la chaîne.
    Exclut :
      - bid == 0 (pas de marché réel)
      - openInterest < 10 (pas de participation)
      - spread bid/ask > 40% du mid-price (illiquidité excessive)
    """
    if df.empty:
        return df
    filtered = df.copy()
    # Exclure bid == 0
    filtered = filtered[filtered["bid"] > 0]
    # Exclure open interest trop faible
    if "openInterest" in filtered.columns:
        filtered = filtered[filtered["openInterest"] >= 10]
    # Exclure spread bid/ask excessif
    mid = (filtered["bid"] + filtered["ask"]) / 2
    spread_pct = (filtered["ask"] - filtered["bid"]) / mid
    filtered = filtered[spread_pct <= 0.40]
    return filtered.reset_index(drop=True)


def build_strategy(spot: float, vix: float, iv_rank: float, bias: str,
                   budget: float, ticker: str, vol_symbol: str = "^VIX"):
    """
    Moteur principal. Sélectionne et construit la stratégie optimale.
    Retourne un dict avec : name, explanation, legs, metrics, exit_plan.
    """

    # --- Récupération de la chaîne d'options ~45 DTE ---
    exp_str, calls, puts, dte = get_options_chain(ticker)

    # --- RISK MANAGER : Filtre de liquidité ---
    calls = filter_liquid_options(calls)
    puts = filter_liquid_options(puts)
    if len(calls) < 3 or len(puts) < 3:
        # Détection horaires de marché US (NYSE : 9h30-16h00 ET)
        import zoneinfo
        now_local = dt.datetime.now().astimezone()
        try:
            now_et = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        except Exception:
            now_et = now_local  # fallback
        market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_weekday = now_et.weekday() < 5
        is_market_open = is_weekday and market_open_et <= now_et <= market_close_et

        market_hint = ""
        if not is_market_open:
            # Calcul de la prochaine ouverture en heure locale
            next_open_et = market_open_et
            if now_et >= market_close_et or not is_weekday:
                # Avancer au prochain jour ouvré
                days_ahead = 1
                next_day = now_et + dt.timedelta(days=days_ahead)
                while next_day.weekday() >= 5:  # skip weekend
                    days_ahead += 1
                    next_day = now_et + dt.timedelta(days=days_ahead)
                next_open_et = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            # Convertir en heure locale
            next_open_local = next_open_et.astimezone(now_local.tzinfo)
            market_hint = (
                f" ⏰ Le marché US (NYSE) est actuellement fermé. "
                f"Les données bid/ask sont indisponibles hors séance. "
                f"Prochaine ouverture : {next_open_local.strftime('%d/%m à %Hh%M')} (heure locale)."
            )

        raise ValueError(
            f"Options trop illiquides sur « {ticker} ». "
            f"Après filtrage (bid>0, OI≥10, spread≤40%), "
            f"il ne reste que {len(puts)} puts et {len(calls)} calls. "
            f"Minimum requis : 3 de chaque côté."
            f"{market_hint}"
        )

    # --- RISK MANAGER : Filtre Anti-Penny Stocks ---
    if spot < 10.0:
        raise ValueError(
            f"Le prix de l'action ({spot:.2f}\\$) est trop bas. "
            f"Les options sur les Penny Stocks offrent un ratio risque/gain désastreux. "
            f"Analyse rejetée par le Risk Manager."
        )

    T = dte / 365.0
    sigma = estimate_sigma(pd.concat([calls, puts]), spot)

    result = {
        "name": "",
        "explanation": "",
        "legs": [],
        "credit_or_debit": 0.0,
        "max_risk": 0.0,
        "max_profit": 0.0,
        "pop": 0.0,
        "exit_plan": {},
        "expiration": exp_str,
        "dte": dte,
    }

    # =============================================
    # CAS A : Volatilité Élevée — VENTE DE PRIME
    # =============================================
    if iv_rank > 50 or vix > 20:

        if bias == "Neutre":
            # ---- Iron Condor ----
            result["name"] = "🦅 Iron Condor"
            result["explanation"] = (
                "La volatilité implicite est élevée (IV Rank {:.0f}%, {} {:.1f}), "
                "ce qui gonfle artificiellement les primes d'options. "
                "L'Iron Condor vend cette prime excessive des deux côtés du marché, "
                "capturant le retour statistique à la moyenne de la volatilité."
            ).format(iv_rank, VOL_INDEX_NAMES.get(vol_symbol, "VIX"), vix)

            # Vente Call/Put à ~0.16 Delta
            sell_put = find_strike_by_delta(puts, spot, T, sigma, -0.16, "put")
            sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.16, "call")

            if sell_put is None or sell_call is None:
                raise ValueError("Impossible de trouver les strikes appropriés dans la chaîne d'options.")

            sell_put_strike = float(sell_put["strike"])
            sell_call_strike = float(sell_call["strike"])

            # --- SYMÉTRIE DES STRIKES ---
            # Forcer une distance OTM égale des deux côtés du spot
            dist_put = spot - sell_put_strike   # distance put (positive)
            dist_call = sell_call_strike - spot  # distance call (positive)
            sym_dist = min(dist_put, dist_call)  # prendre la + petite

            # Recalculer les strikes symétriques
            sym_put_target = spot - sym_dist
            sym_call_target = spot + sym_dist

            # Snapper sur les strikes disponibles les + proches
            put_strikes_all = sorted(puts["strike"].unique())
            call_strikes_all = sorted(calls["strike"].unique())

            sell_put_candidates = [s for s in put_strikes_all if s < spot]
            sell_call_candidates = [s for s in call_strikes_all if s > spot]

            if sell_put_candidates and sell_call_candidates:
                sell_put_strike = min(sell_put_candidates, key=lambda x: abs(x - sym_put_target))
                sell_call_strike = min(sell_call_candidates, key=lambda x: abs(x - sym_call_target))
                # Mettre à jour les rows pour les prix
                sell_put_row = puts[puts["strike"] == sell_put_strike]
                sell_call_row = calls[calls["strike"] == sell_call_strike]
                if not sell_put_row.empty:
                    sell_put = sell_put_row.iloc[0]
                if not sell_call_row.empty:
                    sell_call = sell_call_row.iloc[0]

            # Largeur standardisée (~1.5% du spot, min 5$)
            target_width = max(1.0, round(spot * 0.015))
            put_strikes = sorted(puts["strike"].unique())
            call_strikes = sorted(calls["strike"].unique())

            # Protection put : strike le + proche de sell_put - target_width
            buy_put_target = sell_put_strike - target_width
            candidates_put = [s for s in put_strikes if s < sell_put_strike]
            if not candidates_put:
                raise ValueError("Pas de strikes de protection disponibles pour le Put side de l'Iron Condor.")
            buy_put_strike = min(candidates_put, key=lambda x: abs(x - buy_put_target))

            # Protection call : strike le + proche de sell_call + target_width
            buy_call_target = sell_call_strike + target_width
            candidates_call = [s for s in call_strikes if s > sell_call_strike]
            if not candidates_call:
                raise ValueError("Pas de strikes de protection disponibles pour le Call side de l'Iron Condor.")
            buy_call_strike = min(candidates_call, key=lambda x: abs(x - buy_call_target))

            sell_put_price = get_mid_price(sell_put)
            sell_call_price = get_mid_price(sell_call)

            buy_put_row = puts[puts["strike"] == buy_put_strike]
            buy_call_row = calls[calls["strike"] == buy_call_strike]
            buy_put_price = get_mid_price(buy_put_row.iloc[0]) if not buy_put_row.empty else 0.0
            buy_call_price = get_mid_price(buy_call_row.iloc[0]) if not buy_call_row.empty else 0.0

            net_credit = (sell_put_price + sell_call_price) - (buy_put_price + buy_call_price)
            put_width = sell_put_strike - buy_put_strike
            call_width = buy_call_strike - sell_call_strike
            max_width = max(put_width, call_width)

            # RISK MANAGER : Sanity check crédit — prix physiquement cohérents
            if net_credit <= 0 or net_credit >= max_width:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = (max_width * 100) - (net_credit * 100)

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\$) pour un Iron Condor standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\$."
                )

            result["legs"] = [
                {"action": "SELL", "type": "Put", "strike": sell_put_strike,
                 "exp": exp_str, "dte": dte, "price": sell_put_price},
                {"action": "BUY", "type": "Put", "strike": buy_put_strike,
                 "exp": exp_str, "dte": dte, "price": buy_put_price},
                {"action": "SELL", "type": "Call", "strike": sell_call_strike,
                 "exp": exp_str, "dte": dte, "price": sell_call_price},
                {"action": "BUY", "type": "Call", "strike": buy_call_strike,
                 "exp": exp_str, "dte": dte, "price": buy_call_price},
            ]
            max_profit = net_credit * 100
            result["credit_or_debit"] = round(max_profit, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max_profit, 2)


        elif bias == "Haussier":
            # ---- Bull Put Spread ----
            result["name"] = "🐂 Bull Put Spread"
            result["explanation"] = (
                "La volatilité élevée (IV Rank {:.0f}%) offre des primes gonflées sur les puts. "
                "Ce spread haussier vend un put OTM et achète une protection, profitant du biais "
                "directionnel tout en collectant une prime statistiquement avantageuse."
            ).format(iv_rank)

            sell_put = find_strike_by_delta(puts, spot, T, sigma, -0.20, "put")
            if sell_put is None:
                raise ValueError("Impossible de trouver le strike approprié.")
            sell_put_strike = float(sell_put["strike"])
            sell_put_price = get_mid_price(sell_put)

            # Largeur standardisée (~1.5% du spot, min 5$)
            target_width = max(1.0, round(spot * 0.015))
            put_strikes = sorted([s for s in puts["strike"].unique() if s < sell_put_strike])
            if not put_strikes:
                raise ValueError("Pas de strikes de protection disponibles pour le Bull Put Spread.")

            buy_put_target = sell_put_strike - target_width
            buy_put_strike = min(put_strikes, key=lambda x: abs(x - buy_put_target))
            buy_put_row = puts[puts["strike"] == buy_put_strike]
            buy_put_price = get_mid_price(buy_put_row.iloc[0]) if not buy_put_row.empty else 0.0

            net_credit = sell_put_price - buy_put_price
            width = sell_put_strike - buy_put_strike

            # RISK MANAGER : Sanity check crédit — prix physiquement cohérents
            if net_credit <= 0 or net_credit >= width:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_profit = net_credit * 100
            max_risk = (width * 100) - max_profit

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\$) pour un Bull Put Spread standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\$."
                )

            result["legs"] = [
                {"action": "SELL", "type": "Put", "strike": sell_put_strike,
                 "exp": exp_str, "dte": dte, "price": sell_put_price},
                {"action": "BUY", "type": "Put", "strike": buy_put_strike,
                 "exp": exp_str, "dte": dte, "price": buy_put_price},
            ]
            result["credit_or_debit"] = round(max_profit, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max_profit, 2)


        else:  # Baissier
            # ---- Bear Call Spread ----
            result["name"] = "🐻 Bear Call Spread"
            result["explanation"] = (
                "La volatilité élevée (IV Rank {:.0f}%) rend les calls OTM anormalement chers. "
                "Ce spread baissier vend cette prime excessive tout en limitant le risque "
                "grâce à la protection supérieure."
            ).format(iv_rank)

            sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.20, "call")
            if sell_call is None:
                raise ValueError("Impossible de trouver le strike approprié.")
            sell_call_strike = float(sell_call["strike"])
            sell_call_price = get_mid_price(sell_call)

            # Largeur standardisée (~1.5% du spot, min 5$)
            target_width = max(1.0, round(spot * 0.015))
            call_strikes = sorted([s for s in calls["strike"].unique() if s > sell_call_strike])
            if not call_strikes:
                raise ValueError("Pas de strikes de protection disponibles pour le Bear Call Spread.")

            buy_call_target = sell_call_strike + target_width
            buy_call_strike = min(call_strikes, key=lambda x: abs(x - buy_call_target))
            buy_call_row = calls[calls["strike"] == buy_call_strike]
            buy_call_price = get_mid_price(buy_call_row.iloc[0]) if not buy_call_row.empty else 0.0

            net_credit = sell_call_price - buy_call_price
            width = buy_call_strike - sell_call_strike

            # RISK MANAGER : Sanity check crédit — prix physiquement cohérents
            if net_credit <= 0 or net_credit >= width:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_profit = net_credit * 100
            max_risk = (width * 100) - max_profit

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\$) pour un Bear Call Spread standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\$."
                )

            result["legs"] = [
                {"action": "SELL", "type": "Call", "strike": sell_call_strike,
                 "exp": exp_str, "dte": dte, "price": sell_call_price},
                {"action": "BUY", "type": "Call", "strike": buy_call_strike,
                 "exp": exp_str, "dte": dte, "price": buy_call_price},
            ]
            result["credit_or_debit"] = round(max_profit, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max_profit, 2)


    # =============================================
    # CAS B : Volatilité Faible — ACHAT DE TEMPS
    # =============================================
    elif iv_rank < 20 and vix < 15:

        if bias == "Haussier":
            # ---- PMCC (Poor Man's Covered Call) ----
            result["name"] = "📈 PMCC (Diagonal Spread)"
            result["explanation"] = (
                "La volatilité est historiquement basse (IV Rank {:.0f}%, {} {:.1f}). "
                "Le PMCC reproduit une covered call à moindre coût : achat d'un LEAPS deep ITM "
                "et vente d'un call court terme pour générer du revenu récurrent."
            ).format(iv_rank, VOL_INDEX_NAMES.get(vol_symbol, "VIX"), vix)

            leaps = get_leaps_chain(ticker)
            if leaps is None:
                raise ValueError("Pas d'options LEAPS disponibles (>200 DTE) pour le PMCC.")
            leaps_exp, leaps_calls, _, leaps_dte = leaps

            sigma_leaps = estimate_sigma(leaps_calls, spot)
            leaps_T = leaps_dte / 365.0

            # Achat LEAPS deep ITM (Delta > 0.80)
            buy_call = find_strike_by_delta(leaps_calls, spot, leaps_T, sigma_leaps, 0.80, "call")
            if buy_call is None:
                raise ValueError("Impossible de trouver un LEAPS approprié.")
            buy_call_strike = float(buy_call["strike"])
            buy_call_price = get_mid_price(buy_call)

            # Vente Call court terme (~0.30 delta)
            sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.30, "call")
            if sell_call is None:
                raise ValueError("Impossible de trouver le call court terme.")
            sell_call_strike = float(sell_call["strike"])
            sell_call_price = get_mid_price(sell_call)

            net_debit = buy_call_price - sell_call_price

            # RISK MANAGER : Sanity check débit — la jambe achetée doit coûter plus
            if net_debit <= 0:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = net_debit * 100

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\$) pour le PMCC. "
                    f"Débit net estimé : {max_risk:.0f}\$."
                )

            max_profit = (sell_call_strike - buy_call_strike - net_debit) * 100

            result["legs"] = [
                {"action": "BUY", "type": "Call", "strike": buy_call_strike,
                 "exp": leaps_exp, "dte": leaps_dte, "price": buy_call_price},
                {"action": "SELL", "type": "Call", "strike": sell_call_strike,
                 "exp": exp_str, "dte": dte, "price": sell_call_price},
            ]
            result["credit_or_debit"] = round(-net_debit * 100, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max(max_profit, 0), 2)


        elif bias == "Neutre":
            # ---- Calendar Spread ----
            result["name"] = "📅 Calendar Spread"
            result["explanation"] = (
                "Volatilité basse (IV Rank {:.0f}%). Le Calendar Spread profite de l'accélération "
                "du déclin temporel (Theta) sur la jambe courte vendue, tout en conservant la "
                "valeur de la jambe longue achetée."
            ).format(iv_rank)

            short_chain = get_short_term_chain(ticker)
            if short_chain is None:
                raise ValueError("Pas d'expiration court terme disponible pour le Calendar Spread.")
            short_exp, short_calls, _, short_dte = short_chain

            # Strike ATM
            atm_strike = min(calls["strike"], key=lambda x: abs(x - spot))

            # Vente court (~20 DTE)
            short_row = short_calls[short_calls["strike"] == atm_strike]
            if short_row.empty:
                short_row = short_calls.iloc[(short_calls["strike"] - atm_strike).abs().argsort()[:1]]
                atm_strike = float(short_row["strike"].iloc[0])
            sell_price = get_mid_price(short_row.iloc[0])

            # Achat long (~45-60 DTE)
            long_row = calls[calls["strike"] == atm_strike]
            if long_row.empty:
                long_row = calls.iloc[(calls["strike"] - atm_strike).abs().argsort()[:1]]
            buy_price = get_mid_price(long_row.iloc[0])

            net_debit = buy_price - sell_price

            # RISK MANAGER : Sanity check débit — la jambe achetée doit coûter plus
            if net_debit <= 0:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = net_debit * 100

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\$) pour le Calendar Spread. "
                    f"Débit net estimé : {max_risk:.0f}\$."
                )

            result["legs"] = [
                {"action": "BUY", "type": "Call", "strike": atm_strike,
                 "exp": exp_str, "dte": dte, "price": buy_price},
                {"action": "SELL", "type": "Call", "strike": atm_strike,
                 "exp": short_exp, "dte": short_dte, "price": sell_price},
            ]
            result["credit_or_debit"] = round(-net_debit * 100, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max_risk * 0.5, 2)  # estimation


        else:  # Baissier en basse vol
            # Fallback vers un Bear Put Spread (débit)
            result["name"] = "🐻 Bear Put Spread (Débit)"
            result["explanation"] = (
                "Volatilité basse avec biais baissier. Un Bear Put Spread en débit permet "
                "de profiter d'une baisse tout en limitant le risque au débit payé."
            )

            buy_put = find_strike_by_delta(puts, spot, T, sigma, -0.45, "put")
            if buy_put is None:
                raise ValueError("Impossible de construire le Bear Put Spread.")
            buy_put_strike = float(buy_put["strike"])
            buy_put_price = get_mid_price(buy_put)

            # Largeur standardisée (~1.5% du spot, min 5$)
            target_width = max(1.0, round(spot * 0.015))
            put_strikes = sorted([s for s in puts["strike"].unique() if s < buy_put_strike])
            if not put_strikes:
                raise ValueError("Pas de strikes de protection disponibles pour le Bear Put Spread.")

            sell_put_target = buy_put_strike - target_width
            sell_put_strike = min(put_strikes, key=lambda x: abs(x - sell_put_target))
            sell_put_row = puts[puts["strike"] == sell_put_strike]
            sell_put_price = get_mid_price(sell_put_row.iloc[0]) if not sell_put_row.empty else 0.0

            net_debit = buy_put_price - sell_put_price
            width = buy_put_strike - sell_put_strike

            # RISK MANAGER : Sanity check débit — prix physiquement cohérents
            if net_debit <= 0 or net_debit >= width:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = net_debit * 100
            max_profit = (width * 100) - max_risk

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\$) pour un Bear Put Spread standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\$."
                )

            result["legs"] = [
                {"action": "BUY", "type": "Put", "strike": buy_put_strike,
                 "exp": exp_str, "dte": dte, "price": buy_put_price},
                {"action": "SELL", "type": "Put", "strike": sell_put_strike,
                 "exp": exp_str, "dte": dte, "price": sell_put_price},
            ]
            result["credit_or_debit"] = round(-max_risk, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max_profit, 2)


    # =============================================
    # CAS C : Volatilité Moyenne — CLASSIQUE / WHEEL
    # =============================================
    else:
        can_wheel = budget >= spot * 100

        if can_wheel and bias != "Baissier":
            # ---- Cash Secured Put (The Wheel) ----
            result["name"] = "🎡 Cash Secured Put (The Wheel)"
            result["explanation"] = (
                "Volatilité moyenne (IV Rank {:.0f}%). Votre budget ({:,.0f}$) couvre l'achat de "
                "100 actions. La stratégie Wheel vend un put sécurisé par cash : soit vous collectez "
                "la prime, soit vous achetez l'action à un prix réduit."
            ).format(iv_rank, budget)

            sell_put = find_strike_by_delta(puts, spot, T, sigma, -0.25, "put")
            if sell_put is None:
                raise ValueError("Impossible de trouver le strike approprié.")
            sell_put_strike = float(sell_put["strike"])
            sell_put_price = get_mid_price(sell_put)

            max_risk = (sell_put_strike * 100) - (sell_put_price * 100)
            if max_risk > budget:
                # Baisser le strike
                lower_puts = puts[puts["strike"] * 100 - sell_put_price * 100 <= budget]
                if lower_puts.empty:
                    raise ValueError(f"Budget insuffisant ({budget}\$) pour un Cash Secured Put sur {ticker}.")
                sell_put = lower_puts.iloc[(lower_puts["strike"] - (budget / 100)).abs().argsort()[:1]].iloc[0]
                sell_put_strike = float(sell_put["strike"])
                sell_put_price = get_mid_price(sell_put)
                max_risk = (sell_put_strike * 100) - (sell_put_price * 100)

            result["legs"] = [
                {"action": "SELL", "type": "Put", "strike": sell_put_strike,
                 "exp": exp_str, "dte": dte, "price": sell_put_price},
            ]
            result["credit_or_debit"] = round(sell_put_price * 100, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(sell_put_price * 100, 2)


        else:
            # ---- Spread Directionnel Classique ----
            if bias == "Haussier":
                result["name"] = "📊 Bull Call Spread"
                result["explanation"] = (
                    "Volatilité moyenne (IV Rank {:.0f}%). Un spread d'achat haussier en débit "
                    "offre un profil risque/rendement défini avec un biais long."
                ).format(iv_rank)

                buy_call = find_strike_by_delta(calls, spot, T, sigma, 0.50, "call")
                if buy_call is None:
                    raise ValueError("Impossible de construire le Bull Call Spread.")
                buy_call_strike = float(buy_call["strike"])
                buy_call_price = get_mid_price(buy_call)

                # Largeur standardisée (~1.5% du spot, min 5$)
                target_width = max(1.0, round(spot * 0.015))
                call_strikes = sorted([s for s in calls["strike"].unique() if s > buy_call_strike])
                if not call_strikes:
                    raise ValueError("Pas de strikes de protection disponibles pour le Bull Call Spread.")

                sell_call_target = buy_call_strike + target_width
                sell_call_strike = min(call_strikes, key=lambda x: abs(x - sell_call_target))
                sell_call_row = calls[calls["strike"] == sell_call_strike]
                sell_call_price = get_mid_price(sell_call_row.iloc[0]) if not sell_call_row.empty else 0.0

                net_debit = buy_call_price - sell_call_price
                width = sell_call_strike - buy_call_strike

                # RISK MANAGER : Sanity check débit — prix physiquement cohérents
                if net_debit <= 0 or net_debit >= width:
                    raise ValueError(
                        "Les prix de la chaîne d'options sont illogiques "
                        "(illiquidité majeure ou bid/ask cassé). "
                        "Analyse annulée pour votre sécurité."
                    )

                max_risk = net_debit * 100
                max_profit = (width * 100) - max_risk

                if max_risk > budget:
                    raise ValueError(
                        f"Budget insuffisant ({budget}\$) pour un Bull Call Spread standard sur {ticker}. "
                        f"Risque par contrat : {max_risk:.0f}\$."
                    )

                result["legs"] = [
                    {"action": "BUY", "type": "Call", "strike": buy_call_strike,
                     "exp": exp_str, "dte": dte, "price": buy_call_price},
                    {"action": "SELL", "type": "Call", "strike": sell_call_strike,
                     "exp": exp_str, "dte": dte, "price": sell_call_price},
                ]
                result["credit_or_debit"] = round(-max_risk, 2)
                result["max_risk"] = round(max_risk, 2)
                result["max_profit"] = round(max_profit, 2)


            elif bias == "Baissier":
                # ---- Bear Put Spread ----
                result["name"] = "📊 Bear Put Spread"
                result["explanation"] = (
                    "Volatilité moyenne (IV Rank {:.0f}%). Un spread baissier en débit "
                    "profite de la baisse anticipée tout en définissant un risque maximal strict."
                ).format(iv_rank)

                buy_put = find_strike_by_delta(puts, spot, T, sigma, -0.50, "put")
                if buy_put is None:
                    raise ValueError("Impossible de construire le Bear Put Spread.")
                buy_put_strike = float(buy_put["strike"])
                buy_put_price = get_mid_price(buy_put)

                # Largeur standardisée (~1.5% du spot, min 1$)
                target_width = max(1.0, round(spot * 0.015))
                put_strikes = sorted([s for s in puts["strike"].unique() if s < buy_put_strike])
                if not put_strikes:
                    raise ValueError("Pas de strikes de protection disponibles pour le Bear Put Spread.")

                sell_put_target = buy_put_strike - target_width
                sell_put_strike = min(put_strikes, key=lambda x: abs(x - sell_put_target))
                sell_put_row = puts[puts["strike"] == sell_put_strike]
                sell_put_price = get_mid_price(sell_put_row.iloc[0]) if not sell_put_row.empty else 0.0

                net_debit = buy_put_price - sell_put_price
                width = buy_put_strike - sell_put_strike

                # RISK MANAGER : Sanity check débit — prix physiquement cohérents
                if net_debit <= 0 or net_debit >= width:
                    raise ValueError(
                        "Les prix de la chaîne d'options sont illogiques "
                        "(illiquidité majeure ou bid/ask cassé). "
                        "Analyse annulée pour votre sécurité."
                    )

                max_risk = net_debit * 100
                max_profit = (width * 100) - max_risk

                if max_risk > budget:
                    raise ValueError(
                        f"Budget insuffisant ({budget}\$) pour un Bear Put Spread standard sur {ticker}. "
                        f"Risque par contrat : {max_risk:.0f}\$."
                    )

                result["legs"] = [
                    {"action": "BUY", "type": "Put", "strike": buy_put_strike,
                     "exp": exp_str, "dte": dte, "price": buy_put_price},
                    {"action": "SELL", "type": "Put", "strike": sell_put_strike,
                     "exp": exp_str, "dte": dte, "price": sell_put_price},
                ]
                result["credit_or_debit"] = round(-max_risk, 2)
                result["max_risk"] = round(max_risk, 2)
                result["max_profit"] = round(max_profit, 2)


            else:  # Neutre sans budget Wheel
                # ---- Iron Condor (Volatilité Moyenne, Neutre) ----
                result["name"] = "🦅 Iron Condor"
                result["explanation"] = (
                    "Volatilité moyenne et biais neutre. L'Iron Condor encaisse l'érosion "
                    "du temps des deux côtés en pariant sur une stagnation du prix dans un "
                    "canal défini. Le crédit collecté est le profit maximum si le sous-jacent "
                    "reste entre les strikes vendus à l'expiration."
                )

                # Vente Call/Put à ~0.16 Delta
                sell_put = find_strike_by_delta(puts, spot, T, sigma, -0.16, "put")
                sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.16, "call")

                if sell_put is None or sell_call is None:
                    raise ValueError("Impossible de trouver les strikes appropriés pour l'Iron Condor.")

                sell_put_strike = float(sell_put["strike"])
                sell_call_strike = float(sell_call["strike"])

                # --- SYMÉTRIE DES STRIKES ---
                dist_put = spot - sell_put_strike
                dist_call = sell_call_strike - spot
                sym_dist = min(dist_put, dist_call)

                sym_put_target = spot - sym_dist
                sym_call_target = spot + sym_dist

                put_strikes_all = sorted(puts["strike"].unique())
                call_strikes_all = sorted(calls["strike"].unique())
                sell_put_candidates = [s for s in put_strikes_all if s < spot]
                sell_call_candidates = [s for s in call_strikes_all if s > spot]

                if sell_put_candidates and sell_call_candidates:
                    sell_put_strike = min(sell_put_candidates, key=lambda x: abs(x - sym_put_target))
                    sell_call_strike = min(sell_call_candidates, key=lambda x: abs(x - sym_call_target))
                    sell_put_row = puts[puts["strike"] == sell_put_strike]
                    sell_call_row = calls[calls["strike"] == sell_call_strike]
                    if not sell_put_row.empty:
                        sell_put = sell_put_row.iloc[0]
                    if not sell_call_row.empty:
                        sell_call = sell_call_row.iloc[0]

                # Largeur standardisée (~1.5% du spot, min 1$)
                target_width = max(1.0, round(spot * 0.015))
                put_strikes = sorted(puts["strike"].unique())
                call_strikes = sorted(calls["strike"].unique())

                # Protection put : strike le + proche de sell_put - target_width
                buy_put_target = sell_put_strike - target_width
                candidates_put = [s for s in put_strikes if s < sell_put_strike]
                if not candidates_put:
                    raise ValueError("Pas de strikes de protection disponibles pour le Put side de l'Iron Condor.")
                buy_put_strike = min(candidates_put, key=lambda x: abs(x - buy_put_target))

                # Protection call : strike le + proche de sell_call + target_width
                buy_call_target = sell_call_strike + target_width
                candidates_call = [s for s in call_strikes if s > sell_call_strike]
                if not candidates_call:
                    raise ValueError("Pas de strikes de protection disponibles pour le Call side de l'Iron Condor.")
                buy_call_strike = min(candidates_call, key=lambda x: abs(x - buy_call_target))

                sell_put_price = get_mid_price(sell_put)
                sell_call_price = get_mid_price(sell_call)

                buy_put_row = puts[puts["strike"] == buy_put_strike]
                buy_call_row = calls[calls["strike"] == buy_call_strike]
                buy_put_price = get_mid_price(buy_put_row.iloc[0]) if not buy_put_row.empty else 0.0
                buy_call_price = get_mid_price(buy_call_row.iloc[0]) if not buy_call_row.empty else 0.0

                net_credit = (sell_put_price + sell_call_price) - (buy_put_price + buy_call_price)
                put_width = sell_put_strike - buy_put_strike
                call_width = buy_call_strike - sell_call_strike
                max_width = max(put_width, call_width)

                # RISK MANAGER : Sanity check crédit — prix physiquement cohérents
                if net_credit <= 0 or net_credit >= max_width:
                    raise ValueError(
                        "Les prix de la chaîne d'options sont illogiques "
                        "(illiquidité majeure ou bid/ask cassé). "
                        "Analyse annulée pour votre sécurité."
                    )

                max_risk = (max_width * 100) - (net_credit * 100)

                if max_risk > budget:
                    raise ValueError(
                        f"Budget insuffisant ({budget}\$) pour un Iron Condor standard sur {ticker}. "
                        f"Risque par contrat : {max_risk:.0f}\$."
                    )

                result["legs"] = [
                    {"action": "SELL", "type": "Put", "strike": sell_put_strike,
                     "exp": exp_str, "dte": dte, "price": sell_put_price},
                    {"action": "BUY", "type": "Put", "strike": buy_put_strike,
                     "exp": exp_str, "dte": dte, "price": buy_put_price},
                    {"action": "SELL", "type": "Call", "strike": sell_call_strike,
                     "exp": exp_str, "dte": dte, "price": sell_call_price},
                    {"action": "BUY", "type": "Call", "strike": buy_call_strike,
                     "exp": exp_str, "dte": dte, "price": buy_call_price},
                ]
                max_profit = net_credit * 100
                result["credit_or_debit"] = round(max_profit, 2)
                result["max_risk"] = round(max_risk, 2)
                result["max_profit"] = round(max_profit, 2)


    # --- Plan de vol (exit triggers) ---
    exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
    time_stop_date = exp_date - dt.timedelta(days=21)
    take_profit_amount = round(abs(result["max_profit"]) * 0.5, 2)

    result["exit_plan"] = {
        "take_profit": take_profit_amount,
        "time_stop_date": time_stop_date.strftime("%d/%m/%Y"),
        "time_stop_dte": (time_stop_date - dt.date.today()).days,
    }

    # --- Probabilités Réelles via Intégration Log-Normale (GBM) ---
    result["sigma"] = sigma  # stocké pour la simulation
    # Volatilité historique réalisée pour le mouvement du sous-jacent
    sigma_move = compute_historical_vol(ticker) or sigma
    probs = compute_real_probabilities(
        legs=result["legs"], spot=spot, dte=dte,
        sigma=sigma, qty=1,  # qty=1 car montants non encore multipliés
        take_profit=take_profit_amount,
        max_risk=result["max_risk"],
        sigma_move=sigma_move,
    )
    result["probabilities"] = probs
    result["pop"] = probs["p_breakeven"]  # rétro-compatibilité
    result["win_rate_estime"] = probs["p_take_profit"]

    # --- Espérance Mathématique (EV) via intégration numérique complète ---
    result["ev"] = probs["expected_pnl"]

    # --- Calcul des Grecques agrégées ---
    net_greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": sigma * 100}
    for leg in result["legs"]:
        leg_T = leg["dte"] / 365.0
        greeks = compute_leg_greeks(leg, spot, leg_T, sigma)
        net_greeks["delta"] += greeks["delta"]
        net_greeks["gamma"] += greeks["gamma"]
        net_greeks["theta"] += greeks["theta"]
        net_greeks["vega"] += greeks["vega"]
    # Multiply by 100 for per-contract values
    for k in ["delta", "gamma", "theta", "vega"]:
        net_greeks[k] = round(net_greeks[k] * 100, 2)
    net_greeks["iv"] = round(net_greeks["iv"], 1)
    result["greeks"] = net_greeks

    # --- Multiplicateur de quantité (Position Sizing) ---
    if result["max_risk"] > 0:
        qty = max(1, int(budget // result["max_risk"]))
    else:
        qty = 1
    result["qty"] = qty

    if qty > 1:
        result["max_risk"] = round(result["max_risk"] * qty, 2)
        result["max_profit"] = round(result["max_profit"] * qty, 2)
        result["credit_or_debit"] = round(result["credit_or_debit"] * qty, 2)
        result["exit_plan"]["take_profit"] = round(result["exit_plan"]["take_profit"] * qty, 2)
        # EV intégrée, se multiplie linéairement par la quantité
        result["ev"] = round(result["ev"] * qty, 2)
        for k in ["delta", "gamma", "theta", "vega"]:
            result["greeks"][k] = round(result["greeks"][k] * qty, 2)

    # --- RISK MANAGER : Kill Switch — Rejet EV Fortement Négative ---
    # Le seuil est -20% du risque max. Le modèle BS est conservateur
    # (il ignore la gestion active, le mean-reversion d'IV, les sorties
    # anticipées), donc on ne bloque que les trades structurellement perdants.
    ev_threshold = -0.20 * result["max_risk"]
    if result.get("ev", 0) < ev_threshold:
        raise ValueError(
            f"Espérance Mathématique (EV) trop négative ({result['ev']:.2f}$). "
            f"Le ratio Risque/Gain est structurellement perdant. "
            f"Trade formellement rejeté."
        )

    return result


# ──────────────────────────────────────────────
# 5b. INDICATEURS AVANCÉS — TENDANCE, EARNINGS, ROC
# ──────────────────────────────────────────────

def compute_trend_and_risk_data(ticker: str, spot: float, bias: str,
                                 dte: int, max_risk: float, ev: float,
                                 max_profit: float):
    """
    Calcule les indicateurs avancés pour un trade validé :
    - EV Yield (%) : rendement de l'EV sur le risque
    - ROC Annualisé (%) : Return on Capital annualisé
    - SMA 50 : moyenne mobile 50 jours
    - Alignement Tendance : cohérence biais / SMA
    - Earnings Risk : risque de résultats avant le time stop
    """
    result = {}

    # ── EV Yield (%) ──
    result["ev_yield"] = (ev / max_risk) * 100 if max_risk != 0 else 0.0

    # ── ROC Annualisé (%) ──
    holding_days = max(1, dte - 21)
    result["roc_annualise"] = (max_profit / max_risk) * (365 / holding_days) * 100 if max_risk != 0 else 0.0

    # ── SMA 50 + RSI 14 + Distance SMA ──
    sma50 = None
    current_rsi = None
    dist_sma = None
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")
        if not hist.empty and len(hist) >= 50:
            sma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
        elif not hist.empty:
            sma50 = float(hist["Close"].mean())

        # RSI (14 jours)
        if not hist.empty and len(hist) >= 15:
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])

        # Distance SMA (%)
        if sma50 is not None and sma50 != 0:
            dist_sma = ((spot - sma50) / sma50) * 100
    except Exception:
        pass
    result["sma50"] = sma50
    result["rsi"] = current_rsi
    result["dist_sma"] = dist_sma

    # ── Alignement Tendance (Filtre de Surchauffe) ──
    if sma50 is None or current_rsi is None:
        result["alignement"] = "➖ N/A"
    elif bias == "Haussier":
        if current_rsi > 70 or (dist_sma is not None and dist_sma > 10.0):
            result["alignement"] = "⚠️ Suracheté (Rejet)"
        elif current_rsi < 30:
            result["alignement"] = "🎯 Achat sur Repli (Oversold)"
        elif spot > sma50:
            result["alignement"] = "✅ Validé (Sain)"
        else:
            result["alignement"] = "❌ Contre-tendance"
    elif bias == "Baissier":
        if current_rsi < 30 or (dist_sma is not None and dist_sma < -10.0):
            result["alignement"] = "⚠️ Survendu (Rejet)"
        elif current_rsi > 70 or (dist_sma is not None and dist_sma > 10.0):
            result["alignement"] = "🎯 Mean Reversion"
        elif spot < sma50:
            result["alignement"] = "✅ Validé (Sain)"
        else:
            result["alignement"] = "❌ Contre-tendance"
    elif bias == "Neutre":
        if current_rsi > 70 or current_rsi < 30:
            result["alignement"] = "⚠️ Élastique tendu (Rejet)"
        else:
            result["alignement"] = "✅ Validé (Range)"

    # ── Earnings Risk ──
    time_stop_date = dt.date.today() + dt.timedelta(days=max(1, dte - 21))
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
            # cal peut être un DataFrame ou un dict
            earnings_date = None
            if isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.columns:
                    earnings_date = pd.to_datetime(cal["Earnings Date"].iloc[0]).date()
                elif "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    earnings_date = pd.to_datetime(val).date()
            elif isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earnings_date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = pd.to_datetime(ed[0]).date()
                    else:
                        earnings_date = pd.to_datetime(ed).date()

            if earnings_date and earnings_date <= time_stop_date:
                result["earnings_risk"] = "⚠️ Danger"
            elif earnings_date:
                result["earnings_risk"] = "✅ OK"
            else:
                result["earnings_risk"] = "✅ N/A"
        else:
            result["earnings_risk"] = "✅ N/A"
    except Exception:
        result["earnings_risk"] = "✅ N/A"

    return result


# ──────────────────────────────────────────────
# 6. INTERFACE UTILISATEUR — SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    # Ticker avec auto-complétion
    ticker_input = st.selectbox(
        "🏷️ Ticker",
        options=TICKER_LIST,
        index=None,
        placeholder="Tapez un ticker… (ex: SPY, AAPL)",
        format_func=lambda t: f"{TICKER_CATEGORY[t]}  ·  {t} — {TICKER_NAMES[t]}",
        help="Sélectionnez ou tapez un symbole boursier (ex: SPY, AAPL, TSLA)",
    )

    ticker = ticker_input if ticker_input else "SPY"

    st.markdown("---")

    budget = st.number_input(
        "💰 Budget Maximum Risqué ($)",
        min_value=50,
        max_value=1_000_000,
        value=1000,
        step=100,
        help="Capital maximum absolu que vous êtes prêt à perdre ou bloquer en marge.",
    )

    bias = st.selectbox(
        "🧭 Biais Directionnel",
        options=["Neutre", "Haussier", "Baissier"],
        index=0,
    )

    st.markdown("---")
    analyze_btn = st.button("🔍  Analyser", use_container_width=True, type="primary")
    scan_btn = st.button("🔎  Scanner Tous les Tickers", use_container_width=True)

    st.markdown("---")
    st.caption("📊 Options Robo-Advisor v1.0")
    st.caption("Méthodologie : Tastytrade / VRP")


# ──────────────────────────────────────────────
# 7. INTERFACE UTILISATEUR — DASHBOARD PRINCIPAL
# ──────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <h1>📈 Options Robo-Advisor</h1>
    <p>Analyse quantitative en temps réel · Méthodologie Tastytrade / VRP</p>
</div>
""", unsafe_allow_html=True)

# ── Mode Scanner Multi-Tickers ──
if scan_btn:
    st.markdown("### 🔎 Scanner Multi-Tickers")
    st.markdown(f"Budget : **${budget:,.0f}** · Scan **Haussier + Neutre + Baissier**")
    st.markdown("---")

    scan_results = []
    progress_bar = st.progress(0, text="Initialisation du scan…")
    status_text = st.empty()
    total = len(TICKER_LIST)
    biases = ["Haussier", "Neutre", "Baissier"]

    for i, t in enumerate(TICKER_LIST):
        progress_bar.progress((i + 1) / total, text=f"Scan de {t} ({i+1}/{total})…")
        for b in biases:
            try:
                s = get_spot_price(t)
                v, vs = get_vol_index(t)
                ivr = compute_iv_rank(t)
                strat = build_strategy(s, v, ivr, b, budget, t, vs)
                qty = strat.get("qty", 1)
                unit_risk = round(strat["max_risk"] / qty, 2) if qty > 0 else strat["max_risk"]
                # Indicateurs avancés
                adv = compute_trend_and_risk_data(
                    t, s, b, int(strat["dte"]),
                    strat["max_risk"], strat.get("ev", 0), strat["max_profit"]
                )
                scan_results.append({
                    "Ticker": t,
                    "Nom": TICKER_NAMES.get(t, t),
                    "Budget Min": unit_risk,
                    "Biais": b,
                    "Stratégie": strat["name"],
                    "Perte Max": round(strat["max_risk"], 2),
                    "Gain Max / 2": round(strat["exit_plan"]["take_profit"], 2),
                    "% TP": strat.get("probabilities", {}).get("p_take_profit", 0),
                    "% BE": strat.get("probabilities", {}).get("p_breakeven", 0),
                    "% Perte": strat.get("probabilities", {}).get("p_partial_loss", 0),
                    "% Loss": strat.get("probabilities", {}).get("p_max_loss", 0),
                    "EV": round(strat.get("ev", 0), 2),
                    "EV Yield": round(adv["ev_yield"], 1),
                    "ROC Ann.": round(adv["roc_annualise"], 1),
                    "SMA 50": round(adv["sma50"], 2) if adv["sma50"] else None,
                    "RSI": round(adv["rsi"], 1) if adv["rsi"] is not None else None,
                    "Écart SMA (%)": round(adv["dist_sma"], 2) if adv["dist_sma"] is not None else None,
                    "Tendance": adv["alignement"],
                    "Earnings": adv["earnings_risk"],
                })
            except Exception:
                continue

    progress_bar.empty()

    if scan_results:
        df = pd.DataFrame(scan_results).sort_values("EV", ascending=False).reset_index(drop=True)
        total_found = len(df)
        # Filtre : ne garder que les cibles parfaites (pas de Rejet ni Contre-tendance)
        df = df[~df["Tendance"].str.contains("Rejet|Contre-tendance", na=False)].reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed

        st.success(f"✅ **{len(df)} cibles validées** sur {total_found} stratégies trouvées ({total} tickers scannés).")

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Nom": st.column_config.TextColumn("Nom", width="medium"),
                "Budget Min": st.column_config.NumberColumn("💰 Budget Min ($)", format="$%.0f"),
                "Biais": st.column_config.TextColumn("Biais", width="small"),
                "Stratégie": st.column_config.TextColumn("Stratégie", width="medium"),
                "Perte Max": st.column_config.NumberColumn("Perte Max ($)", format="$%.2f"),
                "Gain Max / 2": st.column_config.NumberColumn("Gain Max / 2 ($)", format="$%.2f"),
                "% TP": st.column_config.NumberColumn("🎯 % TP", format="%.1f%%"),
                "% BE": st.column_config.NumberColumn("⚖️ % BE", format="%.1f%%"),
                "% Perte": st.column_config.NumberColumn("📉 % Perte", format="%.1f%%"),
                "% Loss": st.column_config.NumberColumn("💀 % Max", format="%.1f%%"),
                "EV": st.column_config.NumberColumn("EV ($)", format="$%.2f"),
                "EV Yield": st.column_config.NumberColumn("📈 EV Yield (%)", format="%.1f%%"),
                "ROC Ann.": st.column_config.NumberColumn("🔄 ROC Ann. (%)", format="%.1f%%"),
                "SMA 50": st.column_config.NumberColumn("📊 SMA 50", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("📉 RSI", format="%.1f"),
                "Écart SMA (%)": st.column_config.NumberColumn("📏 Écart SMA (%)", format="%+.2f%%"),
                "Tendance": st.column_config.TextColumn("📈 Tendance", width="medium"),
                "Earnings": st.column_config.TextColumn("📅 Earnings", width="medium"),
            },
        )
    else:
        st.warning("⚠️ Aucune stratégie valide trouvée. Essayez d'augmenter le budget.")

    st.markdown("---")
    st.caption(
        f"📊 Scan exécuté le {dt.datetime.now().strftime('%d/%m/%Y à %H:%M')} · "
        f"Budget: ${budget:,.0f} · Biais: {bias}"
    )
    st.stop()

if not analyze_btn:
    # État initial : instructions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Sélectionnez")
        st.write("Choisissez un ticker et définissez votre budget maximal dans la barre latérale.")
    with col2:
        st.markdown("### 2️⃣ Analysez")
        st.write("Cliquez sur **🔍 Analyser** pour scanner les données de marché en temps réel.")
    with col3:
        st.markdown("### 3️⃣ Exécutez")
        st.write("Suivez le ticket d'ordre et le plan de vol pour exécuter la stratégie recommandée.")
    st.stop()

# ── Exécution de l'analyse ──
try:
    with st.spinner(f"🔄 Analyse de **{ticker}** en cours…"):
        spot = get_spot_price(ticker)
        vix, vol_symbol = get_vol_index(ticker)
        vol_label = VOL_INDEX_NAMES.get(vol_symbol, vol_symbol.replace("^", ""))
        iv_rank = compute_iv_rank(ticker)

    # ─── Section 1 : CONTEXTE MACRO ───
    st.markdown("### 🌍 Contexte Macro")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="💲 Prix Spot",
            value=f"${spot:,.2f}",
            delta=f"{ticker}",
        )
    with c2:
        vix_color = "🔴" if vix > 20 else ("🟡" if vix > 15 else "🟢")
        st.metric(
            label=f"{vix_color} {vol_label}",
            value=f"{vix:.2f}",
            delta="Élevé" if vix > 20 else ("Modéré" if vix > 15 else "Bas"),
            delta_color="inverse" if vix > 20 else "normal",
        )
    with c3:
        iv_color = "🔴" if iv_rank > 50 else ("🟡" if iv_rank > 20 else "🟢")
        st.metric(
            label=f"{iv_color} IV Rank (52 semaines)",
            value=f"{iv_rank:.1f}%",
            delta="Haute vol." if iv_rank > 50 else ("Moyenne" if iv_rank > 20 else "Basse vol."),
            delta_color="inverse" if iv_rank > 50 else "normal",
        )

    st.markdown("---")

    # ─── Section 2 : STRATÉGIE ───
    with st.spinner("🧠 Construction de la stratégie optimale…"):
        strategy = build_strategy(spot, vix, iv_rank, bias, budget, ticker, vol_symbol)
        adv_data = compute_trend_and_risk_data(
            ticker, spot, bias, int(strategy["dte"]),
            strategy["max_risk"], strategy.get("ev", 0), strategy["max_profit"]
        )

    # Verdict
    st.markdown(f"""
    <div class="verdict-card">
        <h2>🎯 LE VERDICT</h2>
        <div class="strategy-name">{strategy['name']}</div>
        <p>{strategy['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Section 3 : TICKET D'ORDRE ───
    st.markdown("### 📋 Ticket d'Ordre (Legs)")

    qty = strategy.get("qty", 1)

    legs_data = []
    for leg in strategy["legs"]:
        legs_data.append({
            "Qté": qty,
            "Action": f"{'🟢 ' if leg['action'] == 'BUY' else '🔴 '}{leg['action']}",
            "Type": leg["type"],
            "Strike": f"${leg['strike']:,.2f}",
            "Expiration": leg["exp"],
            "DTE": f"{leg['dte']}j",
            "Prix unitaire": f"${leg['price']:.2f}",
        })

    legs_df = pd.DataFrame(legs_data)
    st.dataframe(
        legs_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qté": st.column_config.NumberColumn("Qté", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Strike": st.column_config.TextColumn("Strike", width="small"),
            "Expiration": st.column_config.TextColumn("Expiration", width="medium"),
            "DTE": st.column_config.TextColumn("DTE", width="small"),
            "Prix unitaire": st.column_config.TextColumn("Prix", width="small"),
        },
    )

    st.markdown("---")

    # ─── Section 3b : GRECQUES DE LA POSITION ───
    st.markdown('<div class="section-header"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" /></svg><h2>Grecques de la Position (Net)</h2></div>', unsafe_allow_html=True)

    greeks = strategy.get("greeks", {})
    delta_val = greeks.get("delta", 0)
    gamma_val = greeks.get("gamma", 0)
    theta_val = greeks.get("theta", 0)
    vega_val = greeks.get("vega", 0)
    iv_val = greeks.get("iv", 0)

    st.markdown(f'''
    <div class="greeks-container">
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Delta (Δ)</div>
                <div class="greek-hint-text">Sensibilité au prix du sous-jacent. Un delta de +50 signifie que si l'action bouge de 1$, la position gagne/perd ~50$.</div>
            </div>
            <div class="greek-symbol">Delta (Δ)</div>
            <div class="greek-value">{delta_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Gamma (Γ)</div>
                <div class="greek-hint-text">Accélération du Delta. Un gamma élevé signifie que le Delta changera rapidement si le prix bouge. Risque accru proche de l'expiration.</div>
            </div>
            <div class="greek-symbol">Gamma (Γ)</div>
            <div class="greek-value">{gamma_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Theta (Θ)</div>
                <div class="greek-hint-text">Déclin temporel journalier en $. Un theta négatif = la position perd de la valeur chaque jour. Positif = vous profitez du passage du temps.</div>
            </div>
            <div class="greek-symbol">Theta (Θ)</div>
            <div class="greek-value">{theta_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Vega (ν)</div>
                <div class="greek-hint-text">Sensibilité à la volatilité implicite. Indique le gain/perte pour chaque 1% de hausse de l'IV. Vega positif profite d'une hausse de la vol.</div>
            </div>
            <div class="greek-symbol">Vega (ν)</div>
            <div class="greek-value">{vega_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Vol. Implicite</div>
                <div class="greek-hint-text">Volatilité implicite actuelle du marché pour ces options. Elle mesure l'anticipation de mouvement futur du sous-jacent par le marché.</div>
            </div>
            <div class="greek-symbol">IV</div>
            <div class="greek-value">{iv_val:.1f}%</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.caption("💡 Survolez chaque grecque pour comprendre sa signification")

    st.markdown("---")

    # ─── Section 4 : MÉTRIQUES FINANCIÈRES ───
    st.markdown("### 💰 Métriques Financières")

    m1, m2, m3, m4 = st.columns(4)

    cd_val = strategy["credit_or_debit"]
    cd_label = "Crédit Net Reçu" if cd_val > 0 else "Débit Net Payé"
    cd_color = "green" if cd_val > 0 else "red"

    with m1:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">{cd_label}</div>
            <div class="value {cd_color}">${abs(cd_val):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">⚠️ Risque Maximal</div>
            <div class="value red">${strategy['max_risk']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">🎯 Profit Maximal</div>
            <div class="value green">${strategy['max_profit']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        ev_val = strategy.get('ev', 0)
        ev_color = "green" if ev_val > 0 else "red"
        ev_sign = "+" if ev_val > 0 else "-"
        st.markdown(f"""
        <div class="fin-metric" >
            <div class="label">⚖️ Score EV (Espérance)</div>
            <div class="value {ev_color}">{ev_sign}${abs(ev_val):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Vérification budget
    if strategy["max_risk"] <= budget:
        st.success(f"✅ **Budget respecté** — Risque max ({strategy['max_risk']:,.2f}$) ≤ Budget ({budget:,.2f}$)")
    else:
        st.error(f"❌ **ATTENTION** — Risque max ({strategy['max_risk']:,.2f}$) > Budget ({budget:,.2f}$)")

    st.markdown("---")

    # ─── Section 4a : INDICATEURS AVANCÉS ───
    st.markdown("### 📊 Indicateurs Avancés")

    a1, a2, a3, a4, a5, a6, a7 = st.columns(7)

    ev_yield_val = adv_data["ev_yield"]
    ev_yield_color = "green" if ev_yield_val > 0 else "red"
    with a1:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📈 EV Yield</div>
            <div class="value {ev_yield_color}">{ev_yield_val:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    roc_val = adv_data["roc_annualise"]
    roc_color = "green" if roc_val > 0 else "red"
    with a2:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">🔄 ROC Annualisé</div>
            <div class="value {roc_color}">{roc_val:,.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    sma50_val = adv_data["sma50"]
    with a3:
        sma50_display = f"${sma50_val:,.2f}" if sma50_val else "N/A"
        sma50_color = "green" if sma50_val and spot > sma50_val else ("red" if sma50_val else "")
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📊 SMA 50</div>
            <div class="value {sma50_color}">{sma50_display}</div>
        </div>
        """, unsafe_allow_html=True)

    rsi_val = adv_data.get("rsi")
    with a4:
        if rsi_val is not None:
            rsi_color = "red" if rsi_val > 70 else ("green" if rsi_val < 30 else "")
            rsi_display = f"{rsi_val:.1f}"
        else:
            rsi_color = ""
            rsi_display = "N/A"
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📉 RSI (14)</div>
            <div class="value {rsi_color}">{rsi_display}</div>
        </div>
        """, unsafe_allow_html=True)

    dist_sma_val = adv_data.get("dist_sma")
    with a5:
        if dist_sma_val is not None:
            dist_color = "red" if abs(dist_sma_val) > 10 else "green"
            dist_display = f"{dist_sma_val:+.2f}%"
        else:
            dist_color = ""
            dist_display = "N/A"
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📏 Écart SMA (%)</div>
            <div class="value {dist_color}">{dist_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with a6:
        trend_val = adv_data["alignement"]
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📐 Alignement Tendance</div>
            <div class="value" style="font-size: 1rem;">{trend_val}</div>
        </div>
        """, unsafe_allow_html=True)

    with a7:
        earnings_val = adv_data["earnings_risk"]
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📅 Earnings Risk</div>
            <div class="value" style="font-size: 1rem;">{earnings_val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Section 4b : PROBABILITÉS & NIVEAUX DE PRIX ───
    st.markdown("### 📊 Probabilités & Niveaux de Prix")

    probs = strategy.get('probabilities', {})
    p_tp = probs.get('p_take_profit', 0)
    p_be = probs.get('p_breakeven', 0)
    p_pl = probs.get('p_partial_loss', 0)
    p_loss = probs.get('p_max_loss', 0)

    # Calcul des niveaux de prix associés via bisection sur simulate_pnl
    current_sigma = strategy.get("sigma", 0.25)
    qty_prob = strategy.get("qty", 1)
    take_profit_val = strategy["exit_plan"]["take_profit"]
    max_risk_val = strategy["max_risk"]

    def find_nearest_spot_for_pnl(target_pnl, legs, remaining_dte, sigma, qty, current_spot):
        """Sweep pour trouver le spot le plus proche du spot actuel
        où le P&L croise le seuil cible. Gère tous les types de stratégies
        (monotones et non-monotones comme les Iron Condors)."""
        n_pts = 500
        spots = np.linspace(current_spot * 0.50, current_spot * 1.50, n_pts)
        pnls = [simulate_pnl(legs, s, remaining_dte, sigma, qty) for s in spots]

        # Trouver tous les croisements (changement de signe de pnl - target)
        crossings = []
        for i in range(len(pnls) - 1):
            diff_a = pnls[i] - target_pnl
            diff_b = pnls[i + 1] - target_pnl
            if diff_a * diff_b <= 0:  # croisement
                # Interpolation linéaire
                if abs(diff_b - diff_a) > 1e-10:
                    frac = abs(diff_a) / abs(diff_b - diff_a)
                    cross_spot = spots[i] + frac * (spots[i + 1] - spots[i])
                else:
                    cross_spot = (spots[i] + spots[i + 1]) / 2
                crossings.append(cross_spot)

        if crossings:
            # Retourner le croisement le plus proche du spot actuel
            return min(crossings, key=lambda s: abs(s - current_spot))
        else:
            # Pas de croisement : retourner le spot qui donne le P&L le plus proche du target
            closest_idx = min(range(len(pnls)), key=lambda i: abs(pnls[i] - target_pnl))
            return float(spots[closest_idx])

    strat_legs = strategy["legs"]
    spot_tp = find_nearest_spot_for_pnl(take_profit_val, strat_legs, 21, current_sigma, qty_prob, spot)
    spot_be = find_nearest_spot_for_pnl(0, strat_legs, 21, current_sigma, qty_prob, spot)
    spot_ml = find_nearest_spot_for_pnl(-max_risk_val * 0.95, strat_legs, 21, current_sigma, qty_prob, spot)

    pct_tp = ((spot_tp - spot) / spot) * 100
    pct_be = ((spot_be - spot) / spot) * 100
    pct_ml = ((spot_ml - spot) / spot) * 100

    prob_data = [
        {"Scénario": "🎯 Take Profit", "P&L": f"+${take_profit_val:,.0f}", "Spot Cible": f"${spot_tp:,.2f}", "Mouvement": f"{pct_tp:+.1f}%", "Probabilité (%)": p_tp},
        {"Scénario": "⚖️ Break-Even", "P&L": "$0", "Spot Cible": f"${spot_be:,.2f}", "Mouvement": f"{pct_be:+.1f}%", "Probabilité (%)": p_be},
        {"Scénario": "📉 Perte Partielle", "P&L": "—", "Spot Cible": f"${spot_be:,.0f} – ${spot_ml:,.0f}", "Mouvement": "—", "Probabilité (%)": p_pl},
        {"Scénario": "💀 Perte Maximale", "P&L": f"-${max_risk_val:,.0f}", "Spot Cible": f"${spot_ml:,.2f}", "Mouvement": f"{pct_ml:+.1f}%", "Probabilité (%)": p_loss},
    ]
    df_prob = pd.DataFrame(prob_data)
    st.dataframe(
        df_prob,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Scénario": st.column_config.TextColumn("Scénario", width="medium"),
            "P&L": st.column_config.TextColumn("P&L", width="small"),
            "Spot Cible": st.column_config.TextColumn("Spot Cible", width="medium"),
            "Mouvement": st.column_config.TextColumn("Mouvement", width="small"),
            "Probabilité (%)": st.column_config.ProgressColumn("Probabilité", format="%.1f%%", min_value=0, max_value=100),
        },
    )
    hist_vol = compute_historical_vol(ticker)
    hist_vol_str = f"{hist_vol*100:.1f}%" if hist_vol else "N/A"
    st.caption(f"📍 Spot actuel : **${spot:,.2f}** · Évaluation au time-stop (21 DTE restants) · Vol. historique {hist_vol_str}")

    st.markdown("---")

    # ─── Section 4c : GRAPHIQUE HISTORIQUE 6 MOIS ───
    st.markdown(f"### 📈 Historique {ticker} (6 mois)")

    tk_hist = yf.Ticker(ticker)
    hist_data = tk_hist.history(period="6mo")

    if not hist_data.empty:
      try:
        import plotly.graph_objects as go
        import traceback as _tb

        fig = go.Figure()

        # Courbe du prix
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data["Close"],
            mode="lines",
            name="Prix",
            line=dict(color="#60A5FA", width=2),
            hovertemplate="$%{y:,.2f}<extra></extra>",
        ))

        # ── SMA 50 jours (historique — ligne continue) ──
        sma50_series = hist_data["Close"].rolling(window=50).mean()
        sma50_valid = sma50_series.dropna()
        if not sma50_valid.empty:
            fig.add_trace(go.Scatter(
                x=sma50_valid.index,
                y=sma50_valid,
                mode="lines",
                name="SMA 50",
                line=dict(color="#FBBF24", width=1.5),
                hovertemplate="SMA50: $%{y:,.2f}<extra></extra>",
            ))

            # ── Projection SMA 50 (prix flat au spot) — pointillé ──
            close_list = list(hist_data["Close"].values[-50:])  # derniers 50 prix
            proj_sma_values = [float(sma50_valid.iloc[-1])]  # ancrage au dernier SMA connu
            proj_dates = [sma50_valid.index[-1]]
            future_bdays = pd.bdate_range(start=hist_data.index[-1], periods=23)[1:]  # ~1 mois
            for d in future_bdays:
                close_list.pop(0)
                close_list.append(spot)
                proj_sma_values.append(float(np.mean(close_list)))
                proj_dates.append(d)

            fig.add_trace(go.Scatter(
                x=proj_dates,
                y=proj_sma_values,
                mode="lines",
                name="SMA 50 (proj.)",
                line=dict(color="#FBBF24", width=1.5, dash="dash"),
                hovertemplate="SMA50 proj.: $%{y:,.2f}<extra></extra>",
            ))

        # ── Projection linéaire 1 mois (ancrée au dernier prix) ──
        close_vals = hist_data["Close"].values
        x_numeric = np.arange(len(close_vals))
        coeffs = np.polyfit(x_numeric, close_vals, 1)
        slope = coeffs[0]
        last_price = float(close_vals[-1])

        last_date = hist_data.index[-1]
        future_days = 22  # ~1 mois de trading
        future_dates = pd.bdate_range(start=last_date, periods=future_days + 1)  # inclut le dernier jour
        future_prices = [last_price + slope * d for d in range(future_days + 1)]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode="lines",
            name="Projection (1 mois)",
            line=dict(color="#60A5FA", width=2, dash="dot"),
            hovertemplate="$%{y:,.2f} (proj.)<extra></extra>",
        ))

        # Lignes horizontales pour les strikes
        legs = strategy.get("legs", [])
        strikes = sorted(set(leg["strike"] for leg in legs))
        strike_colors = ["#F87171", "#FBBF24", "#34D399", "#A78BFA"]
        for i, s in enumerate(strikes):
            color = strike_colors[i % len(strike_colors)]
            action = next((l["action"] for l in legs if l["strike"] == s), "")
            opt_type = next((l["type"] for l in legs if l["strike"] == s), "")
            label = f"{action} {opt_type} ${s:.0f}"
            fig.add_hline(
                y=s, line_dash="dash", line_color=color, line_width=1,
                annotation_text=label,
                annotation_position="right",
                annotation_font_color=color,
                annotation_font_size=11,
            )

        # Ligne du spot actuel
        fig.add_hline(
            y=spot, line_dash="dot", line_color="#94A3B8", line_width=1,
            annotation_text=f"Spot ${spot:.0f}",
            annotation_position="left",
            annotation_font_color="#94A3B8",
            annotation_font_size=11,
        )

        # ── Zones vertes (profit) et rouges (perte) ──
        # Sweep pour trouver TOUS les breakevens, TP et ML crossings
        n_sweep = 300
        sweep_spots = np.linspace(spot * 0.50, spot * 1.50, n_sweep)
        sweep_pnls = [simulate_pnl(strat_legs, s, 21, current_sigma, qty_prob) for s in sweep_spots]
        ml_threshold = -max_risk_val * 0.95

        def find_crossings(pnls, spots_arr, threshold):
            crossings = []
            for i in range(len(pnls) - 1):
                diff_a = pnls[i] - threshold
                diff_b = pnls[i + 1] - threshold
                if diff_a * diff_b <= 0 and abs(diff_a - diff_b) > 0.01:
                    frac = abs(diff_a) / (abs(diff_a) + abs(diff_b))
                    cross = spots_arr[i] + frac * (spots_arr[i + 1] - spots_arr[i])
                    crossings.append(float(cross))
            return sorted(crossings)

        be_crossings = find_crossings(sweep_pnls, sweep_spots, 0)
        tp_crossings = find_crossings(sweep_pnls, sweep_spots, take_profit_val)
        ml_crossings = find_crossings(sweep_pnls, sweep_spots, ml_threshold)

        # Déterminer les zones du y-axis
        y_min_zone = float(sweep_spots[0])
        y_max_zone = float(sweep_spots[-1])

        GREEN_LIGHT = "rgba(52, 211, 153, 0.07)"
        GREEN_DARK = "rgba(52, 211, 153, 0.18)"
        RED_LIGHT = "rgba(248, 113, 113, 0.07)"
        RED_DARK = "rgba(248, 113, 113, 0.18)"

        if len(be_crossings) == 0:
            is_positive = sweep_pnls[len(sweep_pnls) // 2] > 0
            fig.add_hrect(y0=y_min_zone, y1=y_max_zone,
                fillcolor=GREEN_LIGHT if is_positive else RED_LIGHT, line_width=0, layer="below")
        elif len(be_crossings) == 1:
            # 1 BE = stratégie directionnelle
            be = be_crossings[0]
            pnl_above = simulate_pnl(strat_legs, be + 1, 21, current_sigma, qty_prob)
            profit_above = pnl_above > 0

            if profit_above:
                fig.add_hrect(y0=be, y1=y_max_zone, fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=y_min_zone, y1=be, fillcolor=RED_LIGHT, line_width=0, layer="below")
                # TP dark green above TP spot
                if tp_crossings:
                    fig.add_hrect(y0=tp_crossings[-1], y1=y_max_zone, fillcolor=GREEN_DARK, line_width=0, layer="below")
                # ML dark red below ML spot
                if ml_crossings:
                    fig.add_hrect(y0=y_min_zone, y1=ml_crossings[0], fillcolor=RED_DARK, line_width=0, layer="below")
            else:
                fig.add_hrect(y0=y_min_zone, y1=be, fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be, y1=y_max_zone, fillcolor=RED_LIGHT, line_width=0, layer="below")
                # TP dark green below TP spot
                if tp_crossings:
                    fig.add_hrect(y0=y_min_zone, y1=tp_crossings[0], fillcolor=GREEN_DARK, line_width=0, layer="below")
                # ML dark red above ML spot
                if ml_crossings:
                    fig.add_hrect(y0=ml_crossings[-1], y1=y_max_zone, fillcolor=RED_DARK, line_width=0, layer="below")

            # BE line
            fig.add_hline(y=be, line_dash="dash", line_color="#60A5FA", line_width=1,
                annotation_text=f"BE ${be:.0f}", annotation_position="right",
                annotation_font_color="#60A5FA", annotation_font_size=11)
            # TP line
            fig.add_hline(y=spot_tp, line_dash="dash", line_color="#34D399", line_width=1,
                annotation_text=f"TP ${spot_tp:.0f}", annotation_position="right",
                annotation_font_color="#34D399", annotation_font_size=11)
        else:
            # 2+ BE = Iron Condor
            be_sorted = sorted(be_crossings)
            center = (be_sorted[0] + be_sorted[-1]) / 2
            pnl_center = simulate_pnl(strat_legs, center, 21, current_sigma, qty_prob)
            center_positive = pnl_center > 0

            if center_positive:
                fig.add_hrect(y0=y_min_zone, y1=be_sorted[0], fillcolor=RED_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[0], y1=be_sorted[-1], fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[-1], y1=y_max_zone, fillcolor=RED_LIGHT, line_width=0, layer="below")
                # ML dark red at extremes
                if ml_crossings:
                    fig.add_hrect(y0=y_min_zone, y1=ml_crossings[0], fillcolor=RED_DARK, line_width=0, layer="below")
                    if len(ml_crossings) >= 2:
                        fig.add_hrect(y0=ml_crossings[-1], y1=y_max_zone, fillcolor=RED_DARK, line_width=0, layer="below")
            else:
                fig.add_hrect(y0=y_min_zone, y1=be_sorted[0], fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[0], y1=be_sorted[-1], fillcolor=RED_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[-1], y1=y_max_zone, fillcolor=GREEN_LIGHT, line_width=0, layer="below")

            # BE lines
            for be in be_sorted:
                fig.add_hline(y=be, line_dash="dash", line_color="#60A5FA", line_width=1,
                    annotation_text=f"BE ${be:.0f}", annotation_position="right",
                    annotation_font_color="#60A5FA", annotation_font_size=11)

        # Ligne verticale : date de sortie (time-stop)
        dte_val = int(strategy["dte"])
        exit_date = dt.datetime.now() + dt.timedelta(days=max(1, dte_val - 21))
        exit_date_str = exit_date.strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=exit_date_str, x1=exit_date_str,
            y0=0, y1=1, yref="paper",
            line=dict(color="#FBBF24", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=exit_date_str, y=1, yref="paper",
            text=f"Sortie {exit_date.strftime('%d/%m')}",
            showarrow=False, font=dict(color="#FBBF24", size=11),
            yshift=10,
        )

        # Y-axis range: padding autour du min/max prix
        y_min = float(hist_data["Low"].min())
        y_max = float(hist_data["High"].max())
        # Inclure les strikes, spot, BE, TP et projection dans le range
        all_levels = [y_min, y_max] + strikes + [spot, spot_be, spot_tp] + list(future_prices)
        y_range_min = min(all_levels) * 0.97
        y_range_max = max(all_levels) * 1.03

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=80, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                gridcolor="rgba(51,65,85,0.3)",
                showgrid=True,
            ),
            yaxis=dict(
                range=[y_range_min, y_range_max],
                gridcolor="rgba(51,65,85,0.3)",
                showgrid=True,
                tickprefix="$",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                font=dict(size=11),
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistiques rapides
        price_6m_ago = float(hist_data["Close"].iloc[0])
        price_now = float(hist_data["Close"].iloc[-1])
        change_pct = ((price_now - price_6m_ago) / price_6m_ago) * 100
        high_6m = float(hist_data["High"].max())
        low_6m = float(hist_data["Low"].min())

        st.caption(
            f"📊 **6 mois** : {change_pct:+.1f}% · "
            f"Plus haut : ${high_6m:,.2f} · Plus bas : ${low_6m:,.2f}"
        )
      except Exception as _chart_err:
        st.error(f"Erreur chart : {_chart_err}")
        st.code(_tb.format_exc())

    # ─── Section 4b : SIMULATION P&L À 21 DTE ───
    st.markdown("### 🔮 Simulation P&L à la Clôture (Time Stop à 21 DTE)")

    current_sigma = strategy.get("sigma", 0.25)
    dte_strat = int(strategy["dte"])
    holding_days = max(1, dte_strat - 21)
    qty_sim = strategy.get("qty", 1)
    take_profit_sim = strategy["exit_plan"]["take_profit"]
    max_risk_sim = strategy["max_risk"]

    # Écart-type statistique du mouvement sur la période de détention
    sd_move = spot * current_sigma * np.sqrt(holding_days / 365.0)

    # 5 scénarios
    scenarios = [
        ("-1.5 SD", "Forte Baisse", spot - 1.5 * sd_move),
        ("-0.5 SD", "Baisse", spot - 0.5 * sd_move),
        ("0 SD", "Stagnation", spot),
        ("+0.5 SD", "Hausse", spot + 0.5 * sd_move),
        ("+1.5 SD", "Forte Hausse", spot + 1.5 * sd_move),
    ]

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    scenario_cols = [sc1, sc2, sc3, sc4, sc5]

    for col, (sd_label, move_label, target_spot) in zip(scenario_cols, scenarios):
        sim_pnl = simulate_pnl(
            strategy["legs"], target_spot, 21, current_sigma, qty_sim
        )

        # Label dynamique basé sur le P&L
        if sim_pnl > take_profit_sim:
            result_label = "🚀 Très Positif"
            pnl_border = "rgba(52, 211, 153, 0.4)"
        elif sim_pnl > 0:
            result_label = "🟢 Positif"
            pnl_border = "rgba(52, 211, 153, 0.25)"
        elif sim_pnl == 0:
            result_label = "⚪ Neutre"
            pnl_border = "rgba(255, 255, 255, 0.1)"
        elif sim_pnl < -max_risk_sim * 0.5:
            result_label = "🔴 Très Défavorable"
            pnl_border = "rgba(248, 113, 113, 0.4)"
        else:
            result_label = "🟠 Défavorable"
            pnl_border = "rgba(251, 191, 36, 0.3)"

        pnl_color = "#34D399" if sim_pnl >= 0 else "#F87171"
        pnl_sign = "+" if sim_pnl > 0 else ""

        # Calcul du % de variation du sous-jacent
        pct_change = ((target_spot - spot) / spot) * 100
        pct_sign = "+" if pct_change >= 0 else ""
        pct_color = "#34D399" if pct_change >= 0 else "#F87171"

        with col:
            st.markdown(f"""
            <div class="greek-card" style="border-color: {pnl_border}; padding: 1.2rem 0.8rem;">
                <div class="greek-symbol" style="color: #FBBF24; margin-bottom: 0.5rem;">{sd_label}</div>
                <div style="font-family: 'Fira Code', monospace; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.15rem;">${target_spot:,.2f}</div>
                <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; font-weight: 600; color: {pct_color}; margin-bottom: 0.4rem;">{pct_sign}{pct_change:.1f}%</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-bottom: 0.6rem; font-family: 'Fira Sans', sans-serif;">{result_label}</div>
                <div style="font-family: 'Fira Code', monospace; font-size: 1.25rem; font-weight: 700; color: {pnl_color};">{pnl_sign}${abs(sim_pnl):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(f"📐 Écart-type estimé sur {holding_days}j : ±${sd_move:,.2f} (basé sur IV {current_sigma*100:.1f}%)")

    st.markdown("---")

    # ─── Section 5 : PLAN DE VOL ───
    st.markdown("### 🛫 Plan de Vol (Triggers de Sortie)")

    exit_plan = strategy["exit_plan"]

    # Estimer le prix du sous-jacent pour le take profit
    tp_target_spot = estimate_take_profit_spot(
        strategy["legs"], spot, 21, current_sigma, qty_sim,
        exit_plan["take_profit"]
    )
    if tp_target_spot is not None:
        tp_pct_change = ((tp_target_spot - spot) / spot) * 100
        tp_pct_sign = "+" if tp_pct_change >= 0 else ""
        tp_spot_info = (
            f" · Sous-jacent estimé à **\\${tp_target_spot:,.2f}** "
            f"({tp_pct_sign}{tp_pct_change:.1f}% vs spot actuel)"
        )
    else:
        tp_spot_info = ""

    st.info(
        f"🎯 **TAKE PROFIT** — Placez un ordre limite (GTC) pour racheter la position et "
        f"encaisser dès que le profit atteint **\\${exit_plan['take_profit']:,.2f}** "
        f"(50% du profit maximum).{tp_spot_info}"
    )

    st.warning(
        f"⏱️ **TIME STOP** — Clôturez obligatoirement la position le "
        f"**{exit_plan['time_stop_date']}** (dans {exit_plan['time_stop_dte']} jours, à 21 DTE), "
        f"quels que soient les gains/pertes, pour écraser le risque Gamma."
    )

    # ─── Footer ───
    st.markdown("---")
    st.caption(
        f"📊 Analyse exécutée le {dt.datetime.now().strftime('%d/%m/%Y à %H:%M')} · "
        f"Ticker: {ticker} · Budget: ${budget:,.0f} · Biais: {bias}"
    )

except ValueError as e:
    st.error(f"⚠️ **Erreur** : {e}")
    import zoneinfo as _zi
    try:
        _et = _zi.ZoneInfo("America/New_York")
        _local_tz = dt.datetime.now().astimezone().tzinfo
        _open_local = dt.datetime.now(_et).replace(hour=9, minute=30).astimezone(_local_tz)
        _close_local = dt.datetime.now(_et).replace(hour=16, minute=0).astimezone(_local_tz)
        _hours = f"{_open_local.strftime('%Hh%M')}-{_close_local.strftime('%Hh%M')} (heure locale)"
    except Exception:
        _hours = "9h30-16h00 ET"
    st.info(f"💡 **Conseil** : Vérifiez le ticker, augmentez votre budget, ou réessayez pendant les heures de marché ({_hours}).")

except Exception as e:
    st.error(f"❌ **Erreur inattendue** : {type(e).__name__} — {e}")
    st.info("💡 Cela peut être dû à un ticker invalide, au marché fermé, ou à un problème réseau. Réessayez.")
