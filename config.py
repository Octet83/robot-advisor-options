"""
config.py — Constantes, tickers et mappings de volatilité
==========================================================
"""

import pandas as pd

# ──────────────────────────────────────────────
# Paramètres financiers
# ──────────────────────────────────────────────

RISK_FREE_RATE = 0.05  # ~taux sans risque approximatif


# ──────────────────────────────────────────────
# Tickers & Groupes
# ──────────────────────────────────────────────

TICKER_GROUPS = {
    "🇺🇸 Index US": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
        "DIA": "Dow Jones", "VTI": "US Total Market",
    },
    "🌐 Émergents": {
        "EEM": "Emerging Markets",
    },
    "💻 Tech": {
        "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
        "GOOGL": "Alphabet", "META": "Meta", "NVDA": "NVIDIA", "TSLA": "Tesla",
        "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce",
        "ADBE": "Adobe", "CSCO": "Cisco", "IBM": "IBM",
    },
    "🔬 Semiconducteurs": {
        "AMD": "AMD", "INTC": "Intel", "MU": "Micron", "QCOM": "Qualcomm",
        "TSM": "TSMC", "MRVL": "Marvell", "ARM": "Arm Holdings", "SMCI": "Super Micro",
    },
    "🎬 Média": {
        "NFLX": "Netflix", "DIS": "Disney",
    },
    "🏦 Finance": {
        "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
        "MS": "Morgan Stanley", "WFC": "Wells Fargo", "C": "Citigroup", "SCHW": "Schwab",
        "V": "Visa", "MA": "Mastercard", "AXP": "Amex", "BLK": "BlackRock",
    },
    "⛽ Énergie": {
        "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
        "SLB": "Schlumberger", "OXY": "Occidental", "EOG": "EOG Resources",
    },
    "🏥 Santé / Pharma": {
        "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "PFE": "Pfizer",
        "ABBV": "AbbVie", "LLY": "Eli Lilly", "MRK": "Merck", "BMY": "Bristol-Myers",
        "AMGN": "Amgen", "GILD": "Gilead", "TMO": "Thermo Fisher",
        "ABT": "Abbott", "MRNA": "Moderna",
    },
    "🏭 Industrie": {
        "BA": "Boeing", "CAT": "Caterpillar", "DE": "Deere & Co",
        "GE": "GE Aerospace", "HON": "Honeywell", "LMT": "Lockheed Martin",
        "RTX": "RTX / Raytheon", "UPS": "UPS", "FDX": "FedEx",
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
        "F": "Ford", "GM": "General Motors",
    },
    "🎰 Spéculatif / High-Vol": {
        "COIN": "Coinbase", "PLTR": "Palantir", "SOFI": "SoFi", "RIVN": "Rivian",
        "HOOD": "Robinhood",
        "SNAP": "Snap", "GME": "GameStop", "AMC": "AMC Entertainment",
        "UBER": "Uber", "SHOP": "Shopify", "ROKU": "Roku",
        "DKNG": "DraftKings", "ABNB": "Airbnb",
        "PYPL": "PayPal", "SNOW": "Snowflake", "NET": "Cloudflare",
        "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks", "ZS": "Zscaler",
    },
    "🪙 Matières Premières": {
        "GLD": "Or (Gold)", "SLV": "Argent (Silver)",
        "USO": "Pétrole brut (WTI)",
    },
    "📈 Obligations": {
        "TLT": "Treasuries 20 ans+",
    },
    "📊 Secteurs ETF": {
        "XLF": "Secteur Finance", "XLE": "Secteur Énergie", "XLK": "Secteur Tech",
        "XLV": "Secteur Santé", "XLI": "Secteur Industrie",
        "XLP": "Conso. de base", "XLY": "Conso. discrétionnaire",
        "XLU": "Secteur Utilities",
        "SMH": "Semiconducteurs ETF",
        "ARKK": "ARK Innovation", "SOXX": "Semiconducteurs (iShares)",
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


# ── Mapping ticker → indice de volatilité CBOE spécifique ──
# Fallback : ^VIX si le ticker n'a pas d'indice dédié.
VOL_INDEX_MAP = {
    # S&P 500
    "SPY": "^VIX", "VOO": "^VIX", "IVV": "^VIX",
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
