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
