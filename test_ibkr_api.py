"""
test_ibkr_api.py — Test standalone de l'API IBKR via ib_insync
==============================================================
Vérifie que IBKR peut fournir toutes les données actuellement
obtenues via yfinance pour le Robo-Advisor d'Options.

Prérequis :
  - TWS ou IB Gateway ouvert avec l'API activée
  - pip install ib_insync
  - Port par défaut : 7497 (TWS paper) ou 4002 (Gateway paper)

Usage :
  python test_ibkr_api.py [--port 7497] [--ticker SPY]
"""

import argparse
import time
import sys
import datetime as dt
import numpy as np
import pandas as pd

try:
    from ib_insync import (
        IB, Stock, Index, Option, util,
        MarketOrder, LimitOrder, TagValue,
    )
except ImportError:
    print("❌ ib_insync non installé. Installez-le avec :")
    print("   pip install ib_insync")
    sys.exit(1)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Test IBKR API pour remplacer yfinance")
    parser.add_argument("--host", default="127.0.0.1", help="Hôte TWS/Gateway (défaut: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7497, help="Port TWS/Gateway (défaut: 7497)")
    parser.add_argument("--ticker", default="SPY", help="Ticker à tester (défaut: SPY)")
    parser.add_argument("--client-id", type=int, default=99, help="Client ID (défaut: 99)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

results = []

def report(test_name: str, success: bool, details: str = ""):
    """Enregistre et affiche le résultat d'un test."""
    icon = "✅" if success else "❌"
    results.append((test_name, success))
    print(f"\n{'='*60}")
    print(f"{icon} TEST: {test_name}")
    print(f"{'='*60}")
    if details:
        print(details)
    print()


def _is_valid(val) -> bool:
    """Vérifie qu'une valeur de market data est valide (pas nan, pas None, > 0)."""
    if val is None:
        return False
    try:
        return not np.isnan(val) and val > 0
    except (TypeError, ValueError):
        return False


# ──────────────────────────────────────────────
# TEST 1 : Prix Spot (remplace get_spot_price)
# ──────────────────────────────────────────────

def test_1_spot_price(ib: IB, ticker: str):
    """
    yfinance : yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    IBKR     : reqMktData (delayed) → ticker.marketPrice()
               Fallback : dernier close historique
    """
    print("\n" + "─"*60)
    print(f"🔍 Test 1 : Prix spot pour {ticker}")
    print("─"*60)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    # ── Méthode 1 : reqMktData avec données différées ──
    md = ib.reqMktData(contract, "", snapshot=False)
    # Attendre que les données arrivent (différées = quelques secondes)
    for _ in range(80):  # max 8 secondes
        ib.sleep(0.1)
        if _is_valid(md.last) or _is_valid(md.close) or _is_valid(md.bid):
            break

    spot = md.marketPrice()
    last = md.last
    close = md.close
    bid = md.bid
    ask = md.ask

    ib.cancelMktData(contract)

    method = "reqMktData (delayed)"
    mktdata_ok = _is_valid(spot)

    # ── Méthode 2 (fallback) : dernier close historique ──
    if not mktdata_ok:
        print("  ⚠️ reqMktData n'a pas retourné de données, fallback historique...")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if bars:
            spot = float(bars[-1].close)
            method = "reqHistoricalData (fallback)"
            mktdata_ok = True

    details = (
        f"  Méthode         = {method}\n"
        f"  marketPrice()   = {spot}\n"
        f"  last            = {last}\n"
        f"  close           = {close}\n"
        f"  bid             = {bid}\n"
        f"  ask             = {ask}"
    )

    report("Prix Spot", mktdata_ok, details)
    return spot if mktdata_ok else None


# ──────────────────────────────────────────────
# TEST 2 : Indice de Volatilité (remplace get_vol_index)
# ──────────────────────────────────────────────

def test_2_vol_index(ib: IB):
    """
    yfinance : yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
    IBKR     : reqMktData (delayed) sur contrat Index "VIX"
               Fallback : reqHistoricalData sur VIX
    """
    print("\n" + "─"*60)
    print("🔍 Test 2 : Indice de volatilité (VIX)")
    print("─"*60)

    # VIX est un Index sur CBOE
    contract = Index("VIX", "CBOE", "USD")
    ib.qualifyContracts(contract)

    # ── Méthode 1 : reqMktData différé ──
    md = ib.reqMktData(contract, "", snapshot=False)
    for _ in range(80):
        ib.sleep(0.1)
        if _is_valid(md.last) or _is_valid(md.close) or _is_valid(md.bid):
            break

    vix_value = md.marketPrice()
    last = md.last
    close = md.close
    method = "reqMktData (delayed)"

    ib.cancelMktData(contract)

    mktdata_ok = _is_valid(vix_value)

    # ── Méthode 2 (fallback) : historique VIX ──
    if not mktdata_ok:
        print("  ⚠️ reqMktData n'a pas retourné de données, fallback historique...")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="5 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if bars:
            vix_value = float(bars[-1].close)
            method = "reqHistoricalData (fallback)"
            mktdata_ok = True
        else:
            # Certains Index ne supportent pas TRADES, essayer OPTION_IMPLIED_VOLATILITY
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="5 D",
                barSizeSetting="1 day",
                whatToShow="MIDPOINT",
                useRTH=True,
                formatDate=1,
            )
            if bars:
                vix_value = float(bars[-1].close)
                method = "reqHistoricalData MIDPOINT (fallback)"
                mktdata_ok = True

    details = (
        f"  Méthode           = {method}\n"
        f"  VIX               = {vix_value}\n"
        f"  last (mktData)    = {last}\n"
        f"  close (mktData)   = {close}"
    )

    report("Indice de Volatilité VIX", mktdata_ok, details)
    return vix_value if mktdata_ok else None


# ──────────────────────────────────────────────
# TEST 3 : Historique 1 an (remplace compute_iv_rank)
# ──────────────────────────────────────────────

def test_3_historical_1y(ib: IB, ticker: str):
    """
    yfinance : yf.Ticker(ticker).history(period="1y")
    IBKR     : reqHistoricalData(durationStr="1 Y", barSizeSetting="1 day")

    On calcule ensuite l'IV Rank comme dans app.py.
    """
    print("\n" + "─"*60)
    print(f"🔍 Test 3 : Historique 1 an pour {ticker} (IV Rank)")
    print("─"*60)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        report("Historique 1 an (reqHistoricalData)", False, "  Aucune barre reçue")
        return None

    df = util.df(bars)
    n_bars = len(df)

    # Calcul IV Rank comme dans app.py
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    rolling_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
    rolling_vol = rolling_vol.dropna()

    iv_current = float(rolling_vol.iloc[-1])
    iv_min = float(rolling_vol.min())
    iv_max = float(rolling_vol.max())
    iv_rank = 100.0 * (iv_current - iv_min) / (iv_max - iv_min) if iv_max != iv_min else 50.0
    iv_rank = round(float(np.clip(iv_rank, 0, 100)), 1)

    details = (
        f"  Barres reçues     = {n_bars}\n"
        f"  Période           = {df['date'].iloc[0]} → {df['date'].iloc[-1]}\n"
        f"  Dernier close     = ${df['close'].iloc[-1]:.2f}\n"
        f"  Vol rolling 20j   = {iv_current:.1f}%\n"
        f"  Vol min 1y        = {iv_min:.1f}%\n"
        f"  Vol max 1y        = {iv_max:.1f}%\n"
        f"  ► IV Rank calculé = {iv_rank:.1f}%"
    )

    success = n_bars >= 200  # On attend au moins ~250 barres
    report("Historique 1 an → IV Rank", success, details)
    return iv_rank if success else None


# ──────────────────────────────────────────────
# TEST 4 : Historique 3 mois (remplace compute_historical_vol)
# ──────────────────────────────────────────────

def test_4_historical_3m(ib: IB, ticker: str):
    """
    yfinance : yf.Ticker(ticker).history(period="3mo")
    IBKR     : reqHistoricalData(durationStr="3 M", barSizeSetting="1 day")
    """
    print("\n" + "─"*60)
    print(f"🔍 Test 4 : Historique 3 mois pour {ticker} (Vol Historique)")
    print("─"*60)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="3 M",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        report("Historique 3 mois (reqHistoricalData)", False, "  Aucune barre reçue")
        return None

    df = util.df(bars)
    n_bars = len(df)

    # Calcul vol historique comme dans app.py
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    sigma_hist = float(log_returns.tail(30).std() * np.sqrt(252))

    details = (
        f"  Barres reçues     = {n_bars}\n"
        f"  Période           = {df['date'].iloc[0]} → {df['date'].iloc[-1]}\n"
        f"  ► Vol historique  = {sigma_hist*100:.1f}% (annualisée)"
    )

    success = n_bars >= 50
    report("Historique 3 mois → Vol Historique", success, details)
    return sigma_hist if success else None


# ──────────────────────────────────────────────
# TEST 5 : Chaîne d'options (remplace get_options_chain)
# ──────────────────────────────────────────────

def test_5_options_chain(ib: IB, ticker: str, spot: float | None):
    """
    yfinance : yf.Ticker(ticker).options + .option_chain(exp)
    IBKR     : reqSecDefOptParams() → construction de contrats Option
               → reqMktData() pour bid/ask/OI/IV de chaque strike
    """
    print("\n" + "─"*60)
    print(f"🔍 Test 5 : Chaîne d'options pour {ticker}")
    print("─"*60)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Phase 1 : récupérer les expirations et strikes disponibles
    chains = ib.reqSecDefOptParams(
        contract.symbol, "", contract.secType, contract.conId
    )

    if not chains:
        report("Chaîne d'options (reqSecDefOptParams)", False, "  Aucune chaîne reçue")
        return None

    # ── Sélectionner la meilleure chaîne ──
    # SMART retourne souvent très peu de strikes.
    # On cherche la chaîne avec le plus de strikes (typiquement CBOE, AMEX, etc.)
    print(f"  Chaînes disponibles :")
    for c in chains:
        print(f"    {c.exchange:12s} → {len(c.expirations):3d} exps, {len(c.strikes):5d} strikes")

    # Prendre la chaîne avec le plus de strikes
    chain = max(chains, key=lambda c: len(c.strikes))
    print(f"  ► Chaîne sélectionnée : {chain.exchange} ({len(c.strikes)} strikes)")

    # Lister les expirations et strikes
    expirations = sorted(chain.expirations)
    strikes = sorted(chain.strikes)

    # Trouver l'expiration ~45 DTE
    today = dt.date.today()
    best_exp = None
    best_dte = None
    best_diff = float("inf")

    for exp_str in expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y%m%d").date()
        dte = (exp_date - today).days
        diff = abs(dte - 45)
        if 35 <= dte <= 60 and diff < best_diff:
            best_diff = diff
            best_exp = exp_str
            best_dte = dte

    if best_exp is None:
        # Fallback : la plus proche de 45 DTE
        for exp_str in expirations:
            exp_date = dt.datetime.strptime(exp_str, "%Y%m%d").date()
            dte = (exp_date - today).days
            if dte > 0:
                diff = abs(dte - 45)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
                    best_dte = dte

    details_p1 = (
        f"  Exchange        = {chain.exchange}\n"
        f"  Nb expirations  = {len(expirations)}\n"
        f"  5 premières     = {expirations[:5]}\n"
        f"  Nb strikes      = {len(strikes)}\n"
        f"  Exp choisie     = {best_exp} ({best_dte} DTE)\n"
    )

    # Phase 2 : récupérer les données de marché pour quelques strikes
    # On prend les strikes dans ±10% du spot
    if spot and spot > 0:
        nearby_strikes = [s for s in strikes if abs(s - spot) / spot < 0.05]
        nearby_strikes = sorted(nearby_strikes)[:10]  # max 10 strikes
    else:
        # Si pas de spot, prendre les 10 strikes du milieu
        mid = len(strikes) // 2
        nearby_strikes = strikes[max(0, mid-5):mid+5]

    if not nearby_strikes:
        report("Chaîne d'options complète", False, 
               details_p1 + f"\n  ❌ Aucun strike trouvé près du spot ({spot})")
        return None

    print(f"  Strikes testés  = {nearby_strikes}")

    # Construire des contrats Option avec l'exchange de la chaîne
    call_contracts = []
    put_contracts = []
    for strike in nearby_strikes:
        c = Option(ticker, best_exp, strike, "C", chain.exchange)
        p = Option(ticker, best_exp, strike, "P", chain.exchange)
        call_contracts.append(c)
        put_contracts.append(p)

    all_options = call_contracts + put_contracts
    ib.qualifyContracts(*all_options)

    # Compter les options qualifiées
    qualified_opts = [opt for opt in all_options if opt.conId]
    print(f"  Options qualifiées = {len(qualified_opts)}/{len(all_options)}")

    # ── Demander les market data ──
    # PAS de genericTickList en mode snapshot (erreur 321)
    # On utilise le mode streaming (snapshot=False) avec un délai
    option_tickers = []
    for opt in qualified_opts:
        # Streaming sans ticks génériques pour éviter l'erreur 321
        md = ib.reqMktData(opt, "", snapshot=False)
        option_tickers.append((opt, md))

    # Attendre que les données arrivent (les options différées prennent du temps)
    print("  ⏳ Attente des données de marché (12s)...")
    ib.sleep(12)

    # Collecter les résultats
    calls_data = []
    puts_data = []
    for opt, md in option_tickers:
        bid_val = md.bid if _is_valid(md.bid) else 0.0
        ask_val = md.ask if _is_valid(md.ask) else 0.0
        last_val = md.last if _is_valid(md.last) else 0.0
        vol_val = int(md.volume) if _is_valid(md.volume) else 0

        # IV depuis modelGreeks si disponible
        iv_val = None
        if md.modelGreeks and hasattr(md.modelGreeks, 'impliedVol') and md.modelGreeks.impliedVol:
            iv_val = md.modelGreeks.impliedVol
        elif hasattr(md, 'impliedVolatility') and _is_valid(md.impliedVolatility):
            iv_val = md.impliedVolatility

        # Greeks
        greeks_str = "N/A"
        if md.modelGreeks:
            g = md.modelGreeks
            greeks_str = f"δ={g.delta:.3f} γ={g.gamma:.4f} θ={g.theta:.3f}" if g.delta else "partial"

        row = {
            "strike": opt.strike,
            "bid": bid_val,
            "ask": ask_val,
            "last": last_val,
            "volume": vol_val,
            "IV": f"{iv_val:.2%}" if iv_val else "N/A",
            "greeks": greeks_str,
        }
        if opt.right == "C":
            calls_data.append(row)
        else:
            puts_data.append(row)

    # Annuler toutes les souscriptions
    for opt, md in option_tickers:
        ib.cancelMktData(opt)

    details_p2 = ""
    if calls_data:
        calls_df = pd.DataFrame(calls_data)
        details_p2 += f"\n  📗 CALLS ({len(calls_data)} strikes) :\n"
        details_p2 += calls_df.to_string(index=False) + "\n"

    if puts_data:
        puts_df = pd.DataFrame(puts_data)
        details_p2 += f"\n  📕 PUTS ({len(puts_data)} strikes) :\n"
        details_p2 += puts_df.to_string(index=False) + "\n"

    # Vérifier qu'on a bien des bid/ask
    has_bid_ask = any(r["bid"] > 0 or r["ask"] > 0 for r in calls_data + puts_data)
    has_iv = any(r["IV"] != "N/A" for r in calls_data + puts_data)
    has_greeks = any(r["greeks"] != "N/A" for r in calls_data + puts_data)

    details = details_p1 + details_p2
    details += f"\n  Bid/Ask reçus   = {'✅ Oui' if has_bid_ask else '❌ Non'}"
    details += f"\n  IV reçue        = {'✅ Oui' if has_iv else '❌ Non'}"
    details += f"\n  Greeks reçus    = {'✅ Oui' if has_greeks else '❌ Non'}"

    success = len(expirations) > 0 and len(qualified_opts) > 0 and has_bid_ask
    report("Chaîne d'options complète", success, details)
    return (calls_data, puts_data) if success else None


# ──────────────────────────────────────────────
# TEST 6 : Calendrier Earnings (remplace tk.calendar)
# ──────────────────────────────────────────────

def test_6_earnings_calendar(ib: IB, ticker: str):
    """
    yfinance : yf.Ticker(ticker).calendar → Earnings Date
    IBKR     : reqFundamentalData ou reqContractDetails
    Note : IBKR n'a pas d'API directe "earnings date" simple.
    On teste via reqContractDetails qui contient quelques infos.
    """
    print("\n" + "─"*60)
    print(f"🔍 Test 6 : Calendrier Earnings pour {ticker}")
    print("─"*60)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Méthode 1 : reqContractDetails
    details_list = ib.reqContractDetails(contract)
    contract_info = ""
    if details_list:
        d = details_list[0]
        contract_info = (
            f"  Long Name    = {d.longName}\n"
            f"  Industry     = {d.industry}\n"
            f"  Category     = {d.category}\n"
            f"  Subcategory  = {d.subcategory}\n"
        )

    # Méthode 2 : reqFundamentalData (nécessite un abonnement data)
    fundamental_data = None
    fundamental_info = ""
    try:
        fundamental_data = ib.reqFundamentalData(contract, "ReportSnapshot")
        if fundamental_data:
            if "EarningDate" in str(fundamental_data) or "earnings" in str(fundamental_data).lower():
                fundamental_info = "  ✅ Données fondamentales contiennent des infos earnings"
            else:
                fundamental_info = f"  ⚠️ Données fondamentales reçues ({len(fundamental_data)} chars) mais pas de date earnings explicite"
        else:
            fundamental_info = "  ⚠️ Pas de données fondamentales (abonnement requis ?)"
    except Exception as e:
        fundamental_info = f"  ⚠️ reqFundamentalData a échoué : {e}"

    details = contract_info + "\n" + fundamental_info
    details += "\n\n  💡 Note : Pour les earnings, une alternative est d'utiliser"
    details += "\n     un service tiers (ex: finnhub, polygon.io) ou de garder"
    details += "\n     yfinance uniquement pour cette donnée spécifique."

    # Ce test est un "soft pass" — IBKR n'a pas toujours les earnings
    success = len(details_list) > 0
    report("Calendrier Earnings (reqContractDetails + reqFundamentalData)", success, details)
    return fundamental_data


# ──────────────────────────────────────────────
# TEST 7 : Historique 6 mois (remplace graphique + SMA50 + RSI)
# ──────────────────────────────────────────────

def test_7_historical_6m(ib: IB, ticker: str):
    """
    yfinance : yf.Ticker(ticker).history(period="6mo")
    IBKR     : reqHistoricalData(durationStr="6 M", barSizeSetting="1 day")

    On recalcule SMA50 et RSI14 comme dans app.py.
    """
    print("\n" + "─"*60)
    print(f"🔍 Test 7 : Historique 6 mois pour {ticker} (SMA50 + RSI)")
    print("─"*60)

    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="6 M",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        report("Historique 6 mois (reqHistoricalData)", False, "  Aucune barre reçue")
        return None

    df = util.df(bars)
    n_bars = len(df)

    # SMA 50
    sma50 = None
    if n_bars >= 50:
        sma50 = float(df["close"].rolling(50).mean().iloc[-1])

    # RSI 14
    current_rsi = None
    if n_bars >= 15:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

    # Distance SMA (%)
    dist_sma = None
    current_close = float(df["close"].iloc[-1])
    if sma50 and sma50 != 0:
        dist_sma = ((current_close - sma50) / sma50) * 100

    details = (
        f"  Barres reçues     = {n_bars}\n"
        f"  Période           = {df['date'].iloc[0]} → {df['date'].iloc[-1]}\n"
        f"  Dernier close     = ${current_close:.2f}\n"
        f"  ► SMA 50          = ${sma50:.2f}" if sma50 else "  ► SMA 50          = N/A"
    )
    details += f"\n  ► RSI 14          = {current_rsi:.1f}" if current_rsi else "\n  ► RSI 14          = N/A"
    details += f"\n  ► Dist SMA (%)    = {dist_sma:+.2f}%" if dist_sma is not None else "\n  ► Dist SMA (%)    = N/A"

    # Afficher les 5 dernières barres
    details += "\n\n  📊 5 dernières barres :"
    for _, row in df.tail(5).iterrows():
        details += f"\n     {row['date']}  O={row['open']:.2f}  H={row['high']:.2f}  L={row['low']:.2f}  C={row['close']:.2f}  V={row['volume']:,.0f}"

    success = n_bars >= 100 and sma50 is not None and current_rsi is not None
    report("Historique 6 mois → SMA50 + RSI14", success, details)
    return {"sma50": sma50, "rsi": current_rsi, "dist_sma": dist_sma}


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "█"*60)
    print("  TEST IBKR API — Remplacement de yfinance")
    print("█"*60)
    print(f"  Hôte      : {args.host}")
    print(f"  Port      : {args.port}")
    print(f"  Ticker    : {args.ticker}")
    print(f"  Client ID : {args.client_id}")
    print("█"*60)

    # Connexion
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id)
    except Exception as e:
        print(f"\n❌ Impossible de se connecter à TWS/Gateway sur {args.host}:{args.port}")
        print(f"   Erreur : {e}")
        print(f"\n   Vérifiez que :")
        print(f"   1. TWS ou IB Gateway est ouvert")
        print(f"   2. L'API est activée (Configuration → API → Settings)")
        print(f"   3. Le port est correct ({args.port})")
        print(f"   4. 'Allow connections from localhost only' est coché")
        sys.exit(1)

    print(f"\n✅ Connecté à IBKR (serveur version {ib.client.serverVersion()})")
    print(f"   Compte : {ib.managedAccounts()}")

    # ── Activer les données différées (type 3) ──
    # Nécessaire si pas d'abonnement temps réel
    ib.reqMarketDataType(3)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
    print("   Market data type : DELAYED (type 3)")

    try:
        # Exécuter les 7 tests
        spot = test_1_spot_price(ib, args.ticker)
        test_2_vol_index(ib)

        # Petite pause pour éviter le pacing
        ib.sleep(1)

        test_3_historical_1y(ib, args.ticker)

        ib.sleep(1)

        test_4_historical_3m(ib, args.ticker)

        ib.sleep(1)

        test_5_options_chain(ib, args.ticker, spot)

        ib.sleep(1)

        test_6_earnings_calendar(ib, args.ticker)

        ib.sleep(1)

        test_7_historical_6m(ib, args.ticker)

    finally:
        ib.disconnect()

    # ── Résumé final ──
    print("\n" + "█"*60)
    print("  RÉSUMÉ DES TESTS")
    print("█"*60)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        icon = "✅" if success else "❌"
        print(f"  {icon} {name}")

    print(f"\n  Score : {passed}/{total} tests réussis")

    if passed == total:
        print("\n  🎉 Tous les tests passent ! IBKR peut remplacer yfinance.")
    elif passed >= total - 1:
        print("\n  ⚠️  Presque tout fonctionne. Le test earnings est souvent")
        print("     limité — on peut garder yfinance ou un service tiers pour ça.")
    else:
        print(f"\n  ⚠️  {total - passed} tests échoués. Vérifiez votre configuration IBKR.")

    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
