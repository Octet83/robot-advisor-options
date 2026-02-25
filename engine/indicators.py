"""
engine/indicators.py — Indicateurs techniques (IV Rank, Vol, SMA, RSI, Trend)
==============================================================================
"""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf


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
