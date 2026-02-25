"""
engine/strategy.py — Moteur de stratégie d'options
====================================================
Contient build_strategy() et les helpers de sélection de strikes.
"""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, VOL_INDEX_NAMES
from engine.black_scholes import (
    black_scholes_delta,
    compute_leg_greeks,
    compute_real_probabilities,
    simulate_pnl,
)
from engine.indicators import compute_historical_vol


# ──────────────────────────────────────────────
# Helpers — Sélection de strikes
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
    Fallback sur lastPrice si bid/ask sont absents (fréquent avec yfinance).
    """
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)
    # Fallback : utiliser lastPrice si disponible
    last = row.get("lastPrice", 0) or 0
    if last > 0:
        return round(float(last), 2)
    return 0.0


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


def filter_liquid_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les options illiquides de la chaîne.
    Si bid/ask sont disponibles, applique les filtres stricts.
    Sinon, fallback sur lastPrice + openInterest.
    """
    if df.empty:
        return df
    filtered = df.copy()

    # Synthétiser bid/ask à partir de lastPrice quand absents
    if "lastPrice" in filtered.columns:
        mask_no_bid = filtered["bid"] <= 0
        mask_has_last = filtered["lastPrice"] > 0
        synth = mask_no_bid & mask_has_last
        if synth.any():
            # Créer un spread synthétique de ±2% autour du lastPrice
            filtered.loc[synth, "bid"] = (filtered.loc[synth, "lastPrice"] * 0.98).round(2)
            filtered.loc[synth, "ask"] = (filtered.loc[synth, "lastPrice"] * 1.02).round(2)

    # Exclure bid == 0 (même après synthèse)
    filtered = filtered[filtered["bid"] > 0]
    # Exclure open interest trop faible
    if "openInterest" in filtered.columns:
        filtered = filtered[filtered["openInterest"] >= 10]
    # Exclure spread bid/ask excessif
    mid = (filtered["bid"] + filtered["ask"]) / 2
    spread_pct = (filtered["ask"] - filtered["bid"]) / mid
    filtered = filtered[spread_pct <= 0.40]
    return filtered.reset_index(drop=True)


# ──────────────────────────────────────────────
# Moteur principal
# ──────────────────────────────────────────────

def build_strategy(spot: float, vix: float, iv_rank: float, bias: str,
                   budget: float, ticker: str, vol_symbol: str = "^VIX",
                   *, data_provider=None):
    """
    Moteur principal. Sélectionne et construit la stratégie optimale.
    Retourne un dict avec : name, explanation, legs, metrics, exit_plan.

    data_provider: instance de DataProvider (si None, utilise YFinanceProvider).
    """
    if data_provider is None:
        from data.yfinance_provider import YFinanceProvider
        data_provider = YFinanceProvider()

    # --- Récupération de la chaîne d'options ~45 DTE ---
    exp_str, calls, puts, dte = data_provider.get_options_chain(ticker)

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
            next_open_et = market_open_et
            if now_et >= market_close_et or not is_weekday:
                days_ahead = 1
                next_day = now_et + dt.timedelta(days=days_ahead)
                while next_day.weekday() >= 5:
                    days_ahead += 1
                    next_day = now_et + dt.timedelta(days=days_ahead)
                next_open_et = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
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

            sell_put = find_strike_by_delta(puts, spot, T, sigma, -0.16, "put")
            sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.16, "call")

            if sell_put is None or sell_call is None:
                raise ValueError("Impossible de trouver les strikes appropriés dans la chaîne d'options.")

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

            target_width = max(1.0, round(spot * 0.015))
            put_strikes = sorted(puts["strike"].unique())
            call_strikes = sorted(calls["strike"].unique())

            buy_put_target = sell_put_strike - target_width
            candidates_put = [s for s in put_strikes if s < sell_put_strike]
            if not candidates_put:
                raise ValueError("Pas de strikes de protection disponibles pour le Put side de l'Iron Condor.")
            buy_put_strike = min(candidates_put, key=lambda x: abs(x - buy_put_target))

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

            if put_width > target_width * 3 or call_width > target_width * 3:
                raise ValueError(
                    f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                    f"(put: {put_width:.0f}$, call: {call_width:.0f}$ vs cible: {target_width:.0f}$). "
                    f"Chaîne d'options trop peu liquide pour un Iron Condor fiable."
                )

            if net_credit <= 0 or net_credit >= max_width:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = (max_width * 100) - (net_credit * 100)

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\\$) pour un Iron Condor standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\\$."
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

            if width > target_width * 3:
                raise ValueError(
                    f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                    f"(écart réel : {width:.0f}$ vs cible : {target_width:.0f}$). "
                    f"Chaîne d'options trop peu liquide pour un spread fiable."
                )

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
                    f"Budget insuffisant ({budget}\\$) pour un Bull Put Spread standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\\$."
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

            if width > target_width * 3:
                raise ValueError(
                    f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                    f"(écart réel : {width:.0f}$ vs cible : {target_width:.0f}$). "
                    f"Chaîne d'options trop peu liquide pour un spread fiable."
                )

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
                    f"Budget insuffisant ({budget}\\$) pour un Bear Call Spread standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\\$."
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

            leaps = data_provider.get_leaps_chain(ticker)
            if leaps is None:
                raise ValueError("Pas d'options LEAPS disponibles (>200 DTE) pour le PMCC.")
            leaps_exp, leaps_calls, _, leaps_dte = leaps

            sigma_leaps = estimate_sigma(leaps_calls, spot)
            leaps_T = leaps_dte / 365.0

            buy_call = find_strike_by_delta(leaps_calls, spot, leaps_T, sigma_leaps, 0.80, "call")
            if buy_call is None:
                raise ValueError("Impossible de trouver un LEAPS approprié.")
            buy_call_strike = float(buy_call["strike"])
            buy_call_price = get_mid_price(buy_call)

            sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.30, "call")
            if sell_call is None:
                raise ValueError("Impossible de trouver le call court terme.")
            sell_call_strike = float(sell_call["strike"])
            sell_call_price = get_mid_price(sell_call)

            net_debit = buy_call_price - sell_call_price

            if net_debit <= 0:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = net_debit * 100

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\\$) pour le PMCC. "
                    f"Débit net estimé : {max_risk:.0f}\\$."
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

            short_chain = data_provider.get_short_term_chain(ticker)
            if short_chain is None:
                raise ValueError("Pas d'expiration court terme disponible pour le Calendar Spread.")
            short_exp, short_calls, _, short_dte = short_chain

            atm_strike = min(calls["strike"], key=lambda x: abs(x - spot))

            short_row = short_calls[short_calls["strike"] == atm_strike]
            if short_row.empty:
                short_row = short_calls.iloc[(short_calls["strike"] - atm_strike).abs().argsort()[:1]]
                atm_strike = float(short_row["strike"].iloc[0])
            sell_price = get_mid_price(short_row.iloc[0])

            long_row = calls[calls["strike"] == atm_strike]
            if long_row.empty:
                long_row = calls.iloc[(calls["strike"] - atm_strike).abs().argsort()[:1]]
            buy_price = get_mid_price(long_row.iloc[0])

            net_debit = buy_price - sell_price

            if net_debit <= 0:
                raise ValueError(
                    "Les prix de la chaîne d'options sont illogiques "
                    "(illiquidité majeure ou bid/ask cassé). "
                    "Analyse annulée pour votre sécurité."
                )

            max_risk = net_debit * 100

            if max_risk > budget:
                raise ValueError(
                    f"Budget insuffisant ({budget}\\$) pour le Calendar Spread. "
                    f"Débit net estimé : {max_risk:.0f}\\$."
                )

            result["legs"] = [
                {"action": "BUY", "type": "Call", "strike": atm_strike,
                 "exp": exp_str, "dte": dte, "price": buy_price},
                {"action": "SELL", "type": "Call", "strike": atm_strike,
                 "exp": short_exp, "dte": short_dte, "price": sell_price},
            ]
            result["credit_or_debit"] = round(-net_debit * 100, 2)
            result["max_risk"] = round(max_risk, 2)
            result["max_profit"] = round(max_risk * 0.5, 2)


        else:  # Baissier en basse vol
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

            if width > target_width * 3:
                raise ValueError(
                    f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                    f"(écart réel : {width:.0f}$ vs cible : {target_width:.0f}$). "
                    f"Chaîne d'options trop peu liquide pour un spread fiable."
                )

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
                    f"Budget insuffisant ({budget}\\$) pour un Bear Put Spread standard sur {ticker}. "
                    f"Risque par contrat : {max_risk:.0f}\\$."
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
                lower_puts = puts[puts["strike"] * 100 - sell_put_price * 100 <= budget]
                if lower_puts.empty:
                    raise ValueError(f"Budget insuffisant ({budget}\\$) pour un Cash Secured Put sur {ticker}.")
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

                if width > target_width * 3:
                    raise ValueError(
                        f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                        f"(écart réel : {width:.0f}$ vs cible : {target_width:.0f}$). "
                        f"Chaîne d'options trop peu liquide pour un spread fiable."
                    )

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
                        f"Budget insuffisant ({budget}\\$) pour un Bull Call Spread standard sur {ticker}. "
                        f"Risque par contrat : {max_risk:.0f}\\$."
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

                if width > target_width * 3:
                    raise ValueError(
                        f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                        f"(écart réel : {width:.0f}$ vs cible : {target_width:.0f}$). "
                        f"Chaîne d'options trop peu liquide pour un spread fiable."
                    )

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
                        f"Budget insuffisant ({budget}\\$) pour un Bear Put Spread standard sur {ticker}. "
                        f"Risque par contrat : {max_risk:.0f}\\$."
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

                sell_put = find_strike_by_delta(puts, spot, T, sigma, -0.16, "put")
                sell_call = find_strike_by_delta(calls, spot, T, sigma, 0.16, "call")

                if sell_put is None or sell_call is None:
                    raise ValueError("Impossible de trouver les strikes appropriés pour l'Iron Condor.")

                sell_put_strike = float(sell_put["strike"])
                sell_call_strike = float(sell_call["strike"])

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

                target_width = max(1.0, round(spot * 0.015))
                put_strikes = sorted(puts["strike"].unique())
                call_strikes = sorted(calls["strike"].unique())

                buy_put_target = sell_put_strike - target_width
                candidates_put = [s for s in put_strikes if s < sell_put_strike]
                if not candidates_put:
                    raise ValueError("Pas de strikes de protection disponibles pour le Put side de l'Iron Condor.")
                buy_put_strike = min(candidates_put, key=lambda x: abs(x - buy_put_target))

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

                if put_width > target_width * 3 or call_width > target_width * 3:
                    raise ValueError(
                        f"Les strikes disponibles sur « {ticker} » sont trop espacés "
                        f"(put: {put_width:.0f}$, call: {call_width:.0f}$ vs cible: {target_width:.0f}$). "
                        f"Chaîne d'options trop peu liquide pour un Iron Condor fiable."
                    )

                if net_credit <= 0 or net_credit >= max_width:
                    raise ValueError(
                        "Les prix de la chaîne d'options sont illogiques "
                        "(illiquidité majeure ou bid/ask cassé). "
                        "Analyse annulée pour votre sécurité."
                    )

                max_risk = (max_width * 100) - (net_credit * 100)

                if max_risk > budget:
                    raise ValueError(
                        f"Budget insuffisant ({budget}\\$) pour un Iron Condor standard sur {ticker}. "
                        f"Risque par contrat : {max_risk:.0f}\\$."
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
    result["sigma"] = sigma
    sigma_move = compute_historical_vol(ticker) or sigma
    probs = compute_real_probabilities(
        legs=result["legs"], spot=spot, dte=dte,
        sigma=sigma, qty=1,
        take_profit=take_profit_amount,
        max_risk=result["max_risk"],
        sigma_move=sigma_move,
    )
    result["probabilities"] = probs
    result["pop"] = probs["p_breakeven"]
    result["win_rate_estime"] = probs["p_take_profit"]

    # --- Espérance Mathématique (EV) ---
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
        result["ev"] = round(result["ev"] * qty, 2)
        for k in ["delta", "gamma", "theta", "vega"]:
            result["greeks"][k] = round(result["greeks"][k] * qty, 2)

    # --- RISK MANAGER : Kill Switch — Rejet EV Fortement Négative ---
    ev_threshold = -0.20 * result["max_risk"]
    if result.get("ev", 0) < ev_threshold:
        raise ValueError(
            f"Espérance Mathématique (EV) trop négative ({result['ev']:.2f}$). "
            f"Le ratio Risque/Gain est structurellement perdant. "
            f"Trade formellement rejeté."
        )

    return result
