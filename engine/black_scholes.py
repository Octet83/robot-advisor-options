"""
engine/black_scholes.py — Modèle Black-Scholes et fonctions P&L
================================================================
Fonctions pures : aucune dépendance Streamlit ni data provider.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from config import RISK_FREE_RATE


# ──────────────────────────────────────────────
# Greeks
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Leg Greeks & P&L
# ──────────────────────────────────────────────

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
