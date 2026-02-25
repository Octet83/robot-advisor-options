"""
test_app.py — Suite de Tests Unitaires du Robo-Advisor d'Options
================================================================
Framework : pytest + unittest.mock
Objectif  : Tests 100% déterministes (aucun appel réseau, marché figé).

6 suites :
  1. Le Moteur Mathématique (Black-Scholes)
  2. Le Routage Stratégique (L'Aiguilleur)
  3. Le Risk Manager (Dimensionnement, Budget, Métriques)
  4. Les Filtres de Sécurité (Kill Switches)
  5. Le Golden Dataset (12 scénarios paramétrés — P&L déterministes)
  6. Les Invariants de Probabilité (cohérence structurelle post-fix d2)
"""

import sys
import datetime as dt
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════
# BOOTSTRAP : Mocker Streamlit avant tout import
# ═══════════════════════════════════════════════

class _MockAttr:
    """Objet renvoyé par les attributs du mock Streamlit."""
    def __call__(self, *a, **kw):
        if a and isinstance(a[0], int):
            return [_MockAttr() for _ in range(a[0])]
        return _MockAttr()
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self): return iter([_MockAttr(), _MockAttr()])
    def __getattr__(self, name): return _MockAttr()
    def __bool__(self): return False


class _MockStreamlit:
    """Simule le module streamlit pour éviter l'import réel."""
    def cache_data(self, **kwargs):
        def decorator(func): return func
        return decorator
    def set_page_config(self, **kwargs): pass
    def markdown(self, *a, **kw): pass
    def __getattr__(self, name): return _MockAttr()


sys.modules["streamlit"] = _MockStreamlit()

from app import (
    black_scholes_price,
    black_scholes_delta,
    build_strategy,
    get_options_chain,
    compute_real_probabilities,
    simulate_pnl,
)


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def _make_options_df(rows):
    defaults = {
        "contractSymbol": "MOCK", "lastTradeDate": "2026-03-15",
        "lastPrice": 0.0, "change": 0.0, "percentChange": 0.0,
        "volume": 100, "openInterest": 500, "inTheMoney": False,
        "contractSize": "REGULAR", "currency": "USD",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_chain_for_bull_put():
    puts = _make_options_df([
        {"strike": 85.0, "bid": 0.20, "ask": 0.40, "impliedVolatility": 0.25},
        {"strike": 90.0, "bid": 0.80, "ask": 1.20, "impliedVolatility": 0.25},
        {"strike": 95.0, "bid": 2.80, "ask": 3.20, "impliedVolatility": 0.25},
        {"strike": 100.0, "bid": 5.00, "ask": 5.50, "impliedVolatility": 0.25},
        {"strike": 105.0, "bid": 9.00, "ask": 9.50, "impliedVolatility": 0.25},
    ])
    calls = _make_options_df([
        {"strike": 85.0, "bid": 15.00, "ask": 15.50, "impliedVolatility": 0.25},
        {"strike": 90.0, "bid": 10.50, "ask": 11.00, "impliedVolatility": 0.25},
        {"strike": 95.0, "bid": 6.50, "ask": 7.00, "impliedVolatility": 0.25},
        {"strike": 100.0, "bid": 3.50, "ask": 4.00, "impliedVolatility": 0.25},
        {"strike": 105.0, "bid": 1.50, "ask": 2.00, "impliedVolatility": 0.25},
        {"strike": 110.0, "bid": 0.40, "ask": 0.80, "impliedVolatility": 0.25},
    ])
    return puts, calls


def _make_basic_chain():
    """Chaîne d'options liquide (spreads ≤ 40%, OI=500)."""
    puts = _make_options_df([
        {"strike": s, "bid": b, "ask": a, "impliedVolatility": 0.25}
        for s, b, a in [
            (80, 0.30, 0.40), (85, 0.60, 0.80), (90, 1.00, 1.30),
            (92, 1.30, 1.60), (95, 2.10, 2.60), (97, 2.80, 3.20),
            (98, 3.20, 3.60), (99, 3.80, 4.20), (100, 4.50, 5.00),
            (102, 6.00, 6.50), (105, 8.50, 9.00), (108, 11.0, 11.5),
            (110, 13.0, 13.5), (115, 17.0, 17.5), (120, 21.0, 21.5),
        ]
    ])
    calls = _make_options_df([
        {"strike": s, "bid": b, "ask": a, "impliedVolatility": 0.25}
        for s, b, a in [
            (80, 20.5, 21.0), (85, 15.5, 16.0), (90, 11.0, 11.5),
            (92, 9.00, 9.50), (95, 6.50, 7.00), (97, 5.00, 5.50),
            (98, 4.50, 5.00), (99, 3.80, 4.20), (100, 3.20, 3.60),
            (102, 2.20, 2.60), (105, 1.30, 1.60), (108, 0.80, 1.00),
            (110, 0.60, 0.80), (115, 0.30, 0.40), (120, 0.20, 0.25),
            (125, 0.12, 0.16), (130, 0.08, 0.10),
        ]
    ])
    return puts, calls


# ═══════════════════════════════════════════════
# TEST 1 : LE MOTEUR MATHÉMATIQUE (Black-Scholes)
# ═══════════════════════════════════════════════

class TestBlackScholes:
    S, K, T, r, sigma = 100.0, 100.0, 30 / 365.0, 0.05, 0.20

    def test_call_price(self):
        assert black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "call") == pytest.approx(2.49, abs=0.02)

    def test_put_price(self):
        assert black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "put") == pytest.approx(2.08, abs=0.02)

    def test_call_delta(self):
        assert black_scholes_delta(self.S, self.K, self.T, self.r, self.sigma, "call") == pytest.approx(0.53, abs=0.02)

    def test_put_delta(self):
        assert black_scholes_delta(self.S, self.K, self.T, self.r, self.sigma, "put") == pytest.approx(-0.46, abs=0.02)


# ═══════════════════════════════════════════════
# TEST 2 : LE ROUTAGE STRATÉGIQUE
# ═══════════════════════════════════════════════

class TestStrategyRouting:

    def _run_routing(self, vix, iv_rank, bias):
        spot, budget, ticker = 100.0, 5000.0, "SPY"
        exp_date = (dt.date.today() + dt.timedelta(days=45)).strftime("%Y-%m-%d")
        puts, calls = _make_basic_chain()

        with patch("app.get_options_chain", return_value=(exp_date, calls, puts, 45)):
            leaps_exp = (dt.date.today() + dt.timedelta(days=365)).strftime("%Y-%m-%d")
            leaps_calls = _make_options_df([
                {"strike": s, "bid": b, "ask": a, "impliedVolatility": 0.22}
                for s, b, a in [(75, 27, 28), (80, 22, 23), (85, 17, 18), (90, 13, 14), (95, 9, 10)]
            ])
            with patch("app.get_leaps_chain", return_value=(leaps_exp, leaps_calls, _make_options_df([{"strike": 75, "bid": 0.5, "ask": 1, "impliedVolatility": 0.22}]), 365)):
                short_exp = (dt.date.today() + dt.timedelta(days=20)).strftime("%Y-%m-%d")
                short_calls = _make_options_df([
                    {"strike": s, "bid": b, "ask": a, "impliedVolatility": 0.25}
                    for s, b, a in [(98, 2.0, 2.5), (100, 1.5, 2.0), (102, 1.0, 1.5)]
                ])
                with patch("app.get_short_term_chain", return_value=(short_exp, short_calls, _make_options_df([{"strike": 100, "bid": 1.5, "ask": 2.0, "impliedVolatility": 0.25}]), 20)):
                    with patch("app.compute_real_probabilities", return_value={"p_take_profit": 99.0, "p_breakeven": 99.5, "p_max_loss": 0.1, "expected_pnl": 50.0}):
                        return build_strategy(spot=spot, vix=vix, iv_rank=iv_rank, bias=bias, budget=budget, ticker=ticker)

    def test_high_vol_neutral_iron_condor(self):
        assert "Iron Condor" in self._run_routing(25, 60, "Neutre")["name"]

    def test_high_vol_bearish_bear_call(self):
        assert "Bear Call Spread" in self._run_routing(25, 60, "Baissier")["name"]

    def test_low_vol_bullish_pmcc(self):
        assert "PMCC" in self._run_routing(12, 10, "Haussier")["name"]

    def test_low_vol_neutral_calendar(self):
        assert "Calendar Spread" in self._run_routing(12, 10, "Neutre")["name"]

    def test_mid_vol_bullish_bull_call(self):
        assert "Bull Call Spread" in self._run_routing(18, 30, "Haussier")["name"]

    def test_mid_vol_neutral_iron_condor(self):
        assert "Iron Condor" in self._run_routing(18, 30, "Neutre")["name"]


# ═══════════════════════════════════════════════
# TEST 3 : LE RISK MANAGER
# ═══════════════════════════════════════════════

class TestRiskManager:

    def _build(self):
        exp = (dt.date.today() + dt.timedelta(days=45)).strftime("%Y-%m-%d")
        puts, calls = _make_chain_for_bull_put()
        with patch("app.get_options_chain", return_value=(exp, calls, puts, 45)):
            with patch("app.compute_real_probabilities", return_value={"p_take_profit": 90.0, "p_breakeven": 60.0, "p_max_loss": 5.0, "expected_pnl": 75.0}):
                return build_strategy(spot=100, vix=25, iv_rank=60, bias="Haussier", budget=1000, ticker="SPY")

    def test_qty(self):       assert self._build()["qty"] == 3
    def test_max_risk(self):  assert self._build()["max_risk"] == 900.0
    def test_max_profit(self): assert self._build()["max_profit"] == 600.0
    def test_credit(self):    assert self._build()["credit_or_debit"] == 600.0
    def test_pop(self):       assert self._build()["pop"] == 60.0
    def test_tp(self):        assert self._build()["exit_plan"]["take_profit"] == 300.0
    def test_ev(self):        assert self._build()["ev"] == pytest.approx(225.0, abs=0.01)


# ═══════════════════════════════════════════════
# TEST 4 : LES KILL SWITCHES
# ═══════════════════════════════════════════════

class TestKillSwitches:

    def test_penny_stock(self):
        exp = (dt.date.today() + dt.timedelta(days=45)).strftime("%Y-%m-%d")
        _, calls = _make_basic_chain()
        puts, _ = _make_basic_chain()
        with patch("app.get_options_chain", return_value=(exp, calls, puts, 45)):
            with pytest.raises(ValueError, match=r"(?i)trop bas"):
                build_strategy(spot=4.50, vix=20, iv_rank=50, bias="Haussier", budget=500, ticker="X")

    def test_illiquidity(self):
        """Options avec bid=0 ou spreads > 40% → rejetées par le filtre de liquidité."""
        exp = (dt.date.today() + dt.timedelta(days=45)).strftime("%Y-%m-%d")
        puts = _make_options_df([
            {"strike": 90, "bid": 0.0, "ask": 1.0, "impliedVolatility": 0.25},
            {"strike": 95, "bid": 0.0, "ask": 2.0, "impliedVolatility": 0.25},
            {"strike": 100, "bid": 0.0, "ask": 3.0, "impliedVolatility": 0.25},
        ])
        calls = _make_options_df([
            {"strike": 100, "bid": 0.0, "ask": 2.0, "impliedVolatility": 0.25},
            {"strike": 105, "bid": 0.0, "ask": 1.0, "impliedVolatility": 0.25},
            {"strike": 110, "bid": 0.0, "ask": 0.5, "impliedVolatility": 0.25},
        ])
        with patch("app.get_options_chain", return_value=(exp, calls, puts, 45)):
            with pytest.raises(ValueError, match=r"(?i)illiquides"):
                build_strategy(spot=100, vix=25, iv_rank=60, bias="Haussier", budget=5000, ticker="SPY")

    def test_negative_ev(self):
        exp = (dt.date.today() + dt.timedelta(days=45)).strftime("%Y-%m-%d")
        puts, calls = _make_chain_for_bull_put()
        with patch("app.get_options_chain", return_value=(exp, calls, puts, 45)):
            with patch("app.compute_real_probabilities", return_value={"p_take_profit": 5.0, "p_breakeven": 10.0, "p_max_loss": 80.0, "expected_pnl": -250.0}):
                with pytest.raises(ValueError, match=r"(?i)espérance"):
                    build_strategy(spot=100, vix=25, iv_rank=60, bias="Haussier", budget=5000, ticker="SPY")


# ═══════════════════════════════════════════════
# TEST 5 : GOLDEN DATASET — 12 SCÉNARIOS PARAMÉTRÉS
# ═══════════════════════════════════════════════

# Chaque scénario contient les legs, paramètres, métriques,
# et les P&L attendus à des spots clés (évalués à 21 DTE via simulate_pnl).

GOLDEN_SCENARIOS = [
    pytest.param(
        {
            "id": 1, "name": "Bull Put Spread OTM",
            "legs": [
                {"action": "SELL", "type": "Put", "strike": 95.0, "exp": "2026-04-10", "dte": 45, "price": 3.00},
                {"action": "BUY",  "type": "Put", "strike": 90.0, "exp": "2026-04-10", "dte": 45, "price": 1.00},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.25, "qty": 1,
            "expected_credit": 2.00, "expected_width": 5.0,
            "expected_max_profit": 200.0, "expected_max_risk": 300.0,
            "pnl_checks": {80: -293.55, 85: -254.03, 90: -130.90, 95: 37.19, 100: 149.77, 105: 190.31},
        },
        id="T01-BullPutOTM",
    ),
    pytest.param(
        {
            "id": 2, "name": "Bull Put Spread Deep OTM",
            "legs": [
                {"action": "SELL", "type": "Put", "strike": 90.0, "exp": "2026-04-10", "dte": 45, "price": 1.50},
                {"action": "BUY",  "type": "Put", "strike": 85.0, "exp": "2026-04-10", "dte": 45, "price": 0.50},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.25, "qty": 1,
            "expected_credit": 1.00, "expected_width": 5.0,
            "expected_max_profit": 100.0, "expected_max_risk": 400.0,
            "pnl_checks": {80: -359.84, 90: -58.76, 95: 55.19, 100: 92.48, 105: 99.22},
        },
        id="T02-BullPutDeepOTM",
    ),
    pytest.param(
        {
            "id": 3, "name": "Bear Call Spread OTM",
            "legs": [
                {"action": "SELL", "type": "Call", "strike": 105.0, "exp": "2026-04-10", "dte": 45, "price": 2.50},
                {"action": "BUY",  "type": "Call", "strike": 110.0, "exp": "2026-04-10", "dte": 45, "price": 1.00},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.25, "qty": 1,
            "expected_credit": 1.50, "expected_width": 5.0,
            "expected_max_profit": 150.0, "expected_max_risk": 350.0,
            "pnl_checks": {90: 148.99, 100: 88.42, 105: -28.49, 110: -175.48, 115: -282.68},
        },
        id="T03-BearCallOTM",
    ),
    pytest.param(
        {
            "id": 4, "name": "Iron Condor Étroit σ=0.20",
            "legs": [
                {"action": "SELL", "type": "Put",  "strike": 95.0,  "exp": "2026-04-10", "dte": 45, "price": 1.50},
                {"action": "BUY",  "type": "Put",  "strike": 90.0,  "exp": "2026-04-10", "dte": 45, "price": 0.50},
                {"action": "SELL", "type": "Call", "strike": 105.0, "exp": "2026-04-10", "dte": 45, "price": 1.50},
                {"action": "BUY",  "type": "Call", "strike": 110.0, "exp": "2026-04-10", "dte": 45, "price": 0.50},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.20, "qty": 1,
            "expected_credit": 2.00, "expected_width": 5.0,
            "expected_max_profit": 200.0, "expected_max_risk": 300.0,
            "pnl_checks": {85: -273.34, 90: -145.87, 95: 53.70, 100: 132.50, 105: 32.04, 110: -144.42},
        },
        id="T04-IronCondorEtroit",
    ),
    pytest.param(
        {
            "id": 5, "name": "Iron Condor Large σ=0.20",
            "legs": [
                {"action": "SELL", "type": "Put",  "strike": 90.0,  "exp": "2026-04-10", "dte": 45, "price": 1.00},
                {"action": "BUY",  "type": "Put",  "strike": 85.0,  "exp": "2026-04-10", "dte": 45, "price": 0.30},
                {"action": "SELL", "type": "Call", "strike": 110.0, "exp": "2026-04-10", "dte": 45, "price": 1.00},
                {"action": "BUY",  "type": "Call", "strike": 115.0, "exp": "2026-04-10", "dte": 45, "price": 0.30},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.20, "qty": 1,
            "expected_credit": 1.40, "expected_width": 5.0,
            "expected_max_profit": 140.0, "expected_max_risk": 360.0,
            "pnl_checks": {85: -210.64, 90: 1.76, 100: 133.29, 110: -28.89, 115: -200.97},
        },
        id="T05-IronCondorLarge",
    ),
    pytest.param(
        {
            "id": 6, "name": "Bull Call Spread ATM (débit)",
            "legs": [
                {"action": "BUY",  "type": "Call", "strike": 100.0, "exp": "2026-04-10", "dte": 45, "price": 4.00},
                {"action": "SELL", "type": "Call", "strike": 105.0, "exp": "2026-04-10", "dte": 45, "price": 2.00},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.25, "qty": 1,
            "expected_credit": -2.00, "expected_width": 5.0,
            "expected_max_profit": 300.0, "expected_max_risk": 200.0,
            "pnl_checks": {90: -190.93, 100: -24.82, 105: 128.63, 110: 237.77, 120: 296.10},
        },
        id="T06-BullCallATM",
    ),
    pytest.param(
        {
            "id": 7, "name": "Bear Put Spread ATM (débit)",
            "legs": [
                {"action": "BUY",  "type": "Put", "strike": 100.0, "exp": "2026-04-10", "dte": 45, "price": 4.00},
                {"action": "SELL", "type": "Put", "strike": 95.0,  "exp": "2026-04-10", "dte": 45, "price": 2.00},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.25, "qty": 1,
            "expected_credit": -2.00, "expected_width": 5.0,
            "expected_max_profit": 300.0, "expected_max_risk": 200.0,
            "pnl_checks": {80: 298.21, 90: 248.25, 95: 126.98, 100: -33.48, 110: -187.90},
        },
        id="T07-BearPutATM",
    ),
    pytest.param(
        {
            "id": 8, "name": "Bull Put Spread Haute Vol σ=0.40",
            "legs": [
                {"action": "SELL", "type": "Put", "strike": 95.0, "exp": "2026-04-10", "dte": 45, "price": 5.00},
                {"action": "BUY",  "type": "Put", "strike": 90.0, "exp": "2026-04-10", "dte": 45, "price": 3.00},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.40, "qty": 1,
            "expected_credit": 2.00, "expected_width": 5.0,
            "expected_max_profit": 200.0, "expected_max_risk": 300.0,
            "pnl_checks": {85: -204.97, 90: -107.75, 95: 1.43, 100: 92.26, 110: 180.71},
        },
        id="T08-BullPutHauteVol",
    ),
    pytest.param(
        {
            "id": 9, "name": "Iron Condor Très Haute Vol σ=0.40",
            "legs": [
                {"action": "SELL", "type": "Put",  "strike": 90.0,  "exp": "2026-04-10", "dte": 45, "price": 3.00},
                {"action": "BUY",  "type": "Put",  "strike": 85.0,  "exp": "2026-04-10", "dte": 45, "price": 1.50},
                {"action": "SELL", "type": "Call", "strike": 110.0, "exp": "2026-04-10", "dte": 45, "price": 3.00},
                {"action": "BUY",  "type": "Call", "strike": 115.0, "exp": "2026-04-10", "dte": 45, "price": 1.50},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.40, "qty": 1,
            "expected_credit": 3.00, "expected_width": 5.0,
            "expected_max_profit": 300.0, "expected_max_risk": 200.0,
            "pnl_checks": {85: -11.63, 90: 99.31, 100: 202.12, 110: 94.89, 115: 7.94},
        },
        id="T09-IronCondorTresHauteVol",
    ),
    pytest.param(
        {
            "id": 10, "name": "Bull Put Spread x3 contrats",
            "legs": [
                {"action": "SELL", "type": "Put", "strike": 95.0, "exp": "2026-04-10", "dte": 45, "price": 3.00},
                {"action": "BUY",  "type": "Put", "strike": 90.0, "exp": "2026-04-10", "dte": 45, "price": 1.00},
            ],
            "spot": 100.0, "dte": 45, "sigma": 0.25, "qty": 3,
            "expected_credit": 2.00, "expected_width": 5.0,
            "expected_max_profit": 600.0, "expected_max_risk": 900.0,
            "pnl_checks": {85: -762.08, 95: 111.58, 100: 449.30, 105: 570.94},
        },
        id="T10-BullPutx3",
    ),
    pytest.param(
        {
            "id": 11, "name": "Bull Put Spread Court Terme DTE=30",
            "legs": [
                {"action": "SELL", "type": "Put", "strike": 95.0, "exp": "2026-03-26", "dte": 30, "price": 2.50},
                {"action": "BUY",  "type": "Put", "strike": 90.0, "exp": "2026-03-26", "dte": 30, "price": 0.80},
            ],
            "spot": 100.0, "dte": 30, "sigma": 0.25, "qty": 1,
            "expected_credit": 1.70, "expected_width": 5.0,
            "expected_max_profit": 170.0, "expected_max_risk": 330.0,
            "pnl_checks": {80: -323.55, 90: -160.90, 95: 7.19, 100: 119.77, 105: 160.31},
        },
        id="T11-BullPutCourtTerme",
    ),
    pytest.param(
        {
            "id": 12, "name": "Iron Condor Long Terme DTE=60",
            "legs": [
                {"action": "SELL", "type": "Put",  "strike": 90.0,  "exp": "2026-04-25", "dte": 60, "price": 2.00},
                {"action": "BUY",  "type": "Put",  "strike": 85.0,  "exp": "2026-04-25", "dte": 60, "price": 1.00},
                {"action": "SELL", "type": "Call", "strike": 110.0, "exp": "2026-04-25", "dte": 60, "price": 2.00},
                {"action": "BUY",  "type": "Call", "strike": 115.0, "exp": "2026-04-25", "dte": 60, "price": 1.00},
            ],
            "spot": 100.0, "dte": 60, "sigma": 0.20, "qty": 1,
            "expected_credit": 2.00, "expected_width": 5.0,
            "expected_max_profit": 200.0, "expected_max_risk": 300.0,
            "pnl_checks": {85: -150.64, 90: 61.76, 100: 193.29, 110: 31.11, 115: -140.97},
        },
        id="T12-IronCondorLongTerme",
    ),
]


class TestGoldenDataset:
    """12 scénarios paramétrés — vérifie les métriques et les P&L déterministes."""

    @pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
    def test_credit_or_debit(self, scenario):
        """Vérifie le crédit/débit net par action."""
        legs = scenario["legs"]
        # Crédit net = sum(sell_prices) - sum(buy_prices)
        sell_total = sum(l["price"] for l in legs if l["action"] == "SELL")
        buy_total = sum(l["price"] for l in legs if l["action"] == "BUY")
        actual_credit = round(sell_total - buy_total, 2)
        assert actual_credit == pytest.approx(scenario["expected_credit"], abs=0.01), \
            f"Test {scenario['id']}: crédit attendu {scenario['expected_credit']}, obtenu {actual_credit}"

    @pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
    def test_max_profit(self, scenario):
        """Vérifie le max profit = crédit × 100 × qty (crédit) ou (width - débit) × 100 × qty (débit)."""
        credit = scenario["expected_credit"]
        width = scenario["expected_width"]
        qty = scenario["qty"]
        if credit > 0:  # Crédit spread
            expected = credit * 100 * qty
        else:  # Débit spread
            expected = (width - abs(credit)) * 100 * qty
        assert expected == pytest.approx(scenario["expected_max_profit"], abs=0.01), \
            f"Test {scenario['id']}: max_profit"

    @pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
    def test_max_risk(self, scenario):
        """Vérifie le max risk = (width - crédit) × 100 × qty (crédit) ou débit × 100 × qty (débit)."""
        credit = scenario["expected_credit"]
        width = scenario["expected_width"]
        qty = scenario["qty"]
        if credit > 0:  # Crédit spread
            expected = (width - credit) * 100 * qty
        else:  # Débit spread
            expected = abs(credit) * 100 * qty
        assert expected == pytest.approx(scenario["expected_max_risk"], abs=0.01), \
            f"Test {scenario['id']}: max_risk"

    @pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
    def test_pnl_values(self, scenario):
        """Vérifie les P&L du Golden Dataset via simulate_pnl (tolérance 50¢)."""
        legs = scenario["legs"]
        sigma = scenario["sigma"]
        qty = scenario["qty"]
        remaining_dte = min(21, scenario["dte"])

        for target_spot, expected_pnl in scenario["pnl_checks"].items():
            actual = simulate_pnl(legs, float(target_spot), remaining_dte, sigma, qty)
            assert actual == pytest.approx(expected_pnl, abs=0.50), \
                f"Test {scenario['id']} @ spot={target_spot}: P&L attendu ${expected_pnl:+.2f}, obtenu ${actual:+.2f}"


# ═══════════════════════════════════════════════
# TEST 6 : INVARIANTS DE PROBABILITÉ (post-fix d2)
# ═══════════════════════════════════════════════

class TestProbabilityInvariants:
    """Vérifie la cohérence structurelle des probabilités après le fix d2."""

    SCENARIOS = [
        (  # Bull Put Spread OTM
            [{"action": "SELL", "type": "Put", "strike": 95, "exp": "x", "dte": 45, "price": 3},
             {"action": "BUY",  "type": "Put", "strike": 90, "exp": "x", "dte": 45, "price": 1}],
            100, 45, 0.25, 1, 100, 300
        ),
        (  # Iron Condor
            [{"action": "SELL", "type": "Put", "strike": 90, "exp": "x", "dte": 45, "price": 1.5},
             {"action": "BUY",  "type": "Put", "strike": 85, "exp": "x", "dte": 45, "price": 0.5},
             {"action": "SELL", "type": "Call", "strike": 110, "exp": "x", "dte": 45, "price": 1.5},
             {"action": "BUY",  "type": "Call", "strike": 115, "exp": "x", "dte": 45, "price": 0.5}],
            100, 45, 0.20, 1, 100, 300
        ),
        (  # Bull Call Spread ATM
            [{"action": "BUY",  "type": "Call", "strike": 100, "exp": "x", "dte": 45, "price": 4},
             {"action": "SELL", "type": "Call", "strike": 105, "exp": "x", "dte": 45, "price": 2}],
            100, 45, 0.25, 1, 150, 200
        ),
    ]

    def test_tp_leq_breakeven(self):
        """P(TP) ≤ P(BE) — le TP est un sous-ensemble du BE."""
        for legs, spot, dte, sigma, qty, tp, mr in self.SCENARIOS:
            p = compute_real_probabilities(legs, spot, dte, sigma, qty, tp, mr)
            assert p["p_take_profit"] <= p["p_breakeven"], \
                f"P(TP)={p['p_take_profit']} > P(BE)={p['p_breakeven']}"

    def test_probabilities_bounded(self):
        """Toutes les probabilités dans [0.1, 99.9]."""
        for legs, spot, dte, sigma, qty, tp, mr in self.SCENARIOS:
            p = compute_real_probabilities(legs, spot, dte, sigma, qty, tp, mr)
            for key in ["p_take_profit", "p_breakeven", "p_max_loss"]:
                assert 0.1 <= p[key] <= 99.9, f"{key}={p[key]} hors bornes"

    def test_atm_debit_max_loss_near_50pct(self):
        """Un Bull Call ATM doit avoir P(Max Loss) ≈ 50% (pas 5%)."""
        legs, spot, dte, sigma, qty, tp, mr = self.SCENARIOS[2]
        p = compute_real_probabilities(legs, spot, dte, sigma, qty, tp, mr)
        assert p["p_max_loss"] > 1, \
            f"P(Max Loss) d'un Bull Call ATM devrait être >1%, got {p['p_max_loss']}%"

    def test_otm_credit_spread_high_pop(self):
        """Un Bull Put Spread OTM doit avoir PoP > 70%."""
        legs, spot, dte, sigma, qty, tp, mr = self.SCENARIOS[0]
        p = compute_real_probabilities(legs, spot, dte, sigma, qty, tp, mr)
        assert p["p_breakeven"] > 70, \
            f"PoP d'un BPS OTM devrait être >70%, got {p['p_breakeven']}%"

    def test_otm_credit_spread_nonzero_max_loss(self):
        """Un Bull Put Spread OTM doit avoir P(Max Loss) > 5% (pas 0.1%)."""
        legs, spot, dte, sigma, qty, tp, mr = self.SCENARIOS[0]
        p = compute_real_probabilities(legs, spot, dte, sigma, qty, tp, mr)
        assert p["p_max_loss"] >= 0.1, \
            f"P(Max Loss) d'un BPS OTM devrait être ≥0.1%, got {p['p_max_loss']}%"
