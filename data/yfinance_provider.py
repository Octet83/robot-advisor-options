"""
data/yfinance_provider.py — Implémentation yfinance du DataProvider
====================================================================
Source de données gratuite — utilisée par défaut.
"""

from __future__ import annotations

import datetime as dt
import yfinance as yf

from config import VOL_INDEX_MAP
from data.provider import DataProvider


class YFinanceProvider(DataProvider):
    """Fournisseur de données via l'API Yahoo Finance (gratuit, delayed)."""

    def get_spot_price(self, ticker: str) -> float:
        """Récupère le prix actuel (Spot) du ticker."""
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker « {ticker} ».")
        return float(hist["Close"].iloc[-1])

    def get_vol_index(self, ticker: str) -> tuple[float, str]:
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

    def get_options_chain(self, ticker: str, target_dte: int = 45):
        """
        Récupère la chaîne d'options et filtre l'expiration la plus proche
        de target_dte (fourchette 35-60 jours).
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
            diff = abs(dte - target_dte)
            if 35 <= dte <= 60 and diff < best_diff:
                best_diff = diff
                best_exp = exp_str
                best_dte = dte

        # Si rien dans [35,60], prend l'expiration la plus proche de target_dte
        if best_exp is None:
            for exp_str in expirations:
                exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte > 0:
                    diff = abs(dte - target_dte)
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = exp_str
                        best_dte = dte

        if best_exp is None:
            raise ValueError("Aucune expiration d'options valide trouvée.")

        chain = tk.option_chain(best_exp)
        return best_exp, chain.calls, chain.puts, best_dte

    def get_leaps_chain(self, ticker: str):
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

    def get_short_term_chain(self, ticker: str):
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
