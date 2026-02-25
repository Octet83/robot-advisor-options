"""
data/provider.py — Interface abstraite pour les données de marché
=================================================================
Permet de switcher entre yfinance (gratuit) et IBKR (temps réel).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    """Interface commune pour toutes les sources de données."""

    @abstractmethod
    def get_spot_price(self, ticker: str) -> float:
        """Retourne le prix spot courant du ticker."""
        ...

    @abstractmethod
    def get_vol_index(self, ticker: str) -> tuple[float, str]:
        """Retourne (valeur_vol, symbole_indice) pour le ticker."""
        ...

    @abstractmethod
    def get_options_chain(self, ticker: str, target_dte: int = 45):
        """
        Retourne (expiration_str, calls_df, puts_df, dte)
        pour l'expiration la plus proche de target_dte.
        """
        ...

    @abstractmethod
    def get_leaps_chain(self, ticker: str):
        """Retourne la chaîne LEAPS (>200 DTE) ou None."""
        ...

    @abstractmethod
    def get_short_term_chain(self, ticker: str):
        """Retourne la chaîne court terme (~20 DTE) ou None."""
        ...
