"""
data/hybrid_provider.py — Provider hybride IBKR + yfinance
============================================================
Essaie IBKR en priorité, fallback automatique vers yfinance.
Si IBKR n'est pas disponible (TWS fermé, ib_insync absent), tout
passe par yfinance de manière transparente.
"""

from __future__ import annotations

import logging
import sys

from data.provider import DataProvider
from data.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class HybridProvider(DataProvider):
    """
    Provider principal de l'application.

    Stratégie :
        1. Tente d'utiliser IBKR (temps réel, delayed)
        2. Si IBKR échoue → fallback silencieux vers yfinance
        3. Si ib_insync pas installé → yfinance uniquement

    L'attribut `last_source` trace la dernière source utilisée
    par méthode, pour affichage en UI.
    """

    def __init__(self, ibkr_host: str = "127.0.0.1", ibkr_port: int = 7496,
                 ibkr_client_id: int = None):
        import random
        if ibkr_client_id is None:
            ibkr_client_id = random.randint(100, 999)
        self._yf = YFinanceProvider()
        self._ibkr = None
        self._ibkr_available = False

        # Tenter d'initialiser et connecter IBKR
        try:
            from data.ibkr_provider import IBKRProvider, HAS_IB_INSYNC
            if HAS_IB_INSYNC:
                self._ibkr = IBKRProvider(
                    host=ibkr_host,
                    port=ibkr_port,
                    client_id=ibkr_client_id,
                )
                # Connexion immédiate dans le thread dédié
                try:
                    self._ibkr.connect()
                    self._ibkr_available = True
                    logger.info("HybridProvider : IBKR connecté ✅")
                except Exception as conn_err:
                    logger.info("HybridProvider : TWS non joignable (%s) → yfinance fallback", conn_err)
            else:
                logger.info("HybridProvider : ib_insync absent → yfinance only")
        except Exception as e:
            logger.warning("HybridProvider : IBKR init échoué (%s) → yfinance only", e)

        # Trace de la source utilisée par méthode
        self.last_source: dict[str, str] = {}

    @property
    def ibkr_connected(self) -> bool:
        """True si IBKR est actuellement connecté."""
        return self._ibkr is not None and self._ibkr.is_connected

    def _try_ibkr(self, method_name: str, ticker: str, *args, **kwargs):
        """
        Essaie d'appeler une méthode sur IBKRProvider.
        Les appels sont exécutés dans le thread dédié IBKR.
        Retourne (result, True) si succès, (None, False) sinon.
        """
        if self._ibkr is None:
            return None, False

        # Tenter de reconnecter si désactivé
        if not self._ibkr_available:
            self._ibkr._ensure_connected()
            if self._ibkr.is_connected:
                self._ibkr_available = True
                logger.info("IBKR reconnecté → réactivé")
            else:
                return None, False

        # Vérifier que la socket est vivante
        self._ibkr._ensure_connected()

        try:
            method = getattr(self._ibkr, method_name)
            result = method(ticker, *args, **kwargs)
            self.last_source[method_name] = "IBKR"
            return result, True
        except Exception as e:
            err_msg = str(e) or f"{type(e).__name__}: (no message)"
            print(f"[IBKR] {method_name}({ticker}) échoué: {err_msg}", file=sys.stderr)
            logger.warning(
                "IBKR %s(%s) échoué: %s → fallback yfinance",
                method_name, ticker, err_msg,
            )
            if "connect" in str(e).lower() or "timeout" in str(e).lower():
                logger.warning("IBKR connexion perdue → désactivation")
                self._ibkr_available = False
            return None, False

    # ── Interface DataProvider ──
    # Toutes les données passent par yfinance (IBKR trop lent/instable)
    # IBKR est utilisé uniquement pour les ordres

    def get_spot_price(self, ticker: str) -> float:
        self.last_source["get_spot_price"] = "yfinance"
        return self._yf.get_spot_price(ticker)

    def get_vol_index(self, ticker: str) -> tuple[float, str]:
        self.last_source["get_vol_index"] = "yfinance"
        return self._yf.get_vol_index(ticker)

    def get_options_chain(self, ticker: str, target_dte: int = 45):
        self.last_source["get_options_chain"] = "yfinance"
        return self._yf.get_options_chain(ticker, target_dte)

    def get_leaps_chain(self, ticker: str):
        self.last_source["get_leaps_chain"] = "yfinance"
        return self._yf.get_leaps_chain(ticker)

    def get_short_term_chain(self, ticker: str):
        self.last_source["get_short_term_chain"] = "yfinance"
        return self._yf.get_short_term_chain(ticker)

    # ── Cleanup ──

    def disconnect(self):
        """Déconnecte IBKR proprement."""
        if self._ibkr:
            self._ibkr.disconnect()
