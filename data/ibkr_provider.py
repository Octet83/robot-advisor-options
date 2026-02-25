"""
data/ibkr_provider.py — Implémentation IBKR du DataProvider
=============================================================
Source de données temps réel via Interactive Brokers (TWS / IB Gateway).
Utilise ib_insync avec données delayed (type 3) par défaut.

Architecture thread-safe :
    Un thread dédié possède l'objet IB et son event loop.
    Les appels depuis Streamlit sont délégués à ce thread via
    concurrent.futures et exécutés de manière synchrone.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import numpy as np

from config import VOL_INDEX_MAP
from data.provider import DataProvider

logger = logging.getLogger(__name__)

# Import conditionnel de ib_insync
# ib_insync nécessite un event loop asyncio même à l'import.
# Streamlit's ScriptRunner thread n'en a pas → on en crée un.
try:
    _loop = asyncio.get_event_loop()
    if _loop.is_closed():
        raise RuntimeError("closed")
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock, Index, util
    HAS_IB_INSYNC = True
except Exception:
    HAS_IB_INSYNC = False


def _is_valid(val) -> bool:
    """Vérifie qu'une valeur de market data est valide."""
    if val is None:
        return False
    try:
        return not np.isnan(val) and val > 0
    except (TypeError, ValueError):
        return False


class IBKRProvider(DataProvider):
    """
    Fournisseur de données via l'API Interactive Brokers.

    Thread-safe : un thread dédié possède l'objet IB et son event loop.
    Toutes les opérations IBKR sont exécutées dans ce thread.
    Les chaînes d'options délèguent à yfinance (phase 1).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7496,
                 client_id: int = 10, timeout: float = 8.0):
        if not HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync n'est pas installé. "
                "Installez-le avec : pip install ib_insync"
            )
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout

        # Thread dédié avec son event loop et son objet IB
        self._ib: IB | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ibkr"
        )
        self._connected = False

        # Fallback yfinance pour les chaînes d'options (phase 1)
        from data.yfinance_provider import YFinanceProvider
        self._yf_fallback = YFinanceProvider()

    # ── Thread dédié IBKR ──

    def _ensure_connected(self):
        """Vérifie que la connexion IB est vivante, reconnecte si nécessaire."""
        if not self._connected:
            return

        def _check():
            if self._ib and self._ib.isConnected():
                return True
            return False

        try:
            future = self._executor.submit(_check)
            alive = future.result(timeout=5)
        except Exception:
            alive = False

        if not alive:
            logger.warning("IBKR socket stale — reconnexion…")
            # Recréer l'executor et reconnecter
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # Nouveau clientId pour éviter "already in use"
            import random
            self._client_id = random.randint(100, 999)
            try:
                self.connect()
                logger.info("IBKR reconnexion réussie ✅")
            except Exception as e:
                logger.error("IBKR reconnexion échouée: %s", e)
                self._connected = False

    def _run_in_ibkr_thread(self, fn, timeout=10):
        """Exécute une fonction dans le thread IBKR dédié.
        Le thread crée son propre event loop et connexion IB."""
        future = self._executor.submit(fn)
        return future.result(timeout=timeout)

    def connect(self):
        """Connexion à TWS/Gateway dans le thread dédié."""
        def _connect():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ib = IB()
            self._ib.connect(
                self._host, self._port,
                clientId=self._client_id,
                timeout=10,
            )
            self._ib.reqMarketDataType(3)  # Données delayed
            self._connected = True
            logger.info(
                "Connecté à IBKR sur %s:%s (delayed data) [thread: %s]",
                self._host, self._port, threading.current_thread().name,
            )

        self._run_in_ibkr_thread(_connect)

    def disconnect(self):
        """Déconnexion propre."""
        def _disconnect():
            if self._ib and self._ib.isConnected():
                self._ib.disconnect()
            self._connected = False
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            logger.info("Déconnecté de IBKR")

        try:
            self._run_in_ibkr_thread(_disconnect)
        except Exception:
            pass
        self._executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Spot Price ──

    def get_spot_price(self, ticker: str) -> float:
        """Prix spot via reqMktData (delayed), fallback historique."""
        def _fetch():
            ib = self._ib
            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)

            # Méthode 1 : reqMktData delayed
            md = ib.reqMktData(contract, "", snapshot=False)
            steps = int(self._timeout / 0.1)
            for _ in range(steps):
                ib.sleep(0.1)
                if _is_valid(md.last) or _is_valid(md.close) or _is_valid(md.bid):
                    break

            spot = md.marketPrice()
            ib.cancelMktData(contract)

            if _is_valid(spot):
                logger.debug("Spot %s via reqMktData: %.2f", ticker, spot)
                return float(spot)

            # Méthode 2 : fallback historique
            logger.debug("Spot %s: reqMktData échoué, fallback historique", ticker)
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
                logger.debug("Spot %s via historique: %.2f", ticker, spot)
                return spot

            raise ValueError(
                f"IBKR : impossible d'obtenir le prix spot pour « {ticker} »."
            )

        return self._run_in_ibkr_thread(_fetch)

    # ── Vol Index ──

    def get_vol_index(self, ticker: str) -> tuple[float, str]:
        """Indice de volatilité via reqMktData, fallback historique."""
        def _fetch():
            ib = self._ib
            vol_symbol = VOL_INDEX_MAP.get(ticker, "^VIX")
            ibkr_symbol = vol_symbol.lstrip("^")

            contract = Index(ibkr_symbol, "CBOE", "USD")
            ib.qualifyContracts(contract)

            md = ib.reqMktData(contract, "", snapshot=False)
            steps = int(self._timeout / 0.1)
            for _ in range(steps):
                ib.sleep(0.1)
                if _is_valid(md.last) or _is_valid(md.close) or _is_valid(md.bid):
                    break

            val = md.marketPrice()
            ib.cancelMktData(contract)

            if _is_valid(val):
                logger.debug("Vol %s (%s) via reqMktData: %.2f", ticker, vol_symbol, val)
                return float(val), vol_symbol

            # Fallback historique
            logger.debug("Vol %s: reqMktData échoué, fallback historique", vol_symbol)
            for what_to_show in ("TRADES", "MIDPOINT"):
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="5 D",
                    barSizeSetting="1 day",
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=1,
                )
                if bars:
                    val = float(bars[-1].close)
                    logger.debug("Vol %s via historique (%s): %.2f", vol_symbol, what_to_show, val)
                    return val, vol_symbol

            raise ValueError(
                f"IBKR : impossible d'obtenir l'indice de volatilité ({vol_symbol})."
            )

        return self._run_in_ibkr_thread(_fetch)

    # ── Options Chains ──

    def _fetch_chain(self, ticker: str, target_dte: int, dte_min: int = 0, dte_max: int = 999):
        """Fetche une chaîne d'options IBKR pour le target_dte donné."""
        import pandas as pd

        def _fetch():
            ib = self._ib
            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)

            # 1. Récupérer les expirations et strikes disponibles
            chains = ib.reqSecDefOptParams(
                contract.symbol, "", contract.secType, contract.conId
            )
            if not chains:
                raise ValueError(f"IBKR: aucune chaîne d'options pour « {ticker} ».")

            # Choisir la chaîne avec le plus de strikes (SMART a souvent très peu de données)
            chain = max(chains, key=lambda c: len(c.strikes))
            logger.info(
                "IBKR options %s: exchange=%s, %d exps, %d strikes",
                ticker, chain.exchange, len(chain.expirations), len(chain.strikes),
            )

            # 2. Trouver la meilleure expiration
            import datetime as dt
            today = dt.date.today()
            best_exp = None
            best_dte = None
            best_diff = float("inf")

            for exp_str in sorted(chain.expirations):
                exp_date = dt.datetime.strptime(exp_str, "%Y%m%d").date()
                dte = (exp_date - today).days
                if dte < dte_min or dte > dte_max:
                    continue
                diff = abs(dte - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
                    best_dte = dte

            if best_exp is None:
                raise ValueError(f"IBKR: aucune expiration trouvée pour DTE ~{target_dte}.")

            # 3. Filtrer les strikes autour du spot (±20%)
            spot = contract.marketPrice if hasattr(contract, 'marketPrice') else None
            if not spot or not _is_valid(spot):
                # Récupérer le spot rapidement
                md = ib.reqMktData(contract, "", snapshot=False)
                for _ in range(50):
                    ib.sleep(0.1)
                    if _is_valid(md.last) or _is_valid(md.close):
                        break
                spot = md.marketPrice()
                ib.cancelMktData(contract)

            if not _is_valid(spot):
                raise ValueError("IBKR: impossible d'obtenir le spot pour filtrer la chaîne.")

            # Filtrer les strikes ±10% autour du spot
            # SPY: $1 near ATM (±3%), $5 intervals further out
            all_strikes = sorted(s for s in chain.strikes
                               if spot * 0.90 <= s <= spot * 1.10)

            # Garder uniquement les strikes à intervalles standards
            # $1 pour ATM ±3%, $5 pour le reste
            strikes = []
            for s in all_strikes:
                pct_away = abs(s - spot) / spot
                if pct_away <= 0.03:
                    # Près de l'ATM : garder $1 intervals
                    if s % 1 == 0:
                        strikes.append(s)
                else:
                    # Plus loin : garder $5 intervals
                    if s % 5 == 0:
                        strikes.append(s)

            if not strikes:
                raise ValueError("IBKR: aucun strike valide trouvé autour du spot.")

            logger.info("IBKR chain %s: %d strikes autour de %.0f", ticker, len(strikes), spot)

            # 4. Créer les contrats d'options et qualifier
            from ib_insync import Option
            call_contracts = [Option(ticker, best_exp, s, "C", "SMART") for s in strikes]
            put_contracts = [Option(ticker, best_exp, s, "P", "SMART") for s in strikes]

            all_contracts = call_contracts + put_contracts

            # Qualifier par batch — ne garder que les contrats valides (conId > 0)
            qualified = []
            for i in range(0, len(all_contracts), 50):
                batch = all_contracts[i:i+50]
                result = ib.qualifyContracts(*batch)
                qualified.extend(c for c in result if c.conId > 0)

            if not qualified:
                raise ValueError("IBKR: aucun contrat d'option qualifié.")

            logger.info("IBKR chain %s: %d contrats qualifiés", ticker, len(qualified))

            # 5. Récupérer les market data (par batch de 50)
            tickers_data = []
            for i in range(0, len(qualified), 50):
                batch = qualified[i:i+50]
                tickers_data.extend(ib.reqTickers(*batch))
            ib.sleep(2)  # Laisser le temps aux données d'arriver

            # 6. Construire les DataFrames
            calls_rows = []
            puts_rows = []

            for t in tickers_data:
                c = t.contract
                bid = t.bid if _is_valid(t.bid) else 0.0
                ask = t.ask if _is_valid(t.ask) else 0.0
                last = t.last if _is_valid(t.last) else 0.0
                oi = t.open if hasattr(t, 'open') and t.open else 0

                # IV depuis modelGreeks si dispo
                iv = 0.0
                if t.modelGreeks and hasattr(t.modelGreeks, 'impliedVol'):
                    iv_val = t.modelGreeks.impliedVol
                    if iv_val and _is_valid(iv_val):
                        iv = float(iv_val)

                row = {
                    "strike": float(c.strike),
                    "bid": float(bid),
                    "ask": float(ask),
                    "lastPrice": float(last),
                    "openInterest": int(oi),
                    "impliedVolatility": iv,
                    "volume": 0,
                    "contractSymbol": c.localSymbol or f"{c.symbol}{c.lastTradeDateOrContractMonth}{c.right}{c.strike}",
                }

                if c.right == "C":
                    calls_rows.append(row)
                else:
                    puts_rows.append(row)

            calls_df = pd.DataFrame(calls_rows).sort_values("strike").reset_index(drop=True)
            puts_df = pd.DataFrame(puts_rows).sort_values("strike").reset_index(drop=True)

            # Formater la date comme yfinance (YYYY-MM-DD)
            exp_formatted = f"{best_exp[:4]}-{best_exp[4:6]}-{best_exp[6:]}"
            return exp_formatted, calls_df, puts_df, best_dte

        return self._run_in_ibkr_thread(_fetch)

    def get_options_chain(self, ticker: str, target_dte: int = 45):
        return self._fetch_chain(ticker, target_dte, dte_min=7, dte_max=90)

    def get_leaps_chain(self, ticker: str):
        try:
            return self._fetch_chain(ticker, target_dte=365, dte_min=200, dte_max=730)
        except Exception:
            return None

    def get_short_term_chain(self, ticker: str):
        try:
            return self._fetch_chain(ticker, target_dte=20, dte_min=5, dte_max=45)
        except Exception:
            return None

    # ── Portfolio & Account ──

    def get_portfolio(self) -> list[dict]:
        """Retourne les positions du portefeuille IBKR."""
        def _fetch():
            ib = self._ib
            portfolio = ib.portfolio()
            positions = []
            for item in portfolio:
                c = item.contract
                sec_type = c.secType  # STK, OPT, FUT, etc.
                label = c.localSymbol or c.symbol
                if sec_type == "OPT":
                    label = f"{c.symbol} {c.lastTradeDateOrContractMonth} {c.right}{c.strike}"
                positions.append({
                    "symbol": c.symbol,
                    "label": label,
                    "type": sec_type,
                    "position": float(item.position),
                    "market_price": float(item.marketPrice),
                    "market_value": float(item.marketValue),
                    "avg_cost": float(item.averageCost),
                    "unrealized_pnl": float(item.unrealizedPNL),
                    "realized_pnl": float(item.realizedPNL),
                    "currency": c.currency,
                })
            return positions

        return self._run_in_ibkr_thread(_fetch)

    def get_account_summary(self) -> dict:
        """Retourne un résumé du compte IBKR."""
        def _fetch():
            ib = self._ib
            values = ib.accountValues()

            # Détecter la devise de base du compte
            base_currency = "USD"
            for av in values:
                if av.tag == "NetLiquidation" and av.currency not in ("BASE",):
                    base_currency = av.currency
                    break

            summary = {"currency": base_currency}
            for av in values:
                if av.tag in (
                    "NetLiquidation", "TotalCashValue", "BuyingPower",
                    "GrossPositionValue", "MaintMarginReq",
                ) and av.currency == base_currency:
                    summary[av.tag] = float(av.value)
                elif av.tag == "UnrealizedPnL" and av.currency == "BASE":
                    summary["UnrealizedPnL"] = float(av.value)
                elif av.tag == "RealizedPnL" and av.currency == "BASE":
                    summary["RealizedPnL"] = float(av.value)
            return summary

        return self._run_in_ibkr_thread(_fetch)

    # ── Order Preparation ──

    def _build_combo(self, strategy: dict, ticker: str):
        """Construit le contrat BAG et l'ordre LimitOrder pour la stratégie."""
        from ib_insync import Option, LimitOrder, Contract, ComboLeg, TagValue

        ib = self._ib
        legs = strategy["legs"]
        qty = strategy.get("qty", 1)
        exp_raw = legs[0]["exp"].replace("-", "")  # YYYY-MM-DD → YYYYMMDD

        # 1. Qualifier chaque contrat d'option
        qualified_legs = []
        for leg in legs:
            right = "C" if leg["type"] == "Call" else "P"
            opt = Option(ticker, exp_raw, leg["strike"], right, "SMART")
            result = ib.qualifyContracts(opt)
            if not result or result[0].conId == 0:
                raise ValueError(
                    f"Impossible de qualifier {leg['type']} {leg['strike']} {leg['exp']}"
                )
            qualified_legs.append({
                "contract": result[0],
                "action": leg["action"],
                "price": leg["price"],
            })

        # 2. Construire le contrat BAG
        bag = Contract()
        bag.symbol = ticker
        bag.secType = "BAG"
        bag.currency = "USD"
        bag.exchange = "SMART"

        combo_legs = []
        for ql in qualified_legs:
            cl = ComboLeg()
            cl.conId = ql["contract"].conId
            cl.ratio = 1
            cl.action = ql["action"]
            cl.exchange = "SMART"
            combo_legs.append(cl)
        bag.comboLegs = combo_legs

        # 3. Calculer le prix net
        net_credit = strategy.get("net_credit", None)
        if net_credit is None:
            net_credit = sum(
                ql["price"] if ql["action"] == "SELL" else -ql["price"]
                for ql in qualified_legs
            )

        # 4. Déterminer action et prix limite
        if net_credit > 0:
            action = "SELL"
            limit_price = round(abs(net_credit), 2)
        else:
            action = "BUY"
            limit_price = round(abs(net_credit), 2)

        order = LimitOrder(
            action=action,
            totalQuantity=qty,
            lmtPrice=limit_price,
        )
        order.smartComboRoutingParams = [TagValue(tag='NonGuaranteed', value='1')]

        return bag, order, action, qty, limit_price

    def check_order(self, strategy: dict, ticker: str) -> dict:
        """
        Simule l'ordre (whatIf) pour voir marge et commission
        sans placer l'ordre réellement.
        """
        def _check():
            bag, order, action, qty, limit_price = self._build_combo(strategy, ticker)

            # whatIfOrder = simulation sans exécution
            state = self._ib.whatIfOrder(bag, order)
            self._ib.sleep(1)

            return {
                "action": action,
                "qty": qty,
                "limit_price": limit_price,
                "init_margin": state.initMarginChange or "N/A",
                "maint_margin": state.maintMarginChange or "N/A",
                "commission": state.commission if state.commission < 1e9 else 0,
                "max_commission": state.maxCommission if state.maxCommission < 1e9 else 0,
            }

        return self._run_in_ibkr_thread(_check)

    def place_order(self, strategy: dict, ticker: str) -> dict:
        """
        Place l'ordre pour de vrai dans TWS (transmit=True).
        Pour les Iron Condors (4 legs), on split en deux spreads 2-legs
        car IBKR rejette les combos 4-legs via API.
        """
        def _place():
            from ib_insync import Option, LimitOrder, Contract, ComboLeg, TagValue

            ib = self._ib
            legs = strategy["legs"]
            qty = strategy.get("qty", 1)
            exp_raw = legs[0]["exp"].replace("-", "")

            # 1. Qualifier tous les legs
            qualified = []
            for leg in legs:
                right = "C" if leg["type"] == "Call" else "P"
                opt = Option(ticker, exp_raw, leg["strike"], right, "SMART")
                result = ib.qualifyContracts(opt)
                if not result or result[0].conId == 0:
                    raise ValueError(f"Impossible de qualifier {leg['type']} {leg['strike']}")
                qualified.append({
                    "contract": result[0],
                    "action": leg["action"],
                    "price": leg["price"],
                    "type": leg["type"],
                })

            # 2. Grouper par type (Put legs / Call legs)
            put_legs = [q for q in qualified if q["type"] == "Put"]
            call_legs = [q for q in qualified if q["type"] == "Call"]

            # Si 4 legs → split en deux ordres 2-legs
            groups = []
            if len(put_legs) == 2 and len(call_legs) == 2:
                groups = [("Put Spread", put_legs), ("Call Spread", call_legs)]
            else:
                # 2 legs ou autre → un seul ordre
                groups = [("Spread", qualified)]

            results = []
            for label, group_legs in groups:
                bag = Contract()
                bag.symbol = ticker
                bag.secType = "BAG"
                bag.currency = "USD"
                bag.exchange = "SMART"

                combo_legs = []
                net = 0
                for ql in group_legs:
                    cl = ComboLeg()
                    cl.conId = ql["contract"].conId
                    cl.ratio = 1
                    cl.action = ql["action"]
                    cl.exchange = "SMART"
                    combo_legs.append(cl)
                    if ql["action"] == "SELL":
                        net += ql["price"]
                    else:
                        net -= ql["price"]
                bag.comboLegs = combo_legs

                if net > 0:
                    action = "SELL"
                    limit_price = round(abs(net), 2)
                else:
                    action = "BUY"
                    limit_price = round(abs(net), 2)

                order = LimitOrder(
                    action=action,
                    totalQuantity=qty,
                    lmtPrice=limit_price,
                    transmit=False,  # Ne PAS transmettre — l'utilisateur valide dans TWS
                )
                order.smartComboRoutingParams = [TagValue(tag='NonGuaranteed', value='1')]

                trade = ib.placeOrder(bag, order)

                # Attendre le statut
                for _ in range(10):
                    ib.sleep(0.5)
                    status = trade.orderStatus.status
                    if status and status.lower() not in ("", "presubmitted", "pendingsubmit"):
                        break

                log_msgs = []
                for entry in trade.log:
                    if hasattr(entry, 'message') and entry.message:
                        log_msgs.append(entry.message)

                results.append({
                    "order_id": trade.order.orderId,
                    "status": trade.orderStatus.status,
                    "label": label,
                    "action": action,
                    "limit_price": limit_price,
                    "logs": log_msgs,
                })

            # Combiner les résultats
            all_ids = [r["order_id"] for r in results]
            all_statuses = [r["status"] for r in results]
            all_logs = []
            for r in results:
                all_logs.extend(r.get("logs", []))

            worst_status = "Cancelled" if "Cancelled" in all_statuses else all_statuses[0]
            parts = " + ".join(f"#{r['order_id']} {r['label']} {r['action']}@${r['limit_price']:.2f}" for r in results)

            return {
                "order_id": all_ids[0],
                "status": worst_status,
                "why_held": "",
                "action": results[0]["action"],
                "qty": qty,
                "limit_price": sum(r["limit_price"] for r in results),
                "logs": all_logs,
                "message": f"Ordre(s) placé(s) dans TWS — {parts}",
            }

        return self._run_in_ibkr_thread(_place, timeout=30)

    def cancel_all_orders(self) -> list:
        """Annule tous les ordres ouverts via l'API."""
        def _cancel():
            ib = self._ib
            trades = ib.openTrades()
            cancelled = []
            for t in trades:
                ib.cancelOrder(t.order)
                cancelled.append({
                    "order_id": t.order.orderId,
                    "symbol": t.contract.symbol,
                    "action": t.order.action,
                })
            if cancelled:
                ib.sleep(1)
            return cancelled

        return self._run_in_ibkr_thread(_cancel, timeout=15)
