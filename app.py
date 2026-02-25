"""
Options Trading Robo-Advisor
============================
Analyse les données de marché en temps réel et recommande la stratégie
d'options mathématiquement optimale (méthodologie Tastytrade / VRP).

Installation :
    pip install -r requirements.txt

Lancement :
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ── Modules extraits ──
from config import (
    RISK_FREE_RATE, TICKER_GROUPS, TICKER_LIST, TICKER_NAMES,
    TICKER_CATEGORY, VOL_INDEX_MAP, VOL_INDEX_NAMES,
)
from engine.black_scholes import (
    black_scholes_delta, black_scholes_price, black_scholes_gamma,
    black_scholes_theta, black_scholes_vega,
    compute_leg_greeks, simulate_pnl, estimate_take_profit_spot,
    compute_real_probabilities,
)
from engine.strategy import (
    build_strategy, find_strike_by_delta, get_mid_price,
    estimate_sigma, filter_liquid_options,
)
from engine.indicators import (
    compute_iv_rank, compute_historical_vol, compute_trend_and_risk_data,
)
from data.hybrid_provider import HybridProvider
from ui.styles import inject_css

# ── Data provider singleton (IBKR → yfinance fallback) ──
@st.cache_resource
def _init_provider():
    return HybridProvider()

_provider = _init_provider()

def get_spot_price(ticker: str) -> float:
    return _provider.get_spot_price(ticker)

def get_vol_index(ticker: str) -> tuple[float, str]:
    return _provider.get_vol_index(ticker)

def get_options_chain(ticker: str):
    return _provider.get_options_chain(ticker)

def get_leaps_chain(ticker: str):
    return _provider.get_leaps_chain(ticker)

def get_short_term_chain(ticker: str):
    return _provider.get_short_term_chain(ticker)

# ──────────────────────────────────────────────
# 1. CONFIGURATION & THÈME
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Options Robo-Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom — thème glassmorphism financier premium
inject_css()

# ──────────────────────────────────────────────
# 6. INTERFACE UTILISATEUR — SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")

    # ── Indicateur source de données ──
    if _provider.ibkr_connected:
        st.markdown("🟢 **IBKR** connecté (temps réel)")
    elif _provider._ibkr_available:
        st.markdown("🟠 **IBKR** disponible (pas encore connecté)")
    else:
        _reason = "ib_insync absent" if _provider._ibkr is None else "TWS/Gateway non détecté"
        st.markdown(f"🟡 **yfinance** uniquement — {_reason}")

    st.markdown("---")

    # Ticker avec auto-complétion
    ticker_input = st.selectbox(
        "🏷️ Ticker",
        options=TICKER_LIST,
        index=None,
        placeholder="Tapez un ticker… (ex: SPY, AAPL)",
        format_func=lambda t: f"{TICKER_CATEGORY[t]}  ·  {t} — {TICKER_NAMES[t]}",
        help="Sélectionnez ou tapez un symbole boursier (ex: SPY, AAPL, TSLA)",
    )

    ticker = ticker_input if ticker_input else "SPY"

    st.markdown("---")

    budget = st.number_input(
        "💰 Budget Maximum Risqué ($)",
        min_value=50,
        max_value=1_000_000,
        value=1000,
        step=100,
        help="Capital maximum absolu que vous êtes prêt à perdre ou bloquer en marge.",
    )

    bias = st.selectbox(
        "🧭 Biais Directionnel",
        options=["Neutre", "Haussier", "Baissier"],
        index=0,
    )

    st.markdown("---")

    # ── Détection horaires de marché US (NYSE) ──
    import zoneinfo as _zi
    _market_open = True
    _market_hours_msg = ""
    try:
        _et = _zi.ZoneInfo("America/New_York")
        _now_et = dt.datetime.now(_et)
        _local_tz = dt.datetime.now().astimezone().tzinfo
        _open_et = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        _close_et = _now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        _is_weekday = _now_et.weekday() < 5
        _market_open = _is_weekday and _open_et <= _now_et <= _close_et

        if not _market_open:
            # Calcul de la prochaine ouverture en heure locale
            _next_open_et = _open_et
            if _now_et >= _close_et or not _is_weekday:
                _days = 1
                _nd = _now_et + dt.timedelta(days=_days)
                while _nd.weekday() >= 5:
                    _days += 1
                    _nd = _now_et + dt.timedelta(days=_days)
                _next_open_et = _nd.replace(hour=9, minute=30, second=0, microsecond=0)
            _next_open_local = _next_open_et.astimezone(_local_tz)
            _open_local = dt.datetime.now(_et).replace(hour=9, minute=30).astimezone(_local_tz)
            _close_local = dt.datetime.now(_et).replace(hour=16, minute=0).astimezone(_local_tz)
            _market_hours_msg = (
                f"Le marché US (NYSE) est actuellement **fermé**.\n\n"
                f"Heures d'ouverture : **{_open_local.strftime('%Hh%M')} – {_close_local.strftime('%Hh%M')}** (heure locale), du lundi au vendredi.\n\n"
                f"🕐 Prochaine ouverture : **{_next_open_local.strftime('%A %d/%m à %Hh%M')}**"
            )
    except Exception:
        _market_open = True  # en cas d'erreur, on laisse passer

    # ── Mode hors-séance (bypass si IBKR connecté) ──
    _force_analysis = False
    if not _market_open and _provider.ibkr_connected:
        _force_analysis = st.checkbox("🌙 Mode hors-séance (données IBKR delayed)", value=False)

    _can_analyze = _market_open or _force_analysis

    if not _can_analyze:
        st.warning("🔒 Marché fermé — analyse indisponible")

    analyze_btn = st.button("🔍  Analyser", use_container_width=True, type="primary", disabled=not _can_analyze)
    scan_btn = st.button("🔎  Scanner Tous les Tickers", use_container_width=True, disabled=not _can_analyze)

    st.markdown("---")
    st.caption("📊 Options Robo-Advisor v1.0")
    st.caption("Méthodologie : Tastytrade / VRP")


# ──────────────────────────────────────────────
# 7. INTERFACE UTILISATEUR — DASHBOARD PRINCIPAL
# ──────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <h1>📈 Options Robo-Advisor</h1>
    <p>Analyse quantitative en temps réel · Méthodologie Tastytrade / VRP</p>
</div>
""", unsafe_allow_html=True)

# ── Mode Scanner Multi-Tickers ──
if scan_btn:
    st.markdown("### 🔎 Scanner Multi-Tickers")
    st.markdown(f"Budget : **${budget:,.0f}** · Scan **Haussier + Neutre + Baissier**")
    st.markdown("---")

    scan_results = []
    progress_bar = st.progress(0, text="Initialisation du scan…")
    status_text = st.empty()
    total = len(TICKER_LIST)
    biases = ["Haussier", "Neutre", "Baissier"]

    for i, t in enumerate(TICKER_LIST):
        progress_bar.progress((i + 1) / total, text=f"Scan de {t} ({i+1}/{total})…")
        for b in biases:
            try:
                s = get_spot_price(t)
                v, vs = get_vol_index(t)
                ivr = compute_iv_rank(t)
                strat = build_strategy(s, v, ivr, b, budget, t, vs, data_provider=_provider)
                qty = strat.get("qty", 1)
                unit_risk = round(strat["max_risk"] / qty, 2) if qty > 0 else strat["max_risk"]
                # Indicateurs avancés
                adv = compute_trend_and_risk_data(
                    t, s, b, int(strat["dte"]),
                    strat["max_risk"], strat.get("ev", 0), strat["max_profit"]
                )
                scan_results.append({
                    "Ticker": t,
                    "Nom": TICKER_NAMES.get(t, t),
                    "Budget Min": unit_risk,
                    "Biais": b,
                    "Stratégie": strat["name"],
                    "Perte Max": round(strat["max_risk"], 2),
                    "Gain Max / 2": round(strat["exit_plan"]["take_profit"], 2),
                    "% TP": strat.get("probabilities", {}).get("p_take_profit", 0),
                    "% BE": strat.get("probabilities", {}).get("p_breakeven", 0),
                    "% Perte": strat.get("probabilities", {}).get("p_partial_loss", 0),
                    "% Loss": strat.get("probabilities", {}).get("p_max_loss", 0),
                    "EV": round(strat.get("ev", 0), 2),
                    "EV Yield": round(adv["ev_yield"], 1),
                    "ROC Ann.": round(adv["roc_annualise"], 1),
                    "SMA 50": round(adv["sma50"], 2) if adv["sma50"] else None,
                    "RSI": round(adv["rsi"], 1) if adv["rsi"] is not None else None,
                    "Écart SMA (%)": round(adv["dist_sma"], 2) if adv["dist_sma"] is not None else None,
                    "Tendance": adv["alignement"],
                    "Earnings": adv["earnings_risk"],
                })
            except Exception:
                continue

    progress_bar.empty()

    if scan_results:
        df = pd.DataFrame(scan_results).sort_values("EV", ascending=False).reset_index(drop=True)
        total_found = len(df)
        # Filtre : ne garder que les cibles parfaites (pas de Rejet ni Contre-tendance)
        df = df[~df["Tendance"].str.contains("Rejet|Contre-tendance", na=False)].reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed

        st.success(f"✅ **{len(df)} cibles validées** sur {total_found} stratégies trouvées ({total} tickers scannés).")

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Nom": st.column_config.TextColumn("Nom", width="medium"),
                "Budget Min": st.column_config.NumberColumn("💰 Budget Min ($)", format="$%.0f"),
                "Biais": st.column_config.TextColumn("Biais", width="small"),
                "Stratégie": st.column_config.TextColumn("Stratégie", width="medium"),
                "Perte Max": st.column_config.NumberColumn("Perte Max ($)", format="$%.2f"),
                "Gain Max / 2": st.column_config.NumberColumn("Gain Max / 2 ($)", format="$%.2f"),
                "% TP": st.column_config.NumberColumn("🎯 % TP", format="%.1f%%"),
                "% BE": st.column_config.NumberColumn("⚖️ % BE", format="%.1f%%"),
                "% Perte": st.column_config.NumberColumn("📉 % Perte", format="%.1f%%"),
                "% Loss": st.column_config.NumberColumn("💀 % Max", format="%.1f%%"),
                "EV": st.column_config.NumberColumn("EV ($)", format="$%.2f"),
                "EV Yield": st.column_config.NumberColumn("📈 EV Yield (%)", format="%.1f%%"),
                "ROC Ann.": st.column_config.NumberColumn("🔄 ROC Ann. (%)", format="%.1f%%"),
                "SMA 50": st.column_config.NumberColumn("📊 SMA 50", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("📉 RSI", format="%.1f"),
                "Écart SMA (%)": st.column_config.NumberColumn("📏 Écart SMA (%)", format="%+.2f%%"),
                "Tendance": st.column_config.TextColumn("📈 Tendance", width="medium"),
                "Earnings": st.column_config.TextColumn("📅 Earnings", width="medium"),
            },
        )
    else:
        st.warning("⚠️ Aucune stratégie valide trouvée. Essayez d'augmenter le budget.")

    st.markdown("---")
    st.caption(
        f"📊 Scan exécuté le {dt.datetime.now().strftime('%d/%m/%Y à %H:%M')} · "
        f"Budget: ${budget:,.0f} · Biais: {bias}"
    )
    st.stop()

if analyze_btn:
    # Marquer l'analyse comme faite pour persister entre les reruns
    st.session_state["analysis_done"] = True
    st.session_state["analysis_ticker"] = ticker

_has_analysis = st.session_state.get("analysis_done", False) and st.session_state.get("analysis_ticker") == ticker

if not analyze_btn and not _has_analysis:
    # ── Portfolio IBKR (si connecté) — chargement à la demande ──
    if _provider.ibkr_connected and hasattr(_provider, '_ibkr') and _provider._ibkr:
        if st.button("💼 Charger le portefeuille IBKR", use_container_width=True):
            with st.spinner("📊 Chargement…"):
                try:
                    st.session_state["ibkr_account"] = _provider._ibkr.get_account_summary()
                    st.session_state["ibkr_portfolio"] = _provider._ibkr.get_portfolio()
                except Exception as e:
                    st.warning(f"⚠️ {e}")
                    st.session_state.pop("ibkr_account", None)

        if "ibkr_account" in st.session_state:
            account = st.session_state["ibkr_account"]
            portfolio = st.session_state.get("ibkr_portfolio", [])

            st.markdown("### 💼 Portefeuille IBKR")

            _cur = account.get("currency", "USD")
            _sym = "€" if _cur == "EUR" else "$"
            mc1, mc2, mc3, mc4 = st.columns(4)
            nlv = account.get("NetLiquidation", 0)
            cash = account.get("TotalCashValue", 0)
            bp = account.get("BuyingPower", 0)
            upnl = account.get("UnrealizedPnL", 0)

            with mc1:
                st.metric("🏦 Valeur Nette", f"{_sym}{nlv:,.0f}")
            with mc2:
                st.metric("💵 Cash", f"{_sym}{cash:,.0f}")
            with mc3:
                st.metric("⚡ Buying Power", f"{_sym}{bp:,.0f}")
            with mc4:
                pnl_delta = "Gain" if upnl >= 0 else "Perte"
                st.metric("📈 P&L Non Réalisé", f"{_sym}{upnl:,.0f}",
                         delta=pnl_delta,
                         delta_color="normal" if upnl >= 0 else "inverse")

            if portfolio:
                import pandas as _pd
                rows = []
                for p in portfolio:
                    p_sym = "€" if p.get("currency", "USD") == "EUR" else "$"
                    rows.append({
                        "Instrument": p["label"],
                        "Type": p["type"],
                        "Qté": int(p["position"]) if p["position"] == int(p["position"]) else p["position"],
                        "Prix": f"{p_sym}{p['market_price']:,.2f}",
                        "Valeur": f"{p_sym}{p['market_value']:,.0f}",
                        "Coût Moy.": f"{p_sym}{p['avg_cost']:,.2f}",
                        "P&L": round(p["unrealized_pnl"], 2),
                        "P&L (%)": round(
                            (p["unrealized_pnl"] / (abs(p["avg_cost"]) * abs(p["position"]))) * 100, 1
                        ) if p["avg_cost"] and p["position"] else 0.0,
                    })

                df = _pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "P&L": st.column_config.NumberColumn("P&L", format="%.2f"),
                        "P&L (%)": st.column_config.NumberColumn("P&L (%)", format="%.1f%%"),
                    },
                )
            else:
                st.info("Aucune position ouverte.")

            st.markdown("---")

    # Bannière marché fermé (seulement si pas de bypass hors-séance)
    if not _can_analyze and _market_hours_msg:
        st.markdown("""\
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #e94560;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 2rem 0;
">
    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔒</div>
    <h2 style="color: #e94560; margin: 0 0 1rem 0;">Marché Fermé</h2>
    <p style="color: #ccc; font-size: 1.1rem; line-height: 1.6;">""" + _market_hours_msg.replace('\n\n', '<br>').replace('**', '<b>', 1).replace('**', '</b>', 1).replace('**', '<b>', 1).replace('**', '</b>', 1).replace('**', '<b>', 1).replace('**', '</b>', 1) + """</p>
    <p style="color: #888; font-size: 0.9rem; margin-top: 1.5rem;">Les données d'options (bid/ask) ne sont pas fiables en dehors des heures de séance.<br>
    L'analyse est désactivée pour éviter des résultats incorrects.</p>
</div>""", unsafe_allow_html=True)
        st.stop()

    # État initial : instructions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Sélectionnez")
        st.write("Choisissez un ticker et définissez votre budget maximal dans la barre latérale.")
    with col2:
        st.markdown("### 2️⃣ Analysez")
        st.write("Cliquez sur **🔍 Analyser** pour scanner les données de marché en temps réel.")
    with col3:
        st.markdown("### 3️⃣ Exécutez")
        st.write("Suivez le ticket d'ordre et le plan de vol pour exécuter la stratégie recommandée.")
    st.stop()

# ── Exécution de l'analyse ──
try:
    # Récupérer ou recalculer les données de marché
    if analyze_btn or "analysis_cache" not in st.session_state or st.session_state.get("analysis_ticker") != ticker:
        with st.spinner(f"🔄 Analyse de **{ticker}** en cours…"):
            spot = get_spot_price(ticker)
            vix, vol_symbol = get_vol_index(ticker)
            vol_label = VOL_INDEX_NAMES.get(vol_symbol, vol_symbol.replace("^", ""))
            iv_rank = compute_iv_rank(ticker)
            st.session_state["analysis_cache"] = {
                "spot": spot, "vix": vix, "vol_symbol": vol_symbol,
                "vol_label": vol_label, "iv_rank": iv_rank,
            }
    else:
        # Utiliser le cache pour les reruns (bouton ordre, etc.)
        _cache = st.session_state["analysis_cache"]
        spot = _cache["spot"]
        vix = _cache["vix"]
        vol_symbol = _cache["vol_symbol"]
        vol_label = _cache["vol_label"]
        iv_rank = _cache["iv_rank"]

    # ─── Section 1 : CONTEXTE MACRO ───
    # Badge source de données
    src_spot = _provider.last_source.get("get_spot_price", "yfinance")
    src_vol = _provider.last_source.get("get_vol_index", "yfinance")
    src_chain = _provider.last_source.get("get_options_chain", "yfinance")
    src_icon = lambda s: "🟢" if s == "IBKR" else "🟡"
    st.caption(
        f"📡 Sources : "
        f"{src_icon(src_spot)} Spot **{src_spot}** · "
        f"{src_icon(src_vol)} Vol **{src_vol}** · "
        f"{src_icon(src_chain)} Chaîne **{src_chain}**"
    )
    st.markdown("### 🌍 Contexte Macro")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="💲 Prix Spot",
            value=f"${spot:,.2f}",
            delta=f"{ticker}",
        )
    with c2:
        vix_color = "🔴" if vix > 20 else ("🟡" if vix > 15 else "🟢")
        st.metric(
            label=f"{vix_color} {vol_label}",
            value=f"{vix:.2f}",
            delta="Élevé" if vix > 20 else ("Modéré" if vix > 15 else "Bas"),
            delta_color="inverse" if vix > 20 else "normal",
        )
    with c3:
        iv_color = "🔴" if iv_rank > 50 else ("🟡" if iv_rank > 20 else "🟢")
        st.metric(
            label=f"{iv_color} IV Rank (52 semaines)",
            value=f"{iv_rank:.1f}%",
            delta="Haute vol." if iv_rank > 50 else ("Moyenne" if iv_rank > 20 else "Basse vol."),
            delta_color="inverse" if iv_rank > 50 else "normal",
        )

    st.markdown("---")

    # ─── Section 2 : STRATÉGIE ───
    if analyze_btn or "strategy_cache" not in st.session_state or st.session_state.get("analysis_ticker") != ticker:
        with st.spinner("🧠 Construction de la stratégie optimale…"):
            strategy = build_strategy(spot, vix, iv_rank, bias, budget, ticker, vol_symbol, data_provider=_provider)
            adv_data = compute_trend_and_risk_data(
                ticker, spot, bias, int(strategy["dte"]),
                strategy["max_risk"], strategy.get("ev", 0), strategy["max_profit"]
            )
            st.session_state["strategy_cache"] = strategy
            st.session_state["adv_data_cache"] = adv_data
    else:
        strategy = st.session_state["strategy_cache"]
        adv_data = st.session_state["adv_data_cache"]

    # Verdict
    st.markdown(f"""
    <div class="verdict-card">
        <h2>🎯 LE VERDICT</h2>
        <div class="strategy-name">{strategy['name']}</div>
        <p>{strategy['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Section 3 : TICKET D'ORDRE ───
    st.markdown("### 📋 Ticket d'Ordre (Legs)")

    qty = strategy.get("qty", 1)

    legs_data = []
    for leg in strategy["legs"]:
        legs_data.append({
            "Qté": qty,
            "Action": f"{'🟢 ' if leg['action'] == 'BUY' else '🔴 '}{leg['action']}",
            "Type": leg["type"],
            "Strike": f"${leg['strike']:,.2f}",
            "Expiration": leg["exp"],
            "DTE": f"{leg['dte']}j",
            "Prix unitaire": f"${leg['price']:.2f}",
        })

    legs_df = pd.DataFrame(legs_data)
    st.dataframe(
        legs_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qté": st.column_config.NumberColumn("Qté", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Strike": st.column_config.TextColumn("Strike", width="small"),
            "Expiration": st.column_config.TextColumn("Expiration", width="medium"),
            "DTE": st.column_config.TextColumn("DTE", width="small"),
            "Prix unitaire": st.column_config.TextColumn("Prix", width="small"),
        },
    )

    # ─── Bouton ordres IBKR (si connecté) ───
    if _provider.ibkr_connected and hasattr(_provider, '_ibkr') and _provider._ibkr:
        st.markdown("")

        # Résumé de l'ordre (calculé depuis les legs)
        _net = sum(
            leg["price"] if leg["action"] == "SELL" else -leg["price"]
            for leg in strategy["legs"]
        )
        _action = "SELL" if _net > 0 else "BUY"
        _price = abs(_net)
        st.info(
            f"**Ordre combo prêt** : {_action} {qty}x @ ${_price:.2f} · "
            f"Risque max : ${strategy['max_risk']:,.0f} · "
            f"Profit max : ${strategy['max_profit']:,.0f}"
        )

        if st.button("📋 Préparer l'ordre dans TWS", type="primary", use_container_width=True):
            with st.spinner("📡 Qualification des contrats…"):
                try:
                    _provider._ibkr._ensure_connected()
                    result = _provider._ibkr.place_order(strategy, ticker)
                    status = result.get("status", "Unknown")
                    if status.lower() in ("cancelled", "inactive", "apicancelled"):
                        logs_text = "\n".join(result.get("logs", [])) or "Aucun détail"
                        st.warning(
                            f"⚠️ Ordre #{result['order_id']} **rejeté par TWS** (statut: {status})\n\n"
                            f"```\n{logs_text}\n```\n\n"
                            f"Vérifiez dans TWS : permissions options US, "
                            f"marge disponible, ou limites API."
                        )
                    else:
                        st.success(
                            f"✅ {result['message']}\n\n"
                            f"Statut : **{status}** — "
                            f"L'ordre est prêt dans TWS. "
                            f"**Cliquez sur « Transmettre » dans TWS** pour l'exécuter."
                        )
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

        # Bouton annuler tous les ordres
        if st.button("🗑️ Annuler tous les ordres ouverts", use_container_width=True):
            with st.spinner("❌ Annulation…"):
                try:
                    cancelled = _provider._ibkr.cancel_all_orders()
                    if cancelled:
                        ids = ", ".join(f"#{c['order_id']}" for c in cancelled)
                        st.success(f"✅ {len(cancelled)} ordre(s) annulé(s) : {ids}")
                    else:
                        st.info("Aucun ordre ouvert à annuler.")
                except Exception as e:
                    st.error(f"❌ Erreur annulation : {e}")

    st.markdown("---")

    # ─── Section 3b : GRECQUES DE LA POSITION ───
    st.markdown('<div class="section-header"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" /></svg><h2>Grecques de la Position (Net)</h2></div>', unsafe_allow_html=True)

    greeks = strategy.get("greeks", {})
    delta_val = greeks.get("delta", 0)
    gamma_val = greeks.get("gamma", 0)
    theta_val = greeks.get("theta", 0)
    vega_val = greeks.get("vega", 0)
    iv_val = greeks.get("iv", 0)

    st.markdown(f'''
    <div class="greeks-container">
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Delta (Δ)</div>
                <div class="greek-hint-text">Sensibilité au prix du sous-jacent. Un delta de +50 signifie que si l'action bouge de 1$, la position gagne/perd ~50$.</div>
            </div>
            <div class="greek-symbol">Delta (Δ)</div>
            <div class="greek-value">{delta_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Gamma (Γ)</div>
                <div class="greek-hint-text">Accélération du Delta. Un gamma élevé signifie que le Delta changera rapidement si le prix bouge. Risque accru proche de l'expiration.</div>
            </div>
            <div class="greek-symbol">Gamma (Γ)</div>
            <div class="greek-value">{gamma_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Theta (Θ)</div>
                <div class="greek-hint-text">Déclin temporel journalier en $. Un theta négatif = la position perd de la valeur chaque jour. Positif = vous profitez du passage du temps.</div>
            </div>
            <div class="greek-symbol">Theta (Θ)</div>
            <div class="greek-value">{theta_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Vega (ν)</div>
                <div class="greek-hint-text">Sensibilité à la volatilité implicite. Indique le gain/perte pour chaque 1% de hausse de l'IV. Vega positif profite d'une hausse de la vol.</div>
            </div>
            <div class="greek-symbol">Vega (ν)</div>
            <div class="greek-value">{vega_val:+.2f}</div>
        </div>
        <div class="greek-card">
            <div class="greek-hint">
                <div class="greek-hint-title">Vol. Implicite</div>
                <div class="greek-hint-text">Volatilité implicite actuelle du marché pour ces options. Elle mesure l'anticipation de mouvement futur du sous-jacent par le marché.</div>
            </div>
            <div class="greek-symbol">IV</div>
            <div class="greek-value">{iv_val:.1f}%</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.caption("💡 Survolez chaque grecque pour comprendre sa signification")

    st.markdown("---")

    # ─── Section 4 : MÉTRIQUES FINANCIÈRES ───
    st.markdown("### 💰 Métriques Financières")

    m1, m2, m3, m4 = st.columns(4)

    cd_val = strategy["credit_or_debit"]
    cd_label = "Crédit Net Reçu" if cd_val > 0 else "Débit Net Payé"
    cd_color = "green" if cd_val > 0 else "red"

    with m1:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">{cd_label}</div>
            <div class="value {cd_color}">${abs(cd_val):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">⚠️ Risque Maximal</div>
            <div class="value red">${strategy['max_risk']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">🎯 Profit Maximal</div>
            <div class="value green">${strategy['max_profit']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        ev_val = strategy.get('ev', 0)
        ev_color = "green" if ev_val > 0 else "red"
        ev_sign = "+" if ev_val > 0 else "-"
        st.markdown(f"""
        <div class="fin-metric" >
            <div class="label">⚖️ Score EV (Espérance)</div>
            <div class="value {ev_color}">{ev_sign}${abs(ev_val):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Vérification budget
    if strategy["max_risk"] <= budget:
        st.success(f"✅ **Budget respecté** — Risque max ({strategy['max_risk']:,.2f}$) ≤ Budget ({budget:,.2f}$)")
    else:
        st.error(f"❌ **ATTENTION** — Risque max ({strategy['max_risk']:,.2f}$) > Budget ({budget:,.2f}$)")

    st.markdown("---")

    # ─── Section 4a : INDICATEURS AVANCÉS ───
    st.markdown("### 📊 Indicateurs Avancés")

    a1, a2, a3, a4, a5, a6, a7 = st.columns(7)

    ev_yield_val = adv_data["ev_yield"]
    ev_yield_color = "green" if ev_yield_val > 0 else "red"
    with a1:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📈 EV Yield</div>
            <div class="value {ev_yield_color}">{ev_yield_val:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    roc_val = adv_data["roc_annualise"]
    roc_color = "green" if roc_val > 0 else "red"
    with a2:
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">🔄 ROC Annualisé</div>
            <div class="value {roc_color}">{roc_val:,.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    sma50_val = adv_data["sma50"]
    with a3:
        sma50_display = f"${sma50_val:,.2f}" if sma50_val else "N/A"
        sma50_color = "green" if sma50_val and spot > sma50_val else ("red" if sma50_val else "")
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📊 SMA 50</div>
            <div class="value {sma50_color}">{sma50_display}</div>
        </div>
        """, unsafe_allow_html=True)

    rsi_val = adv_data.get("rsi")
    with a4:
        if rsi_val is not None:
            rsi_color = "red" if rsi_val > 70 else ("green" if rsi_val < 30 else "")
            rsi_display = f"{rsi_val:.1f}"
        else:
            rsi_color = ""
            rsi_display = "N/A"
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📉 RSI (14)</div>
            <div class="value {rsi_color}">{rsi_display}</div>
        </div>
        """, unsafe_allow_html=True)

    dist_sma_val = adv_data.get("dist_sma")
    with a5:
        if dist_sma_val is not None:
            dist_color = "red" if abs(dist_sma_val) > 10 else "green"
            dist_display = f"{dist_sma_val:+.2f}%"
        else:
            dist_color = ""
            dist_display = "N/A"
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📏 Écart SMA (%)</div>
            <div class="value {dist_color}">{dist_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with a6:
        trend_val = adv_data["alignement"]
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📐 Alignement Tendance</div>
            <div class="value" style="font-size: 1rem;">{trend_val}</div>
        </div>
        """, unsafe_allow_html=True)

    with a7:
        earnings_val = adv_data["earnings_risk"]
        st.markdown(f"""
        <div class="fin-metric">
            <div class="label">📅 Earnings Risk</div>
            <div class="value" style="font-size: 1rem;">{earnings_val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Section 4b : PROBABILITÉS & NIVEAUX DE PRIX ───
    st.markdown("### 📊 Probabilités & Niveaux de Prix")

    probs = strategy.get('probabilities', {})
    p_tp = probs.get('p_take_profit', 0)
    p_be = probs.get('p_breakeven', 0)
    p_pl = probs.get('p_partial_loss', 0)
    p_loss = probs.get('p_max_loss', 0)

    # Calcul des niveaux de prix associés via bisection sur simulate_pnl
    current_sigma = strategy.get("sigma", 0.25)
    qty_prob = strategy.get("qty", 1)
    take_profit_val = strategy["exit_plan"]["take_profit"]
    max_risk_val = strategy["max_risk"]

    def find_nearest_spot_for_pnl(target_pnl, legs, remaining_dte, sigma, qty, current_spot):
        """Sweep pour trouver le spot le plus proche du spot actuel
        où le P&L croise le seuil cible. Gère tous les types de stratégies
        (monotones et non-monotones comme les Iron Condors)."""
        n_pts = 500
        spots = np.linspace(current_spot * 0.50, current_spot * 1.50, n_pts)
        pnls = [simulate_pnl(legs, s, remaining_dte, sigma, qty) for s in spots]

        # Trouver tous les croisements (changement de signe de pnl - target)
        crossings = []
        for i in range(len(pnls) - 1):
            diff_a = pnls[i] - target_pnl
            diff_b = pnls[i + 1] - target_pnl
            if diff_a * diff_b <= 0:  # croisement
                # Interpolation linéaire
                if abs(diff_b - diff_a) > 1e-10:
                    frac = abs(diff_a) / abs(diff_b - diff_a)
                    cross_spot = spots[i] + frac * (spots[i + 1] - spots[i])
                else:
                    cross_spot = (spots[i] + spots[i + 1]) / 2
                crossings.append(cross_spot)

        if crossings:
            # Retourner le croisement le plus proche du spot actuel
            return min(crossings, key=lambda s: abs(s - current_spot))
        else:
            # Pas de croisement : retourner le spot qui donne le P&L le plus proche du target
            closest_idx = min(range(len(pnls)), key=lambda i: abs(pnls[i] - target_pnl))
            return float(spots[closest_idx])

    strat_legs = strategy["legs"]
    spot_tp = find_nearest_spot_for_pnl(take_profit_val, strat_legs, 21, current_sigma, qty_prob, spot)
    spot_be = find_nearest_spot_for_pnl(0, strat_legs, 21, current_sigma, qty_prob, spot)
    spot_ml = find_nearest_spot_for_pnl(-max_risk_val * 0.95, strat_legs, 21, current_sigma, qty_prob, spot)

    pct_tp = ((spot_tp - spot) / spot) * 100
    pct_be = ((spot_be - spot) / spot) * 100
    pct_ml = ((spot_ml - spot) / spot) * 100

    prob_data = [
        {"Scénario": "🎯 Take Profit", "P&L": f"+${take_profit_val:,.0f}", "Spot Cible": f"${spot_tp:,.2f}", "Mouvement": f"{pct_tp:+.1f}%", "Probabilité (%)": p_tp},
        {"Scénario": "⚖️ Break-Even", "P&L": "$0", "Spot Cible": f"${spot_be:,.2f}", "Mouvement": f"{pct_be:+.1f}%", "Probabilité (%)": p_be},
        {"Scénario": "📉 Perte Partielle", "P&L": "—", "Spot Cible": f"${spot_be:,.0f} – ${spot_ml:,.0f}", "Mouvement": "—", "Probabilité (%)": p_pl},
        {"Scénario": "💀 Perte Maximale", "P&L": f"-${max_risk_val:,.0f}", "Spot Cible": f"${spot_ml:,.2f}", "Mouvement": f"{pct_ml:+.1f}%", "Probabilité (%)": p_loss},
    ]
    df_prob = pd.DataFrame(prob_data)
    st.dataframe(
        df_prob,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Scénario": st.column_config.TextColumn("Scénario", width="medium"),
            "P&L": st.column_config.TextColumn("P&L", width="small"),
            "Spot Cible": st.column_config.TextColumn("Spot Cible", width="medium"),
            "Mouvement": st.column_config.TextColumn("Mouvement", width="small"),
            "Probabilité (%)": st.column_config.ProgressColumn("Probabilité", format="%.1f%%", min_value=0, max_value=100),
        },
    )
    hist_vol = compute_historical_vol(ticker)
    hist_vol_str = f"{hist_vol*100:.1f}%" if hist_vol else "N/A"
    st.caption(f"📍 Spot actuel : **${spot:,.2f}** · Évaluation au time-stop (21 DTE restants) · Vol. historique {hist_vol_str}")

    st.markdown("---")

    # ─── Section 4c : GRAPHIQUE HISTORIQUE 6 MOIS ───
    st.markdown(f"### 📈 Historique {ticker} (6 mois)")

    tk_hist = yf.Ticker(ticker)
    hist_data = tk_hist.history(period="6mo")

    if not hist_data.empty:
      try:
        import plotly.graph_objects as go
        import traceback as _tb

        fig = go.Figure()

        # Courbe du prix
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data["Close"],
            mode="lines",
            name="Prix",
            line=dict(color="#60A5FA", width=2),
            hovertemplate="$%{y:,.2f}<extra></extra>",
        ))

        # ── SMA 50 jours (historique — ligne continue) ──
        sma50_series = hist_data["Close"].rolling(window=50).mean()
        sma50_valid = sma50_series.dropna()
        if not sma50_valid.empty:
            fig.add_trace(go.Scatter(
                x=sma50_valid.index,
                y=sma50_valid,
                mode="lines",
                name="SMA 50",
                line=dict(color="#FBBF24", width=1.5),
                hovertemplate="SMA50: $%{y:,.2f}<extra></extra>",
            ))

            # ── Projection SMA 50 (prix flat au spot) — pointillé ──
            close_list = list(hist_data["Close"].values[-50:])  # derniers 50 prix
            proj_sma_values = [float(sma50_valid.iloc[-1])]  # ancrage au dernier SMA connu
            proj_dates = [sma50_valid.index[-1]]
            future_bdays = pd.bdate_range(start=hist_data.index[-1], periods=23)[1:]  # ~1 mois
            for d in future_bdays:
                close_list.pop(0)
                close_list.append(spot)
                proj_sma_values.append(float(np.mean(close_list)))
                proj_dates.append(d)

            fig.add_trace(go.Scatter(
                x=proj_dates,
                y=proj_sma_values,
                mode="lines",
                name="SMA 50 (proj.)",
                line=dict(color="#FBBF24", width=1.5, dash="dash"),
                hovertemplate="SMA50 proj.: $%{y:,.2f}<extra></extra>",
            ))

        # ── Projection linéaire 1 mois (ancrée au dernier prix) ──
        close_vals = hist_data["Close"].values
        x_numeric = np.arange(len(close_vals))
        coeffs = np.polyfit(x_numeric, close_vals, 1)
        slope = coeffs[0]
        last_price = float(close_vals[-1])

        last_date = hist_data.index[-1]
        future_days = 22  # ~1 mois de trading
        future_dates = pd.bdate_range(start=last_date, periods=future_days + 1)  # inclut le dernier jour
        future_prices = [last_price + slope * d for d in range(future_days + 1)]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode="lines",
            name="Projection (1 mois)",
            line=dict(color="#60A5FA", width=2, dash="dot"),
            hovertemplate="$%{y:,.2f} (proj.)<extra></extra>",
        ))

        # Lignes horizontales pour les strikes
        legs = strategy.get("legs", [])
        strikes = sorted(set(leg["strike"] for leg in legs))
        strike_colors = ["#F87171", "#FBBF24", "#34D399", "#A78BFA"]
        for i, s in enumerate(strikes):
            color = strike_colors[i % len(strike_colors)]
            action = next((l["action"] for l in legs if l["strike"] == s), "")
            opt_type = next((l["type"] for l in legs if l["strike"] == s), "")
            label = f"{action} {opt_type} ${s:.0f}"
            fig.add_hline(
                y=s, line_dash="dash", line_color=color, line_width=1,
                annotation_text=label,
                annotation_position="right",
                annotation_font_color=color,
                annotation_font_size=11,
            )

        # Ligne du spot actuel
        fig.add_hline(
            y=spot, line_dash="dot", line_color="#94A3B8", line_width=1,
            annotation_text=f"Spot ${spot:.0f}",
            annotation_position="left",
            annotation_font_color="#94A3B8",
            annotation_font_size=11,
        )

        # ── Zones vertes (profit) et rouges (perte) ──
        # Sweep pour trouver TOUS les breakevens, TP et ML crossings
        n_sweep = 300
        sweep_spots = np.linspace(spot * 0.50, spot * 1.50, n_sweep)
        sweep_pnls = [simulate_pnl(strat_legs, s, 21, current_sigma, qty_prob) for s in sweep_spots]
        ml_threshold = -max_risk_val * 0.95

        def find_crossings(pnls, spots_arr, threshold):
            crossings = []
            for i in range(len(pnls) - 1):
                diff_a = pnls[i] - threshold
                diff_b = pnls[i + 1] - threshold
                if diff_a * diff_b <= 0 and abs(diff_a - diff_b) > 0.01:
                    frac = abs(diff_a) / (abs(diff_a) + abs(diff_b))
                    cross = spots_arr[i] + frac * (spots_arr[i + 1] - spots_arr[i])
                    crossings.append(float(cross))
            return sorted(crossings)

        be_crossings = find_crossings(sweep_pnls, sweep_spots, 0)
        tp_crossings = find_crossings(sweep_pnls, sweep_spots, take_profit_val)
        ml_crossings = find_crossings(sweep_pnls, sweep_spots, ml_threshold)

        # Déterminer les zones du y-axis
        y_min_zone = float(sweep_spots[0])
        y_max_zone = float(sweep_spots[-1])

        GREEN_LIGHT = "rgba(52, 211, 153, 0.07)"
        GREEN_DARK = "rgba(52, 211, 153, 0.18)"
        RED_LIGHT = "rgba(248, 113, 113, 0.07)"
        RED_DARK = "rgba(248, 113, 113, 0.18)"

        if len(be_crossings) == 0:
            is_positive = sweep_pnls[len(sweep_pnls) // 2] > 0
            fig.add_hrect(y0=y_min_zone, y1=y_max_zone,
                fillcolor=GREEN_LIGHT if is_positive else RED_LIGHT, line_width=0, layer="below")
        elif len(be_crossings) == 1:
            # 1 BE = stratégie directionnelle
            be = be_crossings[0]
            pnl_above = simulate_pnl(strat_legs, be + 1, 21, current_sigma, qty_prob)
            profit_above = pnl_above > 0

            if profit_above:
                fig.add_hrect(y0=be, y1=y_max_zone, fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=y_min_zone, y1=be, fillcolor=RED_LIGHT, line_width=0, layer="below")
                # TP dark green above TP spot
                if tp_crossings:
                    fig.add_hrect(y0=tp_crossings[-1], y1=y_max_zone, fillcolor=GREEN_DARK, line_width=0, layer="below")
                # ML dark red below ML spot
                if ml_crossings:
                    fig.add_hrect(y0=y_min_zone, y1=ml_crossings[0], fillcolor=RED_DARK, line_width=0, layer="below")
            else:
                fig.add_hrect(y0=y_min_zone, y1=be, fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be, y1=y_max_zone, fillcolor=RED_LIGHT, line_width=0, layer="below")
                # TP dark green below TP spot
                if tp_crossings:
                    fig.add_hrect(y0=y_min_zone, y1=tp_crossings[0], fillcolor=GREEN_DARK, line_width=0, layer="below")
                # ML dark red above ML spot
                if ml_crossings:
                    fig.add_hrect(y0=ml_crossings[-1], y1=y_max_zone, fillcolor=RED_DARK, line_width=0, layer="below")


            # TP line
            fig.add_hline(y=spot_tp, line_dash="dash", line_color="#34D399", line_width=1,
                annotation_text=f"TP ${spot_tp:.0f}", annotation_position="right",
                annotation_font_color="#34D399", annotation_font_size=11)
        else:
            # 2+ BE = Iron Condor
            be_sorted = sorted(be_crossings)
            center = (be_sorted[0] + be_sorted[-1]) / 2
            pnl_center = simulate_pnl(strat_legs, center, 21, current_sigma, qty_prob)
            center_positive = pnl_center > 0

            if center_positive:
                fig.add_hrect(y0=y_min_zone, y1=be_sorted[0], fillcolor=RED_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[0], y1=be_sorted[-1], fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[-1], y1=y_max_zone, fillcolor=RED_LIGHT, line_width=0, layer="below")
                # ML dark red at extremes
                if ml_crossings:
                    fig.add_hrect(y0=y_min_zone, y1=ml_crossings[0], fillcolor=RED_DARK, line_width=0, layer="below")
                    if len(ml_crossings) >= 2:
                        fig.add_hrect(y0=ml_crossings[-1], y1=y_max_zone, fillcolor=RED_DARK, line_width=0, layer="below")
            else:
                fig.add_hrect(y0=y_min_zone, y1=be_sorted[0], fillcolor=GREEN_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[0], y1=be_sorted[-1], fillcolor=RED_LIGHT, line_width=0, layer="below")
                fig.add_hrect(y0=be_sorted[-1], y1=y_max_zone, fillcolor=GREEN_LIGHT, line_width=0, layer="below")



        # Ligne verticale : date de sortie (time-stop)
        dte_val = int(strategy["dte"])
        exit_date = dt.datetime.now() + dt.timedelta(days=max(1, dte_val - 21))
        exit_date_str = exit_date.strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=exit_date_str, x1=exit_date_str,
            y0=0, y1=1, yref="paper",
            line=dict(color="#FBBF24", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=exit_date_str, y=1, yref="paper",
            text=f"Sortie {exit_date.strftime('%d/%m')}",
            showarrow=False, font=dict(color="#FBBF24", size=11),
            yshift=10,
        )

        # Y-axis range: padding autour du min/max prix
        y_min = float(hist_data["Low"].min())
        y_max = float(hist_data["High"].max())
        # Inclure les strikes, spot, BE, TP et projection dans le range
        all_levels = [y_min, y_max] + strikes + [spot, spot_be, spot_tp] + list(future_prices)
        y_range_min = min(all_levels) * 0.97
        y_range_max = max(all_levels) * 1.03

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=80, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                gridcolor="rgba(51,65,85,0.3)",
                showgrid=True,
            ),
            yaxis=dict(
                range=[y_range_min, y_range_max],
                gridcolor="rgba(51,65,85,0.3)",
                showgrid=True,
                tickprefix="$",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                font=dict(size=11),
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistiques rapides
        price_6m_ago = float(hist_data["Close"].iloc[0])
        price_now = float(hist_data["Close"].iloc[-1])
        change_pct = ((price_now - price_6m_ago) / price_6m_ago) * 100
        high_6m = float(hist_data["High"].max())
        low_6m = float(hist_data["Low"].min())

        st.caption(
            f"📊 **6 mois** : {change_pct:+.1f}% · "
            f"Plus haut : ${high_6m:,.2f} · Plus bas : ${low_6m:,.2f}"
        )
      except Exception as _chart_err:
        st.error(f"Erreur chart : {_chart_err}")
        st.code(_tb.format_exc())

    # ─── Section 4b : SIMULATION P&L À 21 DTE ───
    st.markdown("### 🔮 Simulation P&L à la Clôture (Time Stop à 21 DTE)")

    current_sigma = strategy.get("sigma", 0.25)
    dte_strat = int(strategy["dte"])
    holding_days = max(1, dte_strat - 21)
    qty_sim = strategy.get("qty", 1)
    take_profit_sim = strategy["exit_plan"]["take_profit"]
    max_risk_sim = strategy["max_risk"]

    # Écart-type statistique du mouvement sur la période de détention
    sd_move = spot * current_sigma * np.sqrt(holding_days / 365.0)

    # 5 scénarios
    scenarios = [
        ("-1.5 SD", "Forte Baisse", spot - 1.5 * sd_move),
        ("-0.5 SD", "Baisse", spot - 0.5 * sd_move),
        ("0 SD", "Stagnation", spot),
        ("+0.5 SD", "Hausse", spot + 0.5 * sd_move),
        ("+1.5 SD", "Forte Hausse", spot + 1.5 * sd_move),
    ]

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    scenario_cols = [sc1, sc2, sc3, sc4, sc5]

    for col, (sd_label, move_label, target_spot) in zip(scenario_cols, scenarios):
        sim_pnl = simulate_pnl(
            strategy["legs"], target_spot, 21, current_sigma, qty_sim
        )

        # Label dynamique basé sur le P&L
        if sim_pnl > take_profit_sim:
            result_label = "🚀 Très Positif"
            pnl_border = "rgba(52, 211, 153, 0.4)"
        elif sim_pnl > 0:
            result_label = "🟢 Positif"
            pnl_border = "rgba(52, 211, 153, 0.25)"
        elif sim_pnl == 0:
            result_label = "⚪ Neutre"
            pnl_border = "rgba(255, 255, 255, 0.1)"
        elif sim_pnl < -max_risk_sim * 0.5:
            result_label = "🔴 Très Défavorable"
            pnl_border = "rgba(248, 113, 113, 0.4)"
        else:
            result_label = "🟠 Défavorable"
            pnl_border = "rgba(251, 191, 36, 0.3)"

        pnl_color = "#34D399" if sim_pnl >= 0 else "#F87171"
        pnl_sign = "+" if sim_pnl > 0 else ""

        # Calcul du % de variation du sous-jacent
        pct_change = ((target_spot - spot) / spot) * 100
        pct_sign = "+" if pct_change >= 0 else ""
        pct_color = "#34D399" if pct_change >= 0 else "#F87171"

        with col:
            st.markdown(f"""
            <div class="greek-card" style="border-color: {pnl_border}; padding: 1.2rem 0.8rem;">
                <div class="greek-symbol" style="color: #FBBF24; margin-bottom: 0.5rem;">{sd_label}</div>
                <div style="font-family: 'Fira Code', monospace; font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.15rem;">${target_spot:,.2f}</div>
                <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; font-weight: 600; color: {pct_color}; margin-bottom: 0.4rem;">{pct_sign}{pct_change:.1f}%</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-bottom: 0.6rem; font-family: 'Fira Sans', sans-serif;">{result_label}</div>
                <div style="font-family: 'Fira Code', monospace; font-size: 1.25rem; font-weight: 700; color: {pnl_color};">{pnl_sign}${abs(sim_pnl):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(f"📐 Écart-type estimé sur {holding_days}j : ±${sd_move:,.2f} (basé sur IV {current_sigma*100:.1f}%)")

    st.markdown("---")

    # ─── Section 5 : PLAN DE VOL ───
    st.markdown("### 🛫 Plan de Vol (Triggers de Sortie)")

    exit_plan = strategy["exit_plan"]

    # Estimer le prix du sous-jacent pour le take profit
    tp_target_spot = estimate_take_profit_spot(
        strategy["legs"], spot, 21, current_sigma, qty_sim,
        exit_plan["take_profit"]
    )
    if tp_target_spot is not None:
        tp_pct_change = ((tp_target_spot - spot) / spot) * 100
        tp_pct_sign = "+" if tp_pct_change >= 0 else ""
        tp_spot_info = (
            f" · Sous-jacent estimé à **\\${tp_target_spot:,.2f}** "
            f"({tp_pct_sign}{tp_pct_change:.1f}% vs spot actuel)"
        )
    else:
        tp_spot_info = ""

    st.info(
        f"🎯 **TAKE PROFIT** — Placez un ordre limite (GTC) pour racheter la position et "
        f"encaisser dès que le profit atteint **\\${exit_plan['take_profit']:,.2f}** "
        f"(50% du profit maximum).{tp_spot_info}"
    )

    st.warning(
        f"⏱️ **TIME STOP** — Clôturez obligatoirement la position le "
        f"**{exit_plan['time_stop_date']}** (dans {exit_plan['time_stop_dte']} jours, à 21 DTE), "
        f"quels que soient les gains/pertes, pour écraser le risque Gamma."
    )

    # ─── Footer ───
    st.markdown("---")
    st.caption(
        f"📊 Analyse exécutée le {dt.datetime.now().strftime('%d/%m/%Y à %H:%M')} · "
        f"Ticker: {ticker} · Budget: ${budget:,.0f} · Biais: {bias}"
    )

except ValueError as e:
    st.error(f"⚠️ **Erreur** : {e}")
    import zoneinfo as _zi
    try:
        _et = _zi.ZoneInfo("America/New_York")
        _local_tz = dt.datetime.now().astimezone().tzinfo
        _open_local = dt.datetime.now(_et).replace(hour=9, minute=30).astimezone(_local_tz)
        _close_local = dt.datetime.now(_et).replace(hour=16, minute=0).astimezone(_local_tz)
        _hours = f"{_open_local.strftime('%Hh%M')}-{_close_local.strftime('%Hh%M')} (heure locale)"
    except Exception:
        _hours = "9h30-16h00 ET"
    st.info(f"💡 **Conseil** : Vérifiez le ticker, augmentez votre budget, ou réessayez pendant les heures de marché ({_hours}).")

except Exception as e:
    st.error(f"❌ **Erreur inattendue** : {type(e).__name__} — {e}")
    st.info("💡 Cela peut être dû à un ticker invalide, au marché fermé, ou à un problème réseau. Réessayez.")
