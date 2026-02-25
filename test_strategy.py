"""
Script de test pour le moteur de probabilités.
Usage : python3 test_strategy.py
"""
import numpy as np
from scipy.stats import norm

RISK_FREE_RATE = 0.05


def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def simulate_pnl(legs, target_spot, days_to_target, current_sigma, qty):
    T_target = max(days_to_target, 1) / 365.0
    initial_value = 0.0
    for leg in legs:
        sign = 1 if leg["action"] == "BUY" else -1
        initial_value += sign * leg["price"]
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


def compute_real_probabilities(legs, spot, dte, sigma, qty,
                                take_profit, max_risk):
    holding_days = max(1, dte - 21)
    remaining_dte = min(21, dte)
    T_holding = holding_days / 365.0

    drift = (RISK_FREE_RATE - 0.5 * sigma**2) * T_holding
    vol = sigma * np.sqrt(T_holding)

    n_points = 500
    z_values = np.linspace(-4, 4, n_points)
    dz = z_values[1] - z_values[0]

    p_take_profit = 0.0
    p_breakeven = 0.0
    p_max_loss = 0.0

    for z in z_values:
        s_t = spot * np.exp(drift + vol * z)
        prob = norm.pdf(z) * dz
        pnl = simulate_pnl(legs, s_t, remaining_dte, sigma, qty)

        if pnl >= take_profit:
            p_take_profit += prob
        if pnl >= 0:
            p_breakeven += prob
        if pnl <= -max_risk * 0.95:
            p_max_loss += prob

    return {
        "p_take_profit": round(max(0.1, min(99.9, p_take_profit * 100)), 1),
        "p_breakeven": round(max(0.1, min(99.9, p_breakeven * 100)), 1),
        "p_max_loss": round(max(0.1, min(99.9, p_max_loss * 100)), 1),
    }


def make_bs_legs(spot, sigma, dte, leg_specs):
    """
    Crée des legs avec des prix cohérents Black-Scholes (prix d'ouverture = BS à T=dte).
    leg_specs: list of (action, type, strike)
    """
    T = dte / 365.0
    legs = []
    for action, opt_type, strike in leg_specs:
        price = black_scholes_price(spot, strike, T, RISK_FREE_RATE, sigma, opt_type.lower())
        legs.append({
            "action": action,
            "type": opt_type,
            "strike": strike,
            "exp": "2025-05-15",
            "dte": dte,
            "price": round(price, 2),
        })
    return legs


# ══════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════

def test_pnl_sanity():
    """Vérification P&L avec prix BS-cohérents."""
    print("=" * 60)
    print("TEST 1 : Sanity check P&L (prix BS)")
    print("=" * 60)
    spot, sigma, dte = 100.0, 0.20, 45

    legs = make_bs_legs(spot, sigma, dte, [
        ("BUY", "Call", 100.0),
        ("SELL", "Call", 105.0),
    ])
    net_debit = legs[0]["price"] - legs[1]["price"]
    print(f"  Bull Call 100/105: Buy@${legs[0]['price']}, Sell@${legs[1]['price']}")
    print(f"  Net Debit: ${net_debit:.2f}")

    pnl_90 = simulate_pnl(legs, 90.0, 21, sigma, 1)
    pnl_100 = simulate_pnl(legs, 100.0, 21, sigma, 1)
    pnl_110 = simulate_pnl(legs, 110.0, 21, sigma, 1)

    print(f"  P&L à spot=90:  ${pnl_90:+.2f}")
    print(f"  P&L à spot=100: ${pnl_100:+.2f}")
    print(f"  P&L à spot=110: ${pnl_110:+.2f}")

    assert pnl_90 < 0, "Deep OTM = loss"
    assert pnl_110 > 0, "Deep ITM = profit"
    print("  ✅ OK\n")


def test_bull_put_spread():
    """Bull Put Spread OTM (crédit) — haute PoP attendue."""
    print("=" * 60)
    print("TEST 2 : Bull Put Spread SPY (crédit, prix BS)")
    print("=" * 60)
    spot, sigma, dte = 500.0, 0.18, 45

    legs = make_bs_legs(spot, sigma, dte, [
        ("SELL", "Put", 480.0),
        ("BUY", "Put", 475.0),
    ])

    net_credit = legs[0]["price"] - legs[1]["price"]
    width = 5.0
    max_profit = net_credit * 100
    max_risk = (width * 100) - max_profit
    take_profit = max_profit * 0.5

    print(f"  Spot: ${spot}, σ: {sigma}, DTE: {dte}")
    print(f"  Sell Put 480 @ ${legs[0]['price']}, Buy Put 475 @ ${legs[1]['price']}")
    print(f"  Net Credit: ${net_credit:.2f}, Max Risk: ${max_risk:.2f}")
    print(f"  Take Profit: ${take_profit:.2f}")

    # Debug P&L
    print(f"\n  P&L sweep (remaining_dte=21):")
    for sp in [470, 475, 480, 490, 500, 510]:
        pnl = simulate_pnl(legs, sp, 21, sigma, 1)
        print(f"    Spot={sp}: ${pnl:+.2f}")
    print()

    probs = compute_real_probabilities(legs, spot, dte, sigma, 1,
                                        take_profit, max_risk)
    print(f"  → P(Take Profit): {probs['p_take_profit']}%")
    print(f"  → P(Break-Even):  {probs['p_breakeven']}%")
    print(f"  → P(Max Loss):    {probs['p_max_loss']}%")

    assert probs["p_breakeven"] > probs["p_take_profit"]
    assert probs["p_breakeven"] > 50, f"OTM credit spread should have > 50% BE, got {probs['p_breakeven']}"
    print("  ✅ OK\n")


def test_bear_put_spread():
    """Bear Put Spread ATM (débit) — probabilités modérées attendues."""
    print("=" * 60)
    print("TEST 3 : Bear Put Spread (débit, prix BS)")
    print("=" * 60)
    spot, sigma, dte = 62.0, 0.25, 45

    legs = make_bs_legs(spot, sigma, dte, [
        ("BUY", "Put", 62.0),
        ("SELL", "Put", 61.0),
    ])

    net_debit = legs[0]["price"] - legs[1]["price"]
    width = 1.0
    max_risk = net_debit * 100
    max_profit = (width * 100) - max_risk
    take_profit = max_profit * 0.5

    print(f"  Spot: ${spot}, σ: {sigma}, DTE: {dte}")
    print(f"  Buy Put 62 @ ${legs[0]['price']}, Sell Put 61 @ ${legs[1]['price']}")
    print(f"  Net Debit: ${net_debit:.2f}, Max Risk: ${max_risk:.2f}")
    print(f"  Take Profit: ${take_profit:.2f}")

    # Debug P&L
    print(f"\n  P&L sweep (remaining_dte=21):")
    for sp in [58, 59, 60, 61, 62, 63, 64]:
        pnl = simulate_pnl(legs, sp, 21, sigma, 1)
        print(f"    Spot={sp}: ${pnl:+.2f}")
    print()

    probs = compute_real_probabilities(legs, spot, dte, sigma, 1,
                                        take_profit, max_risk)
    print(f"  → P(Take Profit): {probs['p_take_profit']}%")
    print(f"  → P(Break-Even):  {probs['p_breakeven']}%")
    print(f"  → P(Max Loss):    {probs['p_max_loss']}%")

    assert probs["p_breakeven"] > probs["p_take_profit"]
    assert probs["p_take_profit"] < 50, "ATM debit spread TP should be < 50%"
    print("  ✅ OK\n")


def test_iron_condor():
    """Iron Condor (crédit, prix BS) — haute PoP attendue."""
    print("=" * 60)
    print("TEST 4 : Iron Condor SPY (crédit, prix BS)")
    print("=" * 60)
    spot, sigma, dte = 500.0, 0.22, 45

    legs = make_bs_legs(spot, sigma, dte, [
        ("SELL", "Put", 480.0),
        ("BUY", "Put", 475.0),
        ("SELL", "Call", 520.0),
        ("BUY", "Call", 525.0),
    ])

    sold = legs[0]["price"] + legs[2]["price"]
    bought = legs[1]["price"] + legs[3]["price"]
    net_credit = sold - bought
    max_width = 5.0
    max_profit = net_credit * 100
    max_risk = (max_width * 100) - max_profit
    take_profit = max_profit * 0.5

    print(f"  Spot: ${spot}, σ: {sigma}, DTE: {dte}")
    for l in legs:
        print(f"    {l['action']} {l['type']} {l['strike']} @ ${l['price']:.2f}")
    print(f"  Net Credit: ${net_credit:.2f}, Max Risk: ${max_risk:.2f}")
    print(f"  Take Profit: ${take_profit:.2f}")

    # Debug P&L
    print(f"\n  P&L sweep (remaining_dte=21):")
    for sp in [470, 475, 480, 490, 500, 510, 520, 525, 530]:
        pnl = simulate_pnl(legs, sp, 21, sigma, 1)
        print(f"    Spot={sp}: ${pnl:+.2f}")
    print()

    probs = compute_real_probabilities(legs, spot, dte, sigma, 1,
                                        take_profit, max_risk)
    print(f"  → P(Take Profit): {probs['p_take_profit']}%")
    print(f"  → P(Break-Even):  {probs['p_breakeven']}%")
    print(f"  → P(Max Loss):    {probs['p_max_loss']}%")

    assert probs["p_breakeven"] > 50, f"IC should have > 50% BE, got {probs['p_breakeven']}"
    print("  ✅ OK\n")


if __name__ == "__main__":
    print("\n🧪 TEST DU MOTEUR DE PROBABILITÉS (prix BS-cohérents)\n")
    test_pnl_sanity()
    test_bull_put_spread()
    test_bear_put_spread()
    test_iron_condor()
    print("🎉 Tous les tests passent !")
