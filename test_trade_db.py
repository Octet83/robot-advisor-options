"""
test_trade_db.py — Tests unitaires pour le journal de trades SQLite
====================================================================
"""
import json
import os
import tempfile
import unittest

from data.trade_db import TradeDB


class TestTradeDB(unittest.TestCase):
    """Tests CRUD pour TradeDB."""

    def setUp(self):
        """Crée une BDD temporaire pour chaque test."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = TradeDB(db_path=self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    # ── Helpers ──

    def _sample_strategy(self) -> dict:
        return {
            "name": "🦅 Iron Condor",
            "legs": [
                {"action": "SELL", "type": "Put", "strike": 580.0,
                 "exp": "2026-04-17", "dte": 51, "price": 3.50},
                {"action": "BUY", "type": "Put", "strike": 575.0,
                 "exp": "2026-04-17", "dte": 51, "price": 2.80},
                {"action": "SELL", "type": "Call", "strike": 620.0,
                 "exp": "2026-04-17", "dte": 51, "price": 3.20},
                {"action": "BUY", "type": "Call", "strike": 625.0,
                 "exp": "2026-04-17", "dte": 51, "price": 2.60},
            ],
            "qty": 2,
            "expiration": "2026-04-17",
            "max_risk": 740.0,
            "max_profit": 260.0,
            "credit_or_debit": 260.0,
            "ev": 32.50,
            "exit_plan": {
                "take_profit": 130.0,
                "time_stop_date": "27/03/2026",
            },
        }

    def _sample_ibkr_result(self) -> dict:
        return {
            "order_id": 42,
            "status": "PreSubmitted",
            "message": "Ordre Combo 4-legs placé dans TWS",
        }

    # ── Tests ──

    def test_init_creates_table(self):
        """La table trades existe après initialisation."""
        import sqlite3
        conn = sqlite3.connect(self._tmp.name)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        self.assertIn(("trades",), tables)

    def test_save_and_list(self):
        """Round-trip : save → list retourne le trade correct."""
        strat = self._sample_strategy()
        ibkr = self._sample_ibkr_result()
        tid = self.db.save_trade("SPY", "Neutre", strat, ibkr, spot=600.0)

        self.assertIsInstance(tid, int)
        self.assertGreater(tid, 0)

        trades = self.db.list_trades()
        self.assertEqual(len(trades), 1)

        t = trades[0]
        self.assertEqual(t["ticker"], "SPY")
        self.assertEqual(t["bias"], "Neutre")
        self.assertEqual(t["strategy_name"], "🦅 Iron Condor")
        self.assertEqual(t["qty"], 2)
        self.assertAlmostEqual(t["max_risk"], 740.0)
        self.assertAlmostEqual(t["take_profit"], 130.0)
        self.assertEqual(t["ibkr_order_id"], 42)
        self.assertEqual(t["status"], "OPEN")
        self.assertAlmostEqual(t["spot_at_entry"], 600.0)

        # Vérifier les legs JSON
        legs = json.loads(t["legs_json"])
        self.assertEqual(len(legs), 4)
        self.assertEqual(legs[0]["action"], "SELL")

    def test_delete(self):
        """delete_trade supprime uniquement le trade ciblé."""
        strat = self._sample_strategy()
        ibkr = self._sample_ibkr_result()

        id1 = self.db.save_trade("SPY", "Neutre", strat, ibkr, spot=600.0)
        id2 = self.db.save_trade("AAPL", "Haussier", strat, ibkr, spot=180.0)

        self.assertEqual(len(self.db.list_trades()), 2)

        result = self.db.delete_trade(id1)
        self.assertTrue(result)

        remaining = self.db.list_trades()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["ticker"], "AAPL")

    def test_delete_nonexistent(self):
        """delete_trade retourne False pour un id inexistant."""
        self.assertFalse(self.db.delete_trade(999))

    def test_list_empty(self):
        """list_trades retourne une liste vide si aucun trade."""
        self.assertEqual(self.db.list_trades(), [])

    def test_list_order(self):
        """Les trades sont retournés par date décroissante."""
        strat = self._sample_strategy()
        ibkr = self._sample_ibkr_result()

        self.db.save_trade("AAPL", "Haussier", strat, ibkr)
        self.db.save_trade("SPY", "Neutre", strat, ibkr)

        trades = self.db.list_trades()
        self.assertEqual(trades[0]["ticker"], "SPY")  # plus récent
        self.assertEqual(trades[1]["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()
