"""
data/trade_db.py — Journal de trades SQLite
=============================================
Stocke l'historique des trades envoyés à IBKR avec plans d'exit,
métriques financières et statut.
"""

from __future__ import annotations

import json
import os
import sqlite3
import datetime as dt
from pathlib import Path


# Chemin par défaut : data/trades.db (à côté de ce fichier)
_DEFAULT_DB = Path(__file__).parent / "trades.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    bias            TEXT    NOT NULL,
    strategy_name   TEXT    NOT NULL,
    legs_json       TEXT    NOT NULL,
    qty             INTEGER NOT NULL DEFAULT 1,
    expiration      TEXT,
    max_risk        REAL,
    max_profit      REAL,
    credit_debit    REAL,
    ev              REAL,
    take_profit     REAL,
    time_stop_date  TEXT,
    ibkr_order_id   INTEGER,
    ibkr_status     TEXT,
    spot_at_entry   REAL,
    notes           TEXT    DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'OPEN'
);
"""


class TradeDB:
    """Interface CRUD pour le journal de trades SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    # ── Create ──

    def save_trade(
        self,
        ticker: str,
        bias: str,
        strategy: dict,
        ibkr_result: dict,
        spot: float | None = None,
    ) -> int:
        """Insère un trade et retourne l'id créé."""
        row = (
            dt.datetime.now().isoformat(timespec="seconds"),
            ticker,
            bias,
            strategy.get("name", ""),
            json.dumps(strategy.get("legs", []), ensure_ascii=False),
            strategy.get("qty", 1),
            strategy.get("expiration", ""),
            strategy.get("max_risk"),
            strategy.get("max_profit"),
            strategy.get("credit_or_debit"),
            strategy.get("ev"),
            strategy.get("exit_plan", {}).get("take_profit"),
            strategy.get("exit_plan", {}).get("time_stop_date"),
            ibkr_result.get("order_id"),
            ibkr_result.get("status"),
            spot,
            "",   # notes
            "OPEN",
        )
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO trades (
                    created_at, ticker, bias, strategy_name, legs_json,
                    qty, expiration, max_risk, max_profit, credit_debit,
                    ev, take_profit, time_stop_date,
                    ibkr_order_id, ibkr_status, spot_at_entry,
                    notes, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            return cur.lastrowid

    # ── Read ──

    def get_trade(self, trade_id: int) -> dict | None:
        """Retourne un trade par son id, ou None s'il n'existe pas."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_trades(self) -> list[dict]:
        """Retourne tous les trades, les plus récents en premier."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC, id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Delete ──

    def delete_trade(self, trade_id: int) -> bool:
        """Supprime un trade par son id. Retourne True si une ligne a été supprimée."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
            return cur.rowcount > 0
