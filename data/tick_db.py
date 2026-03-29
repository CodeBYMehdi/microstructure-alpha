"""SQLite-based trade/decision storage with WAL mode for crash safety.

Replaces CSV-based logging with ACID-compliant database that survives
power outages and process crashes. Uses Write-Ahead Logging (WAL) for
concurrent read/write without locking.

Usage:
    db = TradeDatabase("trades.db")
    db.record_order(proposal, order_id, equity=100000)
    db.record_fill(result, proposal, equity=100050)
    
    # Query
    trades = db.query_trades(symbol="AAPL", start_date="2024-01-01")
    daily_pnl = db.get_daily_pnl()
"""

import sqlite3
import logging
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager

from core.types import TradeProposal, OrderResult, TradeAction

logger = logging.getLogger(__name__)


class TradeDatabase:
    """ACID-compliant trade database using SQLite WAL mode."""

    def __init__(self, filepath: str = "trades.db"):
        self.filepath = Path(filepath)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
        logger.info(f"TradeDatabase initialized: {self.filepath} (WAL mode)")

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            str(self.filepath),
            check_same_thread=False,
            isolation_level=None,  # Autocommit for WAL
        )
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for crash safety + concurrent reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._conn.execute("PRAGMA busy_timeout=5000")

    @contextmanager
    def _transaction(self):
        """Context manager for explicit transactions."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                order_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                requested_qty REAL NOT NULL,
                filled_qty REAL DEFAULT 0,
                requested_price REAL DEFAULT 0,
                filled_price REAL DEFAULT 0,
                fees REAL DEFAULT 0,
                slippage_pct REAL DEFAULT 0,
                reason TEXT,
                regime_id TEXT,
                net_position REAL DEFAULT 0,
                equity REAL DEFAULT 0,
                kill_switch INTEGER DEFAULT 0,
                metadata_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                tick_count INTEGER,
                regime_id TEXT,
                action TEXT NOT NULL,
                reason TEXT,
                delta_mu REAL DEFAULT 0,
                delta_sigma REAL DEFAULT 0,
                delta_entropy REAL DEFAULT 0,
                delta_skew REAL DEFAULT 0,
                regime_age INTEGER DEFAULT 0,
                transition_strength REAL DEFAULT 0,
                tail_risk REAL DEFAULT 0,
                l2_liquidity_slope REAL DEFAULT 0,
                composite_signal_dir REAL DEFAULT 0,
                composite_signal_str REAL DEFAULT 0,
                result TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                peak_equity REAL NOT NULL,
                drawdown_pct REAL DEFAULT 0,
                net_position REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                regime_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                metadata_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS alpha_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                signal_value REAL NOT NULL,
                predicted_direction REAL DEFAULT 0,
                realized_return REAL DEFAULT 0,
                contribution_pnl REAL DEFAULT 0,
                weight REAL DEFAULT 0,
                is_correct INTEGER DEFAULT 0,
                metadata_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Indices for common queries
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp);
            CREATE INDEX IF NOT EXISTS idx_orders_event ON orders(event_type);
            CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_decisions_result ON decisions(result);
            CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
            CREATE INDEX IF NOT EXISTS idx_attribution_signal ON alpha_attribution(signal_name);
        """)

    # ── Order/Fill Recording ──

    def record_order(
        self,
        proposal: TradeProposal,
        order_id: str,
        equity: float = 0.0,
        kill_switch: bool = False,
        net_position: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        regime_id = str(proposal.regime_state.metadata.get('id', 'UNKNOWN')) if proposal.regime_state else 'UNKNOWN'
        with self._transaction():
            self._conn.execute(
                """INSERT INTO orders (timestamp, order_id, event_type, symbol, side,
                   requested_qty, requested_price, reason, regime_id, net_position,
                   equity, kill_switch, metadata_json)
                   VALUES (?, ?, 'ORDER_SUBMITTED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(), order_id, proposal.symbol,
                    proposal.action.value, proposal.quantity, proposal.price or 0.0,
                    proposal.reason, regime_id, net_position, equity,
                    int(kill_switch), json.dumps(metadata) if metadata else None,
                ),
            )

    def record_fill(
        self,
        result: OrderResult,
        proposal: TradeProposal,
        equity: float = 0.0,
        kill_switch: bool = False,
        net_position: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        slippage = 0.0
        if proposal.price and proposal.price > 0:
            slippage = abs(result.filled_price - proposal.price) / proposal.price

        regime_id = str(proposal.regime_state.metadata.get('id', 'UNKNOWN')) if proposal.regime_state else 'UNKNOWN'
        with self._transaction():
            self._conn.execute(
                """INSERT INTO orders (timestamp, order_id, event_type, symbol, side,
                   requested_qty, filled_qty, requested_price, filled_price, fees,
                   slippage_pct, reason, regime_id, net_position, equity,
                   kill_switch, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(), result.order_id, result.status,
                    proposal.symbol, proposal.action.value, proposal.quantity,
                    result.filled_quantity, proposal.price or 0.0,
                    result.filled_price, result.fees, slippage,
                    proposal.reason, regime_id, net_position, equity,
                    int(kill_switch), json.dumps(metadata) if metadata else None,
                ),
            )

    def record_rejection(
        self,
        proposal: TradeProposal,
        reason: str,
        equity: float = 0.0,
        kill_switch: bool = False,
        net_position: float = 0.0,
    ) -> None:
        regime_id = str(proposal.regime_state.metadata.get('id', 'UNKNOWN')) if proposal.regime_state else 'UNKNOWN'
        with self._transaction():
            self._conn.execute(
                """INSERT INTO orders (timestamp, order_id, event_type, symbol, side,
                   requested_qty, requested_price, reason, regime_id, net_position,
                   equity, kill_switch)
                   VALUES (?, '', 'REJECTED', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(), proposal.symbol,
                    proposal.action.value, proposal.quantity,
                    proposal.price or 0.0, reason, regime_id,
                    net_position, equity, int(kill_switch),
                ),
            )

    # ── Decision Logging ──

    def record_decision(self, decision: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO decisions (timestamp, tick_count, regime_id, action, reason,
               delta_mu, delta_sigma, delta_entropy, delta_skew, regime_age,
               transition_strength, tail_risk, l2_liquidity_slope,
               composite_signal_dir, composite_signal_str, result, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.get('timestamp', 0),
                decision.get('tick_count', 0),
                decision.get('regime_id', ''),
                decision.get('action', 'FLAT'),
                decision.get('reason', ''),
                decision.get('delta_mu', 0),
                decision.get('delta_sigma', 0),
                decision.get('delta_entropy', 0),
                decision.get('delta_skew', 0),
                decision.get('regime_age', 0),
                decision.get('transition_strength', 0),
                decision.get('tail_risk', 0),
                decision.get('l2_liquidity_slope', 0),
                decision.get('composite_signal_dir', 0),
                decision.get('composite_signal_str', 0),
                decision.get('result', 'FLAT'),
                json.dumps(decision.get('metadata')) if decision.get('metadata') else None,
            ),
        )

    # ── Equity Snapshots ──

    def record_equity_snapshot(
        self,
        equity: float,
        peak_equity: float,
        drawdown_pct: float,
        net_position: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        regime_id: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT INTO equity_snapshots (timestamp, equity, peak_equity,
               drawdown_pct, net_position, unrealized_pnl, realized_pnl, regime_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(), equity, peak_equity,
                drawdown_pct, net_position, unrealized_pnl, realized_pnl, regime_id,
            ),
        )

    # ── Alpha Attribution ──

    def record_attribution(
        self,
        signal_name: str,
        signal_value: float,
        predicted_direction: float = 0.0,
        realized_return: float = 0.0,
        contribution_pnl: float = 0.0,
        weight: float = 0.0,
        is_correct: bool = False,
    ) -> None:
        self._conn.execute(
            """INSERT INTO alpha_attribution (timestamp, signal_name, signal_value,
               predicted_direction, realized_return, contribution_pnl, weight, is_correct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(), signal_name, signal_value,
                predicted_direction, realized_return, contribution_pnl,
                weight, int(is_correct),
            ),
        )

    # ── Alerts ──

    def record_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO alerts (timestamp, alert_type, severity, message, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(), alert_type, severity, message,
                json.dumps(metadata) if metadata else None,
            ),
        )

    # ── Queries ──

    def query_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        query = "SELECT * FROM orders WHERE 1=1"
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_daily_pnl(self, n_days: int = 30) -> List[Dict]:
        rows = self._conn.execute(
            """SELECT date(timestamp) as trade_date,
                      COUNT(*) as trade_count,
                      SUM(CASE WHEN event_type='FILLED' THEN fees ELSE 0 END) as total_fees,
                      SUM(CASE WHEN event_type='FILLED' THEN filled_qty * filled_price ELSE 0 END) as total_notional
               FROM orders
               WHERE event_type IN ('FILLED')
               GROUP BY date(timestamp)
               ORDER BY trade_date DESC
               LIMIT ?""",
            (n_days,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_net_position(self, symbol: str) -> float:
        row = self._conn.execute(
            """SELECT SUM(CASE
                   WHEN event_type='FILLED' AND side='BUY' THEN filled_qty
                   WHEN event_type='FILLED' AND side='SELL' THEN -filled_qty
                   ELSE 0 END) as net_pos
               FROM orders WHERE symbol = ?""",
            (symbol,),
        ).fetchone()
        return float(row['net_pos'] or 0.0) if row else 0.0

    def get_all_positions(self) -> Dict[str, float]:
        rows = self._conn.execute(
            """SELECT symbol,
                      SUM(CASE
                          WHEN event_type='FILLED' AND side='BUY' THEN filled_qty
                          WHEN event_type='FILLED' AND side='SELL' THEN -filled_qty
                          ELSE 0 END) as net_pos
               FROM orders
               GROUP BY symbol
               HAVING ABS(net_pos) > 1e-10""",
        ).fetchall()
        return {row['symbol']: float(row['net_pos']) for row in rows}

    def get_total_fees(self) -> float:
        row = self._conn.execute(
            "SELECT SUM(fees) as total FROM orders WHERE event_type='FILLED'"
        ).fetchone()
        return float(row['total'] or 0.0) if row else 0.0

    def get_signal_attribution_summary(self, n_days: int = 30) -> List[Dict]:
        rows = self._conn.execute(
            """SELECT signal_name,
                      COUNT(*) as observations,
                      AVG(signal_value) as avg_signal,
                      SUM(contribution_pnl) as total_pnl_contribution,
                      AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as hit_rate,
                      AVG(weight) as avg_weight
               FROM alpha_attribution
               WHERE created_at >= date('now', '-' || ? || ' days')
               GROUP BY signal_name
               ORDER BY total_pnl_contribution DESC""",
            (n_days,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    @property
    def entry_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM orders").fetchone()
        return int(row['cnt']) if row else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
