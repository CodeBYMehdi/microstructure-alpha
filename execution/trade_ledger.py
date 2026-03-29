"""Persistent trade ledger for audit trail and position reconciliation.

Append-only CSV log of every order and fill. Survives process restarts.
Supports position reconstruction from the ledger for crash recovery.
"""

import csv
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from core.types import TradeProposal, OrderResult, TradeAction

logger = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    timestamp: str
    order_id: str
    event_type: str        # ORDER_SUBMITTED, FILLED, REJECTED, CANCELLED
    symbol: str
    side: str              # BUY / SELL
    requested_qty: float
    filled_qty: float
    requested_price: float
    filled_price: float
    fees: float
    slippage_pct: float
    reason: str
    regime_id: str
    net_position: float    # Position after this event
    equity: float          # Equity after this event
    kill_switch: bool


class TradeLedger:
    """Append-only trade ledger for regulatory audit trail and crash recovery."""

    COLUMNS = [
        'timestamp', 'order_id', 'event_type', 'symbol', 'side',
        'requested_qty', 'filled_qty', 'requested_price', 'filled_price',
        'fees', 'slippage_pct', 'reason', 'regime_id',
        'net_position', 'equity', 'kill_switch',
    ]

    def __init__(self, filepath: str = "trade_ledger.csv"):
        self.filepath = Path(filepath)
        self._entries: List[LedgerEntry] = []
        self._net_positions: Dict[str, float] = {}
        self._total_fees: float = 0.0
        # Session tracking — counts only trades from THIS process lifetime
        self._session_fills: int = 0
        self._session_fees: float = 0.0

        # Create or validate header
        if not self.filepath.exists():
            self._write_header()
            logger.info(f"Trade ledger created: {self.filepath}")
        else:
            self._load_existing()
            logger.info(f"Trade ledger loaded: {len(self._entries)} existing entries")

    def _write_header(self) -> None:
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)

    def _load_existing(self) -> None:
        """Reconstruct position state from existing ledger on startup (crash recovery)."""
        try:
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = LedgerEntry(
                        timestamp=row.get('timestamp', ''),
                        order_id=row.get('order_id', ''),
                        event_type=row.get('event_type', ''),
                        symbol=row.get('symbol', ''),
                        side=row.get('side', ''),
                        requested_qty=float(row.get('requested_qty', 0)),
                        filled_qty=float(row.get('filled_qty', 0)),
                        requested_price=float(row.get('requested_price', 0)),
                        filled_price=float(row.get('filled_price', 0)),
                        fees=float(row.get('fees', 0)),
                        slippage_pct=float(row.get('slippage_pct', 0)),
                        reason=row.get('reason', ''),
                        regime_id=row.get('regime_id', ''),
                        net_position=float(row.get('net_position', 0)),
                        equity=float(row.get('equity', 0)),
                        kill_switch=row.get('kill_switch', 'False') == 'True',
                    )
                    self._entries.append(entry)
                    if entry.event_type == 'FILLED':
                        if entry.side == 'BUY':
                            self._net_positions[entry.symbol] = self._net_positions.get(entry.symbol, 0.0) + entry.filled_qty
                        elif entry.side == 'SELL':
                            self._net_positions[entry.symbol] = self._net_positions.get(entry.symbol, 0.0) - entry.filled_qty
                        self._total_fees += entry.fees
            if self._net_positions:
                logger.info(f"Recovered positions from ledger: {self._net_positions}")
        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")

    def record_order(
        self,
        proposal: TradeProposal,
        order_id: str,
        equity: float = 0.0,
        kill_switch: bool = False,
    ) -> None:
        """Record an order submission."""
        regime_id = str(proposal.regime_state.metadata.get('id', 'UNKNOWN')) if proposal.regime_state else 'UNKNOWN'
        entry = LedgerEntry(
            timestamp=datetime.now().isoformat(),
            order_id=order_id,
            event_type='ORDER_SUBMITTED',
            symbol=proposal.symbol,
            side=proposal.action.value,
            requested_qty=proposal.quantity,
            filled_qty=0.0,
            requested_price=proposal.price or 0.0,
            filled_price=0.0,
            fees=0.0,
            slippage_pct=0.0,
            reason=proposal.reason,
            regime_id=regime_id,
            net_position=self._net_positions.get(proposal.symbol, 0.0),
            equity=equity,
            kill_switch=kill_switch,
        )
        self._append(entry)

    def record_fill(
        self,
        result: OrderResult,
        proposal: TradeProposal,
        equity: float = 0.0,
        kill_switch: bool = False,
    ) -> None:
        """Record a fill event."""
        symbol = proposal.symbol
        side = proposal.action

        # Update position
        if side == TradeAction.BUY:
            self._net_positions[symbol] = self._net_positions.get(symbol, 0.0) + result.filled_quantity
        elif side == TradeAction.SELL:
            self._net_positions[symbol] = self._net_positions.get(symbol, 0.0) - result.filled_quantity
        self._total_fees += result.fees
        self._session_fills += 1
        self._session_fees += result.fees

        slippage = 0.0
        if proposal.price and proposal.price > 0:
            slippage = abs(result.filled_price - proposal.price) / proposal.price

        regime_id = str(proposal.regime_state.metadata.get('id', 'UNKNOWN')) if proposal.regime_state else 'UNKNOWN'
        entry = LedgerEntry(
            timestamp=datetime.now().isoformat(),
            order_id=result.order_id,
            event_type=result.status,
            symbol=symbol,
            side=side.value,
            requested_qty=proposal.quantity,
            filled_qty=result.filled_quantity,
            requested_price=proposal.price or 0.0,
            filled_price=result.filled_price,
            fees=result.fees,
            slippage_pct=slippage,
            reason=proposal.reason,
            regime_id=regime_id,
            net_position=self._net_positions.get(symbol, 0.0),
            equity=equity,
            kill_switch=kill_switch,
        )
        self._append(entry)

    def record_rejection(
        self,
        proposal: TradeProposal,
        reason: str,
        equity: float = 0.0,
        kill_switch: bool = False,
    ) -> None:
        """Record an order rejection."""
        regime_id = str(proposal.regime_state.metadata.get('id', 'UNKNOWN')) if proposal.regime_state else 'UNKNOWN'
        entry = LedgerEntry(
            timestamp=datetime.now().isoformat(),
            order_id='',
            event_type='REJECTED',
            symbol=proposal.symbol,
            side=proposal.action.value,
            requested_qty=proposal.quantity,
            filled_qty=0.0,
            requested_price=proposal.price or 0.0,
            filled_price=0.0,
            fees=0.0,
            slippage_pct=0.0,
            reason=reason,
            regime_id=regime_id,
            net_position=self._net_positions.get(proposal.symbol, 0.0),
            equity=equity,
            kill_switch=kill_switch,
        )
        self._append(entry)

    def _append(self, entry: LedgerEntry) -> None:
        try:
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry.timestamp, entry.order_id, entry.event_type,
                    entry.symbol, entry.side, entry.requested_qty,
                    entry.filled_qty, entry.requested_price, entry.filled_price,
                    entry.fees, entry.slippage_pct, entry.reason,
                    entry.regime_id, entry.net_position, entry.equity,
                    entry.kill_switch,
                ])
                f.flush()
                os.fsync(f.fileno())
            # Append to memory ONLY after disk write succeeds
            self._entries.append(entry)
        except IOError as e:
            logger.critical(f"CRITICAL: Failed to write to trade ledger: {e} — entry NOT recorded in memory")

    def get_orphaned_orders(self) -> List[Dict[str, str]]:
        """Find orders that were SUBMITTED but never FILLED or CANCELLED.

        These represent potential orphaned orders from a previous crash.
        Returns list of dicts with order_id, symbol, side, requested_qty.
        """
        submitted: Dict[str, Dict[str, str]] = {}
        resolved: set = set()

        for entry in self._entries:
            if entry.event_type == 'ORDER_SUBMITTED' and entry.order_id:
                submitted[entry.order_id] = {
                    'order_id': entry.order_id,
                    'symbol': entry.symbol,
                    'side': entry.side,
                    'requested_qty': str(entry.requested_qty),
                    'timestamp': entry.timestamp,
                }
            elif entry.event_type in ('FILLED', 'CANCELLED', 'REJECTED') and entry.order_id:
                resolved.add(entry.order_id)

        return [v for k, v in submitted.items() if k not in resolved]

    def get_net_position(self, symbol: str) -> float:
        return self._net_positions.get(symbol, 0.0)

    def get_all_positions(self) -> Dict[str, float]:
        return dict(self._net_positions)

    def get_total_fees(self) -> float:
        return self._total_fees

    @property
    def fill_count(self) -> int:
        """Count only FILLED entries (actual executed trades, all sessions)."""
        return sum(1 for e in self._entries if e.event_type == 'FILLED')

    @property
    def session_fill_count(self) -> int:
        """Count only fills from THIS session."""
        return self._session_fills

    @property
    def session_fees(self) -> float:
        """Fees from THIS session only."""
        return self._session_fees

    @property
    def entry_count(self) -> int:
        return len(self._entries)
