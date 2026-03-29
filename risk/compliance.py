"""Compliance guard — pre-trade checks required for live trading.

Enforces:
- Price collars (reject orders far from market)
- Daily loss limits (absolute + percentage)
- Max order size per trade
- Per-symbol position limits
- Wash trade prevention (min interval between opposite trades)
- Quote staleness rejection
"""

import logging
import time
from collections import defaultdict, deque
from typing import Tuple, Optional

from core.types import TradeProposal, TradeAction
from config.loader import get_config

logger = logging.getLogger(__name__)


class ComplianceGuard:
    def __init__(self, config=None):
        config = config or get_config()
        cc = config.thresholds.compliance

        # Price collar
        self.price_collar_pct: float = cc.price_collar_pct

        # Order size
        self.max_order_notional: float = cc.max_order_notional
        self.max_position_notional: float = cc.max_position_notional

        # Daily loss
        self.daily_loss_limit_pct: float = cc.daily_loss_limit_pct
        self.daily_loss_limit_abs: float = cc.daily_loss_limit_abs
        self._session_start_equity: float = 0.0
        self._session_realized_pnl: float = 0.0

        # Wash trade prevention
        self.min_trade_interval_s: float = cc.min_trade_interval_s
        self._last_trade: dict = {}  # symbol -> (side, timestamp)
        self._trades_per_symbol: defaultdict = defaultdict(lambda: deque(maxlen=100))

        # Quote staleness
        self.max_quote_age_s: float = cc.max_quote_age_s

        # Violation counters
        self._violations: deque = deque(maxlen=500)

        logger.info(
            f"ComplianceGuard: collar={self.price_collar_pct:.1%}, "
            f"max_order=${self.max_order_notional:.0f}, "
            f"daily_loss={self.daily_loss_limit_pct:.1%}/${self.daily_loss_limit_abs:.0f}, "
            f"wash_interval={self.min_trade_interval_s}s"
        )

    def start_session(self, equity: float) -> None:
        """Call at session start (market open) to reset daily counters."""
        self._session_start_equity = equity
        self._session_realized_pnl = 0.0
        self._last_trade.clear()
        self._trades_per_symbol.clear()
        logger.info(f"Compliance session started: equity=${equity:.2f}")

    def check_order(
        self,
        proposal: TradeProposal,
        last_price: float,
        last_tick_time: Optional[float] = None,
        current_position_notional: float = 0.0,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Pre-trade compliance check. Returns (ok, reason)."""
        symbol = proposal.symbol
        price = proposal.price or last_price
        notional = proposal.quantity * price

        # 1. Price collar: reject if order price deviates too much from last trade
        if last_price > 0 and price > 0:
            deviation = abs(price - last_price) / last_price
            if deviation > self.price_collar_pct:
                reason = (
                    f"Price collar violation: {deviation:.2%} deviation "
                    f"(limit {self.price_collar_pct:.2%})"
                )
                self._record_violation("PRICE_COLLAR", reason)
                return False, reason

        # 2. Max order size
        if notional > self.max_order_notional:
            reason = (
                f"Order too large: ${notional:.0f} > ${self.max_order_notional:.0f}"
            )
            self._record_violation("ORDER_SIZE", reason)
            return False, reason

        # 3. Position limit per symbol
        if proposal.action == TradeAction.BUY:
            projected = current_position_notional + notional
        else:
            projected = current_position_notional - notional
        if abs(projected) > self.max_position_notional:
            reason = (
                f"Position limit: projected ${abs(projected):.0f} > "
                f"${self.max_position_notional:.0f}"
            )
            self._record_violation("POSITION_LIMIT", reason)
            return False, reason

        # 4. Wash trade prevention
        if symbol in self._last_trade:
            last_side, last_time = self._last_trade[symbol]
            now = current_time if current_time is not None else time.time()
            elapsed = now - last_time
            if last_side != proposal.action and elapsed < self.min_trade_interval_s:
                reason = (
                    f"Wash trade: {proposal.action.value} after {last_side.value} "
                    f"on {symbol} ({elapsed:.1f}s < {self.min_trade_interval_s}s)"
                )
                self._record_violation("WASH_TRADE", reason)
                return False, reason

        # 5. Quote staleness
        if last_tick_time is not None:
            age = time.time() - last_tick_time
            if age > self.max_quote_age_s:
                reason = f"Stale quote: {age:.1f}s old (limit {self.max_quote_age_s}s)"
                self._record_violation("STALE_QUOTE", reason)
                return False, reason

        return True, "OK"

    def check_daily_loss(self, current_equity: float) -> Tuple[bool, str]:
        """Check if daily loss limit breached. Returns (ok, reason)."""
        if self._session_start_equity <= 0:
            return True, "OK"

        session_pnl = current_equity - self._session_start_equity
        pct_loss = -session_pnl / self._session_start_equity if session_pnl < 0 else 0.0

        # Percentage check
        if pct_loss > self.daily_loss_limit_pct:
            reason = (
                f"Daily loss limit (pct): {pct_loss:.2%} > {self.daily_loss_limit_pct:.2%}"
            )
            self._record_violation("DAILY_LOSS_PCT", reason)
            return False, reason

        # Absolute check
        if -session_pnl > self.daily_loss_limit_abs:
            reason = (
                f"Daily loss limit (abs): ${-session_pnl:.2f} > ${self.daily_loss_limit_abs:.2f}"
            )
            self._record_violation("DAILY_LOSS_ABS", reason)
            return False, reason

        return True, "OK"

    def record_trade(self, symbol: str, side: TradeAction, pnl: float = 0.0, timestamp: Optional[float] = None) -> None:
        """Record a completed trade for wash-trade tracking and daily PnL."""
        self._last_trade[symbol] = (side, timestamp if timestamp is not None else time.time())
        self._trades_per_symbol[symbol].append(time.time())
        self._session_realized_pnl += pnl

    def get_session_pnl(self) -> float:
        return self._session_realized_pnl

    def get_violations(self) -> list:
        return list(self._violations)

    def get_status(self) -> dict:
        return {
            "session_start_equity": self._session_start_equity,
            "session_realized_pnl": self._session_realized_pnl,
            "violations_count": len(self._violations),
            "tracked_symbols": len(self._last_trade),
        }

    def _record_violation(self, kind: str, reason: str) -> None:
        self._violations.append({
            "kind": kind,
            "reason": reason,
            "timestamp": time.time(),
        })
        logger.warning(f"COMPLIANCE VIOLATION [{kind}]: {reason}")
