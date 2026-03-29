"""Data quality sentinel — detects anomalies in live/historical market data.

Monitors for:
- Tick gaps exceeding configurable threshold
- Stale quotes (no update for N seconds)
- Crossed markets (bid > ask)
- Price jumps exceeding N sigma
- Volume spikes / zero volume periods
- Duplicate timestamps
- Exchange timestamp vs receive timestamp drift

Each check returns a DataQualityEvent that can trigger alerts or
automatic data rejection.
"""

import logging
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Any
from enum import Enum

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    TICK_GAP = "TICK_GAP"
    STALE_QUOTE = "STALE_QUOTE"
    CROSSED_MARKET = "CROSSED_MARKET"
    PRICE_JUMP = "PRICE_JUMP"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    ZERO_VOLUME = "ZERO_VOLUME"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    CLOCK_DRIFT = "CLOCK_DRIFT"
    DATA_OUTAGE = "DATA_OUTAGE"
    BACKWARD_TIMESTAMP = "BACKWARD_TIMESTAMP"


@dataclass
class DataQualityEvent:
    issue: QualityIssue
    severity: str              # "WARNING", "CRITICAL"
    symbol: str
    timestamp: float
    message: str
    value: float = 0.0        # The anomalous value
    threshold: float = 0.0    # The threshold that was breached
    metadata: Dict = field(default_factory=dict)


@dataclass
class DataQualityConfig:
    # Tick gap thresholds
    max_tick_gap_ms: float = 5000.0        # 5 seconds
    critical_tick_gap_ms: float = 30000.0  # 30 seconds

    # Stale quote
    max_quote_age_s: float = 5.0
    critical_quote_age_s: float = 15.0

    # Price jump (in rolling sigma)
    price_jump_warning_sigma: float = 5.0
    price_jump_critical_sigma: float = 10.0

    # Volume anomaly
    volume_spike_mult: float = 20.0   # 20x average volume
    zero_volume_max_ticks: int = 50   # N consecutive zero-volume ticks

    # Clock drift
    max_clock_drift_ms: float = 500.0  # 500ms between exchange and receive time

    # Rolling window for statistics
    stats_window: int = 500


class DataQualitySentinel:
    """Monitors incoming market data for quality issues."""

    def __init__(self, config: Optional[DataQualityConfig] = None):
        self.config = config or DataQualityConfig()

        # Per-symbol state
        self._state: Dict[str, _SymbolState] = {}

        # Global event log
        self._events: Deque[DataQualityEvent] = deque(maxlen=1000)
        self._issues_by_type: Dict[QualityIssue, int] = {q: 0 for q in QualityIssue}

        # Callback for real-time alerting
        self._alert_callback = None

        logger.info("DataQualitySentinel initialized")

    def set_alert_callback(self, callback) -> None:
        """Set callback for real-time quality alerts: callback(event: DataQualityEvent)"""
        self._alert_callback = callback

    def check_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: Optional[float],
        ask: Optional[float],
        exchange_ts: Optional[float] = None,
        receive_ts: Optional[float] = None,
    ) -> List[DataQualityEvent]:
        """Check a single tick for quality issues. Returns list of issues found."""
        now = receive_ts or time.time()
        events: List[DataQualityEvent] = []

        # Get or create symbol state
        if symbol not in self._state:
            self._state[symbol] = _SymbolState(self.config.stats_window)
        state = self._state[symbol]

        # 1. Tick gap detection
        if state.last_tick_time is not None:
            gap_ms = (now - state.last_tick_time) * 1000
            if gap_ms > self.config.critical_tick_gap_ms:
                events.append(DataQualityEvent(
                    issue=QualityIssue.TICK_GAP,
                    severity="CRITICAL",
                    symbol=symbol,
                    timestamp=now,
                    message=f"Critical tick gap: {gap_ms:.0f}ms (threshold: {self.config.critical_tick_gap_ms:.0f}ms)",
                    value=gap_ms,
                    threshold=self.config.critical_tick_gap_ms,
                ))
            elif gap_ms > self.config.max_tick_gap_ms:
                events.append(DataQualityEvent(
                    issue=QualityIssue.TICK_GAP,
                    severity="WARNING",
                    symbol=symbol,
                    timestamp=now,
                    message=f"Tick gap: {gap_ms:.0f}ms (threshold: {self.config.max_tick_gap_ms:.0f}ms)",
                    value=gap_ms,
                    threshold=self.config.max_tick_gap_ms,
                ))

        # 1b. Backward timestamp detection (time going backwards = data corruption)
        if state.last_tick_time is not None and now < state.last_tick_time:
            backward_ms = (state.last_tick_time - now) * 1000
            events.append(DataQualityEvent(
                issue=QualityIssue.BACKWARD_TIMESTAMP,
                severity="CRITICAL",
                symbol=symbol,
                timestamp=now,
                message=f"Backward timestamp: {backward_ms:.0f}ms behind previous tick (data corruption or clock reset)",
                value=backward_ms,
                metadata={"previous_ts": state.last_tick_time, "current_ts": now},
            ))

        # 2. Crossed market detection
        if bid is not None and ask is not None:
            if bid > ask and bid > 0 and ask > 0:
                events.append(DataQualityEvent(
                    issue=QualityIssue.CROSSED_MARKET,
                    severity="CRITICAL",
                    symbol=symbol,
                    timestamp=now,
                    message=f"Crossed market: bid={bid:.4f} > ask={ask:.4f}",
                    value=bid - ask,
                ))

        # 3. Price jump detection
        if state.price_history and price > 0:
            prices = np.array(state.price_history)
            if len(prices) >= 20:
                returns = np.diff(np.log(prices[-100:]))
                if len(returns) > 5:
                    ret_std = np.std(returns)
                    if ret_std > 1e-10:
                        last_ret = np.log(price / prices[-1])
                        z_score = abs(last_ret) / ret_std

                        if z_score > self.config.price_jump_critical_sigma:
                            events.append(DataQualityEvent(
                                issue=QualityIssue.PRICE_JUMP,
                                severity="CRITICAL",
                                symbol=symbol,
                                timestamp=now,
                                message=f"Extreme price jump: {z_score:.1f} sigma (return={last_ret:.4%})",
                                value=z_score,
                                threshold=self.config.price_jump_critical_sigma,
                                metadata={"return": float(last_ret), "price": price, "prev_price": float(prices[-1])},
                            ))
                        elif z_score > self.config.price_jump_warning_sigma:
                            events.append(DataQualityEvent(
                                issue=QualityIssue.PRICE_JUMP,
                                severity="WARNING",
                                symbol=symbol,
                                timestamp=now,
                                message=f"Price jump: {z_score:.1f} sigma (return={last_ret:.4%})",
                                value=z_score,
                                threshold=self.config.price_jump_warning_sigma,
                            ))

        # 4. Volume anomaly detection
        if volume is not None:
            if volume == 0:
                state.zero_volume_count += 1
                if state.zero_volume_count >= self.config.zero_volume_max_ticks:
                    events.append(DataQualityEvent(
                        issue=QualityIssue.ZERO_VOLUME,
                        severity="WARNING",
                        symbol=symbol,
                        timestamp=now,
                        message=f"Zero volume for {state.zero_volume_count} consecutive ticks",
                        value=float(state.zero_volume_count),
                        threshold=float(self.config.zero_volume_max_ticks),
                    ))
            else:
                state.zero_volume_count = 0

            if state.volume_history and volume > 0:
                avg_vol = np.mean(list(state.volume_history))
                if avg_vol > 0 and volume > avg_vol * self.config.volume_spike_mult:
                    events.append(DataQualityEvent(
                        issue=QualityIssue.VOLUME_SPIKE,
                        severity="WARNING",
                        symbol=symbol,
                        timestamp=now,
                        message=f"Volume spike: {volume:.0f} ({volume / avg_vol:.1f}x average)",
                        value=volume / avg_vol,
                        threshold=self.config.volume_spike_mult,
                    ))

        # 5. Clock drift detection
        if exchange_ts is not None and receive_ts is not None:
            drift_ms = abs(receive_ts - exchange_ts) * 1000
            if drift_ms > self.config.max_clock_drift_ms:
                events.append(DataQualityEvent(
                    issue=QualityIssue.CLOCK_DRIFT,
                    severity="WARNING",
                    symbol=symbol,
                    timestamp=now,
                    message=f"Clock drift: {drift_ms:.0f}ms between exchange and receive",
                    value=drift_ms,
                    threshold=self.config.max_clock_drift_ms,
                ))

        # 6. Duplicate timestamp
        if state.last_tick_time is not None and now == state.last_tick_time:
            state.duplicate_ts_count += 1
            if state.duplicate_ts_count > 10:
                events.append(DataQualityEvent(
                    issue=QualityIssue.DUPLICATE_TIMESTAMP,
                    severity="WARNING",
                    symbol=symbol,
                    timestamp=now,
                    message=f"Duplicate timestamps: {state.duplicate_ts_count} consecutive",
                    value=float(state.duplicate_ts_count),
                ))
        else:
            state.duplicate_ts_count = 0

        # Update state
        state.last_tick_time = now
        if price > 0:
            state.price_history.append(price)
        if volume is not None and volume >= 0:
            state.volume_history.append(volume)
        state.tick_count += 1

        # Log and fire callbacks
        for event in events:
            self._events.append(event)
            self._issues_by_type[event.issue] += 1
            if event.severity == "CRITICAL":
                logger.warning(f"DATA QUALITY [{event.issue.value}]: {event.message}")
            else:
                logger.debug(f"DATA QUALITY [{event.issue.value}]: {event.message}")
            if self._alert_callback:
                self._alert_callback(event)

        return events

    def check_quote_staleness(self, symbol: str) -> Optional[DataQualityEvent]:
        """Check if quotes for a symbol are stale."""
        state = self._state.get(symbol)
        if state is None or state.last_tick_time is None:
            return None

        age = time.time() - state.last_tick_time
        if age > self.config.critical_quote_age_s:
            event = DataQualityEvent(
                issue=QualityIssue.STALE_QUOTE,
                severity="CRITICAL",
                symbol=symbol,
                timestamp=time.time(),
                message=f"Quote stale: {age:.1f}s old (threshold: {self.config.critical_quote_age_s}s)",
                value=age,
                threshold=self.config.critical_quote_age_s,
            )
            self._events.append(event)
            return event
        elif age > self.config.max_quote_age_s:
            event = DataQualityEvent(
                issue=QualityIssue.STALE_QUOTE,
                severity="WARNING",
                symbol=symbol,
                timestamp=time.time(),
                message=f"Quote aging: {age:.1f}s old (threshold: {self.config.max_quote_age_s}s)",
                value=age,
                threshold=self.config.max_quote_age_s,
            )
            self._events.append(event)
            return event
        return None

    def is_data_safe(self, symbol: str) -> bool:
        """Returns True if data quality is acceptable for trading."""
        state = self._state.get(symbol)
        if state is None:
            return False
        if state.last_tick_time is None:
            return False

        # Check staleness
        age = time.time() - state.last_tick_time
        if age > self.config.critical_quote_age_s:
            return False

        # Need minimum history
        if state.tick_count < 20:
            return False

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get quality summary across all symbols."""
        return {
            "symbols_tracked": len(self._state),
            "total_issues": len(self._events),
            "issues_by_type": {k.value: v for k, v in self._issues_by_type.items() if v > 0},
            "per_symbol": {
                sym: {
                    "tick_count": state.tick_count,
                    "last_tick_age_s": time.time() - state.last_tick_time if state.last_tick_time else None,
                    "zero_volume_streak": state.zero_volume_count,
                }
                for sym, state in self._state.items()
            },
        }

    def get_recent_events(self, n: int = 50) -> List[DataQualityEvent]:
        return list(self._events)[-n:]


class _SymbolState:
    """Per-symbol tracking state."""

    def __init__(self, window: int = 500):
        self.last_tick_time: Optional[float] = None
        self.price_history: Deque[float] = deque(maxlen=window)
        self.volume_history: Deque[float] = deque(maxlen=window)
        self.tick_count: int = 0
        self.zero_volume_count: int = 0
        self.duplicate_ts_count: int = 0
