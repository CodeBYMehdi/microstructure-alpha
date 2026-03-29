"""Execution quality analytics — implementation shortfall, fill analysis.

Tracks and analyzes execution quality over time to identify:
- Slippage patterns (by time of day, regime, order type)
- Implementation shortfall vs. arrival price  
- Fill rate and rejection reasons
- Market impact estimation from actual fills
- Latency distribution

Usage:
    analytics = ExecutionAnalytics()
    analytics.record_fill(proposal, result, arrival_price=100.0, decision_ts=...)
    report = analytics.generate_report()
"""

import numpy as np
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """Record of a single fill for analysis."""
    timestamp: float
    symbol: str
    side: str                  # BUY / SELL
    requested_qty: float
    filled_qty: float
    arrival_price: float       # Price at decision time
    filled_price: float        # Actual fill price
    fees: float
    slippage_bps: float        # (fill - arrival) / arrival * 10000
    impl_shortfall_bps: float  # Signed IS in bps
    decision_to_fill_ms: float # Latency from decision to fill
    regime_id: str = ""
    order_type: str = ""       # LMT / MKT / TWAP
    hour_of_day: int = 0


class ExecutionAnalytics:
    """Tracks and analyzes execution quality over time."""

    def __init__(self, max_history: int = 5000):
        self._fills: Deque[FillRecord] = deque(maxlen=max_history)
        self._rejections: Deque[Dict] = deque(maxlen=1000)
        self._total_fills = 0
        self._total_rejections = 0

        logger.info("ExecutionAnalytics initialized")

    def record_fill(
        self,
        symbol: str,
        side: str,
        requested_qty: float,
        filled_qty: float,
        arrival_price: float,
        filled_price: float,
        fees: float = 0.0,
        decision_ts: float = 0.0,
        fill_ts: float = 0.0,
        regime_id: str = "",
        order_type: str = "",
    ) -> None:
        """Record a fill for analysis."""
        # Slippage in bps
        if arrival_price > 0:
            raw_slip = (filled_price - arrival_price) / arrival_price * 10000
            if side == "SELL":
                raw_slip = -raw_slip  # Normalize: positive = bad for both sides
            slippage_bps = abs(raw_slip)

            # Implementation shortfall (signed)
            if side == "BUY":
                impl_shortfall = (filled_price - arrival_price) / arrival_price * 10000
            else:
                impl_shortfall = (arrival_price - filled_price) / arrival_price * 10000
        else:
            slippage_bps = 0.0
            impl_shortfall = 0.0

        # Latency
        latency_ms = (fill_ts - decision_ts) * 1000 if fill_ts > decision_ts > 0 else 0.0

        # Hour of day
        import datetime
        hour = datetime.datetime.fromtimestamp(fill_ts).hour if fill_ts > 0 else 0

        record = FillRecord(
            timestamp=fill_ts,
            symbol=symbol,
            side=side,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            arrival_price=arrival_price,
            filled_price=filled_price,
            fees=fees,
            slippage_bps=slippage_bps,
            impl_shortfall_bps=impl_shortfall,
            decision_to_fill_ms=latency_ms,
            regime_id=regime_id,
            order_type=order_type,
            hour_of_day=hour,
        )
        self._fills.append(record)
        self._total_fills += 1

    def record_rejection(
        self,
        symbol: str,
        side: str,
        qty: float,
        reason: str,
        timestamp: float = 0.0,
    ) -> None:
        """Record an order rejection."""
        self._rejections.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'reason': reason,
        })
        self._total_rejections += 1

    def generate_report(self, recent_n: Optional[int] = None) -> Dict:
        """Generate comprehensive execution quality report."""
        fills = list(self._fills)
        if recent_n:
            fills = fills[-recent_n:]

        if not fills:
            return {
                'total_fills': 0,
                'total_rejections': self._total_rejections,
                'fill_rate': 0.0,
            }

        slippages = [f.slippage_bps for f in fills]
        impl_shortfalls = [f.impl_shortfall_bps for f in fills]
        latencies = [f.decision_to_fill_ms for f in fills if f.decision_to_fill_ms > 0]
        fees = [f.fees for f in fills]

        fill_rate = self._total_fills / max(self._total_fills + self._total_rejections, 1)

        report = {
            'total_fills': self._total_fills,
            'total_rejections': self._total_rejections,
            'fill_rate': fill_rate,

            # Slippage
            'avg_slippage_bps': float(np.mean(slippages)),
            'median_slippage_bps': float(np.median(slippages)),
            'p95_slippage_bps': float(np.percentile(slippages, 95)),
            'max_slippage_bps': float(np.max(slippages)),

            # Implementation shortfall
            'avg_impl_shortfall_bps': float(np.mean(impl_shortfalls)),
            'median_impl_shortfall_bps': float(np.median(impl_shortfalls)),

            # Latency
            'avg_latency_ms': float(np.mean(latencies)) if latencies else 0.0,
            'median_latency_ms': float(np.median(latencies)) if latencies else 0.0,
            'p95_latency_ms': float(np.percentile(latencies, 95)) if latencies else 0.0,

            # Fees
            'total_fees': float(np.sum(fees)),
            'avg_fees_per_fill': float(np.mean(fees)),

            # By side
            'by_side': self._analyze_by_group(fills, 'side'),

            # By regime
            'by_regime': self._analyze_by_group(fills, 'regime_id'),

            # By order type
            'by_order_type': self._analyze_by_group(fills, 'order_type'),

            # By hour
            'by_hour': self._analyze_by_group(fills, 'hour_of_day'),

            # Rejection reasons
            'rejection_reasons': self._summarize_rejections(),
        }

        return report

    def _analyze_by_group(self, fills: List[FillRecord], group_field: str) -> Dict:
        groups = defaultdict(list)
        for f in fills:
            key = str(getattr(f, group_field, 'unknown'))
            groups[key].append(f)

        result = {}
        for key, group_fills in groups.items():
            if not group_fills:
                continue
            slippages = [f.slippage_bps for f in group_fills]
            result[key] = {
                'count': len(group_fills),
                'avg_slippage_bps': float(np.mean(slippages)),
                'avg_impl_shortfall_bps': float(np.mean([f.impl_shortfall_bps for f in group_fills])),
            }

        return result

    def _summarize_rejections(self) -> Dict[str, int]:
        reasons = defaultdict(int)
        for r in self._rejections:
            reasons[r.get('reason', 'unknown')] += 1
        return dict(reasons)

    def get_cost_curve(self, cost_bps_range: Optional[List[float]] = None) -> Dict[str, float]:
        """Estimate PnL at different cost assumptions.
        
        Useful for cost sensitivity analysis:
        "Would we still be profitable at 3bps? 5bps?"
        """
        fills = list(self._fills)
        if not fills:
            return {}

        if cost_bps_range is None:
            cost_bps_range = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

        # For each cost level, compute how much total fill cost would change
        total_notional = sum(f.filled_qty * f.filled_price for f in fills)
        actual_cost = sum(f.fees for f in fills) + sum(
            f.slippage_bps / 10000 * f.filled_qty * f.arrival_price for f in fills
        )

        result = {}
        for bps in cost_bps_range:
            hypothetical_cost = total_notional * bps / 10000
            result[f"{bps}bps"] = hypothetical_cost

        result['actual_cost'] = actual_cost
        result['total_notional'] = total_notional

        return result
