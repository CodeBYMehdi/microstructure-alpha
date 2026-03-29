"""Per-signal PnL attribution — tracks which signals actually make money.

Decomposes portfolio PnL into contributions from individual signal channels
(regime_transition, return_prediction, order_flow, momentum, mean_reversion,
orderbook). This is critical for knowing which signals to keep vs. kill.

Usage:
    tracker = AlphaAttribution(signal_names=["regime", "momentum", ...])
    tracker.record_signals(signals={"regime": 0.5, "momentum": -0.2}, weights={"regime": 0.4, ...})
    tracker.record_realized(return=0.001)
    
    summary = tracker.get_attribution_summary()
    # {'regime': {'total_pnl': 1.23, 'hit_rate': 0.55, ...}, ...}
"""

import numpy as np
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Single observation of a signal's prediction and outcome."""
    timestamp: float
    signal_value: float        # Raw signal value
    weight: float              # Weight of this signal in composite
    predicted_direction: float # +1 / -1 / 0
    realized_return: float     # What actually happened
    contribution: float        # weight * signal_value * sign(realized)
    is_correct: bool           # Did direction match?


@dataclass
class SignalSummary:
    """Aggregated performance metrics for a single signal."""
    name: str
    total_pnl_contribution: float = 0.0
    hit_rate: float = 0.0
    avg_signal_magnitude: float = 0.0
    avg_weight: float = 0.0
    information_coefficient: float = 0.0  # Rank correlation with realized returns
    avg_contribution_when_correct: float = 0.0
    avg_contribution_when_wrong: float = 0.0
    observations: int = 0
    is_profitable: bool = False


class AlphaAttribution:
    """Tracks per-signal PnL attribution in real-time."""

    def __init__(
        self,
        signal_names: Optional[List[str]] = None,
        lookback: int = 500,
        db=None,
    ):
        self.signal_names = signal_names or [
            "regime_transition",
            "return_prediction",
            "order_flow",
            "momentum",
            "mean_reversion",
            "orderbook",
        ]
        self.lookback = lookback
        self._db = db  # Optional TradeDatabase for persistence

        # Per-signal tracking
        self._signal_history: Dict[str, Deque[SignalRecord]] = {
            name: deque(maxlen=lookback) for name in self.signal_names
        }

        # Pending signals awaiting realized return
        self._pending_signals: Dict[str, float] = {}
        self._pending_weights: Dict[str, float] = {}
        self._pending_timestamp: float = 0.0
        self._n_observations: int = 0

        # Cumulative PnL per signal
        self._cumulative_pnl: Dict[str, float] = defaultdict(float)

        logger.info(f"AlphaAttribution initialized: {len(self.signal_names)} signals tracked")

    def record_signals(
        self,
        signals: Dict[str, float],
        weights: Dict[str, float],
        timestamp: float = 0.0,
    ) -> None:
        """Record raw signal values and weights before observing outcome."""
        self._pending_signals = dict(signals)
        self._pending_weights = dict(weights)
        self._pending_timestamp = timestamp

    def record_realized(self, realized_return: float) -> None:
        """After observing the market, attribute return to each signal."""
        if not self._pending_signals:
            return

        self._n_observations += 1

        for name in self.signal_names:
            signal_val = self._pending_signals.get(name, 0.0)
            weight = self._pending_weights.get(name, 0.0)

            # Direction prediction
            predicted_dir = np.sign(signal_val)

            # Contribution: how much PnL did this signal's prediction generate?
            # If signal correctly predicted direction, it contributed positively
            contribution = weight * abs(signal_val) * np.sign(signal_val * realized_return)

            is_correct = (
                np.sign(signal_val) == np.sign(realized_return)
                if abs(signal_val) > 1e-10 and abs(realized_return) > 1e-10
                else False
            )

            record = SignalRecord(
                timestamp=self._pending_timestamp,
                signal_value=signal_val,
                weight=weight,
                predicted_direction=predicted_dir,
                realized_return=realized_return,
                contribution=contribution,
                is_correct=is_correct,
            )
            self._signal_history[name].append(record)
            self._cumulative_pnl[name] += contribution

            # Persist to database if available
            if self._db:
                try:
                    self._db.record_attribution(
                        signal_name=name,
                        signal_value=signal_val,
                        predicted_direction=predicted_dir,
                        realized_return=realized_return,
                        contribution_pnl=contribution,
                        weight=weight,
                        is_correct=is_correct,
                    )
                except Exception as e:
                    logger.error("DB write failed, attribution lost: %s", e)

        self._pending_signals = {}
        self._pending_weights = {}

    def get_attribution_summary(self, recent_n: Optional[int] = None) -> Dict[str, SignalSummary]:
        """Get per-signal performance metrics."""
        summaries = {}
        n = recent_n or self.lookback

        for name in self.signal_names:
            records = list(self._signal_history[name])[-n:]
            if not records:
                summaries[name] = SignalSummary(name=name)
                continue

            contributions = [r.contribution for r in records]
            signals = [r.signal_value for r in records]
            weights = [r.weight for r in records]
            correct = [r.is_correct for r in records]
            realized = [r.realized_return for r in records]

            correct_contributions = [r.contribution for r in records if r.is_correct]
            wrong_contributions = [r.contribution for r in records if not r.is_correct]

            # Information Coefficient: rank correlation between signal and realized return
            ic = 0.0
            if len(signals) >= 10:
                try:
                    from scipy.stats import spearmanr
                    ic_val, _ = spearmanr(signals, realized)
                    ic = float(ic_val) if np.isfinite(ic_val) else 0.0
                except (ValueError, FloatingPointError) as e:
                    logger.error("IC computation failed in attribution: %s", e)
                    ic = 0.0

            summaries[name] = SignalSummary(
                name=name,
                total_pnl_contribution=self._cumulative_pnl.get(name, 0.0),
                hit_rate=float(np.mean(correct)) if correct else 0.0,
                avg_signal_magnitude=float(np.mean(np.abs(signals))) if signals else 0.0,
                avg_weight=float(np.mean(weights)) if weights else 0.0,
                information_coefficient=ic,
                avg_contribution_when_correct=float(np.mean(correct_contributions)) if correct_contributions else 0.0,
                avg_contribution_when_wrong=float(np.mean(wrong_contributions)) if wrong_contributions else 0.0,
                observations=len(records),
                is_profitable=self._cumulative_pnl.get(name, 0.0) > 0,
            )

        return summaries

    def get_signal_rankings(self) -> List[str]:
        """Rank signals by cumulative PnL contribution (best first)."""
        return sorted(self.signal_names, key=lambda s: self._cumulative_pnl.get(s, 0.0), reverse=True)

    def get_dead_signals(self, min_observations: int = 100) -> List[str]:
        """Identify signals that are not contributing to PnL."""
        dead = []
        for name in self.signal_names:
            records = list(self._signal_history[name])
            if len(records) < min_observations:
                continue
            summary = self.get_attribution_summary(recent_n=min_observations)
            sig_sum = summary.get(name)
            if sig_sum and sig_sum.hit_rate < 0.45 and sig_sum.total_pnl_contribution < 0:
                dead.append(name)
        return dead

    def get_metrics(self) -> Dict:
        """Get compact metrics dict for monitoring."""
        summary = self.get_attribution_summary()
        return {
            'n_observations': self._n_observations,
            'cumulative_pnl': dict(self._cumulative_pnl),
            'rankings': self.get_signal_rankings(),
            'per_signal': {
                name: {
                    'hit_rate': s.hit_rate,
                    'ic': s.information_coefficient,
                    'pnl': s.total_pnl_contribution,
                    'weight': s.avg_weight,
                    'profitable': s.is_profitable,
                }
                for name, s in summary.items()
            },
        }
