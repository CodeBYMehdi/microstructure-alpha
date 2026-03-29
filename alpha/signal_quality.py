"""Signal quality tracker — measures whether our signals actually predict returns.

Tracks:
- Information Coefficient (IC): rank correlation of predictions vs realized
- Hit rate: % of correct directional calls
- Per-signal IC for adaptive weight tuning
- Decay curve: how quickly signal predictiveness fades
"""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalSnapshot:
    """Single prediction record."""
    timestamp: float
    direction: float        # Predicted direction [-1, 1]
    strength: float         # Signal strength [0, 1]
    components: Dict[str, float] = field(default_factory=dict)  # Per-signal values
    realized_return: Optional[float] = None  # Filled in later


class SignalQualityTracker:
    """Tracks signal quality with rolling IC, hit rate, and per-signal metrics."""

    def __init__(self, lookback: int = 200, lag: int = 1):
        self._lookback = lookback
        self._lag = lag  # How many windows before we measure realized return
        self._pending: deque = deque(maxlen=lookback * 2)
        self._completed: deque = deque(maxlen=lookback)
        self._per_signal: Dict[str, deque] = {}  # signal_name -> deque of (pred, realized)

    def record_prediction(
        self,
        direction: float,
        strength: float,
        components: Optional[Dict[str, float]] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Record a new prediction (realized return filled in later)."""
        snap = SignalSnapshot(
            timestamp=timestamp,
            direction=direction,
            strength=strength,
            components=components or {},
        )
        self._pending.append(snap)

    def record_realized(self, actual_return: float) -> None:
        """Match oldest pending prediction with its realized return."""
        if not self._pending:
            return

        snap = self._pending.popleft()
        snap.realized_return = actual_return
        self._completed.append(snap)

        # Per-signal tracking
        for name, value in snap.components.items():
            if name not in self._per_signal:
                self._per_signal[name] = deque(maxlen=self._lookback)
            self._per_signal[name].append((value, actual_return))

    def get_ic(self) -> float:
        """Spearman rank correlation between predicted direction and realized return."""
        if len(self._completed) < 10:
            return 0.0

        predictions = [s.direction for s in self._completed]
        realized = [s.realized_return for s in self._completed]

        return _spearman_corr(predictions, realized)

    def get_hit_rate(self) -> float:
        """Fraction of correct directional calls."""
        if len(self._completed) < 5:
            return 0.5  # Neutral

        correct = 0
        total = 0
        for s in self._completed:
            if abs(s.direction) < 0.01:
                continue  # Skip neutral predictions
            total += 1
            if (s.direction > 0 and s.realized_return > 0) or \
               (s.direction < 0 and s.realized_return < 0):
                correct += 1

        return correct / total if total > 0 else 0.5

    def get_per_signal_ic(self) -> Dict[str, float]:
        """IC for each individual signal channel."""
        result = {}
        for name, pairs in self._per_signal.items():
            if len(pairs) < 10:
                result[name] = 0.0
                continue
            preds = [p[0] for p in pairs]
            reals = [p[1] for p in pairs]
            result[name] = _spearman_corr(preds, reals)
        return result

    def get_signal_strength_vs_return(self) -> Dict[str, float]:
        """Average |realized return| when signal was strong vs weak."""
        if len(self._completed) < 20:
            return {"strong_avg": 0.0, "weak_avg": 0.0}

        strengths = [s.strength for s in self._completed]
        median_str = np.median(strengths)

        strong_returns = [abs(s.realized_return) for s in self._completed if s.strength > median_str]
        weak_returns = [abs(s.realized_return) for s in self._completed if s.strength <= median_str]

        return {
            "strong_avg": float(np.mean(strong_returns)) if strong_returns else 0.0,
            "weak_avg": float(np.mean(weak_returns)) if weak_returns else 0.0,
        }

    @property
    def _n_realized(self) -> int:
        """Number of completed prediction-realization pairs."""
        return len(self._completed)

    def get_rolling_ic(self, window: int = 20) -> float:
        """IC over the most recent `window` observations."""
        if len(self._completed) < max(5, window // 2):
            return 0.0
        recent = list(self._completed)[-window:]
        predictions = [s.direction for s in recent]
        realized = [s.realized_return for s in recent]
        return _spearman_corr(predictions, realized)

    def get_metrics(self) -> dict:
        """Full quality report."""
        ic = self.get_ic()
        hit = self.get_hit_rate()
        per_signal = self.get_per_signal_ic()
        str_vs_ret = self.get_signal_strength_vs_return()

        return {
            "ic": round(ic, 4),
            "hit_rate": round(hit, 4),
            "n_completed": len(self._completed),
            "n_pending": len(self._pending),
            "per_signal_ic": {k: round(v, 4) for k, v in per_signal.items()},
            "strong_vs_weak": str_vs_ret,
            "edge_detected": ic > 0.03 and hit > 0.52,
        }


def _spearman_corr(x: list, y: list) -> float:
    """Spearman rank correlation using proper average-rank tie handling."""
    n = len(x)
    if n < 3:
        return 0.0

    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
    std_y = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5

    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0

    return cov / (std_x * std_y)
