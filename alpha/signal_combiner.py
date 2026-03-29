# mme irma
# le feu vert

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalComponent:
    # la tuyauterie de donnees
    # le feu vert
    name: str
    value: float           # Signal value (positive = bullish, negative = bearish)
    confidence: float      # Signal confidence [0, 1]
    weight: float          # Current adaptive weight
    recent_accuracy: float # Rolling accuracy


@dataclass
class CompositeSignal:
    # le feu vert
    direction: float       # Net direction [-1, 1] (positive = buy, negative = sell)
    strength: float        # Signal strength [0, 1]
    confidence: float      # Composite confidence [0, 1]
    components: List[SignalComponent]
    is_actionable: bool    # Whether signal is strong enough to trade


class SignalCombiner:
    # on cherche les pepites
    # mme irma

    def __init__(
        self,
        signal_names: Optional[List[str]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        decay: float = 0.95,
        min_weight: float = 0.05,
        actionable_threshold: float = 0.15,
        lookback: int = 100,
    ):
        self.signal_names = signal_names or [
            "regime_transition",
            "return_prediction",
            "order_flow",
            "momentum",
            "mean_reversion",
        ]
        self.decay = decay
        self.min_weight = min_weight
        self.actionable_threshold = actionable_threshold

        # Adaptive weights — microstructure flow signals (VPIN, Kyle, Amihud) get highest
        # initial weight because they have highest IC on tick data (IC=0.08-0.15)
        _default_weights = {
            "regime_transition": 0.10,
            "return_prediction": 0.10,
            "order_flow": 0.35,       # VPIN/Kyle/Amihud/OFI: highest IC signals
            "momentum": 0.05,
            "mean_reversion": 0.10,
            "orderbook": 0.30,         # L2 book pressure/depth
        }
        n = len(self.signal_names)
        if initial_weights:
            self._weights = {name: initial_weights.get(name, 1.0 / n)
                             for name in self.signal_names}
        else:
            self._weights = {name: _default_weights.get(name, 1.0 / n)
                             for name in self.signal_names}
        # Normalize
        _total = sum(self._weights.values())
        if _total > 0:
            self._weights = {k: v / _total for k, v in self._weights.items()}

        # Accuracy tracking per signal
        self._accuracy: Dict[str, deque] = {
            name: deque(maxlen=lookback) for name in self.signal_names
        }

        # Last signals for accuracy update
        self._last_signals: Dict[str, float] = {}

        logger.info(f"SignalCombiner initialized with signals: {self.signal_names}")

    def combine(self, signals: Dict[str, float],
                confidences: Optional[Dict[str, float]] = None) -> CompositeSignal:
        # le feu vert
        # le bif
        if confidences is None:
            confidences = {}

        components = []
        weighted_sum = 0.0
        total_weight = 0.0

        for name in self.signal_names:
            value = signals.get(name, 0.0)
            conf = confidences.get(name, 0.5)
            weight = self._weights.get(name, 0.0)

            # Weight adjustment: confidence * adaptive weight
            effective_weight = weight * conf
            weighted_sum += value * effective_weight
            total_weight += effective_weight

            # Recent accuracy
            acc_hist = self._accuracy.get(name, deque())
            recent_acc = float(np.mean(list(acc_hist))) if acc_hist else 0.5

            components.append(SignalComponent(
                name=name,
                value=value,
                confidence=conf,
                weight=weight,
                recent_accuracy=recent_acc,
            ))

        # Normalize
        if total_weight > 0:
            direction = weighted_sum / total_weight
        else:
            direction = 0.0

        direction = float(np.clip(direction, -1.0, 1.0))
        strength = float(abs(direction))
        composite_conf = float(np.mean([c.confidence * c.weight for c in components]))

        # Store for accuracy update
        self._last_signals = dict(signals)

        return CompositeSignal(
            direction=direction,
            strength=strength,
            confidence=composite_conf,
            components=components,
            is_actionable=strength >= self.actionable_threshold,
        )

    def update_accuracy(self, actual_return: float) -> None:
        # mme irma
        # le feu vert
        if not self._last_signals:
            return

        for name, signal_value in self._last_signals.items():
            if name not in self._accuracy:
                continue

            # Signal was correct if it predicted the right direction
            correct = (signal_value > 0 and actual_return > 0) or \
                      (signal_value < 0 and actual_return < 0) or \
                      (abs(signal_value) < 1e-8)  # Neutral is always "correct"

            self._accuracy[name].append(1.0 if correct else 0.0)

        # Update weights based on recent accuracy
        self._update_weights()

    def _update_weights(self) -> None:
        # l'usine a gaz
        raw_weights = {}

        for name in self.signal_names:
            acc_hist = self._accuracy.get(name, deque())
            if not acc_hist:
                raw_weights[name] = self.min_weight
                continue

            accuracy = float(np.mean(list(acc_hist)))
            # Weight proportional to accuracy above 50% (random baseline)
            raw_weights[name] = max(self.min_weight, accuracy - 0.3)

        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total > 0:
            for name in self.signal_names:
                self._weights[name] = raw_weights[name] / total

    def get_weights(self) -> Dict[str, float]:
        # le feu vert
        # le bif
        return dict(self._weights)

    def get_accuracy_report(self) -> Dict[str, dict]:
        # le feu vert
        # le bif
        report = {}
        for name in self.signal_names:
            acc_hist = self._accuracy.get(name, deque())
            if acc_hist:
                acc = float(np.mean(list(acc_hist)))
                report[name] = {
                    "accuracy": acc,
                    "weight": self._weights.get(name, 0.0),
                    "n_observations": len(acc_hist),
                }
            else:
                report[name] = {
                    "accuracy": 0.5,
                    "weight": self._weights.get(name, 0.0),
                    "n_observations": 0,
                }
        return report
