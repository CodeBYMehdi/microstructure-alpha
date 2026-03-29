# on cherche les pepites
# la grosse machine

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecayProfile:
    # le feu vert
    half_life_windows: float    # Windows until edge halves
    initial_edge: float         # Expected edge at t=0
    decay_rate: float           # Exponential decay constant (lambda)
    n_observations: int         # Number of transitions observed
    confidence: float           # Confidence in the estimate [0, 1]


class AlphaDecayModel:
    # on cherche les pepites
    # la grosse machine

    def __init__(self, max_lookback: int = 50, min_observations: int = 5):
        self.max_lookback = max_lookback
        self.min_observations = min_observations

        # Storage: {(from_regime, to_regime): [[cumret at t=0, t=1, ..., t=max_lookback], ...]}
        self._transition_returns: Dict[tuple, List[List[float]]] = defaultdict(list)

        # Current tracking state
        self._active_transitions: List[dict] = []

        # Cache of fitted profiles
        self._profiles: Dict[tuple, DecayProfile] = {}

        logger.info(f"AlphaDecayModel initialized: max_lookback={max_lookback}")

    def on_transition(self, from_regime: int, to_regime: int, entry_price: float) -> None:
        # l'usine a gaz
        self._active_transitions.append({
            "from": from_regime,
            "to": to_regime,
            "entry_price": entry_price,
            "returns": [],
            "windows": 0,
        })

    def on_window(self, current_price: float) -> None:
        # l'usine a gaz
        completed = []

        for i, t in enumerate(self._active_transitions):
            if t["entry_price"] <= 0:
                completed.append(i)
                continue

            ret = (current_price - t["entry_price"]) / t["entry_price"]
            t["returns"].append(ret)
            t["windows"] += 1

            if t["windows"] >= self.max_lookback:
                # Store and close
                key = (t["from"], t["to"])
                self._transition_returns[key].append(list(t["returns"]))
                completed.append(i)

        # Remove completed (reverse order to maintain indices)
        for i in sorted(completed, reverse=True):
            self._active_transitions.pop(i)

    def get_decay_profile(self, from_regime: int, to_regime: int) -> Optional[DecayProfile]:
        # l'usine a gaz
        key = (from_regime, to_regime)

        # Return cached si dispo
        if key in self._profiles:
            return self._profiles[key]

        # Fit if enough data
        returns_list = self._transition_returns.get(key, [])
        if len(returns_list) < self.min_observations:
            return None

        profile = self._fit_decay(returns_list)
        if profile:
            self._profiles[key] = profile
        return profile

    def get_remaining_edge(self, from_regime: int, to_regime: int,
                           windows_elapsed: int) -> float:
        # le bif
        profile = self.get_decay_profile(from_regime, to_regime)
        if profile is None:
            # Default: assume 20-window half-life
            return float(np.exp(-0.693 / 20.0 * windows_elapsed))

        return float(np.exp(-profile.decay_rate * windows_elapsed))

    def _fit_decay(self, returns_list: List[List[float]]) -> Optional[DecayProfile]:
        # le bif
        # Compute average cumulative return curve
        max_len = min(len(r) for r in returns_list)
        if max_len < 5:
            return None

        avg_curve = np.zeros(max_len)
        for returns in returns_list:
            avg_curve += np.abs(np.array(returns[:max_len]))
        avg_curve /= len(returns_list)

        # Find peak edge (maximum absolute return)
        peak_idx = np.argmax(avg_curve)
        initial_edge = float(avg_curve[peak_idx]) if peak_idx < len(avg_curve) else float(avg_curve[0])

        if initial_edge <= 0:
            return DecayProfile(
                half_life_windows=self.max_lookback,
                initial_edge=0.0,
                decay_rate=0.0,
                n_observations=len(returns_list),
                confidence=0.0,
            )

        # Fit decay from peak onwards
        decay_curve = avg_curve[peak_idx:]
        if len(decay_curve) < 3:
            return None

        # Log-linear fit: log(y) = log(a) - lambda * t
        decay_curve = np.maximum(decay_curve, 1e-10)  # Avoid log(0)
        log_y = np.log(decay_curve)
        t = np.arange(len(decay_curve), dtype=float)

        # Simple linear regression
        n = len(t)
        sum_t = np.sum(t)
        sum_y = np.sum(log_y)
        sum_ty = np.sum(t * log_y)
        sum_t2 = np.sum(t ** 2)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-10:
            return None

        slope = (n * sum_ty - sum_t * sum_y) / denom

        decay_rate = max(0.001, -slope)  # Ensure positive decay
        half_life = 0.693 / decay_rate  # ln(2) / lambda

        # R-squared for confidence
        predicted = log_y[0] + slope * t
        ss_res = np.sum((log_y - predicted) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = max(0.0, 1 - ss_res / max(ss_tot, 1e-10))

        return DecayProfile(
            half_life_windows=float(half_life),
            initial_edge=initial_edge,
            decay_rate=float(decay_rate),
            n_observations=len(returns_list),
            confidence=float(r2),
        )

    def get_summary(self) -> Dict[str, dict]:
        # l'usine a gaz
        summary = {}
        for key, returns_list in self._transition_returns.items():
            profile = self.get_decay_profile(key[0], key[1])
            summary[f"{key[0]}->{key[1]}"] = {
                "observations": len(returns_list),
                "half_life": profile.half_life_windows if profile else None,
                "initial_edge": profile.initial_edge if profile else None,
                "confidence": profile.confidence if profile else None,
            }
        return summary
