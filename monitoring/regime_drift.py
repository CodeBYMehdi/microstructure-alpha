# l'usine a gaz

import numpy as np
from typing import Dict, Optional
from collections import deque
from regime.state_vector import StateVector
from regime.labels import RegimeLabelManager
from config.loader import get_config


class RegimeDriftMonitor:
    """Monitors regime drift using normalized Mahalanobis-like distance.

    Features in the state vector (mu, sigma, skew, kurtosis, tail, entropy)
    have different units and scales. Raw Euclidean distance is meaningless.
    We normalize each dimension by its observed range to make drift detection
    scale-invariant.
    """

    # Per-dimension normalization scale factors (empirically calibrated)
    # These represent typical ranges for each state vector dimension
    _SCALE_FACTORS = np.array([
        0.001,   # mu (return drift, typically ~1e-4 to 1e-3)
        0.005,   # sigma (volatility, typically ~1e-3 to 1e-2)
        2.0,     # skew (typically -3 to +3)
        5.0,     # kurtosis (typically 2 to 10+)
        3.0,     # tail_slope (typically 1 to 5)
        2.0,     # entropy (typically 1 to 5)
    ])

    def __init__(self, label_manager: RegimeLabelManager, window_size: int = None, config=None):
        cfg = config or get_config()
        self.label_manager = label_manager
        self.window_size = window_size or cfg.thresholds.drift.drift_window
        self.threshold = cfg.thresholds.drift.structural_break_threshold
        self._drift_history: Dict[int, deque] = {}
        self._baseline_centroids: Dict[int, np.ndarray] = {}
        # Adaptive scale: track running std per regime per dimension
        self._running_stats: Dict[int, Dict] = {}

    def record_observation(self, regime_id: int, current_state: StateVector) -> None:
        if regime_id == -1:
            return

        profile = self.label_manager.get_profile(regime_id)
        if not profile:
            return

        # Establish baseline if new
        if regime_id not in self._baseline_centroids:
            self._baseline_centroids[regime_id] = profile.centroid
            self._drift_history[regime_id] = deque(maxlen=self.window_size)
            self._running_stats[regime_id] = {'sum': np.zeros(6), 'sum_sq': np.zeros(6), 'n': 0}

        # Compute normalized distance
        current_vec = np.array(current_state.to_array())
        baseline = self._baseline_centroids[regime_id]
        delta = current_vec - baseline

        # Update running stats for adaptive normalization
        stats = self._running_stats[regime_id]
        stats['sum'] += current_vec
        stats['sum_sq'] += current_vec ** 2
        stats['n'] += 1

        # Use adaptive scale when we have enough samples, else use defaults
        if stats['n'] >= 20:
            mean = stats['sum'] / stats['n']
            var = stats['sum_sq'] / stats['n'] - mean ** 2
            adaptive_scale = np.sqrt(np.maximum(var, 1e-12))
            # Blend with default scales to prevent degenerate normalization
            scale = np.maximum(adaptive_scale, self._SCALE_FACTORS * 0.1)
        else:
            scale = self._SCALE_FACTORS

        # Normalized Euclidean distance (each dimension contributes equally)
        normalized_delta = delta / scale
        dist = float(np.sqrt(np.sum(normalized_delta ** 2)))
        self._drift_history[regime_id].append(dist)

    def check_drift(self, regime_id: int) -> float:
        # le bif
        if regime_id not in self._drift_history:
            return 0.0
        history = self._drift_history[regime_id]
        if not history:
            return 0.0
        return float(np.mean(history))

    def detect_structural_break(self, regime_id: int, threshold: float = None) -> bool:
        # le bif
        thresh = threshold if threshold is not None else self.threshold
        return self.check_drift(regime_id) > thresh
