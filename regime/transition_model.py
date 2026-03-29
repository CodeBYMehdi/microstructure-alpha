
import numpy as np
from typing import Optional, Dict
import logging
from dataclasses import dataclass
from core.types import TransitionProbability
from config.loader import get_config

logger = logging.getLogger(__name__)

@dataclass
class TransitionFeatures:
    delta_vector: np.ndarray
    strength: float
    kl_divergence: float
    projection_magnitude: float
    mu_velocity: float
    mu_acceleration: float
    entropy_velocity: float
    entropy_acceleration: float

class TransitionProbabilityModel:
    def __init__(self, alpha: float = 0.05, config=None):
        config = config or get_config()
        self._delta_prior = np.array(config.thresholds.regime.transition_weights, dtype=float)
        self._alpha = alpha
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_var: Optional[np.ndarray] = None
        self._signal_mean: float = 0.0
        self._signal_var: float = 1.0
        self._initialized = False
        logger.info("Init TransitionProbabilityModel (AdaptiveCalibratedLogit)")

    def predict(self, features: TransitionFeatures) -> TransitionProbability:
        x = self._build_features(features)
        if not self._initialized:
            signal = 0.0
            prob = 0.5
            self._update_stats(x, signal)
            return TransitionProbability(
                probability=prob,
                is_significant=False,
                model_used="AdaptiveCalibratedLogit"
            )

        std = np.sqrt(self._feature_var + 1e-8)
        z = (x - self._feature_mean) / std
        weights = self._compute_weights(std)
        signal = float(np.dot(weights, z))
        signal_std = np.sqrt(self._signal_var + 1e-8)
        logit = (signal - self._signal_mean) / signal_std
        prob = 1.0 / (1.0 + np.exp(-logit))
        self._update_stats(x, signal)

        return TransitionProbability(
            probability=prob,
            is_significant=prob > 0.5,
            model_used="AdaptiveCalibratedLogit"
        )

    def _build_features(self, features: TransitionFeatures) -> np.ndarray:
        return np.array([
            np.abs(features.delta_vector[0]),
            np.abs(features.delta_vector[1]),
            np.abs(features.delta_vector[2]),
            np.abs(features.delta_vector[3]),
            np.abs(features.delta_vector[4]),
            np.abs(features.delta_vector[5]),
            features.strength,
            max(0.0, features.kl_divergence),
            max(0.0, features.projection_magnitude),
            np.abs(features.mu_velocity),
            np.abs(features.mu_acceleration),
            np.abs(features.entropy_velocity),
            np.abs(features.entropy_acceleration)
        ], dtype=float)

    def _compute_weights(self, std: np.ndarray) -> np.ndarray:
        # _build_features returns 13 elements: 6 deltas + 7 derived features
        # _delta_prior has 6 weights for the delta components
        # Fill the remaining 7 slots with the mean prior weight
        n_extra = len(std) - len(self._delta_prior)
        prior_tail = np.full(max(0, n_extra), float(np.mean(self._delta_prior)))
        prior = np.concatenate([self._delta_prior, prior_tail])
        weights = prior / (std + 1e-6)
        denom = np.sum(np.abs(weights))
        if denom == 0.0:
            return np.ones_like(weights) / len(weights)
        return weights / denom

    def _update_stats(self, features: np.ndarray, signal: float):
        if not self._initialized:
            self._feature_mean = features.copy()
            self._feature_var = np.ones_like(features)
            self._signal_mean = signal
            self._signal_var = 1.0
            self._initialized = True
            return
        delta = features - self._feature_mean
        self._feature_mean = self._feature_mean + self._alpha * delta
        self._feature_var = (1.0 - self._alpha) * self._feature_var + self._alpha * (delta ** 2)
        signal_delta = signal - self._signal_mean
        self._signal_mean = self._signal_mean + self._alpha * signal_delta
        self._signal_var = (1.0 - self._alpha) * self._signal_var + self._alpha * (signal_delta ** 2)
