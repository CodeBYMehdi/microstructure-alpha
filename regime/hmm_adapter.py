"""
Drop-in adapter: wraps GaussianHMM to match RegimeClustering interface.

Allows main.py to swap from HDBSCAN to HMM with a single import change.
All methods match RegimeClustering's signatures and return types.
"""

import numpy as np
from typing import Dict, Optional
from collections import deque
import logging

from .hmm import GaussianHMM
from .state_vector import StateVector
from core.types import RegimeOutput
from config.loader import get_config

logger = logging.getLogger(__name__)

# Use Rust scaler when available, else sklearn
from core.backend import _USE_RUST_CORE, get_rust_core as _get_rust_core
if _USE_RUST_CORE:
    _rc = _get_rust_core()

    class _OnlineScaler:
        """Wrapper around RustOnlineScaler matching StandardScaler interface used here."""
        def __init__(self):
            self._rust = _rc.RustOnlineScaler(n_features=6)
            self.scale_ = np.ones(6)

        def partial_fit(self, X):
            # RustOnlineScaler.partial_fit_transform does both, but we need
            # partial_fit alone sometimes. Just call it and discard result.
            self._rust.partial_fit_transform(X.flatten().astype(np.float64))
            self.scale_ = np.asarray(self._rust.get_scale())
            return self

        def transform(self, X):
            # After partial_fit, recompute transform manually using current stats
            obs = X.flatten().astype(np.float64)
            # We already called partial_fit, so stats are current
            # Use inverse logic: (obs - mean) / std
            # But RustOnlineScaler doesn't expose raw mean/std separately...
            # Instead, use partial_fit_transform which does both in one call
            # We need a transform-only path. Workaround: just do math in Python
            # Actually let's rethink: _scale_observation calls partial_fit then transform
            # We can combine into one call.
            raise NotImplementedError("Use partial_fit_transform directly")

        def partial_fit_transform(self, X):
            obs = X.flatten().astype(np.float64)
            result = np.asarray(self._rust.partial_fit_transform(obs))
            self.scale_ = np.asarray(self._rust.get_scale())
            return result.reshape(1, -1)

        def inverse_transform(self, X):
            obs = X.flatten().astype(np.float64)
            result = np.asarray(self._rust.inverse_transform(obs))
            return result.reshape(1, -1)

        @property
        def is_fitted(self):
            return self._rust.is_fitted

    logger.info("HMMRegimeAdapter scaler → Rust OnlineScaler")
else:
    from sklearn.preprocessing import StandardScaler


class HMMRegimeAdapter:
    """Drop-in replacement for RegimeClustering using HMM."""

    def __init__(self, config=None):
        config = config or get_config()
        hmm_cfg = config.thresholds.regime.hmm
        self._config = config

        self._hmm = GaussianHMM(
            n_states=hmm_cfg.n_states,
            n_features=6,
            learning_rate=hmm_cfg.learning_rate,
            prior_strength=hmm_cfg.prior_strength,
            emission_reg=hmm_cfg.emission_reg,
            sticky_prior_multiplier=hmm_cfg.sticky_prior_multiplier,
        )
        self._warmup_ticks = hmm_cfg.warmup_ticks
        self._min_confidence = hmm_cfg.min_confidence

        self._current_state: int = -1
        self._current_posterior: np.ndarray = np.ones(hmm_cfg.n_states) / hmm_cfg.n_states
        self._posterior_entropy: float = np.log(hmm_cfg.n_states)  # Max entropy initially
        self._history: deque = deque(maxlen=config.thresholds.regime.max_history)
        self._tick_count: int = 0
        self._last_state_vector: Optional[StateVector] = None

        # Feature scaling — critical for HMM to treat all 6 dims equally
        # Use partial_fit from tick 0 for consistent scaling (no warmup buffer drift)
        if _USE_RUST_CORE:
            self._scaler = _OnlineScaler()
        else:
            self._scaler = StandardScaler()
        self._scaler_fitted = False

        # For compatibility with code that accesses _regime_centroids
        self._regime_centroids: Dict[int, np.ndarray] = {}
        self._regime_stats: Dict[int, Dict] = {}
        # For compatibility with code that checks _last_labels
        self._last_labels: Optional[np.ndarray] = None

        logger.info(
            f"Init HMMRegimeAdapter: {hmm_cfg.n_states} states, "
            f"lr={hmm_cfg.learning_rate}, warmup={hmm_cfg.warmup_ticks}"
        )

    def update(self, state: StateVector) -> None:
        """Append state vector (matches RegimeClustering.update)."""
        self._history.append(state)
        self._last_state_vector = state
        self._tick_count += 1

    def _scale_observation(self, obs: np.ndarray) -> np.ndarray:
        """Scale observation via partial_fit from tick 0 for consistent scaling."""
        obs_2d = obs.reshape(1, -1)
        if _USE_RUST_CORE:
            # Combined partial_fit + transform in one Rust call
            result = self._scaler.partial_fit_transform(obs_2d)
            self._scaler_fitted = True
            return result[0]
        self._scaler.partial_fit(obs_2d)
        self._scaler_fitted = True
        return self._scaler.transform(obs_2d)[0]

    def _inverse_scale(self, scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform from scaled space back to original space."""
        if not self._scaler_fitted:
            return scaled
        if _USE_RUST_CORE:
            return np.asarray(self._scaler.inverse_transform(scaled.reshape(1, -1)))[0]
        return self._scaler.inverse_transform(scaled.reshape(1, -1))[0]

    def _inverse_scale_std(self, scaled_std: np.ndarray) -> np.ndarray:
        """Convert standard deviations from scaled space to original space."""
        if not self._scaler_fitted:
            return scaled_std
        return scaled_std * self._scaler.scale_

    def fit(self) -> np.ndarray:
        """
        Run HMM filter step + online update (matches RegimeClustering.fit).
        Returns array of labels for compatibility.
        """
        if self._last_state_vector is None:
            self._last_labels = np.array([-1])
            return self._last_labels

        obs_raw = np.array(self._last_state_vector.to_array())
        obs = self._scale_observation(obs_raw)

        # Filter step on scaled observation
        state, posterior = self._hmm.filter_step(obs)

        # Online parameter update
        self._hmm.online_update(obs)

        self._current_posterior = posterior
        max_prob = float(np.max(posterior))

        # Compute posterior entropy for uncertainty tracking
        p_safe = np.maximum(posterior, 1e-300)
        self._posterior_entropy = -float(np.sum(p_safe * np.log(p_safe)))

        # During warmup or low confidence: report as noise (-1)
        if not self._hmm.is_initialized or max_prob < self._min_confidence:
            self._current_state = -1
        else:
            self._current_state = state

        # Update regime stats from emission parameters (inverse-scaled)
        self._update_stats_from_hmm()

        self._last_labels = np.array([self._current_state])
        return self._last_labels

    def predict_latest(self) -> int:
        """Return current regime ID (matches RegimeClustering.predict_latest)."""
        return self._current_state

    def get_latest_regime_output(self, current_state: StateVector) -> RegimeOutput:
        """Build RegimeOutput from HMM posterior."""
        if not self._hmm.is_initialized or self._current_state == -1:
            return RegimeOutput(
                regime_id=str(self._current_state),
                confidence=0.0,
                cluster_density=0.0,
                persistence_estimate=0.0,
                is_noise=True,
            )

        confidence = float(self._current_posterior[self._current_state])

        # Use raw posterior confidence — no entropy discount.
        # The old discount (confidence * (1-0.5*uncertainty)) suppressed confidence
        # during regime transitions, which are exactly the best trading opportunities.
        adjusted_confidence = confidence

        # Cluster density from emission covariance
        try:
            cov = self._hmm.get_covariance(self._current_state)
            det = np.linalg.det(cov)
            cluster_density = 1.0 / (np.sqrt(abs(det)) + 1e-10)
        except np.linalg.LinAlgError:
            cluster_density = 0.0

        # Persistence = self-transition probability
        persistence = self._hmm.get_transition_prob(
            self._current_state, self._current_state
        )

        return RegimeOutput(
            regime_id=str(self._current_state),
            confidence=adjusted_confidence,
            cluster_density=cluster_density,
            persistence_estimate=persistence,
            is_noise=False,
        )

    def calculate_confidence(self, state: StateVector, regime_id: int) -> float:
        """Confidence = posterior probability of regime_id, adjusted by uncertainty."""
        if regime_id < 0 or regime_id >= self._hmm.n_states:
            return 0.0
        if not self._hmm.is_initialized:
            return 0.0
        raw_conf = float(self._current_posterior[regime_id])
        # Use raw posterior confidence (entropy discount removed — it was suppressing
        # confidence during transitions, blocking the best trading opportunities)
        return raw_conf

    def get_regime_stats(self, regime_id: int) -> Optional[Dict[str, StateVector]]:
        """Return centroid and std from HMM emission params (inverse-scaled)."""
        return self._regime_stats.get(regime_id)

    def get_cluster_quality(self) -> Dict[str, float]:
        """Quality metrics from HMM."""
        n_active = sum(
            1 for k in range(self._hmm.n_states)
            if self._current_posterior[k] > 0.05
        ) if self._hmm.is_initialized else 0

        return {
            "silhouette_score": 1.0 - self._posterior_entropy / np.log(max(self._hmm.n_states, 2)),
            "n_clusters": self._hmm.n_states,
            "noise_ratio": 0.0 if self._current_state >= 0 else 1.0,
            "silhouette_stability": 0.0,
            "n_samples": self._tick_count,
            "mean_silhouette": 0.0,
            "posterior_entropy": self._posterior_entropy,
            "n_active_states": n_active,
            "model": "HMM",
        }

    def get_transition_info(self) -> Dict:
        """Extra info for TransitionDetector integration."""
        # Compute regime transition volatility (fix #8)
        # High off-diagonal = unstable regime
        trans_volatility = 0.0
        if self._hmm.is_initialized and self._current_state >= 0:
            row = self._hmm.transition_matrix[self._current_state]
            trans_volatility = 1.0 - float(row[self._current_state])  # Sum of off-diagonal

        return {
            "transition_matrix": self._hmm.transition_matrix.copy(),
            "current_state": self._current_state,
            "posterior": self._current_posterior.copy(),
            "posterior_entropy": self._posterior_entropy,
            "n_observations": self._hmm.n_observations,
            "is_initialized": self._hmm.is_initialized,
            "transition_volatility": trans_volatility,
        }

    @property
    def n_regimes(self) -> int:
        return self._hmm.n_states

    def _update_stats_from_hmm(self) -> None:
        """Update regime_stats from HMM emission parameters, inverse-scaled to original space."""
        if not self._hmm.is_initialized:
            return

        self._regime_centroids.clear()
        self._regime_stats.clear()

        for k in range(self._hmm.n_states):
            mean_scaled = self._hmm.get_mean(k)
            cov_scaled = self._hmm.get_covariance(k)
            std_scaled = np.sqrt(np.maximum(np.diag(cov_scaled), 0.0))

            # Inverse-transform to original feature space for regime stats
            mean_orig = self._inverse_scale(mean_scaled)
            std_orig = self._inverse_scale_std(std_scaled)

            self._regime_centroids[k] = mean_orig

            centroid_sv = StateVector(*mean_orig)
            std_sv = StateVector(*np.abs(std_orig))

            self._regime_stats[k] = {
                'centroid': centroid_sv,
                'std': std_sv,
            }
