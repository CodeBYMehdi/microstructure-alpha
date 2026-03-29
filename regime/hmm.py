"""
Gaussian Hidden Markov Model for online regime detection.

Replaces HDBSCAN clustering with a generative model that jointly learns:
- Transition probabilities between regimes (transition matrix A)
- Emission distributions per regime (multivariate Gaussian per state)
- Online parameter updates via incremental Baum-Welch

The 6-dim state vector (mu, sigma, skew, kurtosis, tail_slope, entropy)
is the observation/emission. Hidden states are the regimes.
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GaussianHMM:
    """Online Gaussian HMM for regime detection."""

    def __init__(
        self,
        n_states: int = 4,
        n_features: int = 6,
        learning_rate: float = 0.01,
        prior_strength: float = 1.0,
        emission_reg: float = 1e-4,
        seed: int = 42,
        sticky_prior_multiplier: float = 4.0,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.emission_reg = emission_reg
        self._rng = np.random.RandomState(seed)
        self._sticky_prior_mult = sticky_prior_multiplier

        # Uniform start probability
        self.start_prob = np.ones(n_states) / n_states

        # Transition matrix: sticky prior (regimes persist)
        self.transition_matrix = np.full((n_states, n_states), prior_strength)
        for i in range(n_states):
            self.transition_matrix[i, i] += prior_strength * self._sticky_prior_mult
        self._normalize_rows(self.transition_matrix)

        # Emission parameters: mean and covariance per state
        self.means = np.zeros((n_states, n_features))
        self.covariances = np.array([np.eye(n_features) for _ in range(n_states)])

        # Break symmetry with small random perturbation
        for k in range(n_states):
            self.means[k] = self._rng.randn(n_features) * 0.01

        # Forward algorithm state
        self._alpha: Optional[np.ndarray] = None
        self._prev_alpha: Optional[np.ndarray] = None
        self._prev_state: int = -1
        self._n_updates: int = 0

        # Adaptive learning rate state (fix #3)
        self._innovation_ema: float = 0.0
        self._innovation_alpha: float = 0.05  # EMA decay for innovation tracking

        # Entropy regularization threshold (fix #2)
        # Max KL-to-uniform before we regularize a transition row
        # ln(n_states) is the theoretical max; use 80% of that as threshold
        self._max_row_kl: float = 0.8 * np.log(max(n_states, 2))
        self._entropy_reg_strength: float = 0.03  # Was 0.1 — lighter touch preserves sticky regimes

        # Running observation buffer for initialization
        self._obs_buffer = []
        self._initialized = False
        self._warmup_size = max(50, n_states * 10)

    def filter_step(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Single forward-filter step. Returns (most_likely_state, posterior).
        O(K^2) where K = n_states.
        """
        obs = np.asarray(observation, dtype=np.float64)

        # NaN/Inf guard
        if not np.all(np.isfinite(obs)):
            logger.debug("HMM filter_step: non-finite observation, returning previous state")
            if self._alpha is not None:
                return self._prev_state if self._prev_state >= 0 else 0, self._alpha.copy()
            return 0, np.ones(self.n_states) / self.n_states

        if not self._initialized:
            self._obs_buffer.append(obs.copy())
            if len(self._obs_buffer) >= self._warmup_size:
                self._initialize_from_data()
            else:
                posterior = np.ones(self.n_states) / self.n_states
                return 0, posterior

        # Emission probabilities: P(obs | state=k)
        emission_probs = self._compute_emissions(obs)

        # Store previous alpha for transition matrix update
        self._prev_alpha = self._alpha.copy() if self._alpha is not None else None

        if self._alpha is None:
            self._alpha = self.start_prob * emission_probs
        else:
            # Forward step: alpha_t = emission * (A^T @ alpha_{t-1})
            predicted = self.transition_matrix.T @ self._alpha
            self._alpha = emission_probs * predicted

        # Normalize
        alpha_sum = np.sum(self._alpha)
        if alpha_sum < 1e-300:
            self._alpha = np.ones(self.n_states) / self.n_states
            logger.debug("HMM forward underflow — reset to uniform")
        else:
            self._alpha = self._alpha / alpha_sum

        # Adaptive learning rate: track innovation (fix #3)
        # Innovation = negative log-likelihood of observation under current model
        total_emission = float(np.dot(emission_probs, self._alpha))
        innovation = -np.log(max(total_emission, 1e-300))
        self._innovation_ema = (
            (1 - self._innovation_alpha) * self._innovation_ema
            + self._innovation_alpha * innovation
        )
        # Scale lr: base * clamp(innovation_ratio, 1, 2)
        # Capped at 2x (was 5x — aggressive rewrites destabilized transition matrix)
        if self._innovation_ema > 1e-10:
            innovation_ratio = innovation / self._innovation_ema
            lr_scale = float(np.clip(innovation_ratio, 1.0, 2.0))
        else:
            lr_scale = 1.0
        self.learning_rate = self.base_learning_rate * lr_scale

        most_likely = int(np.argmax(self._alpha))
        self._prev_state = most_likely
        self._n_updates += 1

        return most_likely, self._alpha.copy()

    def online_update(self, observation: np.ndarray) -> None:
        """
        Incremental Baum-Welch using current and previous posteriors.
        Includes entropy regularization to prevent degenerate collapse.
        """
        if not self._initialized or self._alpha is None:
            return

        obs = np.asarray(observation, dtype=np.float64)

        # NaN/Inf guard
        if not np.all(np.isfinite(obs)):
            return

        gamma = self._alpha
        lr = self.learning_rate  # Uses adaptive lr from filter_step

        for k in range(self.n_states):
            g_k = gamma[k]
            if g_k < 1e-10:
                continue

            # Update mean
            delta = obs - self.means[k]
            self.means[k] += lr * g_k * delta

            # Update covariance (without accumulating regularization)
            outer = np.outer(delta, delta)
            self.covariances[k] = (1 - lr * g_k) * self.covariances[k] + lr * g_k * outer

            # Floor diagonal to prevent singularity (no accumulation)
            diag = np.diag(self.covariances[k])
            floor_mask = diag < self.emission_reg
            if np.any(floor_mask):
                idx = np.where(floor_mask)[0]
                for i in idx:
                    self.covariances[k][i, i] = self.emission_reg

        # Update transition matrix using prev_alpha and current posterior
        if self._prev_alpha is not None:
            emission_probs = self._compute_emissions(obs)
            # xi(i,j) ~ prev_alpha(i) * A(i,j) * b_j(obs) — correct Baum-Welch
            xi = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[i, j] = self._prev_alpha[i] * self.transition_matrix[i, j] * emission_probs[j]

            xi_sum = xi.sum()
            if xi_sum > 1e-300:
                xi /= xi_sum

            self.transition_matrix = (1 - lr) * self.transition_matrix + lr * xi
            self._normalize_rows(self.transition_matrix)

        # Entropy regularization on transition matrix (fix #2)
        # Prevent any row from becoming too peaked (degenerate state collapse)
        self._regularize_transition_matrix()

    def _regularize_transition_matrix(self) -> None:
        """
        Prevent degenerate state collapse by regularizing transition rows
        that are too peaked (KL divergence to uniform exceeds threshold).
        """
        uniform = np.ones(self.n_states) / self.n_states
        for i in range(self.n_states):
            row = self.transition_matrix[i]
            # KL(row || uniform) = sum(row * log(row / uniform))
            row_safe = np.maximum(row, 1e-300)
            kl = float(np.sum(row_safe * np.log(row_safe * self.n_states)))
            if kl > self._max_row_kl:
                # Blend toward uniform to reduce peakedness
                self.transition_matrix[i] = (
                    (1 - self._entropy_reg_strength) * row
                    + self._entropy_reg_strength * uniform
                )
        self._normalize_rows(self.transition_matrix)

    def get_transition_prob(self, from_state: int, to_state: int) -> float:
        """Get transition probability A[from_state, to_state]."""
        if 0 <= from_state < self.n_states and 0 <= to_state < self.n_states:
            return float(self.transition_matrix[from_state, to_state])
        return 0.0

    def get_state_posterior(self) -> np.ndarray:
        """Current filtered posterior over states."""
        if self._alpha is None:
            return np.ones(self.n_states) / self.n_states
        return self._alpha.copy()

    def get_mean(self, state: int) -> np.ndarray:
        """Emission mean for a given state."""
        return self.means[state].copy()

    def get_covariance(self, state: int) -> np.ndarray:
        """Emission covariance for a given state."""
        return self.covariances[state].copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def n_observations(self) -> int:
        return self._n_updates

    def _compute_emissions(self, obs: np.ndarray) -> np.ndarray:
        """Compute P(obs | state=k) for each state k, with regularized covariance."""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            try:
                # Apply regularization only for PDF computation, not stored
                cov_reg = self.covariances[k] + self.emission_reg * np.eye(self.n_features)
                probs[k] = multivariate_normal.pdf(obs, mean=self.means[k], cov=cov_reg)
            except (np.linalg.LinAlgError, ValueError):
                probs[k] = 1e-300
        probs = np.maximum(probs, 1e-300)
        return probs

    def _initialize_from_data(self) -> None:
        """Initialize emission parameters from buffered observations using k-means++."""
        X = np.array(self._obs_buffer)
        n = len(X)

        global_cov = np.cov(X.T)
        if global_cov.ndim == 0:
            global_cov = np.eye(self.n_features)
        # Ensure positive definite
        global_cov += self.emission_reg * 10 * np.eye(self.n_features)

        # k-means++ initialization for means
        self.means[0] = X[self._rng.randint(n)]
        for k in range(1, self.n_states):
            dists = np.array([
                min(np.linalg.norm(X[i] - self.means[j]) for j in range(k))
                for i in range(n)
            ])
            probs = dists ** 2
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
                idx = self._rng.choice(n, p=probs)
            else:
                idx = self._rng.randint(n)
            self.means[k] = X[idx]

        # Post-init mean separation validation (fix #5)
        # If two means are too close, perturb the second one
        self._enforce_mean_separation(X)

        # Initialize covariances from global covariance
        for k in range(self.n_states):
            self.covariances[k] = global_cov.copy()

        self._initialized = True
        self._obs_buffer = []
        logger.info(
            f"HMM initialized from {n} observations. "
            f"States={self.n_states}, Features={self.n_features}"
        )

    def _enforce_mean_separation(self, X: np.ndarray) -> None:
        """Ensure all state means are sufficiently separated after initialization."""
        global_std = np.std(X, axis=0)
        # Minimum separation = 0.5 * global std norm, with a hard floor
        min_dist = max(0.5 * np.linalg.norm(global_std), 0.1)

        max_attempts = 10
        for _ in range(max_attempts):
            needs_fix = False
            for k in range(self.n_states):
                for j in range(k + 1, self.n_states):
                    dist = np.linalg.norm(self.means[k] - self.means[j])
                    if dist < min_dist:
                        # Perturb the second mean away
                        direction = self._rng.randn(self.n_features)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        self.means[j] = self.means[k] + direction * min_dist * 1.5
                        needs_fix = True
            if not needs_fix:
                break

    def get_state(self) -> dict:
        """Serialize full HMM state for checkpointing."""
        return {
            "transition_matrix": self.transition_matrix.tolist(),
            "means":             self.means.tolist(),
            "covariances":       self.covariances.tolist(),
            "start_prob":        self.start_prob.tolist(),
            "alpha":             self._alpha.tolist() if self._alpha is not None else None,
            "prev_alpha":        self._prev_alpha.tolist() if self._prev_alpha is not None else None,
            "prev_state":        int(self._prev_state),
            "n_updates":         self._n_updates,
            "innovation_ema":    float(self._innovation_ema),
            "initialized":       self._initialized,
        }

    def restore_state(self, state: dict) -> None:
        """Restore HMM state from checkpoint. Requires same n_states/n_features."""
        self.transition_matrix = np.array(state["transition_matrix"])
        self.means             = np.array(state["means"])
        self.covariances       = np.array(state["covariances"])
        self.start_prob        = np.array(state["start_prob"])
        self._alpha            = np.array(state["alpha"]) if state["alpha"] is not None else None
        self._prev_alpha       = np.array(state["prev_alpha"]) if state["prev_alpha"] is not None else None
        self._prev_state       = state["prev_state"]
        self._n_updates        = state["n_updates"]
        self._innovation_ema   = state["innovation_ema"]
        self._initialized      = state["initialized"]

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> None:
        """Normalize each row to sum to 1 (in-place)."""
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        matrix[:] = matrix / row_sums


# --- Rust backend swap ---
# GaussianHMM has a compatible PyO3 interface: filter_step, online_update,
# get_transition_prob, get_state_posterior, get_mean, get_covariance.
# The hmm_adapter.py wraps whichever backend is active.
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    _PyGaussianHMM = GaussianHMM
    RustGaussianHMM = get_rust_core().GaussianHMM
    logger.info("GaussianHMM Rust backend available (use via hmm_adapter)")
