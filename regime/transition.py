import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Deque, Tuple
from collections import deque
from itertools import islice
from .state_vector import StateVector
from .transition_model import TransitionProbabilityModel, TransitionFeatures
from microstructure.entropy import EntropyCalculator
import logging
from config.loader import get_config

logger = logging.getLogger(__name__)

# backend Rust si dispo
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    KalmanDerivativeTracker = get_rust_core().KalmanDerivativeTracker
    logger.info("KalmanDerivativeTracker → Rust backend")
else:
    KalmanDerivativeTracker = None  # défini ci-dessous


@dataclass
class TransitionEvent:
    from_regime: int
    to_regime: int
    strength: float
    delta_vector: np.ndarray  # [d_mu, d_sigma, d_skew, d_kurt, d_tail, d_entropy]
    is_significant: bool
    kl_divergence: float = 0.0
    reason: str = ""
    ml_probability: float = 0.0
    mu_velocity: float = 0.0
    mu_acceleration: float = 0.0
    entropy_velocity: float = 0.0
    entropy_acceleration: float = 0.0
    projection_magnitude: float = 0.0


class SimpleKalmanFilter1D:

    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 1e-2):
        self._estimate: float = 0.0
        self._error_estimate: float = 1.0
        self._process_var: float = process_variance
        self._measurement_var: float = measurement_variance
        self._initialized: bool = False

    def update(self, measurement: float) -> float:
        if not self._initialized:
            self._estimate = measurement
            self._initialized = True
            return self._estimate

        prediction = self._estimate
        prediction_error = self._error_estimate + self._process_var

        kalman_gain = prediction_error / (prediction_error + self._measurement_var)
        self._estimate = prediction + kalman_gain * (measurement - prediction)
        self._error_estimate = (1 - kalman_gain) * prediction_error

        return self._estimate


class _PyKalmanDerivativeTracker:
    """Kalman 3 états [position, vitesse, accélération] — ~70% moins de bruit qu'une diff finie"""

    def __init__(self, process_noise: float = 1e-6, measurement_noise: float = 1e-4,
                 vel_noise_mult: float = 10.0, acc_noise_mult: float = 100.0):
        self._x = np.zeros(3)
        self._P = np.eye(3) * 1.0
        self._F = np.array([
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        self._H = np.array([[1.0, 0.0, 0.0]])
        self._Q = np.diag([process_noise, process_noise * vel_noise_mult, process_noise * acc_noise_mult])
        self._R = np.array([[measurement_noise]])
        self._initialized = False

    def update(self, measurement: float) -> Tuple[float, float]:
        """retourne (vitesse, accélération)"""
        if not np.isfinite(measurement):
            return float(self._x[1]), float(self._x[2])

        if not self._initialized:
            self._x[0] = measurement
            self._initialized = True
            return 0.0, 0.0

        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # gain scalaire (mesure = scalaire)
        y = float(np.asarray(self._H @ x_pred).flat[0])
        y = measurement - y
        S = float(np.asarray(self._H @ P_pred @ self._H.T).flat[0] + self._R[0, 0])
        K = (P_pred @ self._H.T) / max(S, 1e-30)
        K = K.flatten()

        self._x = x_pred + K * y
        I_KH = np.eye(3) - np.outer(K, self._H[0])
        self._P = I_KH @ P_pred

        return float(self._x[1]), float(self._x[2])

    @property
    def velocity(self) -> float:
        return float(self._x[1])

    @property
    def acceleration(self) -> float:
        return float(self._x[2])

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_state(self) -> dict:
        """Serialize Kalman filter state for checkpointing."""
        return {
            "x":           self._x.tolist(),
            "P":           self._P.tolist(),
            "initialized": self._initialized,
        }

    def restore_state(self, state: dict) -> None:
        """Restore Kalman filter state. Requires same constructor params."""
        self._x           = np.array(state["x"])
        self._P           = np.array(state["P"])
        self._initialized = state["initialized"]


if not _USE_RUST_CORE:
    KalmanDerivativeTracker = _PyKalmanDerivativeTracker


class TransitionDetector:

    def __init__(self, strength_threshold: float = None, config=None):
        config = config or get_config()
        self.strength_threshold = strength_threshold or config.thresholds.regime.transition_strength_min
        self.weights = np.array(config.thresholds.regime.transition_weights)
        self.kl_boost_threshold = config.thresholds.regime.kl_boost_threshold
        self.kl_boost_amount = config.thresholds.regime.kl_boost_amount

        self._prev_regime: int = -1
        self._prev_state: Optional[StateVector] = None
        self._prev_kde = None
        self._last_event: Optional[TransitionEvent] = None

        self._state_history: Deque[StateVector] = deque(maxlen=5)

        # Kalman 3 états pour mu et entropy — position/vitesse/accélération conjointement
        _hmm_cfg = config.thresholds.regime.hmm
        _kf_kwargs = {}
        if KalmanDerivativeTracker is _PyKalmanDerivativeTracker:
            _kf_kwargs = dict(vel_noise_mult=_hmm_cfg.kalman_vel_noise_mult,
                              acc_noise_mult=_hmm_cfg.kalman_acc_noise_mult)
        self._kf_mu = KalmanDerivativeTracker(process_noise=1e-8, measurement_noise=1e-5, **_kf_kwargs)
        self._kf_entropy = KalmanDerivativeTracker(process_noise=1e-6, measurement_noise=1e-3, **_kf_kwargs)

        # PCA borné en mémoire (maxlen évite la fuite)
        self._state_accumulator: Deque[np.ndarray] = deque(maxlen=2000)
        self._accumulator_count: int = 0
        self._pca_components: Optional[np.ndarray] = None
        self._pca_fitted: bool = False

        self.ml_model = TransitionProbabilityModel(config=config)

        logger.info(f"Initialized TransitionDetector with threshold={self.strength_threshold}")

    def update(
        self,
        curr_regime: int,
        curr_state: StateVector,
        curr_kde=None,
        hmm_transition_prob: Optional[float] = None,
        hmm_weight: float = 0.0,
    ) -> Optional[TransitionEvent]:
        self._state_history.append(curr_state)
        self._state_accumulator.append(np.array(curr_state.to_array()))
        self._accumulator_count += 1

        # PCA réévalué tous les 50 updates — coût CPU négligeable
        if self._accumulator_count >= 50 and self._accumulator_count % 50 == 0:
            self._fit_pca()

        if self._prev_state is None:
            self._prev_regime = curr_regime
            self._prev_state = curr_state
            self._prev_kde = curr_kde
            return None

        if curr_regime == self._prev_regime:
            self._prev_state = curr_state
            self._prev_kde = curr_kde
            return None

        # --- CHANGEMENT DE RÉGIME ---
        v_curr = np.array(curr_state.to_array())
        v_prev = np.array(self._prev_state.to_array())
        delta = v_curr - v_prev

        # dérivées via Kalman (position/vitesse/accélération) — moins bruité que diff finie
        mu_vel, mu_acc = self._kf_mu.update(curr_state.mu)
        ent_vel, ent_acc = self._kf_entropy.update(curr_state.entropy)

        kl_div = self._compute_kl(curr_kde)
        projection_mag = self._compute_projection(v_curr)

        # force heuristique: delta normalisé par std glissante — évite la div par ~0 pct
        if len(self._state_accumulator) >= 10:
            start = max(0, len(self._state_accumulator) - 200)
            recent = np.array(list(islice(self._state_accumulator, start, None)))
            feature_scales = np.std(recent, axis=0)
            feature_scales = np.where(feature_scales < 1e-9, 1e-9, feature_scales)
        else:
            feature_scales = np.ones(len(delta))
        normalized_delta = np.abs(delta) / feature_scales
        raw_score = np.dot(normalized_delta, self.weights)
        heuristic_strength = float(np.tanh(raw_score))

        ml_result = self.ml_model.predict(TransitionFeatures(
            delta_vector=delta,
            strength=heuristic_strength,
            kl_divergence=kl_div,
            projection_magnitude=projection_mag,
            mu_velocity=mu_vel,
            mu_acceleration=mu_acc,
            entropy_velocity=ent_vel,
            entropy_acceleration=ent_acc,
        ))
        ml_prob = ml_result.probability

        # blend ML + HMM (max 50% HMM à pleine maturité)
        if hmm_transition_prob is not None and hmm_weight > 0:
            w = hmm_weight * 0.5
            strength = (1.0 - w) * ml_prob + w * hmm_transition_prob
        else:
            strength = ml_prob
        if kl_div > self.kl_boost_threshold:
            strength = min(1.0, strength + self.kl_boost_amount)

        is_significant = strength > self.strength_threshold

        reason_parts = []
        if is_significant:
            reason_parts.append(f"Strength {strength:.4f} > Threshold")
        else:
            reason_parts.append(f"Strength {strength:.4f} <= Threshold")
        reason_parts.append(f"(ML: {ml_prob:.2f}, KL: {kl_div:.4f})")
        reason_parts.append(f"[MuAcc: {mu_acc:.2e}, EntAcc: {ent_acc:.2e}]")
        reason = " ".join(reason_parts)

        if is_significant:
            logger.info(f"Significant Regime Transition: {self._prev_regime} -> {curr_regime} | {reason}")
        else:
            logger.debug(f"Insignificant Regime Transition: {self._prev_regime} -> {curr_regime} | {reason}")

        event = TransitionEvent(
            from_regime=self._prev_regime,
            to_regime=curr_regime,
            strength=strength,
            delta_vector=delta,
            is_significant=is_significant,
            kl_divergence=kl_div,
            reason=reason,
            ml_probability=ml_prob,
            mu_velocity=mu_vel,
            mu_acceleration=mu_acc,
            entropy_velocity=ent_vel,
            entropy_acceleration=ent_acc,
            projection_magnitude=projection_mag,
        )

        self._prev_regime = curr_regime
        self._prev_state = curr_state
        self._prev_kde = curr_kde
        self._last_event = event
        return event

    def _compute_derivatives(self, attr: str) -> Tuple[float, float]:
        if len(self._state_history) < 3:
            return 0.0, 0.0

        values = [getattr(s, attr) for s in self._state_history]
        n = len(values)

        # diff 5-points quand dispo — plus lisse que 3-points
        if n >= 5:
            vel = (-values[-5] + 8 * values[-4] - 8 * values[-2] + values[-1]) / 12.0
            acc = (-values[-5] + 16 * values[-4] - 30 * values[-3] + 16 * values[-2] - values[-1]) / 12.0
        else:
            vel = values[-1] - values[-2]
            acc = values[-1] - 2 * values[-2] + values[-3]
        return vel, acc

    def _compute_kl(self, curr_kde) -> float:
        if curr_kde is None or self._prev_kde is None:
            return 0.0

        try:
            p_min, p_max = self._prev_kde.get_bounds()
            q_min, q_max = curr_kde.get_bounds()
            x_min = min(p_min, q_min)
            x_max = max(p_max, q_max)
            x_grid = np.linspace(x_min, x_max, 1000)
            dx = x_grid[1] - x_grid[0]

            pdf_prev = self._prev_kde.evaluate(x_grid)
            pdf_curr = curr_kde.evaluate(x_grid)

            sum_prev = np.sum(pdf_prev) * dx
            sum_curr = np.sum(pdf_curr) * dx
            if sum_prev > 0:
                pdf_prev = pdf_prev / sum_prev
            if sum_curr > 0:
                pdf_curr = pdf_curr / sum_curr

            return EntropyCalculator.compute_kl_divergence(pdf_curr, pdf_prev, dx)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(f"KL divergence computation failed: {e}")
            return float('inf')  # "inconnu" plutôt que "pas de divergence"

    def _fit_pca(self) -> None:
        try:
            start = max(0, len(self._state_accumulator) - 200)
            X = np.array(list(islice(self._state_accumulator, start, None)))
            if X.shape[0] < 10:
                return
            X_centered = X - np.mean(X, axis=0)
            cov = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            self._pca_components = eigenvectors[:, idx[:2]]  # 2 composantes principales
            self._pca_fitted = True
        except np.linalg.LinAlgError:
            pass

    def _compute_projection(self, v: np.ndarray) -> float:
        if self._pca_fitted and self._pca_components is not None:
            projection = self._pca_components.T @ v
            return float(np.linalg.norm(projection))
        return float(np.linalg.norm(v))  # proxy L2 avant warmup PCA

    @staticmethod
    def _normalize_kf_state(kf_state: dict) -> dict:
        """Ensure Kalman state dict is JSON-serializable (converts ndarrays to lists)."""
        x = kf_state["x"]
        P = kf_state["P"]
        return {
            "x":           x.tolist() if hasattr(x, "tolist") else list(x),
            "P":           [row.tolist() if hasattr(row, "tolist") else list(row) for row in P],
            "initialized": bool(kf_state["initialized"]),
        }

    def get_state(self) -> dict:
        """Serialize TransitionDetector state for checkpointing."""
        return {
            "kf_mu":             self._normalize_kf_state(self._kf_mu.get_state()),
            "kf_entropy":        self._normalize_kf_state(self._kf_entropy.get_state()),
            "prev_regime":       self._prev_regime,
            "accumulator_count": self._accumulator_count,
            "pca_components":    self._pca_components.tolist() if self._pca_components is not None else None,
            "pca_fitted":        self._pca_fitted,
            "state_history":     [sv.to_array() for sv in self._state_history],
            "state_accumulator": [a.tolist() for a in self._state_accumulator],
        }

    def restore_state(self, state: dict) -> None:
        """Restore TransitionDetector state from checkpoint."""
        self._kf_mu.restore_state(state["kf_mu"])
        self._kf_entropy.restore_state(state["kf_entropy"])
        self._prev_regime       = state["prev_regime"]
        self._accumulator_count = state["accumulator_count"]
        self._pca_components    = np.array(state["pca_components"]) if state["pca_components"] is not None else None
        self._pca_fitted        = state["pca_fitted"]
        self._state_history     = deque(
            [StateVector(mu=s[0], sigma=s[1], skew=s[2], kurtosis=s[3],
                         tail_slope=s[4], entropy=s[5]) for s in state["state_history"]],
            maxlen=self._state_history.maxlen,
        )
        self._state_accumulator = deque(
            [np.array(a) for a in state["state_accumulator"]],
            maxlen=self._state_accumulator.maxlen,
        )
