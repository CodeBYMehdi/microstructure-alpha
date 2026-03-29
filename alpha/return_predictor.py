import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    expected_return: float
    confidence: float
    prediction_std: float
    feature_importance: Optional[np.ndarray] = None
    is_valid: bool = True


class ReturnPredictor:

    def __init__(
        self,
        n_features: int = 38,
        learning_rate: float = 0.001,
        l2_lambda: float = 0.01,
        forgetting_factor: float = 0.995,  # décroissance exponentielle — empirique
        min_samples: int = 30,
        max_history: int = 500,
    ):
        self.n_features = n_features
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.forgetting = forgetting_factor
        self.min_samples = min_samples

        self._weights = np.zeros(n_features)
        self._bias = 0.0

        self._feature_mean = np.zeros(n_features)
        self._feature_var = np.ones(n_features)
        self._target_mean = 0.0
        self._target_var = 1.0
        self._n_updates = 0

        self._errors: deque = deque(maxlen=max_history)
        self._predictions: deque = deque(maxlen=max_history)
        self._actuals: deque = deque(maxlen=max_history)

        # stocke la prédiction courante pour l'update au prochain tick
        self._pending_features: Optional[np.ndarray] = None
        self._pending_prediction: Optional[float] = None

        logger.info(
            f"ReturnPredictor initialized: n_features={n_features}, "
            f"lr={learning_rate}, l2={l2_lambda}"
        )

    def predict(self, features: np.ndarray) -> PredictionResult:
        if len(features) != self.n_features:
            logger.warning(f"Feature dimension mismatch: got {len(features)}, expected {self.n_features}")
            return PredictionResult(0.0, 0.0, 0.0, is_valid=False)

        x = self._normalize_features(features)
        raw_pred = float(np.dot(self._weights, x) + self._bias)

        if self._n_updates < self.min_samples:
            confidence = 0.0
            pred_std = 1.0
        else:
            errors = np.array(self._errors)
            pred_std = float(np.std(errors)) if len(errors) > 5 else 1.0
            # confiance = 1/(1 + ratio_erreur) — croît quand le modèle améliore
            target_std = np.sqrt(max(self._target_var, 1e-10))
            error_ratio = pred_std / target_std if target_std > 0 else 1.0
            confidence = float(np.clip(1.0 / (1.0 + error_ratio), 0.0, 1.0))

        abs_weights = np.abs(self._weights)
        total = np.sum(abs_weights)
        importance = abs_weights / total if total > 0 else np.ones(self.n_features) / self.n_features

        self._pending_features = x.copy()
        self._pending_prediction = raw_pred

        return PredictionResult(
            expected_return=raw_pred,
            confidence=confidence,
            prediction_std=pred_std,
            feature_importance=importance,
            is_valid=self._n_updates >= self.min_samples,
        )

    def update(self, actual_return: float) -> None:
        if self._pending_features is None:
            return

        x = self._pending_features
        pred = self._pending_prediction

        error = actual_return - pred
        self._errors.append(error)
        self._predictions.append(pred)
        self._actuals.append(actual_return)

        # SGD + L2 — lr décroissant avec le nb d'updates
        grad_w = -error * x + self.l2_lambda * self._weights
        grad_b = -error
        adaptive_lr = self.lr / (1.0 + 0.001 * self._n_updates)

        self._weights -= adaptive_lr * grad_w
        self._bias -= adaptive_lr * grad_b

        self._weights *= self.forgetting  # oubli exponentiel des vieilles infos

        self._update_stats(self._pending_features, actual_return)

        self._pending_features = None
        self._pending_prediction = None
        self._n_updates += 1

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._feature_var + 1e-8)
        return (features - self._feature_mean) / std

    def _update_stats(self, features: np.ndarray, target: float) -> None:
        alpha = 0.01  # EMA — faible alpha = mémoire longue
        if self._n_updates == 0:
            self._feature_mean = features.copy()
            self._feature_var = np.ones(self.n_features)
            self._target_mean = target
            self._target_var = 1.0
            return

        delta = features - self._feature_mean
        self._feature_mean += alpha * delta
        self._feature_var = (1 - alpha) * self._feature_var + alpha * (delta ** 2)

        t_delta = target - self._target_mean
        self._target_mean += alpha * t_delta
        self._target_var = (1 - alpha) * self._target_var + alpha * (t_delta ** 2)

    def get_metrics(self) -> dict:
        if len(self._errors) < 5:
            return {
                "n_updates": self._n_updates,
                "mae": 0.0,
                "rmse": 0.0,
                "directional_accuracy": 0.0,
                "r_squared": 0.0,
                "ic": 0.0,
            }

        errors = np.array(self._errors)
        preds = np.array(self._predictions)
        actuals = np.array(self._actuals)

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        correct_dir = np.sum(np.sign(preds) == np.sign(actuals))
        dir_acc = correct_dir / len(preds)

        # IC = rank corr (Spearman) — meilleur que Pearson sur données asymétriques
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(preds, actuals)
            ic = float(ic) if np.isfinite(ic) else 0.0
        except (ValueError, FloatingPointError) as e:
            logger.error("IC computation failed: %s", e)
            ic = 0.0

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / max(ss_tot, 1e-10))

        return {
            "n_updates": self._n_updates,
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": float(dir_acc),
            "r_squared": float(np.clip(r2, -1.0, 1.0)),
            "ic": ic,
        }

    def get_top_features(self, feature_names: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        abs_w = np.abs(self._weights)
        indices = np.argsort(abs_w)[::-1][:top_k]
        return [(feature_names[i], float(abs_w[i])) for i in indices if i < len(feature_names)]

    def reset(self) -> None:
        self._weights = np.zeros(self.n_features)
        self._bias = 0.0
        self._feature_mean = np.zeros(self.n_features)
        self._feature_var = np.ones(self.n_features)
        self._n_updates = 0
        self._errors.clear()
        self._predictions.clear()
        self._actuals.clear()
        self._pending_features = None
        self._pending_prediction = None

    def get_state(self) -> dict:
        """Serialize predictor state for checkpointing."""
        return {
            "weights":      self._weights.tolist(),
            "bias":         float(self._bias),
            "feature_mean": self._feature_mean.tolist(),
            "feature_var":  self._feature_var.tolist(),
            "target_mean":  float(self._target_mean),
            "target_var":   float(self._target_var),
            "n_updates":    self._n_updates,
            "errors":       list(self._errors),
            "predictions":  list(self._predictions),
            "actuals":      list(self._actuals),
        }

    def restore_state(self, state: dict) -> None:
        """Restore predictor state from checkpoint."""
        restored_weights = np.array(state["weights"])
        if len(restored_weights) != self.n_features:
            logger.warning(
                "checkpoint_dim_mismatch: saved=%d current=%d — skipping restore",
                len(restored_weights), self.n_features,
            )
            return
        self._weights      = restored_weights
        self._bias         = state["bias"]
        self._feature_mean = np.array(state["feature_mean"])
        self._feature_var  = np.array(state["feature_var"])
        self._target_mean  = state["target_mean"]
        self._target_var   = state["target_var"]
        self._n_updates    = state["n_updates"]
        self._errors       = deque(state["errors"], maxlen=self._errors.maxlen)
        self._predictions  = deque(state["predictions"], maxlen=self._predictions.maxlen)
        self._actuals      = deque(state["actuals"], maxlen=self._actuals.maxlen)
        self._pending_features = None
        self._pending_prediction = None


# backend Rust si dispo — hot path ~10x plus rapide
from core.backend import _USE_RUST_CORE, get_rust_core as _get_rust_core
if _USE_RUST_CORE:
    _rc = _get_rust_core()
    _PyReturnPredictor = ReturnPredictor

    class ReturnPredictor(_PyReturnPredictor):  # type: ignore[no-redef]

        def __init__(self, n_features=38, learning_rate=0.001, l2_lambda=0.01,
                     forgetting_factor=0.995, min_samples=30, max_history=500):
            super().__init__(n_features, learning_rate, l2_lambda,
                             forgetting_factor, min_samples, max_history)
            self._rust = _rc.RustReturnPredictor(
                n_features=n_features,
                learning_rate=learning_rate,
                l2_lambda=l2_lambda,
                forgetting_factor=forgetting_factor,
                min_samples=min_samples,
                max_history=max_history,
            )

        def predict(self, features: np.ndarray) -> PredictionResult:
            if len(features) != self.n_features:
                return PredictionResult(0.0, 0.0, 0.0, is_valid=False)
            feat = np.asarray(features, dtype=np.float64)
            expected_return, confidence, pred_std, is_valid = self._rust.predict(feat)
            importance = np.asarray(self._rust.get_feature_importance())
            # sync état Python pour get_metrics()
            self._pending_prediction = expected_return
            self._pending_features = feat
            return PredictionResult(
                expected_return=expected_return,
                confidence=confidence,
                prediction_std=pred_std,
                feature_importance=importance,
                is_valid=is_valid,
            )

        def update(self, actual_return: float) -> None:
            self._rust.update(actual_return)
            if self._pending_prediction is not None:
                error = actual_return - self._pending_prediction
                self._errors.append(error)
                self._predictions.append(self._pending_prediction)
                self._actuals.append(actual_return)
                self._pending_features = None
                self._pending_prediction = None
            self._n_updates = self._rust.n_updates

        def reset(self) -> None:
            super().reset()
            self._rust.reset()

    logger.info("ReturnPredictor → Rust backend")
