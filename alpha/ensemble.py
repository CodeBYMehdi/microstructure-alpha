import numpy as np
import logging
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    expected_return: float
    confidence: float
    prediction_std: float
    model_contributions: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    is_valid: bool = True
    disagreement: float = 0.0  # variance inter-modèles


class OnlineRidgeModel:

    def __init__(self, n_features: int, l2_lambda: float = 0.01,
                 learning_rate: float = 0.001, forgetting_factor: float = 0.995):
        self.n_features = n_features
        self.l2_lambda = l2_lambda
        self.lr = learning_rate
        self.forgetting = forgetting_factor

        self._weights = np.zeros(n_features)
        self._bias = 0.0

        self._feature_mean = np.zeros(n_features)
        self._feature_var = np.ones(n_features)
        self._n_updates = 0

        self._errors: Deque[float] = deque(maxlen=500)
        self._predictions: Deque[float] = deque(maxlen=500)
        self._actuals: Deque[float] = deque(maxlen=500)

        self._pending_features: Optional[np.ndarray] = None
        self._pending_prediction: Optional[float] = None

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        x = self._normalize(features)
        pred = float(np.dot(self._weights, x) + self._bias)

        if self._n_updates < 30:
            confidence = 0.0
        else:
            errors = np.array(self._errors)
            pred_std = float(np.std(errors)) if len(errors) > 5 else 1.0
            target_std = max(np.std(list(self._actuals)), 1e-10)
            confidence = float(np.clip(1.0 - pred_std / target_std, 0.0, 1.0))

        self._pending_features = x.copy()
        self._pending_prediction = pred
        return pred, confidence

    def update(self, actual: float) -> None:
        if self._pending_features is None:
            return

        error = actual - self._pending_prediction
        self._errors.append(error)
        self._actuals.append(actual)
        self._predictions.append(self._pending_prediction)

        x = self._pending_features
        grad_w = -2.0 * error * x + 2.0 * self.l2_lambda * self._weights
        adaptive_lr = self.lr / (1.0 + 0.001 * self._n_updates)

        self._weights -= adaptive_lr * grad_w
        self._bias -= adaptive_lr * (-2.0 * error)
        self._weights *= self.forgetting

        self._update_stats(self._pending_features, actual)
        self._pending_features = None
        self._pending_prediction = None
        self._n_updates += 1

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._feature_var + 1e-8)
        return (features - self._feature_mean) / std

    def _update_stats(self, features: np.ndarray, target: float) -> None:
        alpha = 0.01
        if self._n_updates == 0:
            self._feature_mean = features.copy()
            self._feature_var = np.ones(self.n_features)
            return
        delta = features - self._feature_mean
        self._feature_mean += alpha * delta
        self._feature_var = (1 - alpha) * self._feature_var + alpha * (delta ** 2)

    def get_state(self) -> Dict:
        return {
            'weights': self._weights.copy(),
            'bias': self._bias,
            'feature_mean': self._feature_mean.copy(),
            'feature_var': self._feature_var.copy(),
            'n_updates': self._n_updates,
        }

    def set_state(self, state: Dict) -> None:
        self._weights = state['weights'].copy()
        self._bias = state['bias']
        self._feature_mean = state['feature_mean'].copy()
        self._feature_var = state['feature_var'].copy()
        self._n_updates = state['n_updates']


class RegimeConditionalModel:
    """modèle linéaire séparé par régime — s'adapte à chaque contexte"""

    def __init__(self, n_features: int, max_regimes: int = 10,
                 l2_lambda: float = 0.05, learning_rate: float = 0.002):
        self.n_features = n_features
        self.max_regimes = max_regimes
        self.l2_lambda = l2_lambda
        self.lr = learning_rate

        self._models: Dict[int, Dict] = {}
        self._default_model = {
            'weights': np.zeros(n_features),
            'bias': 0.0,
            'n_updates': 0,
            'errors': deque(maxlen=200),
            'actuals': deque(maxlen=200),
        }

        self._pending_regime: Optional[int] = None
        self._pending_features: Optional[np.ndarray] = None
        self._pending_prediction: Optional[float] = None

        # normalisation partagée entre régimes
        self._feature_mean = np.zeros(n_features)
        self._feature_var = np.ones(n_features)
        self._total_updates = 0

    def predict(self, features: np.ndarray, regime_id: int) -> Tuple[float, float]:
        x = self._normalize(features)
        model = self._get_model(regime_id)

        pred = float(np.dot(model['weights'], x) + model['bias'])

        if model['n_updates'] < 15:
            confidence = 0.0
        else:
            errors = np.array(model['errors'])
            pred_std = float(np.std(errors)) if len(errors) > 5 else 1.0
            target_std = max(np.std(list(model['actuals'])), 1e-10)
            confidence = float(np.clip(1.0 - pred_std / target_std, 0.0, 1.0))

        self._pending_regime = regime_id
        self._pending_features = x.copy()
        self._pending_prediction = pred
        return pred, confidence

    def update(self, actual: float) -> None:
        if self._pending_features is None or self._pending_regime is None:
            return

        model = self._get_model(self._pending_regime)
        error = actual - self._pending_prediction
        model['errors'].append(error)
        model['actuals'].append(actual)

        x = self._pending_features
        grad_w = -error * x + self.l2_lambda * model['weights']
        adaptive_lr = self.lr / (1.0 + 0.001 * model['n_updates'])
        model['weights'] -= adaptive_lr * grad_w
        model['bias'] -= adaptive_lr * (-error)
        model['n_updates'] += 1

        self._update_stats(self._pending_features, actual)
        self._pending_features = None
        self._pending_regime = None
        self._pending_prediction = None

    def _get_model(self, regime_id: int) -> Dict:
        if regime_id not in self._models:
            if len(self._models) >= self.max_regimes:
                # éviction du régime le moins utilisé
                least_used = min(self._models, key=lambda r: self._models[r]['n_updates'])
                del self._models[least_used]
            self._models[regime_id] = {
                'weights': np.zeros(self.n_features),
                'bias': 0.0,
                'n_updates': 0,
                'errors': deque(maxlen=200),
                'actuals': deque(maxlen=200),
            }
        return self._models[regime_id]

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._feature_var + 1e-8)
        return (features - self._feature_mean) / std

    def _update_stats(self, features: np.ndarray, target: float) -> None:
        alpha = 0.01
        self._total_updates += 1
        if self._total_updates == 1:
            self._feature_mean = features.copy()
            self._feature_var = np.ones(self.n_features)
            return
        delta = features - self._feature_mean
        self._feature_mean += alpha * delta
        self._feature_var = (1 - alpha) * self._feature_var + alpha * (delta ** 2)


class MomentumMeanReversionModel:
    """signaux hand-crafted — pas d'entraînement, règles microstructure directes"""

    def __init__(self):
        self._price_history: Deque[float] = deque(maxlen=500)
        self._return_history: Deque[float] = deque(maxlen=500)
        self._pending_prediction: Optional[float] = None
        self._errors: Deque[float] = deque(maxlen=200)
        self._actuals: Deque[float] = deque(maxlen=200)
        self._n_updates = 0

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        # indices feature: 0=mu, 6=ofi, 13=mom_short, 14=mom_long, 15=mean_rev
        if len(features) < 16:
            self._pending_prediction = 0.0
            return 0.0, 0.0

        mu = features[0]
        ofi = features[6]
        mom_short = features[13]
        mom_long = features[14]
        mean_rev = features[15]

        # momentum confirmé par OFI — filtre les faux signaux
        momentum_signal = 0.0
        if abs(mom_short) > 1e-6:
            ofi_sign = np.sign(ofi) if abs(ofi) > 0.01 else 0
            mom_sign = np.sign(mom_short)
            if ofi_sign == mom_sign:
                momentum_signal = mom_short * 0.3

        # MR: fade les Z-scores extrêmes seulement (>1.5σ)
        mr_signal = 0.0
        if abs(mean_rev) > 1.5:
            mr_signal = -mean_rev * 0.002

        # trending → momentum dominant, ranging → MR dominant
        trend_strength = abs(mom_long) if len(features) > 14 else 0
        if trend_strength > 0.005:
            pred = momentum_signal * 0.7 + mr_signal * 0.3
        else:
            pred = momentum_signal * 0.3 + mr_signal * 0.7

        signal_magnitude = abs(momentum_signal) + abs(mr_signal)
        confidence = min(1.0, signal_magnitude * 100) if self._n_updates > 30 else 0.0

        self._pending_prediction = pred
        return pred, confidence

    def update(self, actual: float) -> None:
        if self._pending_prediction is not None:
            error = actual - self._pending_prediction
            self._errors.append(error)
            self._actuals.append(actual)
            self._n_updates += 1
            self._pending_prediction = None


class GradientBoostedModel:
    """GBT offline — réentraîné périodiquement sur buffer glissant"""

    def __init__(self, n_features: int, retrain_interval: int = 500,
                 min_train_samples: int = 200, max_buffer: int = 5000):
        self.n_features = n_features
        self.retrain_interval = retrain_interval
        self.min_train_samples = min_train_samples

        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=max_buffer)
        self._target_buffer: Deque[float] = deque(maxlen=max_buffer)
        self._model = None
        self._n_updates = 0
        self._last_train = 0

        self._pending_features: Optional[np.ndarray] = None
        self._pending_prediction: Optional[float] = None
        self._errors: Deque[float] = deque(maxlen=500)
        self._actuals: Deque[float] = deque(maxlen=500)

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        if self._model is None:
            self._pending_features = features.copy()
            self._pending_prediction = 0.0
            return 0.0, 0.0

        try:
            pred = float(self._model.predict(features.reshape(1, -1))[0])
        except Exception:
            pred = 0.0

        confidence = 0.0
        if len(self._errors) > 20:
            pred_std = np.std(list(self._errors))
            target_std = max(np.std(list(self._actuals)), 1e-10)
            confidence = float(np.clip(1.0 - pred_std / target_std, 0.0, 1.0))

        self._pending_features = features.copy()
        self._pending_prediction = pred
        return pred, confidence

    def update(self, actual: float) -> None:
        if self._pending_features is not None:
            self._feature_buffer.append(self._pending_features)
            self._target_buffer.append(actual)

            if self._pending_prediction is not None:
                error = actual - self._pending_prediction
                self._errors.append(error)
            self._actuals.append(actual)

            self._n_updates += 1
            self._pending_features = None
            self._pending_prediction = None

        if (self._n_updates - self._last_train >= self.retrain_interval
                and len(self._feature_buffer) >= self.min_train_samples):
            self._retrain()

    def _retrain(self) -> None:
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            X = np.array(list(self._feature_buffer))
            y = np.array(list(self._target_buffer))

            # 80/20 train/val — pas de shuffle (données temporelles)
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            )
            model.fit(X_train, y_train)

            if split < len(X):
                X_val, y_val = X[split:], y[split:]
                val_pred = model.predict(X_val)
                val_mse = np.mean((val_pred - y_val) ** 2)
                baseline_mse = np.var(y_val)

                if baseline_mse > 1e-15 and val_mse < baseline_mse:
                    self._model = model
                    logger.info(
                        f"GBT retrained: val_mse={val_mse:.8f}, baseline={baseline_mse:.8f}, "
                        f"R²={1 - val_mse / baseline_mse:.4f}"
                    )
                else:
                    logger.info(f"GBT retrain rejected: no improvement (val_mse={val_mse:.8f} vs baseline={baseline_mse:.8f})")
            else:
                self._model = model
                logger.info("GBT retrained (no validation split)")

            self._last_train = self._n_updates

        except ImportError:
            logger.warning("scikit-learn not available, GBT model disabled")
        except Exception as e:
            logger.error(f"GBT retrain failed: {e}")


class AlphaEnsemble:
    """4 modèles + méta-learner à poids dynamiques + rollback sur dégradation"""

    def __init__(
        self,
        n_features: int = 38,
        min_samples: int = 50,
        weight_decay: float = 0.98,       # transition lente des poids — stabilité
        rollback_threshold: float = 0.3,
        rollback_window: int = 100,
    ):
        self.n_features = n_features
        self.min_samples = min_samples
        self.weight_decay = weight_decay
        self.rollback_threshold = rollback_threshold
        self.rollback_window = rollback_window

        self._models: Dict[str, object] = {
            'ridge': OnlineRidgeModel(n_features, l2_lambda=0.01, learning_rate=0.001),
            'regime_cond': RegimeConditionalModel(n_features),
            'momentum_mr': MomentumMeanReversionModel(),
            'gbt': GradientBoostedModel(n_features, retrain_interval=500),
        }

        # poids initiaux égaux
        self._meta_weights: Dict[str, float] = {name: 1.0 / len(self._models) for name in self._models}

        self._model_errors: Dict[str, Deque[float]] = {
            name: deque(maxlen=500) for name in self._models
        }
        self._model_correct: Dict[str, Deque[bool]] = {
            name: deque(maxlen=500) for name in self._models
        }

        self._model_checkpoints: Dict[str, Dict] = {}
        self._model_oos_performance: Dict[str, Deque[float]] = {
            name: deque(maxlen=rollback_window) for name in self._models
        }

        self._n_predictions = 0
        self._pending_predictions: Dict[str, float] = {}
        self._current_regime_id: int = -1

        logger.info(
            f"AlphaEnsemble initialized: {len(self._models)} models, "
            f"n_features={n_features}"
        )

    def predict(self, features: np.ndarray, regime_id: int = -1) -> EnsemblePrediction:
        self._current_regime_id = regime_id
        predictions: Dict[str, Tuple[float, float]] = {}

        for name, model in self._models.items():
            try:
                if name == 'regime_cond':
                    pred, conf = model.predict(features, regime_id)
                else:
                    pred, conf = model.predict(features)
                predictions[name] = (pred, conf)
            except Exception as e:
                logger.debug(f"Model {name} prediction failed: {e}")
                predictions[name] = (0.0, 0.0)

        self._pending_predictions = {name: pred for name, (pred, _) in predictions.items()}

        if self._n_predictions < self.min_samples:
            # pas encore assez d'historique — moyenne simple
            avg_pred = np.mean([p for p, _ in predictions.values()])
            return EnsemblePrediction(
                expected_return=avg_pred,
                confidence=0.0,
                prediction_std=1.0,
                model_contributions={name: pred / len(predictions) for name, (pred, _) in predictions.items()},
                model_weights=dict(self._meta_weights),
                is_valid=False,
            )

        total_weight = sum(self._meta_weights.values())
        if total_weight < 1e-10:
            total_weight = 1.0

        weighted_pred = 0.0
        contributions = {}
        for name, (pred, conf) in predictions.items():
            w = self._meta_weights[name] / total_weight
            weighted_pred += w * pred
            contributions[name] = w * pred

        weighted_conf = sum(
            (self._meta_weights[name] / total_weight) * conf
            for name, (_, conf) in predictions.items()
        )

        pred_values = np.array([p for p, _ in predictions.values()])
        disagreement = float(np.std(pred_values))

        # désaccord inter-modèles réduit la confiance
        adj_confidence = weighted_conf * max(0.1, 1.0 - disagreement * 10)

        return EnsemblePrediction(
            expected_return=weighted_pred,
            confidence=float(np.clip(adj_confidence, 0.0, 1.0)),
            prediction_std=disagreement,
            model_contributions=contributions,
            model_weights={name: w / total_weight for name, w in self._meta_weights.items()},
            is_valid=True,
            disagreement=disagreement,
        )

    def update(self, actual_return: float) -> None:
        self._n_predictions += 1

        for name, model in self._models.items():
            try:
                model.update(actual_return)
            except Exception as e:
                logger.debug(f"Model {name} update failed: {e}")

            if name in self._pending_predictions:
                pred = self._pending_predictions[name]
                error = abs(actual_return - pred)
                self._model_errors[name].append(error)

                correct = (np.sign(pred) == np.sign(actual_return)) if abs(actual_return) > 1e-10 else False
                self._model_correct[name].append(correct)
                self._model_oos_performance[name].append(error)

        self._pending_predictions = {}

        self._update_meta_weights()

        if self._n_predictions % self.rollback_window == 0:
            self._check_rollbacks()

        if self._n_predictions % (self.rollback_window * 2) == 0:
            self._save_checkpoints()

    def _update_meta_weights(self) -> None:
        """poids ∝ 1/MAE — le meilleur modèle récent prend plus de place"""
        if self._n_predictions < self.min_samples:
            return

        inv_mae = {}
        for name in self._models:
            errors = list(self._model_errors[name])
            if len(errors) >= 10:
                mae = np.mean(np.abs(errors[-100:]))
                inv_mae[name] = 1.0 / max(mae, 1e-10)
            else:
                inv_mae[name] = 1.0

        total = sum(inv_mae.values())
        if total > 0:
            for name in self._models:
                new_w = inv_mae[name] / total
                # blend lent pour éviter les oscillations de poids
                self._meta_weights[name] = (
                    self.weight_decay * self._meta_weights[name]
                    + (1 - self.weight_decay) * new_w
                )

    def _save_checkpoints(self) -> None:
        for name, model in self._models.items():
            if hasattr(model, 'get_state'):
                self._model_checkpoints[name] = model.get_state()

    def _check_rollbacks(self) -> None:
        # ratio + Mann-Whitney: ratio détecte les gros shifts, U-test les changements distribut.
        from scipy.stats import mannwhitneyu

        for name in self._models:
            perf = list(self._model_oos_performance[name])
            if len(perf) < self.rollback_window:
                continue

            half = self.rollback_window // 2
            earlier = perf[-self.rollback_window:-half]
            recent = perf[-half:]

            mean_earlier = np.mean(earlier)
            mean_recent = np.mean(recent)

            ratio_degraded = (
                mean_earlier > 1e-10
                and mean_recent / mean_earlier > (1 + self.rollback_threshold)
            )

            stat_degraded = False
            if len(earlier) >= 10 and len(recent) >= 10:
                try:
                    _, p_value = mannwhitneyu(recent, earlier, alternative='greater')
                    stat_degraded = p_value < 0.05
                except ValueError:
                    pass

            if (ratio_degraded or stat_degraded):
                if name in self._model_checkpoints and hasattr(self._models[name], 'set_state'):
                    logger.warning(
                        f"Rolling back model '{name}': recent MAE {mean_recent:.6f} vs earlier {mean_earlier:.6f} "
                        f"(ratio={mean_recent / max(mean_earlier, 1e-10):.2f}x, stat_test={stat_degraded})"
                    )
                    self._models[name].set_state(self._model_checkpoints[name])
                    self._model_oos_performance[name].clear()

    def get_metrics(self) -> Dict:
        metrics = {
            'n_predictions': self._n_predictions,
            'meta_weights': dict(self._meta_weights),
            'per_model': {},
        }

        for name in self._models:
            errors = list(self._model_errors[name])
            correct = list(self._model_correct[name])
            if errors:
                metrics['per_model'][name] = {
                    'mae': float(np.mean(np.abs(errors[-100:]))),
                    'directional_accuracy': float(np.mean(correct[-100:])) if correct else 0.0,
                    'n_updates': len(errors),
                    'weight': self._meta_weights.get(name, 0.0),
                }

        return metrics

    def get_top_model(self) -> str:
        if not self._meta_weights:
            return 'ridge'
        return max(self._meta_weights, key=self._meta_weights.get)
