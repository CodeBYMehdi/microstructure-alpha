"""Online GARCH(1,1) conditional volatility estimator."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GarchVolatility:
    """GARCH(1,1) conditional vol with periodic MLE refit.

    Between refits, updates variance via the GARCH recursion:
        sigma_sq_t = omega + alpha * r_sq_{t-1} + beta * sigma_sq_{t-1}
    Falls back to rolling std when model hasn't converged.
    """

    def __init__(self, refit_interval: int = 200, min_obs: int = 100):
        self._omega: float | None = None
        self._alpha: float | None = None
        self._beta: float | None = None
        self._last_variance: float | None = None
        self._tick_count: int = 0
        self._refit_interval = refit_interval
        self._min_obs = min_obs

    def conditional_vol(self, returns: np.ndarray) -> float:
        """Return conditional sigma. Falls back to rolling std if not fitted."""
        self._tick_count += 1

        if len(returns) < self._min_obs:
            return float(np.std(returns, ddof=1))

        # Refit periodically
        if self._omega is None or self._tick_count % self._refit_interval == 0:
            self._fit(returns)

        # If fit failed, fallback
        if self._omega is None:
            return float(np.std(returns, ddof=1))

        # GARCH recursion
        r_prev = returns[-1]
        sigma_sq = self._omega + self._alpha * r_prev**2 + self._beta * self._last_variance
        sigma_sq = max(sigma_sq, 1e-12)
        self._last_variance = sigma_sq
        return float(np.sqrt(sigma_sq))

    def _fit(self, returns: np.ndarray) -> None:
        """Refit GARCH(1,1) via arch library MLE."""
        try:
            from arch import arch_model
            scaled = returns * 100  # percentage for numerical stability
            model = arch_model(scaled, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
            result = model.fit(disp='off', show_warning=False)
            params = result.params
            omega = params['omega'] / 1e4  # undo percentage scaling
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            cond_vol = result.conditional_volatility
            last_vol = float(cond_vol[-1] if hasattr(cond_vol, '__getitem__') else cond_vol.iloc[-1])
            # Commit all at once so state is never partially updated
            self._omega = omega
            self._alpha = alpha
            self._beta = beta
            self._last_variance = (last_vol / 100) ** 2
        except Exception as e:
            logger.warning("garch_fit_failed: error=%s n_obs=%d fallback=rolling_std", e, len(returns))

    def get_state(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "last_variance": self._last_variance,
            "tick_count": self._tick_count,
        }

    def restore_state(self, state: dict) -> None:
        """Restore from checkpoint."""
        self._omega = state["omega"]
        self._alpha = state["alpha"]
        self._beta = state["beta"]
        self._last_variance = state["last_variance"]
        self._tick_count = state["tick_count"]
