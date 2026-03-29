import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class MicrostructureMoments:
    mu: float
    sigma: float
    skew: float
    kurtosis: float
    tail_slope: float

class MomentsCalculator:
    # l'usine a gaz

    def __init__(self, garch_refit_interval: int = 200, garch_min_obs: int = 100):
        from microstructure.garch import GarchVolatility
        self._garch = GarchVolatility(
            refit_interval=garch_refit_interval,
            min_obs=garch_min_obs,
        )

    def compute(self, data: np.ndarray) -> MicrostructureMoments:
        # la calculette
        if len(data) < 10:
            # Retourne état zéro/neutre si données insuffisantes
            return MicrostructureMoments(0.0, 0.0, 0.0, 0.0, 0.0)

        mu = np.mean(data)
        sigma = self._garch.conditional_vol(data)
        
        if sigma == 0:
            return MicrostructureMoments(mu, 0.0, 0.0, 0.0, 0.0)
            
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data) # Kurtosis Fisher (excès, normal = 0)
        
        # Hill estimator for tail index: measures true tail heaviness
        # Uses top 10% of absolute returns to estimate Pareto tail exponent
        # tail_slope = 1/alpha where alpha is the Hill tail index
        # Higher tail_slope = heavier tails (more extreme events)
        abs_returns = np.abs(data)
        n_tail = max(5, len(abs_returns) // 10)  # Top 10%, minimum 5
        sorted_abs = np.sort(abs_returns)[::-1]  # Descending
        tail_values = sorted_abs[:n_tail]
        x_min = sorted_abs[n_tail] if n_tail < len(sorted_abs) else sorted_abs[-1]

        if x_min > 1e-15 and np.all(tail_values > 0):
            # Hill estimator: alpha = n / sum(log(x_i / x_min))
            log_ratios = np.log(tail_values / x_min)
            sum_log = np.sum(log_ratios)
            if sum_log > 1e-10:
                hill_alpha = n_tail / sum_log
                tail_slope = 1.0 / max(hill_alpha, 0.1)  # Invert: heavier tails = higher value
            else:
                tail_slope = 0.0  # Very thin tails (near-Gaussian)
        else:
            tail_slope = 0.0
        
        return MicrostructureMoments(
            mu=mu,
            sigma=sigma,
            skew=skewness,
            kurtosis=kurt,
            tail_slope=tail_slope
        )


# --- Rust backend swap ---
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    import logging as _logging
    _rc = get_rust_core()
    _PyMomentsCalculator = MomentsCalculator
    MomentsCalculator = _rc.MomentsCalculator
    # Keep Python MicrostructureMoments dataclass — Rust MomentsCalculator.compute()
    # returns a Rust MicrostructureMoments that has the same .mu/.sigma/.skew etc attrs
    _logging.getLogger(__name__).info("MomentsCalculator → Rust backend")
