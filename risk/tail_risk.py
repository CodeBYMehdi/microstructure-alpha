# attention aux degats
# ce qu'on a dans le sac

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import logging
from config.loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class TailRiskMetrics:
    # attention aux degats
    var: float       # Value at Risk
    cvar: float      # Conditional VaR (Expected Shortfall)
    tail_slope: float  # Hill estimator alpha
    kurtosis: float
    confidence: float


class TailRiskAnalyzer:
    # attention aux degats
    # le bif

    def __init__(self, window: int = 100, confidence: float = 0.95, min_points: int = 20, config=None):
        cfg = config or get_config()
        tail_cfg = cfg.thresholds.tail_risk

        self.window = window
        self.confidence = confidence
        self.min_points = min_points
        self.returns: List[float] = []

        # Config-driven thresholds
        self.min_tail_points = tail_cfg.min_tail_points
        self.tail_fraction = tail_cfg.tail_fraction
        self.default_tail_slope = tail_cfg.default_tail_slope
        self.alpha_clamp_min = tail_cfg.alpha_clamp_min
        self.alpha_clamp_max = tail_cfg.alpha_clamp_max
        self.fat_tail_threshold = tail_cfg.fat_tail_kurtosis_threshold
        self.severe_kurtosis = tail_cfg.severe_kurtosis
        self.low_tail_slope = tail_cfg.low_tail_slope
        self.severe_cvar = tail_cfg.severe_cvar

        logger.info(f"Initialized TailRiskAnalyzer: window={window}, confidence={confidence}")

    def update(self, ret: float) -> Optional[TailRiskMetrics]:
        # la calculette
        # le bif
        if ret is None or not np.isfinite(ret):
            return None

        self.returns.append(ret)
        if len(self.returns) > self.window:
            self.returns = self.returns[-self.window:]

        if len(self.returns) < self.min_points:
            return None

        return self.compute()

    def compute(self) -> Optional[TailRiskMetrics]:
        # attention aux degats
        # la calculette
        if len(self.returns) < self.min_points:
            return None

        arr = self._clean_returns(np.array(self.returns, dtype=float))
        if arr.size < self.min_points:
            return None

        # Historical VaR
        var_pct = (1 - self.confidence) * 100
        var = float(np.nanpercentile(arr, var_pct))

        # CVaR (mean of returns below VaR)
        tail_returns = arr[arr <= var]
        cvar = float(np.nanmean(tail_returns)) if tail_returns.size > 0 else var

        # Excess kurtosis
        std = np.nanstd(arr)
        if std > 1e-10:
            kurt = float(np.nanmean(((arr - np.nanmean(arr)) / std) ** 4) - 3.0)
        else:
            kurt = 0.0

        # Hill estimator for tail slope
        tail_slope = self._estimate_tail_slope(arr)

        return TailRiskMetrics(
            var=var,
            cvar=cvar,
            tail_slope=tail_slope,
            kurtosis=kurt,
            confidence=self.confidence,
        )

    def _estimate_tail_slope(self, arr: np.ndarray) -> float:
        # on cherche les pepites
        # le bif
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        k = max(self.min_tail_points, int(n * self.tail_fraction))

        tail = sorted_arr[:k]
        tail = -tail  # Flip sign for left tail

        # FIX: validate tail values before log
        if np.min(tail) <= 0:
            return self.default_tail_slope

        log_tail = np.log(tail)
        log_min = np.log(tail[-1])
        denom = np.sum(log_tail - log_min)
        if denom <= 0:
            return self.default_tail_slope

        alpha = k / denom
        return max(self.alpha_clamp_min, min(alpha, self.alpha_clamp_max))

    def is_fat_tail(self) -> bool:
        # l'usine a gaz
        metrics = self.compute()
        if metrics is None:
            return False
        return metrics.kurtosis > self.fat_tail_threshold

    def get_tail_warning(self) -> Optional[str]:
        # attention aux degats
        metrics = self.compute()
        if metrics is None:
            return None

        warnings = []
        if metrics.kurtosis > self.severe_kurtosis:
            warnings.append(f"High kurtosis: {metrics.kurtosis:.2f}")
        if metrics.tail_slope < self.low_tail_slope:
            warnings.append(f"Low tail slope: {metrics.tail_slope:.2f}")
        if metrics.cvar < self.severe_cvar:
            warnings.append(f"Severe CVaR: {metrics.cvar:.4f}")

        return " | ".join(warnings) if warnings else None

    # ── Regime-Conditional VaR ──

    def compute_regime_conditional(self, regime_id: int,
                                    regime_returns: Optional[List[float]] = None) -> Optional[TailRiskMetrics]:
        # la tuyauterie de donnees
        # la calculette
        if not hasattr(self, '_regime_returns'):
            self._regime_returns = {}

        if regime_returns is not None:
            self._regime_returns.setdefault(regime_id, []).extend(regime_returns)

        rets = self._regime_returns.get(regime_id, [])
        if len(rets) < self.min_points:
            return self.compute()  # Fallback to unconditional

        arr = self._clean_returns(np.array(rets[-self.window:], dtype=float))
        if arr.size < self.min_points:
            return self.compute()

        var_pct = (1 - self.confidence) * 100
        var = float(np.nanpercentile(arr, var_pct))
        tail_returns = arr[arr <= var]
        cvar = float(np.nanmean(tail_returns)) if tail_returns.size > 0 else var
        std = np.nanstd(arr)
        kurt = float(np.nanmean(((arr - np.nanmean(arr)) / std) ** 4) - 3.0) if std > 1e-10 else 0.0
        tail_slope = self._estimate_tail_slope(arr)

        return TailRiskMetrics(var=var, cvar=cvar, tail_slope=tail_slope,
                               kurtosis=kurt, confidence=self.confidence)

    def update_regime(self, ret: float, regime_id: int) -> None:
        # le bif
        if ret is None or not np.isfinite(ret):
            return
        if not hasattr(self, '_regime_returns'):
            self._regime_returns = {}
        self._regime_returns.setdefault(regime_id, []).append(ret)
        # Cap at 2x window per regime
        if len(self._regime_returns[regime_id]) > self.window * 2:
            self._regime_returns[regime_id] = self._regime_returns[regime_id][-self.window:]

    # ── Gap Risk Modeling ──

    def estimate_gap_risk(self, hours_closed: float = 16.0,
                          annual_vol: float = 0.16) -> dict:
        # attention aux degats
        # la calculette
        trading_hours_per_day = 6.5
        trading_days_per_year = 252

        # Intraday vol per hour
        hourly_vol = annual_vol / np.sqrt(trading_days_per_year * trading_hours_per_day)

        # Gap vol = hourly_vol * sqrt(hours_closed) * gap_factor
        # Gap factor accounts for information arrival during close (typically 0.5-0.8)
        cfg = get_config()
        gap_factor = cfg.thresholds.tail_risk.gap_risk_factor
        gap_vol = hourly_vol * np.sqrt(hours_closed) * gap_factor

        # Gap VaR at confidence level
        from scipy.stats import norm
        z = norm.ppf(1 - self.confidence)
        gap_var = float(z * gap_vol)
        gap_cvar = float(z * gap_vol * 1.3)  # CVaR ≈ 1.3x VaR for normal

        return {
            "gap_vol": float(gap_vol),
            "gap_var": gap_var,
            "gap_cvar": gap_cvar,
            "hours_closed": hours_closed,
            "gap_factor": gap_factor,
        }

    # ── Historical Scenario Replay ──

    @staticmethod
    def get_historical_scenarios() -> dict:
        # verif rapide
        # ce qu'on a dans le sac
        return {
            "covid_crash_2020": {
                "description": "COVID-19 market crash (March 2020)",
                "worst_day_return": -0.1198,   # March 16, 2020 SPX
                "week_return": -0.1453,
                "month_return": -0.3389,
            },
            "gfc_2008": {
                "description": "Global Financial Crisis (Oct 2008)",
                "worst_day_return": -0.0947,   # Oct 15, 2008
                "week_return": -0.1815,
                "month_return": -0.1694,
            },
            "flash_crash_2010": {
                "description": "Flash Crash (May 6, 2010)",
                "worst_day_return": -0.0347,
                "week_return": -0.0637,
                "month_return": -0.0821,
            },
            "volmageddon_2018": {
                "description": "VIX spike / short-vol blowup (Feb 2018)",
                "worst_day_return": -0.0410,
                "week_return": -0.0539,
                "month_return": -0.0369,
            },
            "rate_shock": {
                "description": "Hypothetical sudden rate shock (+100bps)",
                "worst_day_return": -0.0500,
                "week_return": -0.0800,
                "month_return": -0.1200,
            },
            "liquidity_crisis": {
                "description": "Hypothetical liquidity evaporation",
                "worst_day_return": -0.0700,
                "week_return": -0.1000,
                "month_return": -0.1500,
            },
        }

    def stress_test_position(self, position_value: float,
                             scenarios: Optional[dict] = None) -> dict:
        # verif rapide
        # ce qu'on a dans le sac
        if scenarios is None:
            scenarios = self.get_historical_scenarios()

        results = {}
        for name, scenario in scenarios.items():
            worst_day = scenario["worst_day_return"]
            week = scenario["week_return"]

            # For long positions, negative returns = loss
            # For short positions, negative returns = gain
            day_pnl = position_value * worst_day
            week_pnl = position_value * week

            results[name] = {
                "description": scenario["description"],
                "worst_day_pnl": float(day_pnl),
                "week_pnl": float(week_pnl),
                "survival": abs(day_pnl) < abs(position_value) * 0.5,  # Survive if <50% loss
            }

        return results

    def reset(self) -> None:
        # dans quel etat j'erre
        self.returns.clear()
        if hasattr(self, '_regime_returns'):
            self._regime_returns.clear()
        logger.info("TailRiskAnalyzer reset")

    @staticmethod
    def _clean_returns(arr: np.ndarray) -> np.ndarray:
        # le bif
        return arr[np.isfinite(arr)]


# --- Rust backend: fast compute_tail_risk + RustTailRiskTracker ---
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    _rc = get_rust_core()
    compute_tail_risk_rust = _rc.compute_tail_risk
    RustTailRiskTracker = _rc.RustTailRiskTracker
    logger.info("TailRisk Rust accelerators available")
