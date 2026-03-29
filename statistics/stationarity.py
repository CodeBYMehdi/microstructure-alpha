# verif rapide
# le bif

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StationarityResult:
    # verif rapide
    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Optional[Dict[str, float]] = None
    n_observations: int = 0
    lags_used: int = 0


class StationarityTester:
    # verif rapide

    def __init__(self, significance: float = 0.05):
        self.significance = significance

    def test_adf(self, series: np.ndarray, max_lags: Optional[int] = None) -> StationarityResult:
        # verif rapide
        series = self._clean(series)
        if len(series) < 20:
            return StationarityResult("ADF", 0.0, 1.0, False, n_observations=len(series))

        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, maxlag=max_lags, autolag='AIC')
            stat, p_value, lags = result[0], result[1], result[2]
            crit = {k: v for k, v in result[4].items()}

            return StationarityResult(
                test_name="ADF",
                statistic=float(stat),
                p_value=float(p_value),
                is_stationary=p_value < self.significance,
                critical_values=crit,
                n_observations=len(series),
                lags_used=int(lags),
            )
        except ImportError:
            # Fallback: simple variance ratio test
            return self._simple_stationarity_test(series, "ADF_fallback")
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return StationarityResult("ADF", 0.0, 1.0, False, n_observations=len(series))

    def test_kpss(self, series: np.ndarray) -> StationarityResult:
        # verif rapide
        series = self._clean(series)
        if len(series) < 20:
            return StationarityResult("KPSS", 0.0, 1.0, True, n_observations=len(series))

        try:
            from statsmodels.tsa.stattools import kpss
            stat, p_value, lags, crit = kpss(series, regression='c', nlags='auto')

            return StationarityResult(
                test_name="KPSS",
                statistic=float(stat),
                p_value=float(p_value),
                is_stationary=p_value >= self.significance,  # Fail to reject H0 => stationary
                critical_values={str(k): float(v) for k, v in crit.items()},
                n_observations=len(series),
                lags_used=int(lags),
            )
        except ImportError:
            return self._simple_stationarity_test(series, "KPSS_fallback")
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            return StationarityResult("KPSS", 0.0, 0.5, True, n_observations=len(series))

    def test_both(self, series: np.ndarray) -> Dict[str, StationarityResult]:
        # verif rapide
        adf = self.test_adf(series)
        kpss = self.test_kpss(series)

        return {
            "adf": adf,
            "kpss": kpss,
            "joint_conclusion": self._joint_conclusion(adf, kpss),
        }

    def test_regime_stationarity(
        self, returns: np.ndarray, labels: np.ndarray,
    ) -> Dict[int, Dict[str, StationarityResult]]:
        # verif rapide
        # le bif
        results = {}
        unique_regimes = np.unique(labels)

        for regime_id in unique_regimes:
            if regime_id == -1:  # Skip noise
                continue

            mask = labels == regime_id
            regime_returns = returns[mask]

            if len(regime_returns) < 20:
                continue

            results[int(regime_id)] = self.test_both(regime_returns)

        return results

    def _simple_stationarity_test(self, series: np.ndarray, name: str) -> StationarityResult:
        # verif rapide
        # la grosse machine
        n = len(series)
        half = n // 2

        var_first = np.var(series[:half])
        var_second = np.var(series[half:])
        mean_first = np.mean(series[:half])
        mean_second = np.mean(series[half:])

        # Check if mean and variance are roughly constant
        if var_first > 0:
            var_ratio = var_second / var_first
        else:
            var_ratio = 1.0

        mean_diff = abs(mean_second - mean_first)
        overall_std = np.std(series)
        mean_z = mean_diff / max(overall_std, 1e-10)

        # Rough: stationary if variance ratio near 1 and means similar
        is_stationary = (0.5 < var_ratio < 2.0) and (mean_z < 2.0)

        return StationarityResult(
            test_name=name,
            statistic=float(var_ratio),
            p_value=0.05 if is_stationary else 0.5,
            is_stationary=is_stationary,
            n_observations=n,
        )

    @staticmethod
    def _joint_conclusion(adf: StationarityResult, kpss: StationarityResult) -> str:
        if adf.is_stationary and kpss.is_stationary:
            return "STATIONARY (both tests agree)"
        elif not adf.is_stationary and not kpss.is_stationary:
            return "NON-STATIONARY (both tests agree)"
        elif adf.is_stationary and not kpss.is_stationary:
            return "TREND-STATIONARY (ADF rejects unit root, KPSS rejects level stationarity)"
        else:
            return "INCONCLUSIVE (ADF fails to reject unit root, KPSS fails to reject stationarity)"

    @staticmethod
    def _clean(series: np.ndarray) -> np.ndarray:
        # l'usine a gaz
        return series[np.isfinite(series)]
