# attention aux degats
# la grosse machine

import numpy as np
from typing import Optional
import logging
from core.types import RiskAdjustments
from regime.state_vector import StateVector
from config.loader import get_config

logger = logging.getLogger(__name__)


class RiskCalibrationModel:
    # attention aux degats
    # la grosse machine

    def __init__(self, config=None):
        self.config = config or get_config()
        cal = self.config.thresholds.calibration
        self.tail_slope_tighten = cal.tail_slope_tighten
        self.kurtosis_tighten = cal.kurtosis_tighten
        self.entropy_reduce = cal.entropy_reduce_threshold
        self.vol_high = cal.volatility_high_threshold
        self.vol_trend_scale = cal.volatility_trend_scale
        logger.info("Initialized RiskCalibrationModel")

    def predict(self, state: StateVector, volatility_trend: float = 0.0,
                is_mean_reverting: bool = False) -> RiskAdjustments:
        # attention aux degats
        # la calculette
        if not np.isfinite(state.sigma) or not np.isfinite(state.entropy):
            logger.warning("Invalid state values, returning neutral adjustments")
            return RiskAdjustments(1.0, 1.0, 1.0, valid=False)

        # 1. Stop multiplier -- regime-aware tightening
        #    High kurtosis in mean-reverting regimes is GOOD (fat tails = bigger reversals)
        #    Only tighten stops in trending/non-reverting regimes
        stop_mult = 1.0
        if state.tail_slope > self.tail_slope_tighten:
            if is_mean_reverting:
                stop_mult *= 0.95  # Gentle tightening — fat tails help mean reversion
            else:
                stop_mult *= 0.8   # Aggressive tightening in trending regimes
        if state.kurtosis > self.kurtosis_tighten:
            if is_mean_reverting:
                pass  # Don't penalize — high kurtosis = stronger reversal edge
            else:
                stop_mult *= 0.9

        # 2. Size scaler -- regime-aware sizing
        size_scaler = 1.0
        if volatility_trend > 0:
            if is_mean_reverting:
                # Rising vol in mean-reversion = opportunity, reduce less
                size_scaler *= (1.0 - min(volatility_trend * self.vol_trend_scale * 0.3, 0.2))
            else:
                size_scaler *= (1.0 - min(volatility_trend * self.vol_trend_scale, 0.5))
        if state.entropy > self.entropy_reduce:
            if is_mean_reverting:
                size_scaler *= 0.85  # Mild reduction — high entropy is expected during reversals
            else:
                size_scaler *= 0.7

        # 3. Max exposure ratio
        max_exposure_ratio = 1.0
        if state.sigma > self.vol_high:
            if is_mean_reverting:
                max_exposure_ratio = 0.75  # Allow more exposure for mean-reversion edge
            else:
                max_exposure_ratio = 0.5

        return RiskAdjustments(
            stop_multiplier=max(0.5, min(stop_mult, 1.5)),
            size_scaler=max(0.1, min(size_scaler, 1.0)),
            max_exposure=max_exposure_ratio,
            valid=True,
        )
