# la grosse machine
# on passe a la caisse

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImpactEstimate:
    # verif rapide
    # on passe a la caisse
    temporary_impact: float     # Price impact during execution (reverts partially)
    permanent_impact: float     # Lasting price impact
    total_impact_bps: float     # Total impact in basis points
    cost_estimate: float        # Estimated dollar cost of impact
    is_acceptable: bool         # Whether impact is within tolerance


class MarketImpactModel:
    """Almgren-Chriss market impact model with calibration tracking."""

    def __init__(
        self,
        temporary_coeff: float = 0.5,
        permanent_coeff: float = 0.1,
        max_impact_bps: float = 50.0,
        is_calibrated: bool = False,
    ):
        if temporary_coeff < 0 or permanent_coeff < 0:
            raise ValueError("Impact coefficients must be non-negative")
        if max_impact_bps <= 0:
            raise ValueError("max_impact_bps must be positive")

        self.temporary_coeff = temporary_coeff
        self.permanent_coeff = permanent_coeff
        self.max_impact_bps = max_impact_bps
        self.is_calibrated = is_calibrated
        _cal_tag = "CALIBRATED" if is_calibrated else "UNCALIBRATED (conservative priors)"
        logger.info(
            f"Initialized MarketImpactModel [{_cal_tag}]: temp_k={temporary_coeff}, "
            f"perm_k={permanent_coeff}, max_bps={max_impact_bps}"
        )

    def estimate(
        self,
        order_qty: float,
        price: float,
        volatility: float,
        avg_volume: float = 1000.0,
    ) -> ImpactEstimate:
        # verif rapide
        # on passe a la caisse
        if price <= 0 or not np.isfinite(price):
            logger.warning(f"Invalid price for impact estimate: {price}")
            return ImpactEstimate(0.0, 0.0, 0.0, 0.0, is_acceptable=True)

        qty = abs(order_qty)
        if qty == 0:
            return ImpactEstimate(0.0, 0.0, 0.0, 0.0, is_acceptable=True)

        vol = max(volatility, 1e-8)
        adv = max(avg_volume, 1.0)

        # Participation rate
        participation = qty / adv

        # Square-root model: impact proportional to sqrt(participation)
        sqrt_participation = np.sqrt(participation)

        # Temporary impact (price pressure during execution, partially reverts)
        temp_impact = self.temporary_coeff * vol * sqrt_participation

        # Permanent impact (information content of the trade)
        perm_impact = self.permanent_coeff * vol * participation

        # Total in basis points
        total_bps = (temp_impact + perm_impact) * 10000

        # Dollar cost estimate
        cost = (temp_impact + perm_impact) * price * qty

        acceptable = total_bps <= self.max_impact_bps

        if not acceptable:
            _cal = "UNCALIBRATED" if not self.is_calibrated else ""
            logger.warning(
                f"High impact estimate {_cal}: {total_bps:.1f} bps "
                f"(qty={qty:.2f}, vol={vol:.6f}, adv={adv:.0f})"
            )

        return ImpactEstimate(
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_impact_bps=total_bps,
            cost_estimate=cost,
            is_acceptable=acceptable,
        )

    def calibrate_from_fills(
        self,
        fill_prices: np.ndarray,
        arrival_prices: np.ndarray,
        fill_quantities: np.ndarray,
        avg_volumes: np.ndarray,
        volatilities: np.ndarray,
        min_fills: int = 200,
    ) -> None:
        """Calibrate impact coefficients from historical fill data.

        Accepts raw arrays or a fills dict with keys:
        fill_prices, arrival_prices, fill_quantities, avg_volumes, volatilities.

        Requires >= min_fills (default 200). If fewer: keep conservative priors,
        is_calibrated stays False.
        """
        n = len(fill_prices)
        if n < min_fills:
            logger.warning(
                "Insufficient fills for calibration (%d < %d) — keeping conservative priors",
                n, min_fills,
            )
            self.is_calibrated = False
            return

        participations = fill_quantities / np.maximum(avg_volumes, 1.0)
        realized_impacts = (fill_prices - arrival_prices) / np.maximum(arrival_prices, 1e-8)
        vols = np.maximum(volatilities, 1e-8)
        sqrt_parts = np.sqrt(np.maximum(participations, 1e-10))

        # OLS: implied_eta = |realized_impact| / (sigma * sqrt(participation))
        implied_eta = np.abs(realized_impacts) / (vols * sqrt_parts)
        valid = np.isfinite(implied_eta) & (implied_eta > 0) & (implied_eta < 10.0)

        if valid.sum() < 50:
            logger.warning("Too few valid implied-eta samples (%d) — keeping conservative priors", valid.sum())
            self.is_calibrated = False
            return

        # Fit via OLS: Y = eta * X, where Y = |impact|, X = sigma * sqrt(part)
        Y = np.abs(realized_impacts[valid])
        X = (vols * sqrt_parts)[valid]
        # OLS eta = X'Y / X'X
        eta_ols = float(np.dot(X, Y) / np.dot(X, X)) if np.dot(X, X) > 1e-20 else float(np.median(implied_eta[valid]))
        eta_ols = max(0.01, min(5.0, eta_ols))

        self.temporary_coeff = eta_ols
        self.permanent_coeff = eta_ols * 0.5  # Almgren-Chriss prior ratio
        self.is_calibrated = True
        logger.info(
            "Impact model CALIBRATED from %d fills (OLS): "
            "temp_k=%.4f, perm_k=%.4f",
            valid.sum(), self.temporary_coeff, self.permanent_coeff,
        )

    @classmethod
    def from_fills_dict(cls, fills: dict, **kwargs) -> 'MarketImpactModel':
        """Convenience: calibrate_from_fills(fills_db.load(lookback_days=90)).

        Expects dict with keys: fill_prices, arrival_prices, fill_quantities,
        avg_volumes, volatilities (all np.ndarray).
        """
        model = cls(**kwargs)
        model.calibrate_from_fills(
            fill_prices=np.asarray(fills['fill_prices']),
            arrival_prices=np.asarray(fills['arrival_prices']),
            fill_quantities=np.asarray(fills['fill_quantities']),
            avg_volumes=np.asarray(fills['avg_volumes']),
            volatilities=np.asarray(fills['volatilities']),
        )
        return model
