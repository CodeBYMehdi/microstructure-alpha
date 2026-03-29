# la grosse machine
# execution sans pitie

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    # l'usine a gaz
    spread_cost: float      # Half-spread crossing cost
    impact_cost: float      # Size-dependent market impact
    noise_component: float  # Random execution noise (can be positive or negative)
    total_slippage: float   # Net signed slippage (positive = adverse for trader)
    execution_price: float  # Final estimated execution price


class SlippageModel:
    """Regime-conditional slippage model with Almgren-Chriss impact."""

    def __init__(
        self,
        base_spread_bps: float = 1.0,
        impact_coeff: float = 0.00001,
        noise_std_bps: float = 1.0,
        seed: Optional[int] = None,
        regime_slippage_config=None,
    ):
        self.base_spread_bps = base_spread_bps
        self.impact_coeff = impact_coeff
        self.noise_std_bps = noise_std_bps
        self.rng = np.random.RandomState(seed)
        self._regime_cfg = regime_slippage_config
        logger.info(
            f"Initialized SlippageModel: spread={base_spread_bps}bps, "
            f"impact_k={impact_coeff}, noise={noise_std_bps}bps"
        )

    def get_regime_multiplier(self, volatility: float) -> float:
        """Regime-conditional slippage multiplier based on current volatility."""
        if self._regime_cfg is None:
            return max(1.0, volatility * 1000)
        if volatility >= self._regime_cfg.vol_threshold_crisis:
            return self._regime_cfg.crisis_mult
        elif volatility >= self._regime_cfg.vol_threshold_high:
            return self._regime_cfg.high_vol_mult
        elif volatility >= self._regime_cfg.vol_threshold_high * 0.5:
            return self._regime_cfg.normal_mult
        else:
            return self._regime_cfg.low_vol_mult

    def estimate(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        volatility: float = 0.0001,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> SlippageEstimate:
        # on passe a la caisse
        # execution sans pitie
        if price <= 0 or not np.isfinite(price):
            return SlippageEstimate(0.0, 0.0, 0.0, 0.0, price)

        qty = abs(quantity)
        if qty == 0:
            return SlippageEstimate(0.0, 0.0, 0.0, 0.0, price)

        # 1. Spread cost
        if bid is not None and ask is not None and ask > bid:
            half_spread = (ask - bid) / 2.0
        else:
            half_spread = price * self.base_spread_bps / 10000.0

        # 2. Market impact (square-root model)
        impact = self.impact_coeff * np.sqrt(qty)

        # 3. Volatility-scaled noise — regime-conditional multiplier
        vol_penalty = self.get_regime_multiplier(volatility)
        noise = self.rng.normal(0, self.noise_std_bps / 10000.0 * vol_penalty)

        # Total slippage (directional)
        if is_buy:
            # Buy pushes price up
            directional_slip = half_spread + impact + noise
            exec_price = price + directional_slip
        else:
            # Sell pushes price down
            directional_slip = half_spread + impact - noise
            exec_price = price - directional_slip

        # Signed slippage: positive = adverse for trader (bought higher / sold lower)
        signed_slippage = abs(exec_price - price)

        return SlippageEstimate(
            spread_cost=half_spread,
            impact_cost=impact,
            noise_component=noise,
            total_slippage=signed_slippage,
            execution_price=exec_price,
        )

    def estimate_round_trip(
        self,
        price: float,
        quantity: float,
        volatility: float = 0.0001,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        regime_vol_mult: float = 1.0,
    ) -> float:
        """Estimate round-trip cost as fraction of price.

        Returns total entry + exit cost in price-fraction units.
        """
        if price <= 0 or quantity <= 0:
            return 0.0
        buy_est = self.estimate(price, quantity, is_buy=True, volatility=volatility, bid=bid, ask=ask)
        sell_est = self.estimate(price, quantity, is_buy=False, volatility=volatility, bid=bid, ask=ask)
        # Round-trip cost = entry slippage + exit slippage, as fraction of price
        rt_cost = (buy_est.total_slippage + sell_est.total_slippage) / price
        return rt_cost * regime_vol_mult
