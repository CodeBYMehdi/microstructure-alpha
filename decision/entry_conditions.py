# entrainement intensif
# on passe a la caisse

import numpy as np
from regime.state_vector import StateVector
from regime.transition import TransitionEvent
from core.types import TradeAction, LiquidityState
from typing import Optional, Tuple, Dict
import logging
from config.loader import get_config

logger = logging.getLogger(__name__)


class EntryConditions:
    # l'usine a gaz

    def __init__(self, config=None):
        self.config = config or get_config()
        self.long_thresholds = self.config.thresholds.decision.long
        self.short_thresholds = self.config.thresholds.decision.short
        self.liquidity_config = self.config.thresholds.decision.liquidity
        self.regime_config = self.config.thresholds.regime
        self.exit_config = self.config.thresholds.decision.exit
        self.entry_gate = self.config.thresholds.decision.entry_gate

    def evaluate(
        self,
        transition: TransitionEvent,
        state: StateVector,
        liquidity: Optional[LiquidityState] = None,
        regime_stats: Optional[Dict[str, StateVector]] = None,
        autocorrelation: float = 0.0,
    ) -> Tuple[Optional[TradeAction], str]:
        # Input validation
        if transition is None:
            return None, "No transition event"

        if transition.delta_vector is None or len(transition.delta_vector) < 6:
            return None, "Invalid delta vector (expected 6 dimensions)"

        # 0. Gating: Information gradient & projection
        if transition.kl_divergence < self.regime_config.kl_min:
            return None, (
                f"Insufficient Information Gradient "
                f"(KL {transition.kl_divergence:.4f} < {self.regime_config.kl_min})"
            )

        if transition.projection_magnitude < self.regime_config.projection_min:
            return None, (
                f"Weak Regime Projection "
                f"({transition.projection_magnitude:.2f} < {self.regime_config.projection_min})"
            )

        # 1. Liquidity conditions
        if liquidity:
            if not liquidity.is_liquid:
                return None, "Liquidity Check Failed: Market Illiquid"
            if liquidity.spread > self.liquidity_config.spread_max:
                return None, (
                    f"Liquidity Check Failed: Spread "
                    f"{liquidity.spread:.5f} > {self.liquidity_config.spread_max}"
                )

        # Determine regime type from autocorrelation (thresholds from config)
        _ac_thresh = self.entry_gate.autocorr_trending_threshold
        is_trending = autocorrelation > _ac_thresh
        is_reverting = autocorrelation < -_ac_thresh

        # Extract second-order dynamics
        mu_vel = transition.mu_velocity
        mu_acc = transition.mu_acceleration
        ent_acc = transition.entropy_acceleration

        d_mu = transition.delta_vector[0]
        d_skew = transition.delta_vector[2]

        alignment = False
        if d_mu != 0 and d_skew != 0:
            alignment = np.sign(d_mu) == np.sign(d_skew)

        ent_acc_threshold = self.exit_config.entropy_acceleration_threshold

        # Entropy acceleration: block at extreme values only
        ent_acc_extreme = abs(ent_acc) > ent_acc_threshold * self.entry_gate.entropy_acc_extreme_mult

        # --- REGIME-ADAPTIVE SIGNAL LOGIC ---

        if is_trending:
            # TRENDING REGIME: Go WITH the trend on deceleration (pullback entry)
            if mu_vel > 0 and (mu_acc < 0 or abs(mu_acc) < abs(mu_vel) * 0.5):
                if not ent_acc_extreme:
                    return TradeAction.BUY, (
                        f"Trend Pullback BUY: +Vel (uptrend), decel (pullback), "
                        f"autocorr={autocorrelation:+.3f}. KL={transition.kl_divergence:.2f}"
                    )

            if mu_vel < 0 and (mu_acc > 0 or abs(mu_acc) < abs(mu_vel) * 0.5):
                if not ent_acc_extreme:
                    return TradeAction.SELL, (
                        f"Trend Pullback SELL: -Vel (downtrend), decel (bounce), "
                        f"autocorr={autocorrelation:+.3f}. KL={transition.kl_divergence:.2f}"
                    )

        else:
            # MEAN-REVERTING or NEUTRAL REGIME: Fade the move
            if mu_vel > 0 and (mu_acc < 0 or abs(mu_acc) < abs(mu_vel) * 0.5):
                if not ent_acc_extreme:
                    detail = "reverting" if is_reverting else "neutral"
                    return TradeAction.SELL, (
                        f"Mean Revert SELL: +Vel, decel, {detail}, "
                        f"autocorr={autocorrelation:+.3f}. KL={transition.kl_divergence:.2f}"
                    )

            if mu_vel < 0 and (mu_acc > 0 or abs(mu_acc) < abs(mu_vel) * 0.5):
                if not ent_acc_extreme:
                    detail = "reverting" if is_reverting else "neutral"
                    return TradeAction.BUY, (
                        f"Mean Revert BUY: -Vel, decel, {detail}, "
                        f"autocorr={autocorrelation:+.3f}. KL={transition.kl_divergence:.2f}"
                    )

        # FALLBACK — strength-based entry (regime-aware direction)
        fallback_threshold = self.exit_config.fallback_strength_threshold
        if transition.strength >= fallback_threshold:
            flow_imbalance = liquidity.depth_imbalance if liquidity else 0.0
            _fi_thresh = self.entry_gate.flow_imbalance_threshold
            flow_confirms_sell = (liquidity is not None) and (flow_imbalance < _fi_thresh)
            flow_confirms_buy = (liquidity is not None) and (flow_imbalance > -_fi_thresh)

            if is_trending:
                # Trending fallback: go WITH velocity
                if mu_vel > 0 and flow_confirms_buy:
                    return TradeAction.BUY, (
                        f"Fallback Trend BUY: +Vel, trending, "
                        f"Strength={transition.strength:.3f}, autocorr={autocorrelation:+.3f}"
                    )
                elif mu_vel < 0 and flow_confirms_sell:
                    return TradeAction.SELL, (
                        f"Fallback Trend SELL: -Vel, trending, "
                        f"Strength={transition.strength:.3f}, autocorr={autocorrelation:+.3f}"
                    )
            else:
                # Reverting/neutral fallback: fade velocity (mean-revert)
                if mu_vel > 0 and flow_confirms_sell:
                    return TradeAction.SELL, (
                        f"Fallback Mean Revert SELL: +Vel, "
                        f"Strength={transition.strength:.3f}, autocorr={autocorrelation:+.3f}"
                    )
                elif mu_vel < 0 and flow_confirms_buy:
                    return TradeAction.BUY, (
                        f"Fallback Mean Revert BUY: -Vel, "
                        f"Strength={transition.strength:.3f}, autocorr={autocorrelation:+.3f}"
                    )

            if mu_vel != 0:
                return None, (
                    f"Fallback BLOCKED by flow: Vel={mu_vel:+.8f}, "
                    f"flow={flow_imbalance:+.3f} opposes direction"
                )

        return None, "No directional edge identified (Second-Order constraints not met)"
