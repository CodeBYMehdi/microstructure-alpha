# la grosse machine
# le feu vert

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from regime.transition import TransitionEvent
from regime.state_vector import StateVector
from core.types import LiquidityState
from config.loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    # l'usine a gaz
    regime_confidence: float    # Clustering distance-based confidence
    transition_strength: float  # ML model transition probability
    information_gradient: float # KL divergence signal
    dynamics_alignment: float   # Second-order dynamics coherence
    liquidity_score: float      # Market liquidity quality
    composite: float            # Final weighted score [0, 1]


class ConfidenceScorer:
    # la calculette
    # le feu vert

    def __init__(
        self,
        regime_weight: float = 0.25,
        transition_weight: float = 0.30,
        info_gradient_weight: float = 0.20,
        dynamics_weight: float = 0.15,
        liquidity_weight: float = 0.10,
        config=None,
    ):
        total = regime_weight + transition_weight + info_gradient_weight + dynamics_weight + liquidity_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")

        self.regime_weight = regime_weight
        self.transition_weight = transition_weight
        self.info_gradient_weight = info_gradient_weight
        self.dynamics_weight = dynamics_weight
        self.liquidity_weight = liquidity_weight

        _cfg = (config or get_config()).thresholds.decision.confidence
        self._da_high = _cfg.dynamics_alignment_high
        self._da_low = _cfg.dynamics_alignment_low
        self._entropy_boost = _cfg.entropy_stable_boost
        self._spread_norm = _cfg.spread_normalizer

    def score(
        self,
        regime_confidence: float,
        transition: Optional[TransitionEvent],
        state: Optional[StateVector] = None,
        liquidity: Optional[LiquidityState] = None,
    ) -> ConfidenceBreakdown:
        # la calculette
        # le feu vert
        # 1. Regime confidence (already [0, 1])
        rc = float(np.clip(regime_confidence, 0.0, 1.0))

        # 2. Transition strength
        ts = 0.0
        if transition is not None and transition.is_significant:
            ts = float(np.clip(transition.strength, 0.0, 1.0))

        # 3. Information gradient (KL divergence, mapped to [0, 1])
        ig = 0.0
        if transition is not None:
            # KL > 0.5 is very strong, use sigmoid-like mapping
            kl = max(0.0, transition.kl_divergence)
            ig = float(np.tanh(kl * 2.0))  # Maps ~0.5 -> 0.76, ~1.0 -> 0.96

        # 4. Dynamics alignment (velocity and acceleration coherence)
        da = 0.0
        if transition is not None:
            mu_vel = transition.mu_velocity
            mu_acc = transition.mu_acceleration
            # Aligned if velocity and acceleration have the same sign
            if mu_vel != 0.0 and mu_acc != 0.0:
                same_sign = np.sign(mu_vel) == np.sign(mu_acc)
                da = self._da_high if same_sign else self._da_low
                if abs(transition.entropy_acceleration) < 0.5:
                    da = min(1.0, da + self._entropy_boost)

        # 5. Liquidity score
        ls = 0.5  # Default moderate
        if liquidity is not None:
            _spread = liquidity.spread if np.isfinite(liquidity.spread) else 0.0
            _depth_imb = liquidity.depth_imbalance if np.isfinite(liquidity.depth_imbalance) else 0.0
            if liquidity.is_liquid and _spread > 0:
                # Lower spread = higher confidence
                spread_score = float(np.clip(1.0 - _spread * self._spread_norm, 0.0, 1.0))
                # Balanced book = higher confidence (clamp imbalance to [0,1])
                imbalance_score = float(np.clip(1.0 - abs(_depth_imb), 0.0, 1.0))
                ls = 0.6 * spread_score + 0.4 * imbalance_score
            elif not liquidity.is_liquid:
                ls = 0.0

        # Composite
        composite = (
            self.regime_weight * rc
            + self.transition_weight * ts
            + self.info_gradient_weight * ig
            + self.dynamics_weight * da
            + self.liquidity_weight * ls
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        return ConfidenceBreakdown(
            regime_confidence=rc,
            transition_strength=ts,
            information_gradient=ig,
            dynamics_alignment=da,
            liquidity_score=ls,
            composite=composite,
        )
