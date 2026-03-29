# l'usine a gaz

from dataclasses import dataclass
from typing import Optional
from regime.transition import TransitionEvent
from regime.state_vector import StateVector
import logging

logger = logging.getLogger(__name__)


@dataclass
class EligibilityResult:
    # l'usine a gaz
    is_eligible: bool
    reason: str


class TradeEligibility:
    # l'usine a gaz

    def __init__(self, config=None):
        self.config = config

    def check(
        self,
        transition: Optional[TransitionEvent],
        current_state: StateVector,
        risk_status: bool,
    ) -> EligibilityResult:
        # attention aux degats
        # le bif
        # Gate 1: Transition exists
        if transition is None:
            return EligibilityResult(False, "No regime transition detected")

        # Gate 2: Transition strength (removed is_significant check — caller
        # main.py:479 already gates on this; keeping it here created a
        # redundant triple-gate that blocked trades unnecessarily)

        # Gate 3: Risk status
        if not risk_status:
            return EligibilityResult(False, "Risk engine invalidation")

        # Gate 4: State validity
        if current_state.sigma <= 0 or current_state.entropy == -float('inf'):
            logger.error(f"Invalid microstructure state detected: {current_state}")
            return EligibilityResult(False, "Invalid microstructure state")

        return EligibilityResult(True, "Transition valid and risk cleared")
