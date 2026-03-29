# l'usine a gaz

from regime.state_vector import StateVector
from regime.transition import TransitionEvent
from core.types import TradeAction
from typing import Tuple
from config.loader import get_config


class ExitConditions:
    # le feu vert
    # ce qu'on a dans le sac

    def __init__(self, config=None):
        self.config = config or get_config()
        self.kl_stable = self.config.thresholds.decision.exit.kl_stable_threshold
        # Track consecutive ticks of elevated entropy for persistence check
        self._entropy_vel_streak: int = 0
        self._entropy_persistence_ticks: int = 2  # exit after 2+ consecutive ticks of entropy increase

    def check_exit(
        self,
        position_side: TradeAction,
        transition: TransitionEvent,
        state: StateVector,
    ) -> Tuple[bool, str]:
        # ce qu'on a dans le sac
        # le bif
        if transition is None:
            return False, ""

        # 1. Information gradient collapse (dI/dt < 0)
        if transition.kl_divergence < self.kl_stable:
            return True, (
                f"Edge Decay: Information Gradient Collapsed "
                f"(KL {transition.kl_divergence:.4f} < {self.kl_stable})"
            )

        # 2. Second-order dynamics reversal
        #    Entry is now on DECELERATION, so exit when RE-ACCELERATION starts
        #    (the mean-reversion has played out and a new impulse is forming)
        mu_vel = transition.mu_velocity
        mu_acc = transition.mu_acceleration

        if position_side == TradeAction.BUY:
            # Bought on downward deceleration (vel < 0, acc > 0).
            # Exit if price re-accelerates downward (acc flips negative again)
            if mu_vel < 0 and mu_acc < 0:
                return True, f"Edge Decay: Re-acceleration down (Vel {mu_vel:.6f}, Acc {mu_acc:.6f})"

        elif position_side == TradeAction.SELL:
            # Sold on upward deceleration (vel > 0, acc < 0).
            # Exit if price re-accelerates upward (acc flips positive again)
            if mu_vel > 0 and mu_acc > 0:
                return True, f"Edge Decay: Re-acceleration up (Vel {mu_vel:.6f}, Acc {mu_acc:.6f})"

        # 3. Entropy increasing = growing disorder (with persistence check)
        #    Only exit after 2+ consecutive ticks of elevated entropy velocity
        if transition.entropy_velocity > 0:
            self._entropy_vel_streak += 1
        else:
            self._entropy_vel_streak = 0

        if self._entropy_vel_streak >= self._entropy_persistence_ticks:
            return True, (
                f"Edge Decay: Entropy Increasing for {self._entropy_vel_streak} ticks "
                f"(Vel {transition.entropy_velocity:.4f})"
            )

        return False, ""
