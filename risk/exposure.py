# les thunes
# ce qu'on a dans le sac

import numpy as np
from typing import Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Epsilon for float comparison (replaces Decimal precision)
_EPS = 0.01


@dataclass
class ExposureState:
    # les thunes
    # dans quel etat j'erre
    total_notional: float   # Gross notional
    net_exposure: float     # Long - Short
    gross_exposure: float   # Long + Short
    exposure_ratio: float   # Gross / Capital
    by_symbol: Dict[str, float]


class ExposureTracker:
    # ce qu'on a dans le sac

    def __init__(self, capital: float = 100000.0, max_ratio: float = 2.0):
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if max_ratio <= 0:
            raise ValueError("max_ratio must be positive")
        self.capital: float = capital
        self.max_ratio: float = max_ratio
        self.positions: Dict[str, float] = {}  # symbol -> signed notional
        logger.info(f"Initialized ExposureTracker: capital={capital}, max_ratio={max_ratio}")

    def update(self, symbol: str, qty: float, price: float, is_buy: bool) -> ExposureState:
        if qty == 0 or price == 0:
            return self.get_state()

        notional = abs(qty) * abs(price)
        signed = notional if is_buy else -notional

        current = self.positions.get(symbol, 0.0)
        new_val = current + signed

        if abs(new_val) < _EPS:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_val

        return self.get_state()

    def get_state(self) -> ExposureState:
        if not self.positions:
            return ExposureState(
                total_notional=0.0, net_exposure=0.0,
                gross_exposure=0.0, exposure_ratio=0.0, by_symbol={},
            )

        values = np.fromiter(self.positions.values(), dtype=np.float64, count=len(self.positions))
        long_expo = float(np.sum(values[values > 0]))
        short_expo = float(np.abs(np.sum(values[values < 0])))

        net = long_expo - short_expo
        gross = long_expo + short_expo
        ratio = gross / self.capital if self.capital > 0 else 0.0

        return ExposureState(
            total_notional=gross,
            net_exposure=net,
            gross_exposure=gross,
            exposure_ratio=ratio,
            by_symbol=dict(self.positions),
        )

    def can_add(self, symbol: str, qty: float, price: float, is_buy: bool = True) -> bool:
        if qty == 0 or price == 0:
            return True

        notional = abs(qty) * abs(price)
        signed = notional if is_buy else -notional

        current = self.positions.get(symbol, 0.0)
        new_val = current + signed

        # Simulate the position change
        sim_positions = dict(self.positions)
        if abs(new_val) < _EPS:
            sim_positions.pop(symbol, None)
        else:
            sim_positions[symbol] = new_val

        if not sim_positions:
            return True

        values = np.fromiter(sim_positions.values(), dtype=np.float64, count=len(sim_positions))
        new_gross = float(np.sum(np.abs(values[values > 0]))) + float(np.abs(np.sum(values[values < 0])))

        return (new_gross / self.capital) <= self.max_ratio

    def get_available_notional(self) -> float:
        current_gross = self.get_state().gross_exposure
        max_allowed = self.capital * self.max_ratio
        return max(0.0, max_allowed - current_gross)

    def reset(self) -> None:
        self.positions.clear()
        logger.info("ExposureTracker reset")
