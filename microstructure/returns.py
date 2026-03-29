# le flux en scred
# le bif

import numpy as np
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ReturnCalculator:
    # le flux en scred
    # la calculette

    def __init__(self, max_window_size: int = 5000):
        if max_window_size <= 0:
            raise ValueError("max_window_size must be positive")
        self.max_window_size = max_window_size
        # FIX: Removed unused _prices deque
        self._returns: deque = deque(maxlen=max_window_size)
        self._last_price: Optional[float] = None

    def update(self, price: float) -> Optional[float]:
        # la calculette
        # le bif
        if not np.isfinite(price) or price <= 0:
            logger.error(f"Invalid price received: {price}")
            raise ValueError(f"Price must be positive and finite, got {price}")

        if self._last_price is None:
            self._last_price = price
            return None

        try:
            ret = np.log(price / self._last_price)
        except (ValueError, FloatingPointError) as e:
            logger.error(f"Error calculating log return for price {price}, last {self._last_price}: {e}")
            raise

        if not np.isfinite(ret):
            logger.warning(f"Infinite return calculated: price={price}, last={self._last_price}")
            ret = 0.0

        self._returns.append(ret)
        self._last_price = price

        return ret

    def get_window(self, size: int) -> np.ndarray:
        # aie aie aie
        # le bif
        if size <= 0:
            raise ValueError("Window size must be positive")
        if size > len(self._returns):
            raise ValueError(
                f"Requested window size {size} larger than available history {len(self._returns)}"
            )

        # Efficient tail extraction: only convert the needed slice to numpy
        from itertools import islice
        start = len(self._returns) - size
        return np.fromiter(islice(self._returns, start, None), dtype=float, count=size)

    @property
    def count(self) -> int:
        # le bif
        return len(self._returns)

    def reset(self) -> None:
        # dans quel etat j'erre
        self._returns.clear()
        self._last_price = None


# --- Rust backend swap ---
from core.backend import _USE_RUST_CORE, get_rust_core
if _USE_RUST_CORE:
    _PyReturnCalculator = ReturnCalculator
    ReturnCalculator = get_rust_core().ReturnCalculator
    logger.info("ReturnCalculator → Rust backend")
