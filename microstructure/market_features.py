"""Institutional microstructure features: VPIN, Kyle's lambda, Amihud illiquidity."""
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class VPINEstimator:
    """VPIN via BVC classification with tick-rule fallback."""

    def __init__(self, volume_bucket_size: int = 500, n_buckets: int = 50):
        self._bucket_size = volume_bucket_size
        self._n_buckets = n_buckets
        self._prices: deque = deque(maxlen=10000)
        self._volumes: deque = deque(maxlen=10000)
        self._current_bucket_volume: float = 0.0
        self._bucket_imbalances: deque = deque(maxlen=n_buckets)

    def update(self, price: float, volume: float) -> None:
        self._prices.append(price)
        self._volumes.append(volume)
        self._current_bucket_volume += volume
        if self._current_bucket_volume >= self._bucket_size:
            self._close_bucket()

    def _close_bucket(self) -> None:
        if not self._prices:
            return
        prices = np.array(self._prices)
        volumes = np.array(self._volumes)

        try:
            import pandas as pd
            from mlfinlab.microstructural_features.misc import get_bvc_buy_volume
            buy_vol = get_bvc_buy_volume(
                close=pd.Series(prices), volume=pd.Series(volumes), window=len(prices),
            ).sum()
        except (ImportError, Exception):
            # Fallback: tick rule
            buy_vol = sum(
                v for p, v, pp in zip(prices[1:], volumes[1:], prices[:-1]) if p >= pp
            )

        total_vol = volumes.sum()
        sell_vol = total_vol - buy_vol
        imbalance = abs(buy_vol - sell_vol) / max(total_vol, 1e-10)
        self._bucket_imbalances.append(imbalance)

        self._prices.clear()
        self._volumes.clear()
        self._current_bucket_volume = 0.0

    def estimate(self) -> float:
        if not self._bucket_imbalances:
            return 0.0
        return float(np.mean(self._bucket_imbalances))


class KyleLambdaEstimator:
    """Kyle's lambda: price impact per unit signed order flow."""

    def __init__(self, window: int = 100):
        self._mid_changes: deque = deque(maxlen=window)
        self._signed_volumes: deque = deque(maxlen=window)

    def update(self, mid_change: float, signed_volume: float) -> None:
        self._mid_changes.append(mid_change)
        self._signed_volumes.append(signed_volume)

    def estimate(self) -> float:
        if len(self._signed_volumes) < 20:
            return 0.0
        sv = np.array(self._signed_volumes)
        dm = np.array(self._mid_changes)
        var_sv = np.var(sv)
        if var_sv < 1e-15:
            return 0.0
        return float(np.cov(dm, sv)[0, 1] / var_sv)


class AmihudEstimator:
    """Amihud illiquidity: mean(|return| / volume)."""

    _MIN_VOLUME: float = 1e-8

    def __init__(self, window: int = 200):
        self._ratios: deque = deque(maxlen=window)

    def update(self, abs_return: float, volume: float) -> None:
        if volume > self._MIN_VOLUME:
            self._ratios.append(abs_return / volume)

    def estimate(self) -> float:
        if not self._ratios:
            return 0.0
        return float(np.mean(self._ratios))
